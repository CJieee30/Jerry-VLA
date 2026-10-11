"""Keep SmolVLA VLM K/V, drop the layer's residual update.

SmolVLMWithExpertModel does not call HuggingFace DecoderLayer.forward. A
dropped VLM layer matches π0.5 Appendix B: input LN, K/V, RoPE, and the cache
write. Q, attention, o_proj, post-LN, and the MLP do not run on that stream.
The action expert at the same index is never dropped. When the expert is in
the same forward (training loss), it still attends to those K/V.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch
from transformers.cache_utils import DynamicCache

from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel, apply_rope

_DROP: set[int] = set()
_ORIGINAL = SmolVLMWithExpertModel.forward


def _proj_kv(layer, hidden_states, position_ids):
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = hidden_states.to(dtype=layer.self_attn.k_proj.weight.dtype)
    hidden_shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)
    if position_ids is not None and position_ids.shape[1] != key_state.shape[1]:
        position_ids = position_ids[:, : key_state.shape[1]]
    key_state = apply_rope(key_state, position_ids)
    return key_state, value_state


def _proj_qkv(layer, hidden_states):
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
    hidden_shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
    query = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
    key = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
    value = layer.self_attn.v_proj(hidden_states).view(hidden_shape)
    return query, key, value


def _expert_out(layer, hidden_states, att_output):
    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
    out_emb = layer.self_attn.o_proj(att_output)
    out_emb = out_emb + hidden_states
    after_first_residual = out_emb.clone()
    out_emb = layer.post_attention_layernorm(out_emb)
    out_emb = layer.mlp(out_emb)
    return out_emb + after_first_residual


def _uses_self_attn(self, layer_idx: int, fill_kv_cache: bool) -> bool:
    return bool(
        fill_kv_cache
        or "cross" not in self.attention_mode
        or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
    )


def _prefix_kv_only(self, model_layers, inputs_embeds, layer_idx, position_ids, use_cache, past_key_values):
    """VLM-only pass: write K/V and leave the residual unchanged."""
    layer = model_layers[0][layer_idx]
    hidden = inputs_embeds[0]
    pos = position_ids
    if pos is not None and pos.shape[1] != hidden.shape[1]:
        pos = pos[:, : hidden.shape[1]]
    key_state, value_state = _proj_kv(layer, hidden, pos)
    if use_cache and past_key_values is not None:
        past_key_values.update(key_state.transpose(1, 2), value_state.transpose(1, 2), layer_idx)
    return inputs_embeds, past_key_values


def _dropped_self_with_expert(
    self, model_layers, inputs_embeds, layer_idx, position_ids, attention_mask,
    batch_size, head_dim, use_cache, past_key_values,
):
    vlm_layer = model_layers[0][layer_idx]
    expert_layer = model_layers[1][layer_idx]
    vlm_h, exp_h = inputs_embeds[0], inputs_embeds[1]
    vlm_len = vlm_h.shape[1]
    exp_len = exp_h.shape[1] if expert_layer is not None else 0
    seq_len = vlm_len + exp_len
    if position_ids.shape[1] > seq_len:
        pos = position_ids[:, :seq_len]
        mask = attention_mask[:, :seq_len, :seq_len]
    else:
        pos = position_ids
        mask = attention_mask
    key_v, value_v = _proj_kv(vlm_layer, vlm_h, pos[:, :vlm_len])
    if expert_layer is None:
        if use_cache and past_key_values is not None:
            past_key_values.update(key_v.transpose(1, 2), value_v.transpose(1, 2), layer_idx)
        return [vlm_h, exp_h], past_key_values
    query_e, key_e, value_e = _proj_qkv(expert_layer, exp_h)
    pos_e = pos[:, vlm_len : vlm_len + exp_len]
    query_e = apply_rope(query_e, pos_e)
    key_e = apply_rope(key_e, pos_e)
    key_states = torch.cat([key_v, key_e], dim=1)
    value_states = torch.cat([value_v, value_e], dim=1)
    if use_cache and past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states.transpose(1, 2), value_states.transpose(1, 2), layer_idx
        )
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
    exp_mask = mask[:, vlm_len : vlm_len + exp_len, :]
    if exp_mask.shape[-1] != key_states.shape[1] and attention_mask.shape[-1] == key_states.shape[1]:
        exp_mask = attention_mask[:, -exp_len:, :]
    att = self.get_attention_interface()(
        exp_mask, batch_size, head_dim, query_e, key_states, value_states
    )
    return [vlm_h, _expert_out(expert_layer, exp_h, att)], past_key_values


def _dropped_cross_with_expert(
    self, model_layers, inputs_embeds, layer_idx, position_ids, attention_mask,
    batch_size, head_dim, use_cache, past_key_values,
):
    vlm_layer = model_layers[0][layer_idx]
    expert_layer = model_layers[1][layer_idx]
    vlm_h, exp_h = inputs_embeds[0], inputs_embeds[1]
    vlm_len = vlm_h.shape[1]
    if past_key_values is None:
        key_states, value_states = _proj_kv(vlm_layer, vlm_h, position_ids[:, :vlm_len])
        expert_position_id = position_ids[:, vlm_len:]
    else:
        expert_position_id = position_ids
        key_states = value_states = None
    if use_cache and past_key_values is not None:
        key_states = past_key_values.layers[layer_idx].keys.transpose(1, 2)
        value_states = past_key_values.layers[layer_idx].values.transpose(1, 2)
    if expert_layer is None:
        return [vlm_h, exp_h], past_key_values
    attention_interface = self.get_attention_interface()
    expert_hidden_states = expert_layer.input_layernorm(exp_h)
    expert_hidden_shape = (*expert_hidden_states.shape[:-1], -1, expert_layer.self_attn.head_dim)
    expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
    expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)
    _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).reshape(
        *key_states.shape[:2], -1
    )
    expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
        *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
    )
    _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).reshape(
        *value_states.shape[:2], -1
    )
    expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
        *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
    )
    expert_position_id = expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
    expert_attention_mask = attention_mask[:, -exp_h.shape[1] :, : expert_key_states.shape[1]]
    expert_query_states = apply_rope(expert_query_state, expert_position_id)
    att = attention_interface(
        expert_attention_mask,
        batch_size,
        head_dim,
        expert_query_states,
        expert_key_states,
        expert_value_states,
    )
    return [vlm_h, _expert_out(expert_layer, exp_h, att)], past_key_values


def _forward_keep_kv(
    self,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: "DynamicCache | None" = None,
    inputs_embeds: list[torch.FloatTensor] = None,
    use_cache: bool | None = None,
):
    models = [self.get_vlm_model().text_model, self.lm_expert]
    model_layers = self.get_model_layers(models)
    for hidden_states in inputs_embeds:
        if hidden_states is None:
            continue
        batch_size = hidden_states.shape[0]

    fill_kv_cache = use_cache and past_key_values is None
    if fill_kv_cache:
        past_key_values = DynamicCache()

    num_layers = self.num_vlm_layers
    head_dim = self.vlm.config.text_config.head_dim
    for layer_idx in range(num_layers):
        vlm_hidden = inputs_embeds[0]
        expert_hidden = inputs_embeds[1] if len(inputs_embeds) > 1 else None
        if layer_idx in _DROP and vlm_hidden is not None:
            if expert_hidden is None:
                inputs_embeds, past_key_values = _prefix_kv_only(
                    self, model_layers, inputs_embeds, layer_idx, position_ids, use_cache, past_key_values
                )
                continue
            if _uses_self_attn(self, layer_idx, fill_kv_cache):
                inputs_embeds, past_key_values = _dropped_self_with_expert(
                    self, model_layers, inputs_embeds, layer_idx, position_ids, attention_mask,
                    batch_size, head_dim, use_cache, past_key_values,
                )
            else:
                inputs_embeds, past_key_values = _dropped_cross_with_expert(
                    self, model_layers, inputs_embeds, layer_idx, position_ids, attention_mask,
                    batch_size, head_dim, use_cache, past_key_values,
                )
            continue
        if (
            fill_kv_cache
            or "cross" not in self.attention_mode
            or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
        ):
            att_outputs, past_key_values = self.forward_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
        else:
            att_outputs, past_key_values = self.forward_cross_attn_layer(
                model_layers,
                inputs_embeds,
                layer_idx,
                position_ids,
                attention_mask,
                batch_size,
                head_dim,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
        outputs_embeds = []
        start = 0
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
            if hidden_states is not None:
                if layer is None:
                    outputs_embeds.append(hidden_states)
                    continue
                end = start + hidden_states.shape[1]
                if i == 0 and layer_idx in _DROP:
                    outputs_embeds.append(hidden_states)
                    start = end if len(att_outputs) == 1 else 0
                    continue
                if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                att_out = att_output[:, start:end]
                out_emb = layer.self_attn.o_proj(att_out)

                out_emb += hidden_states
                after_first_residual = out_emb.clone()

                out_emb = layer.post_attention_layernorm(out_emb)
                out_emb = layer.mlp(out_emb)

                out_emb += after_first_residual

                outputs_embeds.append(out_emb)

                start = end if len(att_outputs) == 1 else 0
            else:
                outputs_embeds.append(None)

        inputs_embeds = outputs_embeds

    outputs_embeds = []
    for i, hidden_states in enumerate(inputs_embeds):
        if hidden_states is not None:
            out_emb = models[i].norm(hidden_states)
            outputs_embeds.append(out_emb)
        else:
            outputs_embeds.append(None)
    return outputs_embeds, past_key_values


def _forward_maybe_skip(self, *args, **kwargs):
    if not _DROP:
        return _ORIGINAL(self, *args, **kwargs)
    return _forward_keep_kv(self, *args, **kwargs)


def install() -> None:
    if SmolVLMWithExpertModel.forward is not _forward_maybe_skip:
        SmolVLMWithExpertModel.forward = _forward_maybe_skip


@contextmanager
def skip_vlm_layers(layer_indices: Iterable[int]):
    global _DROP
    install()
    previous = set(_DROP)
    _DROP = {int(i) for i in layer_indices}
    try:
        yield
    finally:
        _DROP = previous

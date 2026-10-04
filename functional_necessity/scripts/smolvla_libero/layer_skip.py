"""Keep SmolVLA VLM K/V, drop the layer's residual update.

SmolVLMWithExpertModel does not call HuggingFace DecoderLayer.forward. The
joint graph runs QKV inside forward_attn_layer / forward_cross_attn_layer,
then applies o_proj + MLP on each stream. A dropped VLM layer still produces
K/V (so the action expert can cross-attend) and leaves the VLM residual
unchanged. The expert stream is never skipped.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch
from transformers.cache_utils import DynamicCache

from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

_DROP: set[int] = set()
_ORIGINAL = SmolVLMWithExpertModel.forward


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

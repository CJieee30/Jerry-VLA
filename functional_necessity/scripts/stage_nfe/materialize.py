"""Replace dropped VLM layers with real K/V-only modules and free the unused weights.

The compute matches the current K/V-only skip. Inference no longer branches on a
Python drop set: the module that sits in the layer list only has K/V.
"""

from __future__ import annotations

import copy
import gc

import torch
import torch.nn as nn
from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb as gemma_rope

from lerobot.policies.pi_gemma import _gated_residual


class KVOnlyGemmaLayer(nn.Module):
    """π0.5 prefix layer: input LN + K/V + RoPE + cache write. Residual is identity."""

    def __init__(self, norm, k_proj, v_proj, head_dim: int, layer_idx: int):
        super().__init__()
        self.input_layernorm = norm
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.head_dim = int(head_dim)
        self.layer_idx = int(layer_idx)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        adarms_cond=None,
        **kwargs,
    ):
        del attention_mask, position_ids, use_cache, cache_position, kwargs
        residual = hidden_states
        hidden_ln, gate = self.input_layernorm(hidden_states, cond=adarms_cond)
        if position_embeddings is None or past_key_values is None:
            raise RuntimeError("materialized π0.5 layer requires position embeddings and a KV cache")
        shape = (*hidden_ln.shape[:-1], -1, self.head_dim)
        key_states = self.k_proj(hidden_ln).view(shape).transpose(1, 2)
        value_states = self.v_proj(hidden_ln).view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        _, key_states = gemma_rope(key_states, key_states, cos, sin)
        past_key_values.update(key_states, value_states, self.layer_idx)
        if adarms_cond is not None:
            return _gated_residual(residual, hidden_ln, gate)
        return residual


class KVOnlyMolmoLayer(nn.Module):
    """Molmo pre-norm text layer: attn RMSNorm, K/V rows, K-norm, RoPE, cache write."""

    def __init__(self, src):
        super().__init__()
        attn = src.self_attn
        q_dim, k_dim, v_dim = (int(x) for x in attn.fused_dims)
        weight = attn.att_proj.weight.detach()
        self.attn_norm = copy.deepcopy(src.attn_norm)
        self.k_norm = copy.deepcopy(attn.k_norm) if attn.k_norm is not None else None
        self.k_proj = nn.Linear(weight.shape[1], k_dim, bias=False, device=weight.device, dtype=weight.dtype)
        self.v_proj = nn.Linear(weight.shape[1], v_dim, bias=False, device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            self.k_proj.weight.copy_(weight[q_dim : q_dim + k_dim])
            self.v_proj.weight.copy_(weight[q_dim + k_dim : q_dim + k_dim + v_dim])
        self.head_dim = int(attn.head_dim)
        self.layer_idx = int(attn.layer_idx)
        self.qk_norm_type = attn.qk_norm_type

    def _kv(self, hidden_states, position_embeddings, past_key_values, cache_position):
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        value_states = value_states.view(hidden_shape)
        if self.k_norm is not None and self.qk_norm_type != "qwen3":
            key_states = self.k_norm(key_states)
        key_states = key_states.view(hidden_shape)
        if self.k_norm is not None and self.qk_norm_type == "qwen3":
            key_states = self.k_norm(key_states)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        cos, sin = position_embeddings
        _, key_states = _molmo_rope(key_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        return key_states, value_states

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        del attention_mask, position_ids, use_cache
        collect = bool(kwargs.pop("collect_layer_kv_states", False))
        normed = self.attn_norm(hidden_states)
        key_states, value_states = self._kv(normed, position_embeddings, past_key_values, cache_position)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if collect:
            outputs += (key_states, value_states)
        return outputs


def _molmo_rope(q, k, cos, sin):
    from lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 import apply_rotary_pos_emb

    return apply_rotary_pos_emb(q, k, cos, sin)


def _deepcopy_linear(src: nn.Linear) -> nn.Linear:
    dst = nn.Linear(
        src.in_features,
        src.out_features,
        bias=src.bias is not None,
        device=src.weight.device,
        dtype=src.weight.dtype,
    )
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if src.bias is not None:
            dst.bias.copy_(src.bias)
    return dst


def materialize_pi05(policy, indices) -> dict:
    from layer_skip import _paligemma_layers

    layers, _ = _paligemma_layers(policy)
    before = _nparams(policy)
    for idx in indices:
        src = layers[int(idx)]
        attn = src.self_attn
        layers[int(idx)] = KVOnlyGemmaLayer(
            copy.deepcopy(src.input_layernorm),
            _deepcopy_linear(attn.k_proj),
            _deepcopy_linear(attn.v_proj),
            attn.head_dim,
            attn.layer_idx,
        )
    _free()
    after = _nparams(policy)
    return {"params_before": before, "params_after": after, "freed": before - after}


def materialize_molmo(policy, indices) -> dict:
    blocks = policy._backbone().transformer.blocks
    before = _nparams(policy)
    for idx in indices:
        blocks[int(idx)] = KVOnlyMolmoLayer(blocks[int(idx)])
    _free()
    after = _nparams(policy)
    return {"params_before": before, "params_after": after, "freed": before - after}


def materialize_smol(policy, indices) -> dict:
    import layer_skip

    text = policy.model.vlm_with_expert.get_vlm_model().text_model
    before = _nparams(policy)
    for idx in indices:
        layer = text.layers[int(idx)]
        layer.self_attn._modules.pop("q_proj", None)
        layer.self_attn._modules.pop("o_proj", None)
        layer._modules.pop("mlp", None)
        layer._modules.pop("post_attention_layernorm", None)
    layer_skip.install()
    layer_skip._DROP = {int(i) for i in indices}
    _free()
    after = _nparams(policy)
    return {"params_before": before, "params_after": after, "freed": before - after}


def _nparams(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

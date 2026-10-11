"""Keep MolmoAct2 VLM K/V, drop the layer's residual update.

The action expert reads each text layer's K/V and does not read that layer's
residual. A dropped text layer matches π0.5 Appendix B: input RMSNorm (pre-norm
blocks only), K/V rows of ``att_proj``, K norm, and RoPE, then the cache write.
Q, attention, ``attn_out``, and the MLP do not run.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch.nn.functional as F

from lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 import (
    MolmoAct2DecoderLayer,
    MolmoAct2PostNormDecoderLayer,
    apply_rotary_pos_emb,
)

_DROP: set[int] = set()
_ORIGINAL = MolmoAct2DecoderLayer.forward
_ORIGINAL_POST = MolmoAct2PostNormDecoderLayer.forward


def _project_kv(attn, hidden_states):
    """K/V rows of the fused QKV projection. Q rows are not multiplied."""
    q_dim, k_dim, _v_dim = attn.fused_dims
    weight = attn.att_proj.weight
    bias = attn.att_proj.bias
    k_w = weight[q_dim : q_dim + k_dim]
    v_w = weight[q_dim + k_dim :]
    k_b = None if bias is None else bias[q_dim : q_dim + k_dim]
    v_b = None if bias is None else bias[q_dim + k_dim :]
    key_states = F.linear(hidden_states, k_w, k_b)
    value_states = F.linear(hidden_states, v_w, v_b)
    hidden_shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
    value_states = value_states.view(hidden_shape)
    if attn.q_norm is not None and attn.k_norm is not None and attn.qk_norm_type != "qwen3":
        key_states = attn.k_norm(key_states)
    key_states = key_states.view(hidden_shape)
    if attn.q_norm is not None and attn.k_norm is not None and attn.qk_norm_type == "qwen3":
        key_states = attn.k_norm(key_states)
    return key_states.transpose(1, 2), value_states.transpose(1, 2)


def _write_kv(attn, hidden_states, position_embeddings, past_key_values, cache_position):
    key_states, value_states = _project_kv(attn, hidden_states)
    cos, sin = position_embeddings
    _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, attn.layer_idx, cache_kwargs
        )
    return key_states, value_states


def _pack(hidden_states, output_attentions, collect, key_states, value_states):
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (None,)
    if collect:
        outputs += (key_states, value_states)
    return outputs


def _skip_prenorm(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None,
                  past_key_values=None, output_attentions=False, use_cache=False, cache_position=None, **kwargs):
    layer_idx = int(self.self_attn.layer_idx)
    if layer_idx not in _DROP:
        return _ORIGINAL(
            self, hidden_states, position_embeddings, attention_mask, position_ids,
            past_key_values, output_attentions, use_cache, cache_position, **kwargs,
        )
    collect = bool(kwargs.pop("collect_layer_kv_states", False))
    normed = self.attn_norm(hidden_states)
    key_states, value_states = _write_kv(
        self.self_attn, normed, position_embeddings, past_key_values, cache_position
    )
    return _pack(hidden_states, output_attentions, collect, key_states, value_states)


def _skip_postnorm(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None,
                   past_key_values=None, output_attentions=False, use_cache=False, cache_position=None, **kwargs):
    layer_idx = int(self.self_attn.layer_idx)
    if layer_idx not in _DROP:
        return _ORIGINAL_POST(
            self, hidden_states, position_embeddings, attention_mask, position_ids,
            past_key_values, output_attentions, use_cache, cache_position, **kwargs,
        )
    collect = bool(kwargs.pop("collect_layer_kv_states", False))
    key_states, value_states = _write_kv(
        self.self_attn, hidden_states, position_embeddings, past_key_values, cache_position
    )
    return _pack(hidden_states, output_attentions, collect, key_states, value_states)


def install() -> None:
    if MolmoAct2DecoderLayer.forward is not _skip_prenorm:
        MolmoAct2DecoderLayer.forward = _skip_prenorm
    if MolmoAct2PostNormDecoderLayer.forward is not _skip_postnorm:
        MolmoAct2PostNormDecoderLayer.forward = _skip_postnorm


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

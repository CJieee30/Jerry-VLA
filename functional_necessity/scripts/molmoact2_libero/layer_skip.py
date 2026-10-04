"""Keep MolmoAct2 VLM K/V, drop the layer's residual update.

The action expert cross-attends to each text layer's K/V. A dropped text layer
therefore still runs input RMSNorm and the attention projection that produces
K/V. It does not add the attention output or the MLP into the residual stream.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

from lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 import (
    MolmoAct2DecoderLayer,
    MolmoAct2PostNormDecoderLayer,
)

_DROP: set[int] = set()
_ORIGINAL = MolmoAct2DecoderLayer.forward
_ORIGINAL_POST = MolmoAct2PostNormDecoderLayer.forward


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
    attn_outputs = self.self_attn(
        hidden_states=normed,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        collect_layer_kv_states=True,
        **kwargs,
    )
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_outputs[1],)
    if collect:
        outputs += (attn_outputs[2], attn_outputs[3])
    return outputs


def _skip_postnorm(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None,
                   past_key_values=None, output_attentions=False, use_cache=False, cache_position=None, **kwargs):
    layer_idx = int(self.self_attn.layer_idx)
    if layer_idx not in _DROP:
        return _ORIGINAL_POST(
            self, hidden_states, position_embeddings, attention_mask, position_ids,
            past_key_values, output_attentions, use_cache, cache_position, **kwargs,
        )
    collect = bool(kwargs.pop("collect_layer_kv_states", False))
    attn_outputs = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        collect_layer_kv_states=True,
        **kwargs,
    )
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_outputs[1],)
    if collect:
        outputs += (attn_outputs[2], attn_outputs[3])
    return outputs


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

"""Identity skip for OpenVLA-OFT Llama decoder layers.

The action head reads the final residual stream. Nothing else consumes a layer's
K/V, so a dropped layer returns its input hidden state and runs no projections.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import transformers.models.llama.modeling_llama as llama_mod

_DROP: set[int] = set()
_ORIGINAL_FORWARD = llama_mod.LlamaDecoderLayer.forward


def _identity_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position=None,
    **kwargs,
):
    if int(self.self_attn.layer_idx) not in _DROP:
        return _ORIGINAL_FORWARD(
            self,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
    # DynamicCache.append is positional. A skipped layer must still occupy its
    # index, or the next layer reads past the end of the cache.
    if past_key_value is not None and hasattr(past_key_value, "update"):
        attn = self.self_attn
        bsz, q_len, _ = hidden_states.shape
        dummy = hidden_states.new_zeros(bsz, attn.num_key_value_heads, q_len, attn.head_dim)
        past_key_value.update(dummy, dummy, int(attn.layer_idx))
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (None,)
    if use_cache:
        outputs += (past_key_value,)
    return outputs


def install_identity_skip() -> None:
    if llama_mod.LlamaDecoderLayer.forward is not _identity_forward:
        llama_mod.LlamaDecoderLayer.forward = _identity_forward


@contextmanager
def skip_llama_layers(layer_indices: Iterable[int]):
    """Drop whole Llama blocks for the duration of the context."""
    global _DROP
    install_identity_skip()
    indices = {int(i) for i in layer_indices}
    n_layers = 32
    for idx in indices:
        if idx < 0 or idx >= n_layers:
            raise IndexError(f"drop index {idx} out of range for {n_layers} Llama layers")
    previous = set(_DROP)
    _DROP = indices
    try:
        yield
    finally:
        _DROP = previous


def dropped_layers() -> list[int]:
    return sorted(_DROP)

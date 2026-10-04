"""Whole-layer identity skip for DB-CogACT Qwen2.5 decoder blocks.

The DiT action head reads the last unmasked hidden state, not per-layer K/V,
so a dropped layer returns its input residual unchanged.
"""

from __future__ import annotations


def llm_layers(model):
    return list(model.model.llm.layers)


def install_identity_skip(model, drop_layers: list[int]) -> None:
    layers = llm_layers(model)
    for index in drop_layers:
        layer = layers[index]

        def _identity(hidden_states, *args, **kwargs):
            del args, kwargs
            return hidden_states

        layer.forward = _identity

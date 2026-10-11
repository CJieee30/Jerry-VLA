"""DTR Appendix B skip for LeRobot PI05: drop language Q/O/MLP, keep K/V + input LN.

Two hooks, same Case 3 semantics:

- Joint graph (PPL / training ``compute_loss``): monkeypatch
  ``compute_layer_complete``. Official ``lerobot-eval`` does **not** enter this
  function (prefix is ``language_model.forward`` + KV-cache; denoise is expert
  only).
- Prefix / closed-loop inference: wrap ``language_model.layers[i].forward``.
  Residual is identity. ``keep_kv=True`` (paper operator) still writes K/V so
  expert denoise can cross-attend. ``keep_kv=False`` writes zeros into those
  cache slots. Do not replace ``sample_actions``.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Iterable

import torch
from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb

import lerobot.policies.pi05.modeling_pi05 as pi05_mod
from lerobot.policies.pi_gemma import _gated_residual, layernorm_forward


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


_DROP_LLM_IDS: set[int] = set()
_KEEP_KV = True
_ORIGINAL_COMPUTE = pi05_mod.compute_layer_complete


def _is_dropped(layer) -> bool:
    return id(layer) in _DROP_LLM_IDS


def _compute_layer_complete_maybe_skip(
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    layers,
    rotary_emb,
):
    paligemma_layer = layers[0]
    if not _is_dropped(paligemma_layer):
        return _ORIGINAL_COMPUTE(
            inputs_embeds,
            attention_mask,
            position_ids,
            adarms_cond,
            layers,
            rotary_emb,
        )
    return _joint_attention_llm_dropped(
        inputs_embeds,
        attention_mask,
        position_ids,
        adarms_cond,
        layers,
        rotary_emb,
    )


def _joint_attention_llm_dropped(
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    layers,
    rotary_emb,
):
    """Port of VLADrop `_joint_attention_llm_dropped` + language MLP skip."""
    llm_layer, expert_layer = layers
    llm_hidden = inputs_embeds[0]
    expert_hidden = inputs_embeds[1]
    llm_seq_len = llm_hidden.shape[1]

    llm_normed, llm_gate = layernorm_forward(llm_layer.input_layernorm, llm_hidden, adarms_cond[0])
    llm_input_shape = llm_normed.shape[:-1]
    llm_hidden_shape = (*llm_input_shape, -1, llm_layer.self_attn.head_dim)
    if _KEEP_KV:
        llm_key = llm_layer.self_attn.k_proj(llm_normed).view(llm_hidden_shape).transpose(1, 2)
        llm_value = llm_layer.self_attn.v_proj(llm_normed).view(llm_hidden_shape).transpose(1, 2)
    else:
        llm_key = torch.zeros(
            llm_normed.shape[0],
            llm_layer.self_attn.config.num_key_value_heads,
            llm_normed.shape[1],
            llm_layer.self_attn.head_dim,
            device=llm_normed.device,
            dtype=llm_normed.dtype,
        )
        llm_value = torch.zeros_like(llm_key)

    expert_normed, expert_gate = layernorm_forward(
        expert_layer.input_layernorm, expert_hidden, adarms_cond[1]
    )
    expert_input_shape = expert_normed.shape[:-1]
    expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)
    expert_query = expert_layer.self_attn.q_proj(expert_normed).view(expert_hidden_shape).transpose(1, 2)
    expert_key = expert_layer.self_attn.k_proj(expert_normed).view(expert_hidden_shape).transpose(1, 2)
    expert_value = expert_layer.self_attn.v_proj(expert_normed).view(expert_hidden_shape).transpose(1, 2)

    full_key = torch.cat([llm_key, expert_key], dim=2)
    full_value = torch.cat([llm_value, expert_value], dim=2)

    total_seq = llm_seq_len + expert_hidden.shape[1]
    dummy = torch.zeros(
        expert_query.shape[0],
        total_seq,
        expert_query.shape[-1],
        device=expert_query.device,
        dtype=expert_query.dtype,
    )
    cos, sin = rotary_emb(dummy, position_ids)
    cos_expert = cos[:, llm_seq_len:]
    sin_expert = sin[:, llm_seq_len:]
    expert_query = (expert_query * cos_expert.unsqueeze(1)) + (
        _rotate_half(expert_query) * sin_expert.unsqueeze(1)
    )
    full_key = (full_key * cos.unsqueeze(1)) + (_rotate_half(full_key) * sin.unsqueeze(1))

    expert_attn_mask = attention_mask[:, :, llm_seq_len:, :] if attention_mask is not None else None
    scaling = expert_layer.self_attn.scaling
    # HuggingFace GemmaAttention stores GQA as `num_key_value_groups`, not `num_heads`.
    # Do not pass `attn.num_heads` as getattr's default: Python evaluates it before the call.
    attn = expert_layer.self_attn
    groups = getattr(attn, "num_key_value_groups", None)
    if groups is None:
        n_heads = attn.config.num_attention_heads
        n_kv = max(attn.config.num_key_value_heads, 1)
        groups = n_heads // n_kv
    full_key_expanded = _repeat_kv(full_key, groups)
    full_value_expanded = _repeat_kv(full_value, groups)
    attn_weights = torch.matmul(expert_query, full_key_expanded.transpose(2, 3)) * scaling
    if expert_attn_mask is not None:
        causal_mask = expert_attn_mask[:, :, :, : full_key_expanded.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        expert_query.dtype
    )
    expert_attn_output = torch.matmul(attn_weights, full_value_expanded)
    expert_attn_output = expert_attn_output.transpose(1, 2).contiguous()

    batch_size = expert_query.shape[0]
    expert_seq_len = expert_hidden.shape[1]
    expert_attn_output = expert_attn_output.reshape(batch_size, expert_seq_len, -1)
    if expert_attn_output.dtype != expert_layer.self_attn.o_proj.weight.dtype:
        expert_attn_output = expert_attn_output.to(expert_layer.self_attn.o_proj.weight.dtype)
    expert_out = expert_layer.self_attn.o_proj(expert_attn_output)
    expert_out = _gated_residual(expert_hidden, expert_out, expert_gate)

    if adarms_cond[0] is not None:
        llm_out = _gated_residual(llm_hidden, llm_normed, llm_gate)
    else:
        llm_out = llm_hidden

    # Language MLP skipped (identity). Expert MLP kept.
    after_first = expert_out.clone()
    expert_mlp_in, expert_mlp_gate = layernorm_forward(
        expert_layer.post_attention_layernorm, expert_out, adarms_cond[1]
    )
    if expert_layer.mlp.up_proj.weight.dtype == torch.bfloat16:
        expert_mlp_in = expert_mlp_in.to(dtype=torch.bfloat16)
    expert_mlp_out = expert_layer.mlp(expert_mlp_in)
    expert_out = _gated_residual(after_first, expert_mlp_out, expert_mlp_gate)
    return [llm_out, expert_out]


def _write_language_kv_to_cache(layer, hidden_ln, position_embeddings, past_key_values) -> None:
    """Write RoPE'd K/V for a dropped language layer so expert denoise can attend."""
    attn = layer.self_attn
    input_shape = hidden_ln.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    key_states = attn.k_proj(hidden_ln).view(hidden_shape).transpose(1, 2)
    value_states = attn.v_proj(hidden_ln).view(hidden_shape).transpose(1, 2)
    if position_embeddings is None:
        raise RuntimeError(
            "Dropped prefix layer needs position_embeddings to RoPE K before cache.update"
        )
    cos, sin = position_embeddings
    _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
    if past_key_values is None:
        raise RuntimeError(
            "Dropped prefix layer got past_key_values=None; expert denoise would miss language K/V"
        )
    past_key_values.update(key_states, value_states, attn.layer_idx)


def _dropped_prefix_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    adarms_cond: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Identity residual; keep input LN + K/V cache write; skip Q/O/MLP."""
    del attention_mask, position_ids, use_cache, cache_position, kwargs
    residual = hidden_states
    hidden_ln, gate = self.input_layernorm(hidden_states, cond=adarms_cond)
    _write_language_kv_to_cache(self, hidden_ln, position_embeddings, past_key_values)
    if adarms_cond is not None:
        return _gated_residual(residual, hidden_ln, gate)
    return residual


def _dropped_prefix_forward_no_kv(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    adarms_cond: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Identity residual and zero K/V. Expert cannot read this layer."""
    del attention_mask, position_ids, use_cache, cache_position, position_embeddings, adarms_cond, kwargs
    attn = self.self_attn
    bsz, slen, _ = hidden_states.shape
    n_kv = attn.config.num_key_value_heads
    key_states = torch.zeros(
        bsz, n_kv, slen, attn.head_dim, device=hidden_states.device, dtype=hidden_states.dtype
    )
    value_states = torch.zeros_like(key_states)
    if past_key_values is None:
        raise RuntimeError("no-KV skip still needs a cache slot so denoise layer_idx stays aligned")
    past_key_values.update(key_states, value_states, attn.layer_idx)
    return hidden_states


@contextmanager
def skip_llm_layers(policy, layer_indices: Iterable[int], keep_kv: bool = True):
    """Temporarily drop the given PaliGemma layer indices (attn Q/O + MLP)."""
    global _DROP_LLM_IDS, _KEEP_KV
    layers, _ = _paligemma_layers(policy)
    indices = [int(i) for i in layer_indices]
    if not indices:
        yield
        return
    n_layers = len(layers)
    for idx in indices:
        if idx < 0 or idx >= n_layers:
            raise IndexError(f"drop index {idx} out of range for {n_layers} language layers")
    _DROP_LLM_IDS = {id(layers[i]) for i in indices}
    _KEEP_KV = bool(keep_kv)
    fwd = _dropped_prefix_forward if _KEEP_KV else _dropped_prefix_forward_no_kv
    patched = []
    for idx in indices:
        layer = layers[idx]
        layer.forward = types.MethodType(fwd, layer)
        patched.append(layer)
    pi05_mod.compute_layer_complete = _compute_layer_complete_maybe_skip
    try:
        yield
    finally:
        pi05_mod.compute_layer_complete = _ORIGINAL_COMPUTE
        for layer in patched:
            try:
                del layer.forward
            except AttributeError:
                pass
        _DROP_LLM_IDS = set()
        _KEEP_KV = True


def dropped_prefix_indices(policy) -> list[int]:
    layers, _ = _paligemma_layers(policy)
    out = []
    for i, layer in enumerate(layers):
        fwd = getattr(layer, "forward", None)
        func = getattr(fwd, "__func__", None)
        if func in (_dropped_prefix_forward, _dropped_prefix_forward_no_kv):
            out.append(i)
    return out


def _paligemma_layers(policy):
    core = policy.model._paligemma_language_model_core()
    return core.layers, core.norm

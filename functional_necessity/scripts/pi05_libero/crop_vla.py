"""CROP-VLA scout: camera-role visual token selection on π0.5.

Training-free. Vision is encoded once. A Drop-6 keep-KV prefix (installed by
the caller) produces layer-2 language→vision attention. Selected original
embeddings are packed into a shorter prefix; RoPE / action offsets keep the
unpruned position indices. Empty-camera slots are not modified.

LIBERO mapping (lerobot/envs/libero.py):
  agentview_image           → observation.images.image  → global
  robot0_eye_in_hand_image  → observation.images.image2 → wrist
  empty_camera_*            → padded slot               → unchanged
"""

from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from typing import Any

import torch
import torch.nn.functional as F

from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import (
    clone_past_key_values,
    make_att_2d_masks,
    prepare_attention_masks_4d,
)

from prefix_trim import TrimInfo, trim_prefix_inputs

TOKENS_PER_VIEW = 256
GRID = 16
SCOUT_LAYER = 2
FISHER_DROP6 = [1, 13, 14, 15, 16, 17]
BUDGET_PRESETS = {
    "role": {
        "wrist": {"relevance": 96, "diversity": 32},
        "global": {"relevance": 32, "diversity": 96},
    },
    "shared_mix": {
        "wrist": {"relevance": 64, "diversity": 64},
        "global": {"relevance": 64, "diversity": 64},
    },
}
BUDGETS = BUDGET_PRESETS["role"]
SELECTORS = ("greedy", "lite", "grid")
LITE_N_ANCHORS = 4
_UNIFORM_GRID_KEEP: dict[str, torch.Tensor] = {}
_UNIFORM_GRID_PREFIX: dict[tuple, torch.Tensor] = {}


def resolve_budgets(name: str | dict | None = None) -> dict[str, dict[str, int]]:
    if name is None:
        return BUDGET_PRESETS["role"]
    if isinstance(name, dict):
        return name
    key = str(name).strip().lower()
    if key in ("", "role", "crop"):
        return BUDGET_PRESETS["role"]
    if key not in BUDGET_PRESETS:
        raise ValueError(f"unknown CROP budget {name!r}; expected one of {sorted(BUDGET_PRESETS)}")
    return BUDGET_PRESETS[key]


def resolve_selector(name: str | None = None) -> str:
    key = "greedy" if name is None else str(name).strip().lower()
    if key in ("", "greedy", "crop", "fps"):
        return "greedy"
    if key in ("lite", "crop-lite", "crop_lite"):
        return "lite"
    if key in ("grid", "uniform", "uniform-grid", "uniform_grid"):
        return "grid"
    raise ValueError(f"unknown CROP selector {name!r}; expected greedy, lite, or grid")


@dataclass
class CameraSlot:
    index: int
    key: str
    role: str
    start: int
    n_tokens: int
    keep_all: bool


@dataclass
class CropStats:
    full_len: int
    short_len: int
    kv_len: int
    orig_action_offset: int
    short_pad_sum: int
    cameras: list[dict[str, Any]] = field(default_factory=list)
    scout_layer: int = SCOUT_LAYER
    keep_all: bool = False
    timing_ms: dict[str, float] | None = None
    scout_layer_calls: list[int] | None = None
    budget_name: str = "role"
    selector: str = "greedy"
    num_steps: int | None = None
    velocity_calls: int | None = None
    deferred: dict[str, Any] = field(default_factory=dict)
    prefix_trim: bool = False
    n_views_in: int | None = None
    n_views_out: int | None = None
    n_lang_in: int | None = None
    n_lang_out: int | None = None
    valid_prefix_len: int | None = None
    untrimmed_physical: int | None = None
    trimmed_physical: int | None = None


def uniform_grid_local_indices(
    device: torch.device | str,
    n_tokens: int = TOKENS_PER_VIEW,
    grid: int = GRID,
) -> torch.Tensor:
    """Keep (row + col) % 2 == 0 on a grid x grid view. Cached per device."""
    if n_tokens != grid * grid:
        raise ValueError(f"uniform-grid expects {grid}x{grid}={grid * grid} tokens, got {n_tokens}")
    device = torch.device(device)
    cache_key = f"{device}:{n_tokens}:{grid}"
    cached = _UNIFORM_GRID_KEEP.get(cache_key)
    if cached is not None and cached.device == device:
        return cached
    rows = torch.arange(grid, device=device)
    cols = torch.arange(grid, device=device)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    parity = (rr + cc) % 2
    idx = (rr * grid + cc)[parity == 0].to(dtype=torch.long).reshape(-1)
    if int(idx.numel()) != n_tokens // 2:
        raise RuntimeError(f"uniform-grid keep {int(idx.numel())} != {n_tokens // 2}")
    _UNIFORM_GRID_KEEP[cache_key] = idx
    return idx


def packed_grid_keep_idx(
    device: torch.device | str,
    n_views: int,
    n_lang: int,
    empty_flags: tuple[bool, ...],
    *,
    keep_all: bool = False,
    n_tokens: int = TOKENS_PER_VIEW,
    grid: int = GRID,
) -> torch.Tensor:
    """Packed prefix indices: live cameras use the cached checkerboard, empty/lang stay full."""
    device = torch.device(device)
    key = (str(device), n_views, n_lang, empty_flags, keep_all, n_tokens, grid)
    cached = _UNIFORM_GRID_PREFIX.get(key)
    if cached is not None and cached.device == device:
        return cached
    local = uniform_grid_local_indices(device, n_tokens, grid)
    parts = []
    for i, is_empty in enumerate(empty_flags):
        start = i * n_tokens
        if keep_all or is_empty:
            parts.append(torch.arange(n_tokens, device=device, dtype=torch.long) + start)
        else:
            parts.append(local + start)
    lang_start = n_tokens * n_views
    parts.append(torch.arange(n_lang, device=device, dtype=torch.long) + lang_start)
    idx = torch.cat(parts, dim=0)
    _UNIFORM_GRID_PREFIX[key] = idx
    return idx


def role_for_image_key(key: str) -> str:
    name = key.rsplit(".", 1)[-1]
    if "empty_camera" in key or name.startswith("empty_camera"):
        return "empty"
    if name == "image2":
        return "wrist"
    if name == "image":
        return "global"
    raise ValueError(f"unrecognized image key {key}")


def camera_slots(policy, batch: dict[str, Any], tokens_per_view: int = TOKENS_PER_VIEW) -> list[CameraSlot]:
    keys = list(policy.config.image_features.keys())
    present = [k for k in keys if k in batch]
    missing = [k for k in keys if k not in batch]
    ordered = present + missing
    slots = []
    start = 0
    for i, key in enumerate(ordered):
        role = role_for_image_key(key)
        slots.append(
            CameraSlot(
                index=i,
                key=key,
                role=role,
                start=start,
                n_tokens=tokens_per_view,
                keep_all=role == "empty",
            )
        )
        start += tokens_per_view
    return slots


def greedy_diversity(feats: torch.Tensor, seed: list[int] | torch.Tensor, n_extra: int) -> list[int]:
    """Add tokens with the smallest nearest-neighbor cosine to the seed set.

    Same rule as the naive loop (recompute max cosine to the whole selected
    set). Incremental: keep a running nearest-cosine vector and only compare
    against the newly added token. Indices stay on-device until the end.
    """
    if n_extra <= 0:
        return []
    n = int(feats.shape[0])
    if isinstance(seed, torch.Tensor):
        selected = seed.to(device=feats.device, dtype=torch.long)
    else:
        selected = torch.tensor(list(seed), device=feats.device, dtype=torch.long)
    n_taken = int(selected.numel())
    n_extra = min(int(n_extra), n - n_taken)
    if n_extra <= 0:
        return []
    taken = torch.zeros(n, dtype=torch.bool, device=feats.device)
    taken[selected] = True
    nearest = (feats @ feats[selected].T).max(dim=1).values
    extra_idx: list[torch.Tensor] = []
    for _ in range(n_extra):
        pick = nearest.masked_fill(taken, 2.0).argmin()
        extra_idx.append(pick)
        taken[pick] = True
        nearest = torch.maximum(nearest, feats @ feats[pick])
    return torch.stack(extra_idx).tolist()


def select_one_camera(
    score: torch.Tensor,
    feats: torch.Tensor,
    *,
    n_rel: int,
    n_div: int,
    valid: torch.Tensor | None = None,
    keep_all: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """Return (keep_sorted, relevance, diversity) local indices in 0..N-1."""
    n = int(score.shape[0])
    if keep_all or n_rel + n_div >= n:
        keep = list(range(n))
        return keep, keep, []
    work = score.detach().float().clone()
    if valid is not None:
        work = work.masked_fill(~valid, -1e9)
    n_rel = min(n_rel, n)
    rel = torch.topk(work, k=n_rel).indices.tolist()
    n_div = min(n_div, n - len(rel))
    div = greedy_diversity(feats, rel, n_div)
    keep = sorted(set(rel) | set(div))
    return keep, rel, div


def select_one_camera_lite(
    score: torch.Tensor,
    feats: torch.Tensor,
    *,
    n_rel: int,
    n_div: int,
    valid: torch.Tensor | None = None,
    keep_all: bool = False,
    n_anchors: int = LITE_N_ANCHORS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-shot diversity vs greedy FPS. Stays on GPU; no .item/.tolist/nonzero."""
    n = int(score.shape[0])
    device = score.device
    empty = torch.empty(0, device=device, dtype=torch.long)
    if keep_all or n_rel + n_div >= n:
        keep = torch.arange(n, device=device, dtype=torch.long)
        return keep, keep, empty
    work = score.detach().float()
    if valid is not None:
        work = work.masked_fill(~valid, -1e9)
    n_rel = min(int(n_rel), n)
    rel = torch.topk(work, k=n_rel).indices
    n_div = min(int(n_div), n - n_rel)
    if n_div <= 0:
        keep = torch.sort(rel).values
        return keep, rel, empty
    n_anchor = min(int(n_anchors), n_rel)
    anchors = torch.topk(work, k=n_anchor).indices
    sim = feats @ feats.index_select(0, anchors).transpose(0, 1)
    div_score = 1.0 - sim.max(dim=1).values
    taken = torch.zeros(n, dtype=torch.bool, device=device)
    taken.scatter_(0, rel, True)
    div_score = div_score.masked_fill(taken, -1e9)
    if valid is not None:
        div_score = div_score.masked_fill(~valid, -1e9)
    div = torch.topk(div_score, k=n_div).indices
    keep = torch.sort(torch.cat([rel, div], dim=0)).values
    return keep, rel, div


@contextmanager
def _capture_layer_attention(language_model, layer_idx: int):
    layers = language_model.layers
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"scout layer {layer_idx} out of range for {len(layers)} layers")
    attn = layers[layer_idx].self_attn
    prev_impl = getattr(attn.config, "_attn_implementation", None)
    prev_lm = getattr(language_model.config, "_attn_implementation", None)
    attn.config._attn_implementation = "eager"
    language_model.config._attn_implementation = "eager"
    orig = attn.forward
    box: dict[str, torch.Tensor] = {}

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        if not (isinstance(out, tuple) and len(out) >= 2):
            raise RuntimeError(f"GemmaAttention.forward did not return weights; got {type(out)}")
        weights = out[1]
        if weights is None:
            raise RuntimeError(
                f"layer {layer_idx} attention weights are None; need eager attention for CROP scout"
            )
        box["weights"] = weights
        return out

    attn.forward = wrapped
    try:
        yield box
    finally:
        attn.forward = orig
        if prev_impl is not None:
            attn.config._attn_implementation = prev_impl
        if prev_lm is not None:
            language_model.config._attn_implementation = prev_lm


@contextmanager
def _scout_stop_after(language_model, scout_layer: int):
    """Run prefix layers 0..scout_layer only; abort if a deeper layer is entered.

    Official inference is untouched: this context is installed only around the
    scout forward, then layers and forwards are restored before the compressed
    prefix. Dropped-layer keep-KV wrappers already on those modules stay in
    place for layers that do run (layer 1 under Fisher Drop-6).
    """
    orig_layers = language_model.layers
    n_layers = len(orig_layers)
    if scout_layer < 0 or scout_layer >= n_layers:
        raise IndexError(f"scout layer {scout_layer} out of range for {n_layers} layers")
    calls = [0] * n_layers
    restored: list[tuple[Any, Any]] = []
    for idx, layer in enumerate(orig_layers):
        orig_fwd = layer.forward

        def _counted(hidden_states, *args, _idx=idx, _orig=orig_fwd, **kwargs):
            calls[_idx] += 1
            if _idx > scout_layer:
                raise RuntimeError(
                    f"CROP scout executed language layer {_idx}; expected stop after {scout_layer}"
                )
            return _orig(hidden_states, *args, **kwargs)

        layer.forward = _counted
        restored.append((layer, orig_fwd))
    language_model.layers = orig_layers.__class__(list(orig_layers)[: scout_layer + 1])
    try:
        yield calls
    finally:
        language_model.layers = orig_layers
        for layer, orig_fwd in restored:
            layer.forward = orig_fwd


def _prefix_vlm(model, embs, pad_masks, att_masks, position_ids=None):
    if position_ids is None:
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_2d = make_att_2d_masks(pad_masks, att_masks)
    att_4d = prepare_attention_masks_4d(att_2d)
    model._paligemma_language_model_core().config._attn_implementation = "eager"
    _, past = model.paligemma_with_expert.forward(
        attention_mask=att_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[embs, None],
        use_cache=True,
    )
    return past, int(embs.shape[1])


def _kv_seq_len(past) -> int:
    if past is None:
        return 0
    getter = getattr(past, "get_seq_length", None)
    if callable(getter):
        return int(getter())
    if isinstance(past, (list, tuple)) and past:
        return int(past[0][0].shape[-2])
    raise TypeError(f"unknown past_key_values type {type(past)}")


def _relevance_scores(
    attn_weights: torch.Tensor,
    lang_index: torch.Tensor,
    vis_start: int,
    vis_n: int,
) -> torch.Tensor:
    """Mean over heads and valid language queries. attn: [B, H, Q, K]."""
    w = attn_weights.float()
    if w.ndim == 3:
        w = w.unsqueeze(0)
    if w.ndim != 4:
        raise RuntimeError(f"expected attention [B,H,Q,K], got {tuple(w.shape)}")
    bsz = w.shape[0]
    scores = []
    for b in range(bsz):
        q = lang_index[b].nonzero(as_tuple=False).squeeze(-1)
        if q.numel() == 0:
            scores.append(w.new_zeros(vis_n))
            continue
        chunk = w[b, :, q, vis_start : vis_start + vis_n]
        scores.append(chunk.mean(dim=(0, 1)))
    return torch.stack(scores, dim=0)


def _relevance_scores_dense(
    attn_weights: torch.Tensor,
    lang_index: torch.Tensor,
    vis_start: int,
    vis_n: int,
) -> torch.Tensor:
    """Same mean as `_relevance_scores`, without nonzero()."""
    w = attn_weights.float()
    if w.ndim == 3:
        w = w.unsqueeze(0)
    if w.ndim != 4:
        raise RuntimeError(f"expected attention [B,H,Q,K], got {tuple(w.shape)}")
    vis = w[:, :, :, vis_start : vis_start + vis_n]
    q_mask = lang_index.to(dtype=vis.dtype)[:, None, :, None]
    num = (vis * q_mask).sum(dim=(1, 2))
    n_q = lang_index.to(dtype=vis.dtype).sum(dim=1).clamp(min=1.0)
    den = n_q * vis.shape[1]
    return num / den[:, None]


def _camera_is_live(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 1:
        return mask.bool()
    return mask.reshape(mask.shape[0], -1)[:, -1].bool()


def _img_mask_1d(mask: torch.Tensor, n_tokens: int) -> torch.Tensor:
    return _camera_is_live(mask)[:, None].expand(mask.shape[0], n_tokens)


def _select_indices(
    slots: list[CameraSlot],
    img_embs: list[torch.Tensor],
    img_masks: list[torch.Tensor],
    attn_weights: torch.Tensor,
    lang_q_full: torch.Tensor,
    n_lang: int,
    *,
    keep_all: bool,
    budgets: dict[str, dict[str, int]] | None = None,
    selector: str = "greedy",
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """lang_q_full is bool [B, full_len] marking valid language queries."""
    budgets = resolve_budgets(budgets)
    selector = resolve_selector(selector)
    bsz = int(img_embs[0].shape[0])
    device = img_embs[0].device
    rows: list[torch.Tensor] = []
    meta_b0: list[dict[str, Any]] = []
    lang_start = slots[-1].start + slots[-1].n_tokens
    if selector == "lite":
        for b in range(bsz):
            parts: list[torch.Tensor] = []
            cam_meta = []
            for slot, emb, mask in zip(slots, img_embs, img_masks, strict=True):
                if slot.keep_all or keep_all:
                    local = torch.arange(slot.n_tokens, device=device, dtype=torch.long)
                    chosen_local, rel, div = local, local, local.new_empty(0)
                else:
                    feats = F.normalize(emb[b].float().reshape(slot.n_tokens, -1), dim=-1)
                    valid = _camera_is_live(mask)[b].expand(slot.n_tokens)
                    score = _relevance_scores_dense(
                        attn_weights[b : b + 1], lang_q_full[b : b + 1], slot.start, slot.n_tokens
                    )[0]
                    budget = budgets[slot.role]
                    chosen_local, rel, div = select_one_camera_lite(
                        score,
                        feats,
                        n_rel=budget["relevance"],
                        n_div=budget["diversity"],
                        valid=valid,
                        keep_all=False,
                    )
                parts.append(chosen_local + slot.start)
                if b == 0:
                    cam_meta.append(
                        {
                            "key": slot.key,
                            "role": slot.role,
                            "n_keep": int(chosen_local.shape[0]),
                            "n_full": slot.n_tokens,
                            "relevance": rel,
                            "diversity": div,
                            "keep": chosen_local,
                        }
                    )
            lang_ids = torch.arange(lang_start, lang_start + n_lang, device=device, dtype=torch.long)
            rows.append(torch.cat(parts + [lang_ids], dim=0))
            if b == 0:
                meta_b0 = cam_meta
        n_keep = int(rows[0].shape[0])
        for row in rows:
            if int(row.shape[0]) != n_keep:
                raise RuntimeError("CROP keep count differs across the batch; eval uses batch_size=1")
        return torch.stack(rows, dim=0), meta_b0

    for b in range(bsz):
        keep: list[int] = []
        cam_meta = []
        for slot, emb, mask in zip(slots, img_embs, img_masks, strict=True):
            local = list(range(slot.n_tokens))
            if slot.keep_all or keep_all:
                chosen, rel, div = local, local, []
            else:
                feats = F.normalize(emb[b].float().reshape(slot.n_tokens, -1), dim=-1)
                live = bool(_camera_is_live(mask)[b].item())
                valid = torch.full((slot.n_tokens,), live, dtype=torch.bool, device=device)
                score = _relevance_scores(
                    attn_weights[b : b + 1], lang_q_full[b : b + 1], slot.start, slot.n_tokens
                )[0]
                budget = budgets[slot.role]
                chosen, rel, div = select_one_camera(
                    score,
                    feats,
                    n_rel=budget["relevance"],
                    n_div=budget["diversity"],
                    valid=valid,
                    keep_all=keep_all,
                )
            keep.extend(slot.start + i for i in chosen)
            if b == 0:
                cam_meta.append(
                    {
                        "key": slot.key,
                        "role": slot.role,
                        "n_keep": len(chosen),
                        "n_full": slot.n_tokens,
                        "relevance": rel,
                        "diversity": div,
                        "keep": chosen,
                    }
                )
        keep.extend(range(lang_start, lang_start + n_lang))
        rows.append(torch.tensor(keep, device=device, dtype=torch.long))
        if b == 0:
            meta_b0 = cam_meta
    n_keep = int(rows[0].numel())
    for row in rows:
        if int(row.numel()) != n_keep:
            raise RuntimeError("CROP keep count differs across the batch; eval uses batch_size=1")
    return torch.stack(rows, dim=0), meta_b0


def _denoise_with_offset(model, prefix_pad_masks, past_key_values, x_t, timestep, prefix_offset):
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(x_t, timestep)
    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    position_ids = prefix_offset.to(dtype=torch.long)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
    full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    past_key_values = clone_past_key_values(past_key_values)
    outputs_embeds, _ = model.paligemma_with_expert.forward(
        attention_mask=full_att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    suffix_out = outputs_embeds[1][:, -model.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


def adjacent_mark_pairs(marks):
    """Pair consecutive timer marks. `zip(marks, marks[1:], strict=True)` raises."""
    return zip(marks[:-1], marks[1:], strict=True)


class _CudaTimer:
    def __init__(self, enabled: bool, device: torch.device | None = None):
        self.device = device if device is not None and device.type == "cuda" else None
        self.enabled = bool(enabled) and self.device is not None
        self.marks: list[tuple[str, torch.cuda.Event]] = []

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self.device))
        self.marks.append((name, event))

    def ms(self) -> dict[str, float] | None:
        if not self.enabled or len(self.marks) < 2:
            return None
        torch.cuda.synchronize(self.device)
        out: dict[str, float] = {}
        for (name0, ev0), (name1, ev1) in adjacent_mark_pairs(self.marks):
            out[name1] = float(ev0.elapsed_time(ev1))
        out["total"] = float(self.marks[0][1].elapsed_time(self.marks[-1][1]))
        return out


def sample_actions_crop(
    model,
    images,
    img_masks,
    tokens,
    masks,
    states=None,
    state_masks=None,
    noise=None,
    num_steps=None,
    *,
    keep_all: bool = False,
    scout_layer: int = SCOUT_LAYER,
    drop_layers: list[int] | None = None,
    profile: bool = False,
    budgets: dict[str, dict[str, int]] | str | None = None,
    budget_name: str = "role",
    selector: str = "greedy",
    trim_info: TrimInfo | None = None,
    **kwargs,
):
    """Scout + select + shortened official prefix + expert. Same noise/NFE as baseline."""
    del kwargs
    if num_steps is None:
        num_steps = model.config.num_inference_steps
    num_steps = int(num_steps)
    selector = resolve_selector(selector)
    drop_layers = list(drop_layers or [])
    if (not keep_all) and scout_layer in drop_layers:
        raise RuntimeError(
            f"CROP scout layer {scout_layer} is in drop_layers={drop_layers}; it must still run attention"
        )
    if getattr(model, "proprio_history_proj", None) is not None:
        raise RuntimeError("CROP assumes use_proprioceptive_memory=False (cameras then language)")
    budgets = resolve_budgets(budgets)

    bsize = tokens.shape[0]
    device = tokens.device
    if noise is None:
        noise = model.sample_noise(
            (bsize, model.config.chunk_size, model.config.max_action_dim),
            device,
        )

    timer = _CudaTimer(profile, device=device)
    timer.mark("start")
    prefix_embs, pad_masks, att_masks = model.embed_prefix(
        images, img_masks, tokens, masks, states, state_masks
    )
    timer.mark("embed")
    n_tok = TOKENS_PER_VIEW
    n_views = len(images)
    n_lang = int(tokens.shape[1])
    if int(prefix_embs.shape[1]) != n_tok * n_views + n_lang:
        raise RuntimeError(
            f"prefix layout {prefix_embs.shape[1]} != {n_tok}*{n_views}+{n_lang}"
        )
    img_embs = [prefix_embs[:, i * n_tok : (i + 1) * n_tok] for i in range(n_views)]
    if int(img_embs[0].shape[1]) != TOKENS_PER_VIEW:
        raise RuntimeError(f"expected {TOKENS_PER_VIEW} tokens per view, got {img_embs[0].shape[1]}")

    slots = []
    start = 0
    live_i = 0
    for i, mask in enumerate(img_masks):
        is_empty = not bool(_camera_is_live(mask).any().item())
        if is_empty:
            role, key = "empty", f"observation.images.empty_camera_{max(i - 2, 0)}"
        elif live_i == 0:
            role, key, live_i = "global", "observation.images.image", 1
        else:
            role, key, live_i = "wrist", "observation.images.image2", live_i + 1
        slots.append(
            CameraSlot(
                index=i,
                key=key,
                role=role,
                start=start,
                n_tokens=n_tok,
                keep_all=role == "empty",
            )
        )
        start += n_tok

    orig_pos = torch.cumsum(pad_masks, dim=1) - 1
    orig_offset = pad_masks.sum(dim=-1)
    att_len = int(prefix_embs.shape[1])

    lang_model = model._paligemma_language_model_core()
    with _scout_stop_after(lang_model, scout_layer) as scout_calls:
        with _capture_layer_attention(lang_model, scout_layer) as box:
            scout_past, _ = _prefix_vlm(model, prefix_embs, pad_masks, att_masks, orig_pos)
        del scout_past
    timer.mark("scout")
    attn_weights = box["weights"]
    if any(scout_calls[i] for i in range(scout_layer + 1, len(scout_calls))):
        raise RuntimeError(f"CROP scout ran layers beyond {scout_layer}: {scout_calls}")
    if scout_calls[scout_layer] < 1:
        raise RuntimeError(f"CROP scout layer {scout_layer} was not executed: {scout_calls}")
    lang_start = n_tok * n_views
    lang_q_full = torch.zeros(bsize, att_len, dtype=torch.bool, device=device)
    lang_q_full[:, lang_start:] = masks.bool()

    keep_idx, cam_meta = _select_indices(
        slots,
        img_embs,
        img_masks,
        attn_weights,
        lang_q_full,
        n_lang,
        keep_all=keep_all,
        budgets=budgets,
        selector=selector,
    )
    timer.mark("select")
    gather = keep_idx.unsqueeze(-1).expand(-1, -1, prefix_embs.shape[-1])
    short_embs = torch.gather(prefix_embs, 1, gather)
    short_pad = torch.gather(pad_masks, 1, keep_idx)
    short_att = torch.gather(att_masks, 1, keep_idx)
    short_pos = torch.gather(orig_pos, 1, keep_idx)

    past, short_len = _prefix_vlm(model, short_embs, short_pad, short_att, short_pos)
    timer.mark("prefix")
    kv_len = _kv_seq_len(past)
    if kv_len != short_len:
        raise RuntimeError(f"KV length {kv_len} != packed prefix {short_len}; sequence was not shortened")

    n_vel = [0]

    def _body(input_x_t, current_timestep):
        n_vel[0] += 1
        return _denoise_with_offset(
            model, short_pad, past, input_x_t, current_timestep, orig_offset
        )

    actions = euler_integrate(
        _body,
        noise,
        num_steps,
        rtc_processor=model.rtc_processor,
        rtc_enabled=False,
    )
    timer.mark("expert")
    if n_vel[0] != num_steps:
        raise RuntimeError(
            f"CROP velocity evaluations {n_vel[0]} != num_steps={num_steps} passed to euler_integrate"
        )
    deferred = {
        "orig_offset": orig_offset,
        "short_pad": short_pad,
        "cameras": cam_meta,
    }
    if selector == "lite":
        orig_off, pad_sum, cameras_out = 0, 0, []
    else:
        orig_off = int(orig_offset[0].item())
        pad_sum = int(short_pad[0].sum().item())
        cameras_out = cam_meta
        deferred = {}
    stats = CropStats(
        full_len=att_len,
        short_len=short_len,
        kv_len=kv_len,
        orig_action_offset=orig_off,
        short_pad_sum=pad_sum,
        cameras=cameras_out,
        scout_layer=scout_layer,
        keep_all=keep_all,
        timing_ms=timer.ms(),
        scout_layer_calls=list(scout_calls),
        budget_name=budget_name,
        selector=selector,
        num_steps=num_steps,
        velocity_calls=n_vel[0],
        deferred=deferred,
        prefix_trim=trim_info is not None,
        n_views_in=None if trim_info is None else trim_info.n_views_in,
        n_views_out=None if trim_info is None else trim_info.n_views_out,
        n_lang_in=None if trim_info is None else trim_info.n_lang_in,
        n_lang_out=None if trim_info is None else trim_info.n_lang_out,
        valid_prefix_len=None if trim_info is None else trim_info.valid_prefix_len,
        untrimmed_physical=None if trim_info is None else trim_info.untrimmed_physical,
        trimmed_physical=None if trim_info is None else trim_info.trimmed_physical,
    )
    model._crop_stats = stats
    return actions


def _camera_slots_from_masks(img_masks, n_tok: int) -> list[CameraSlot]:
    slots = []
    start = 0
    live_i = 0
    for i, mask in enumerate(img_masks):
        is_empty = not bool(_camera_is_live(mask).any().item())
        if is_empty:
            role, key = "empty", f"observation.images.empty_camera_{max(i - 2, 0)}"
        elif live_i == 0:
            role, key, live_i = "global", "observation.images.image", 1
        else:
            role, key, live_i = "wrist", "observation.images.image2", live_i + 1
        slots.append(
            CameraSlot(
                index=i,
                key=key,
                role=role,
                start=start,
                n_tokens=n_tok,
                keep_all=role == "empty",
            )
        )
        start += n_tok
    return slots


def sample_actions_grid(
    model,
    images,
    img_masks,
    tokens,
    masks,
    states=None,
    state_masks=None,
    noise=None,
    num_steps=None,
    *,
    keep_all: bool = False,
    profile: bool = False,
    trim_info: TrimInfo | None = None,
    **kwargs,
):
    """Encode once, pack fixed checkerboard indices, official short prefix, expert. No scout."""
    del kwargs
    if num_steps is None:
        num_steps = model.config.num_inference_steps
    num_steps = int(num_steps)
    if getattr(model, "proprio_history_proj", None) is not None:
        raise RuntimeError("CROP assumes use_proprioceptive_memory=False (cameras then language)")

    bsize = tokens.shape[0]
    device = tokens.device
    if noise is None:
        noise = model.sample_noise(
            (bsize, model.config.chunk_size, model.config.max_action_dim),
            device,
        )

    timer = _CudaTimer(profile, device=device)
    timer.mark("start")
    prefix_embs, pad_masks, att_masks = model.embed_prefix(
        images, img_masks, tokens, masks, states, state_masks
    )
    timer.mark("embed")
    n_tok = TOKENS_PER_VIEW
    n_views = len(images)
    n_lang = int(tokens.shape[1])
    if int(prefix_embs.shape[1]) != n_tok * n_views + n_lang:
        raise RuntimeError(
            f"prefix layout {prefix_embs.shape[1]} != {n_tok}*{n_views}+{n_lang}"
        )
    slots = _camera_slots_from_masks(img_masks, n_tok)
    orig_pos = torch.cumsum(pad_masks, dim=1) - 1
    orig_offset = pad_masks.sum(dim=-1)
    att_len = int(prefix_embs.shape[1])
    empty_flags = tuple(bool(slot.keep_all) for slot in slots)
    row = packed_grid_keep_idx(
        device, n_views, n_lang, empty_flags, keep_all=keep_all, n_tokens=n_tok, grid=GRID
    )
    keep_idx = row.unsqueeze(0).expand(bsize, -1).contiguous()
    gather = keep_idx.unsqueeze(-1).expand(-1, -1, prefix_embs.shape[-1])
    short_embs = torch.gather(prefix_embs, 1, gather)
    short_pad = torch.gather(pad_masks, 1, keep_idx)
    short_att = torch.gather(att_masks, 1, keep_idx)
    short_pos = torch.gather(orig_pos, 1, keep_idx)
    past, short_len = _prefix_vlm(model, short_embs, short_pad, short_att, short_pos)
    timer.mark("prefix")
    kv_len = _kv_seq_len(past)
    if kv_len != short_len:
        raise RuntimeError(f"KV length {kv_len} != packed prefix {short_len}; sequence was not shortened")

    n_vel = [0]

    def _body(input_x_t, current_timestep):
        n_vel[0] += 1
        return _denoise_with_offset(model, short_pad, past, input_x_t, current_timestep, orig_offset)

    actions = euler_integrate(
        _body,
        noise,
        num_steps,
        rtc_processor=model.rtc_processor,
        rtc_enabled=False,
    )
    timer.mark("expert")
    timing_ms = timer.ms()
    if n_vel[0] != num_steps:
        raise RuntimeError(
            f"grid velocity evaluations {n_vel[0]} != num_steps={num_steps} passed to euler_integrate"
        )
    local_grid = uniform_grid_local_indices(device, n_tok, GRID)
    meta_b0 = []
    for slot in slots:
        if slot.keep_all or keep_all:
            local = torch.arange(slot.n_tokens, device=device, dtype=torch.long)
        else:
            local = local_grid
        meta_b0.append(
            {
                "key": slot.key,
                "role": slot.role,
                "n_keep": int(local.shape[0]),
                "n_full": slot.n_tokens,
                "relevance": local.new_empty(0),
                "diversity": local.new_empty(0),
                "keep": local,
            }
        )
    n_layers = len(model._paligemma_language_model_core().layers)
    stats = CropStats(
        full_len=att_len,
        short_len=short_len,
        kv_len=kv_len,
        orig_action_offset=0,
        short_pad_sum=0,
        cameras=[],
        scout_layer=-1,
        keep_all=keep_all,
        timing_ms=timing_ms,
        scout_layer_calls=[0] * n_layers,
        budget_name="grid",
        selector="grid",
        num_steps=num_steps,
        velocity_calls=n_vel[0],
        deferred={"orig_offset": orig_offset, "short_pad": short_pad, "cameras": meta_b0},
        prefix_trim=trim_info is not None,
        n_views_in=None if trim_info is None else trim_info.n_views_in,
        n_views_out=None if trim_info is None else trim_info.n_views_out,
        n_lang_in=None if trim_info is None else trim_info.n_lang_in,
        n_lang_out=None if trim_info is None else trim_info.n_lang_out,
        valid_prefix_len=None if trim_info is None else trim_info.valid_prefix_len,
        untrimmed_physical=None if trim_info is None else trim_info.untrimmed_physical,
        trimmed_physical=None if trim_info is None else trim_info.trimmed_physical,
    )
    model._crop_stats = stats
    return actions


PROFILE_ALIGN_MAX_ABS = 2e-2


@contextmanager
def profile_official_actions(policy):
    """Time official sample_actions by wrapping the original methods.

    CUDA events only. embed_prefix, paligemma prefix forward, denoise_step, and
    euler_integrate stay the original implementations.
    """
    model = policy.model
    orig_sample = model.sample_actions
    orig_embed = model.embed_prefix
    orig_denoise = model.denoise_step
    box: dict[str, Any] = {}

    def embed_prefix(*args, **kwargs):
        out = orig_embed(*args, **kwargs)
        timer = box.get("timer")
        if timer is not None:
            timer.mark("embed")
            box["prefix_embs"] = out[0]
            box["prefix_pad"] = out[1]
        return out

    def denoise_step(*args, **kwargs):
        timer = box.get("timer")
        n_vel = int(box.get("n_vel") or 0)
        if n_vel == 0 and timer is not None:
            timer.mark("prefix")
        box["n_vel"] = n_vel + 1
        return orig_denoise(*args, **kwargs)

    def sample_actions(images, img_masks, tokens, masks, states=None, state_masks=None, noise=None, num_steps=None, **kwargs):
        device = tokens.device
        timer = _CudaTimer(True, device=device)
        box.clear()
        box["timer"] = timer
        box["n_vel"] = 0
        timer.mark("start")
        actions = orig_sample(
            images,
            img_masks,
            tokens,
            masks,
            states=states,
            state_masks=state_masks,
            noise=noise,
            num_steps=num_steps,
            **kwargs,
        )
        timer.mark("expert")
        timing_ms = timer.ms()
        n_vel = int(box.get("n_vel") or 0)
        if num_steps is None:
            num_steps = model.config.num_inference_steps
        num_steps = int(num_steps)
        if n_vel != num_steps:
            raise RuntimeError(f"official velocity evaluations {n_vel} != num_steps={num_steps}")
        embs = box.get("prefix_embs")
        pads = box.get("prefix_pad")
        full_len = int(embs.shape[1]) if embs is not None else 0
        pad_sum = int(pads[0].sum().item()) if pads is not None else 0
        n_layers = len(model._paligemma_language_model_core().layers)
        model._crop_stats = CropStats(
            full_len=full_len,
            short_len=full_len,
            kv_len=full_len,
            orig_action_offset=pad_sum,
            short_pad_sum=pad_sum,
            cameras=[],
            scout_layer=-1,
            keep_all=True,
            timing_ms=timing_ms,
            scout_layer_calls=[0] * n_layers,
            budget_name="baseline",
            selector="baseline",
            num_steps=num_steps,
            velocity_calls=n_vel,
        )
        return actions

    model.embed_prefix = embed_prefix
    model.denoise_step = denoise_step
    model.sample_actions = sample_actions
    try:
        yield
    finally:
        model.embed_prefix = orig_embed
        model.denoise_step = orig_denoise
        model.sample_actions = orig_sample


def align_profiled_baseline(policy, batch, noise, num_steps: int) -> dict[str, Any]:
    """Same input/noise: original sample_actions vs profile-wrapped original path."""
    num_steps = int(num_steps)
    lang = policy.model._paligemma_language_model_core()
    param = next(policy.model.parameters())
    policy.reset()
    official = policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)
    with profile_official_actions(policy):
        policy.reset()
        profiled = policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)
        stats = getattr(policy.model, "_crop_stats", None)
    x = official[:, :10, :7].float()
    y = profiled[:, :10, :7].float()
    max_abs = float((x - y).abs().max())
    timing_keys = sorted((stats.timing_ms or {}).keys()) if stats is not None else []
    return {
        "method": "wrap original embed_prefix, denoise_step, sample_actions; CUDA events only",
        "max_abs": max_abs,
        "max_abs_limit": PROFILE_ALIGN_MAX_ABS,
        "aligned": max_abs <= PROFILE_ALIGN_MAX_ABS,
        "finite": bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
        "official_dtype": str(official.dtype),
        "profiled_dtype": str(profiled.dtype),
        "param_dtype": str(param.dtype),
        "attn_implementation": getattr(lang.config, "_attn_implementation", None),
        "velocity_calls": None if stats is None else stats.velocity_calls,
        "num_steps": num_steps,
        "timing_keys": timing_keys,
    }


@contextmanager
def crop_visual_tokens(
    policy,
    *,
    enabled: bool = True,
    keep_all: bool = False,
    scout_layer: int = SCOUT_LAYER,
    drop_layers: list[int] | None = None,
    profile: bool = False,
    budgets: dict[str, dict[str, int]] | str | None = None,
    budget_name: str = "role",
    selector: str = "greedy",
    prefix_trim: bool = False,
):
    """Replace PI05Pytorch.sample_actions with the CROP scout / grid path."""
    if not enabled:
        yield
        return
    model = policy.model
    orig = model.sample_actions
    selector = resolve_selector(selector)

    def patched(images, img_masks, tokens, masks, states=None, state_masks=None, noise=None, num_steps=None, **kwargs):
        trim_info = None
        if prefix_trim:
            images, img_masks, tokens, masks, trim_info = trim_prefix_inputs(
                images, img_masks, tokens, masks
            )
            model._trim_stats = trim_info
        if selector == "grid":
            return sample_actions_grid(
                model,
                images,
                img_masks,
                tokens,
                masks,
                states=states,
                state_masks=state_masks,
                noise=noise,
                num_steps=num_steps,
                keep_all=keep_all,
                profile=profile,
                trim_info=trim_info,
                **kwargs,
            )
        return sample_actions_crop(
            model,
            images,
            img_masks,
            tokens,
            masks,
            states=states,
            state_masks=state_masks,
            noise=noise,
            num_steps=num_steps,
            keep_all=keep_all,
            scout_layer=scout_layer,
            drop_layers=drop_layers,
            profile=profile,
            budgets=resolve_budgets(budgets if budgets is not None else budget_name),
            budget_name=budget_name,
            selector=selector,
            trim_info=trim_info,
            **kwargs,
        )

    model.sample_actions = patched
    try:
        yield
    finally:
        model.sample_actions = orig


def bind_cuda_device(physical: int) -> torch.device:
    """Use physical GPU `physical` as the only visible device when possible.

    `--gpu N` is the nvidia-smi index. If CUDA_VISIBLE_DEVICES is unset, hide
    every other card before the first CUDA allocation. If it is already set,
    N must be in that list; the process then uses the corresponding logical
    device. Events and synchronize go through the returned `torch.device`.
    """
    physical = int(physical)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is None or vis.strip() == "":
        if torch.cuda.is_initialized():
            torch.cuda.set_device(physical)
            return torch.device(f"cuda:{physical}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical)
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    allowed = [int(part.strip()) for part in vis.split(",") if part.strip() != ""]
    if physical not in allowed:
        raise RuntimeError(
            f"requested GPU {physical} is not in CUDA_VISIBLE_DEVICES={vis}; "
            "refusing to address another device"
        )
    logical = allowed.index(physical)
    torch.cuda.set_device(logical)
    return torch.device(f"cuda:{logical}")


def gpu_occupancy(gpu: int) -> list[dict[str, Any]]:
    """Processes currently using `gpu`, with usernames."""
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        pid_s, name, mem = [p.strip() for p in line.split(",", 2)]
        pid = int(pid_s)
        try:
            user = subprocess.check_output(["ps", "-o", "user=", "-p", str(pid)], text=True).strip()
        except subprocess.CalledProcessError:
            user = "?"
        rows.append({"pid": pid, "user": user, "name": name, "mem_mib": mem})
    return rows


def gpu_is_exclusive(gpu: int, *, allow_users: tuple[str, ...] | None = None) -> tuple[bool, str]:
    """True iff the card has no compute apps. Formal CROP work needs a fully idle GPU."""
    del allow_users
    if os.environ.get("CROP_ALLOW_OCCUPIED") == "1":
        rows = gpu_occupancy(gpu)
        desc = ", ".join(f"{r['user']}:{r['pid']}" for r in rows) or "idle"
        return True, f"CROP_ALLOW_OCCUPIED=1 ({desc})"
    rows = gpu_occupancy(gpu)
    if not rows:
        return True, "idle"
    desc = ", ".join(f"{r['user']}:{r['pid']}:{r['name']}" for r in rows)
    return False, f"occupied ({desc})"


def require_idle_gpu(gpu: int | None = None, gpus: tuple[int, ...] = (0, 1, 2, 3)) -> int:
    if gpu is not None:
        ok, why = gpu_is_exclusive(gpu)
        if not ok:
            raise SystemExit(
                f"GPU{gpu} is not idle ({why}). CROP eval/efficiency waits for an exclusive card."
            )
        return gpu
    found, note = first_exclusive_gpu(gpus)
    if found is None:
        raise SystemExit("NO_IDLE_GPU: " + note)
    return found


def first_exclusive_gpu(gpus: tuple[int, ...] = (0, 1, 2, 3)) -> tuple[int | None, str]:
    notes = []
    for g in gpus:
        ok, why = gpu_is_exclusive(g)
        notes.append(f"gpu{g} {why}")
        if ok:
            return g, "; ".join(notes)
    return None, "; ".join(notes)


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _materialize_cam(cam: dict[str, Any]) -> dict[str, Any]:
    out = dict(cam)
    for key in ("relevance", "diversity", "keep"):
        if key in out:
            out[key] = _as_int_list(out[key])
    if "n_keep" in out:
        out["n_keep"] = int(out["n_keep"])
    if "n_full" in out:
        out["n_full"] = int(out["n_full"])
    return out


def stats_dict(stats: CropStats) -> dict[str, Any]:
    """CPU/debug export. Call after the timed chunk, not inside it."""
    payload = {f.name: getattr(stats, f.name) for f in fields(stats) if f.name != "deferred"}
    deferred = stats.deferred or {}
    cameras = deferred.get("cameras", payload.get("cameras") or [])
    payload["cameras"] = [_materialize_cam(cam) for cam in cameras]
    off = deferred.get("orig_offset")
    pad = deferred.get("short_pad")
    if off is not None:
        payload["orig_action_offset"] = int(off[0].detach().cpu())
    if pad is not None:
        payload["short_pad_sum"] = int(pad[0].sum().detach().cpu())
    return payload


def count_official_velocity(policy, batch, noise, num_steps: int) -> int:
    """Count PI05 denoise_step calls on the official (non-CROP) path."""
    model = policy.model
    orig = model.denoise_step
    n = [0]

    def wrapped(*args, **kwargs):
        n[0] += 1
        return orig(*args, **kwargs)

    model.denoise_step = wrapped
    try:
        policy.reset()
        policy.predict_action_chunk(batch, noise=noise, num_steps=int(num_steps))
    finally:
        model.denoise_step = orig
    return n[0]


def save_stats_json(path, stats: CropStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats_dict(stats), indent=2) + "\n")


def _greedy_diversity_naive(feats: torch.Tensor, seed: list[int], n_extra: int) -> list[int]:
    """Reference greedy diversity (recompute full similarity each round). CPU tests only."""
    if n_extra <= 0:
        return []
    n = int(feats.shape[0])
    selected = list(seed)
    taken = torch.zeros(n, dtype=torch.bool, device=feats.device)
    taken[selected] = True
    extra: list[int] = []
    for _ in range(n_extra):
        if bool(taken.all()):
            break
        nearest = (feats @ feats[selected].T).max(dim=1).values
        nearest = nearest.masked_fill(taken, 2.0)
        pick = int(nearest.argmin().item())
        extra.append(pick)
        selected.append(pick)
        taken[pick] = True
    return extra


def cpu_selftest() -> None:
    """Timer pairing, incremental diversity, and scout layer-range. No GPU."""
    marks = [("start", 0), ("embed", 1), ("scout", 2), ("select", 3)]
    pairs = list(adjacent_mark_pairs(marks))
    if [a[0] for a, _ in pairs] != ["start", "embed", "scout"]:
        raise RuntimeError(f"adjacent_mark_pairs left names {pairs}")
    if [b[0] for _, b in pairs] != ["embed", "scout", "select"]:
        raise RuntimeError(f"adjacent_mark_pairs right names {pairs}")
    try:
        list(zip(marks, marks[1:], strict=True))
        raise RuntimeError("zip(marks, marks[1:], strict=True) should raise")
    except ValueError:
        pass

    torch.manual_seed(0)
    feats = F.normalize(torch.randn(64, 8), dim=-1)
    seed = list(range(8))
    got = greedy_diversity(feats, seed, 12)
    ref = _greedy_diversity_naive(feats, seed, 12)
    if got != ref:
        raise RuntimeError(f"incremental diversity {got} != naive {ref}")

    torch.manual_seed(1)
    n = 32
    feats = F.normalize(torch.randn(n, 8), dim=-1)
    score = torch.linspace(1.0, 0.0, n)
    keep, rel, div = select_one_camera_lite(score, feats, n_rel=12, n_div=8)
    if int(keep.shape[0]) != 20 or int(rel.shape[0]) != 12 or int(div.shape[0]) != 8:
        raise RuntimeError(f"lite keep/rel/div shapes {tuple(keep.shape)} {tuple(rel.shape)} {tuple(div.shape)}")
    keep_l, rel_l, div_l = keep.tolist(), rel.tolist(), div.tolist()
    if keep_l != sorted(keep_l) or len(set(keep_l)) != 20 or set(rel_l) & set(div_l):
        raise RuntimeError(f"lite uniqueness/sort failed keep={keep_l}")
    if rel_l != list(range(12)):
        raise RuntimeError(f"lite relevance should be top-12 scores, got {rel_l}")
    anchors = set(torch.topk(score, k=4).indices.tolist())
    naive_div_score = 1.0 - (feats @ feats[list(anchors)].T).max(dim=1).values
    naive_div_score[list(range(12))] = -1e9
    naive_div = set(torch.topk(naive_div_score, k=8).indices.tolist())
    if set(div_l) != naive_div:
        raise RuntimeError(f"lite diversity {set(div_l)} != naive {naive_div}")
    keep_all, _, _ = select_one_camera_lite(score, feats, n_rel=12, n_div=8, keep_all=True)
    if keep_all.tolist() != list(range(n)):
        raise RuntimeError("lite keep_all did not return every token")

    grid_idx = uniform_grid_local_indices("cpu")
    expect = [i for i in range(TOKENS_PER_VIEW) if ((i // GRID) + (i % GRID)) % 2 == 0]
    if grid_idx.tolist() != expect:
        raise RuntimeError(f"uniform-grid indices {grid_idx[:8].tolist()}... != checkerboard")
    if int(grid_idx.numel()) != TOKENS_PER_VIEW // 2 or len(set(expect)) != TOKENS_PER_VIEW // 2:
        raise RuntimeError(f"uniform-grid keep {int(grid_idx.numel())} not 128 unique even-parity")
    packed = packed_grid_keep_idx("cpu", 3, 200, (False, False, True), keep_all=False)
    if int(packed.numel()) != 128 + 128 + 256 + 200:
        raise RuntimeError(f"packed grid prefix {int(packed.numel())} != 712")
    if packed_grid_keep_idx("cpu", 3, 200, (False, False, True), keep_all=False) is not packed:
        raise RuntimeError("packed_grid_keep_idx did not reuse cache")
    full_pack = packed_grid_keep_idx("cpu", 3, 200, (False, False, True), keep_all=True)
    if int(full_pack.numel()) != 3 * TOKENS_PER_VIEW + 200:
        raise RuntimeError("grid keep-all pack length is not the full prefix")
    trimmed_pack = packed_grid_keep_idx("cpu", 2, 48, (False, False), keep_all=False)
    if int(trimmed_pack.numel()) != 128 + 128 + 48:
        raise RuntimeError(f"trim+grid pack {int(trimmed_pack.numel())} != 304")
    trimmed_all = packed_grid_keep_idx("cpu", 2, 48, (False, False), keep_all=True)
    if int(trimmed_all.numel()) != 2 * TOKENS_PER_VIEW + 48:
        raise RuntimeError("trim+grid keep-all pack is not 2*256+lang")

    class _Block(torch.nn.Module):
        def __init__(self, idx: int):
            super().__init__()
            self.idx = idx

        def forward(self, x, **kwargs):
            return x + 1.0

    fake = torch.nn.Module()
    fake.layers = torch.nn.ModuleList([_Block(i) for i in range(6)])
    deep = fake.layers[5]
    x = torch.zeros(1)
    with _scout_stop_after(fake, 2) as calls:
        for layer in fake.layers:
            x = layer(x)
        if len(fake.layers) != 3:
            raise RuntimeError(f"scout ModuleList was {len(fake.layers)}, expected 3")
        try:
            deep(x)
            raise RuntimeError("layer 5 should not run during scout")
        except RuntimeError as exc:
            if "executed language layer 5" not in str(exc):
                raise
    if list(calls) != [1, 1, 1, 0, 0, 1]:
        raise RuntimeError(f"scout stop calls {calls}")
    if len(fake.layers) != 6:
        raise RuntimeError("scout did not restore language layers")
    if float(x.item()) != 3.0:
        raise RuntimeError(f"scout ran {float(x.item())} layers worth of residual")
    print("cpu selftest ok (timer pairs, diversity, scout range, uniform-grid)", flush=True)

#!/usr/bin/env python3
"""Offline action-discontinuity weights for Stage-2 calibration.

This is the DySL continuity statistic, applied to demonstration actions that
already sit in the loss's normalized space. It is not DySL's online skip
controller.

w_i = mean_j ||a_j - a_{j-1}||_2 over the first H valid steps of the window.
H is the eval action horizon (10), not a closed-loop-tuned k.
High w = less continuous = more important for scoring.

π0.5 / SmolVLA: the window is the first 10 valid steps of the 50-step chunk
already stored with the calibration frame (same tensor the loss sees).

Molmo: the stored loss action is one 7-d step. The window is the same
episode's next H normalized actions, read separately. The loss still uses
the original single step.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

HORIZON = 10
ACTION_DIM = 7
ARM_DIM = 6


def _as_seq(action: torch.Tensor, pad: torch.Tensor | None, horizon: int) -> torch.Tensor:
    a = action.detach().float().cpu()
    while a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    if a.ndim == 1:
        a = a.unsqueeze(0)
    if a.shape[-1] > ACTION_DIM:
        a = a[..., :ACTION_DIM]
    if pad is not None:
        p = pad.detach().cpu().bool().reshape(-1)
        n = min(p.numel(), a.shape[0])
        valid = (~p[:n]).nonzero(as_tuple=False).flatten()
        a = a[valid] if valid.numel() else a[:0]
    return a[:horizon]


def mean_adjacent_l2(seq: torch.Tensor, dims: int | None = None) -> float | None:
    if seq.ndim != 2 or seq.shape[0] < 2:
        return None
    x = seq[:, :dims] if dims is not None else seq
    delta = (x[1:] - x[:-1]).norm(dim=-1)
    return float(delta.mean().item())


def chunk_discontinuity(
    action: torch.Tensor,
    pad: torch.Tensor | None = None,
    *,
    horizon: int = HORIZON,
    dims: int | None = ACTION_DIM,
) -> tuple[float | None, int]:
    seq = _as_seq(action, pad, horizon)
    return mean_adjacent_l2(seq, dims=dims), int(seq.shape[0])


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 3:
        return float("nan")
    ra = rankdata_average(a)
    rb = rankdata_average(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom <= 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def ess(weights) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0:
        return 0.0
    w = w / s
    return float(1.0 / (w * w).sum())


def fill_and_mean_normalize(raw: list[float | None]) -> tuple[np.ndarray, dict[str, Any]]:
    finite = np.array([x for x in raw if x is not None], dtype=np.float64)
    if finite.size == 0:
        raise RuntimeError("no finite discontinuity values")
    median = float(np.median(finite))
    filled = np.array([median if x is None else float(x) for x in raw], dtype=np.float64)
    mean = float(filled.mean())
    if mean <= 0:
        raise RuntimeError(f"discontinuity mean must be > 0, got {mean}")
    weights = filled / mean
    qs = np.quantile(filled, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    stats = {
        "n": int(len(raw)),
        "n_filled": int(sum(x is None for x in raw)),
        "median_raw": median,
        "mean_raw": mean,
        "quantiles_raw": [float(v) for v in qs],
        "ess": ess(weights),
        "cv": float(filled.std() / mean) if mean else float("nan"),
        "weight_mean": float(weights.mean()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
    }
    return weights, stats


def drop_from_redundancy(order: list[int], k: int, *, exclude_l0: bool) -> list[int]:
    seq = [int(i) for i in order]
    if exclude_l0:
        seq = [i for i in seq if i != 0]
    return sorted(seq[:k])


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

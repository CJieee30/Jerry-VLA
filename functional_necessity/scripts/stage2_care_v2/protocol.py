"""CARE v2 protocol constants and ranking helpers.

Frozen before E0: 40 tasks × 12 frames, H=10, per-dimension whitened adjacent
L2 on raw 7-d demonstration actions, mean-normalized α, per-frame scores.
No 6-d branch. Do not tune these on closed-loop SR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

FN_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = FN_ROOT / "results" / "stage2_care_v2"
COMPOSE_DIR = OUT_DIR / "composition"
MOLMO_ROOT = OUT_DIR / "molmo_root"
LIBERO_REPO = "lerobot/libero"
MOLMO_REPO = "allenai/MolmoAct2-LIBERO-Dataset"

SEED = 42
HORIZON = 10
ACTION_DIM = 7
FRAMES_PER_TASK = 12
N_TASKS = 40
N_FRAMES = N_TASKS * FRAMES_PER_TASK  # 480

STAGE1 = {
    "pi05": {
        "k": 6,
        "exclude_l0": True,
        "n_layers": 18,
        "fisher": [1, 13, 14, 15, 16, 17],
        "taylor": [1, 2, 14, 15, 16, 17],
        "gateprobe": [2, 4, 14, 15, 16, 17],
        "score_dir": FN_ROOT / "results" / "scores",
        "score_key": "paligemma",
    },
    "molmoact2": {
        "k": 16,
        "exclude_l0": True,
        "n_layers": 36,
        "fisher": [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        "taylor": [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        "gateprobe": [2, 3, 4, 6, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        "score_dir": FN_ROOT / "results" / "molmoact2_libero" / "scores",
        "score_key": "scores",
    },
    "smolvla": {
        "k": 4,
        "exclude_l0": False,
        "n_layers": 16,
        "fisher": [10, 11, 12, 14],
        "taylor": [12, 13, 14, 15],
        "gateprobe": [7, 13, 14, 15],
        "score_dir": FN_ROOT / "results" / "smolvla_libero" / "scores",
        "score_key": "scores",
    },
}

METHODS = ("fisher", "taylor", "gateprobe")
STACKS = ("pi05", "smolvla", "molmoact2")


def hub_snapshot(repo_id: str) -> Path:
    cache = Path.home() / ".cache" / "huggingface" / "lerobot" / "hub"
    name = "datasets--" + repo_id.replace("/", "--")
    snaps = sorted((cache / name / "snapshots").glob("*"))
    if not snaps:
        raise FileNotFoundError(f"no hub snapshot for {repo_id}")
    for snap in snaps:
        if (snap / "meta" / "info.json").is_file():
            return snap
    raise FileNotFoundError(f"no meta/info.json under {cache / name}")


def molmo_parquet_rels(records, *, max_file_index: int | None = None) -> list[str]:
    """Parquet shards that can contain the shared frames.

    MolmoAct2 episode metadata ``data/file_index`` is the writer-assigned start
    file. About 1/5 of CARE v2 frames live in the next shard, never further.
    """
    files: set[str] = set()
    for rec in records:
        chunk = int(rec["chunk_index"])
        fid = int(rec["file_index"])
        for file_index in (fid, fid + 1):
            if file_index < 0:
                continue
            if max_file_index is not None and file_index > max_file_index:
                continue
            files.add(f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet")
    return sorted(files)


def importance_ranking(scores) -> list[int]:
    values = [float(x) for x in scores]
    keep_order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return list(reversed(keep_order))


def drop_from_redundancy(order: list[int], k: int, *, exclude_l0: bool) -> list[int]:
    seq = [int(i) for i in order]
    if exclude_l0:
        seq = [i for i in seq if i != 0]
    return sorted(seq[:k])


def jaccard(a, b) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def ess(weights) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0:
        return 0.0
    w = w / s
    return float(1.0 / (w * w).sum())


def mean_normalize(raw: np.ndarray) -> np.ndarray:
    mean = float(np.nanmean(raw))
    if not np.isfinite(mean) or mean <= 0:
        raise RuntimeError(f"discontinuity mean must be > 0, got {mean}")
    filled = np.where(np.isfinite(raw), raw, float(np.nanmedian(raw)))
    return filled / mean


def weighted_scores(frame_scores: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """frame_scores: (N, L). α mean is 1, so this is (1/N) Σ α_i S_i."""
    a = alpha.reshape(-1, 1).to(dtype=frame_scores.dtype)
    if int(a.shape[0]) != int(frame_scores.shape[0]):
        raise ValueError(f"alpha {tuple(a.shape)} vs scores {tuple(frame_scores.shape)}")
    return (frame_scores * a).mean(dim=0)


def drop_set(scores, stack: str) -> list[int]:
    spec = STAGE1[stack]
    order = importance_ranking(scores)
    return drop_from_redundancy(order, spec["k"], exclude_l0=spec["exclude_l0"])

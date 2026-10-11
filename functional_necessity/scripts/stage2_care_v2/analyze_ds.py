#!/usr/bin/env python3
"""Compare uniform / discontinuous / smooth calibration drop sets.

CPU only. Uniform scores are the existing CARE v2 480-frame means (no α).
Does not open closed loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from scipy.stats import kendalltau

from protocol import (
    COMPOSE_DIR,
    METHODS,
    OUT_DIR,
    STACKS,
    STAGE1,
    drop_from_redundancy,
    importance_ranking,
    jaccard,
)


EXTRA_K = {
    "pi05": (6, 8),
    "smolvla": (4, 8),
    "molmoact2": (16, 17),
}

KNOWN = {
    ("pi05", 6, "fisher"): ("S1 Fisher-6 97.75", [1, 13, 14, 15, 16, 17]),
    ("pi05", 6, "taylor"): ("S1 Taylor-6 97.25", [1, 2, 14, 15, 16, 17]),
    ("pi05", 6, "gateprobe"): ("CARE-v1 GP-6 97.00", [2, 13, 14, 15, 16, 17]),
    ("pi05", 8, "fisher"): ("S1 Fisher-8 85.50", [1, 2, 3, 13, 14, 15, 16, 17]),
    ("pi05", 8, "taylor"): ("S1 Taylor-8 85.50", [1, 2, 3, 13, 14, 15, 16, 17]),
    ("pi05", 8, "gateprobe"): ("S1 GP-8 85.75", [2, 4, 10, 13, 14, 15, 16, 17]),
    ("molmoact2", 16, "fisher"): ("S1 Fisher-16 72.75", [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]),
    ("molmoact2", 16, "taylor"): ("S1 Taylor-16 98.10", [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]),
    ("molmoact2", 16, "gateprobe"): ("S1 GP-16 87.00", [2, 3, 4, 6, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]),
    ("molmoact2", 17, "fisher"): ("S1 Fisher-17 70.60", [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]),
    ("molmoact2", 17, "taylor"): ("S1 Taylor-17 89.60", [1, 2, 3, 4, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]),
    ("molmoact2", 17, "gateprobe"): ("S1 GP-17 82.65", [2, 3, 4, 6, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]),
    ("smolvla", 4, "taylor"): ("S1 Taylor-4 78.50", [12, 13, 14, 15]),
}


def _load(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return torch.load(path, map_location="cpu", weights_only=False)["frame_scores"].float()


def _drop(scores, stack: str, k: int) -> list[int]:
    spec = STAGE1[stack]
    return drop_from_redundancy(importance_ranking(scores), k, exclude_l0=spec["exclude_l0"])


def _tag(stack: str, k: int, method: str, drop: list[int]) -> str:
    known = KNOWN.get((stack, k, method))
    if known and drop == known[1]:
        return known[0]
    for (st, kk, meth), (label, layers) in KNOWN.items():
        if st == stack and kk == k and drop == layers:
            return label
    if drop == [2, 3, 14, 15, 16, 17] and stack == "pi05" and k == 6:
        return "high-α GP-6 (running)"
    return "new"


def main() -> None:
    rows = []
    for stack in STACKS:
        for method in METHODS:
            uni = _load(OUT_DIR / "scores" / f"{stack}_{method}.pt").mean(0)
            disc = _load(COMPOSE_DIR / "scores" / "disc" / f"{stack}_{method}.pt").mean(0)
            smooth = _load(COMPOSE_DIR / "scores" / "smooth" / f"{stack}_{method}.pt").mean(0)
            tau_du = float(kendalltau(disc.numpy(), uni.numpy()).correlation)
            tau_su = float(kendalltau(smooth.numpy(), uni.numpy()).correlation)
            tau_ds = float(kendalltau(disc.numpy(), smooth.numpy()).correlation)
            for k in EXTRA_K[stack]:
                du, dd, ds = _drop(uni, stack, k), _drop(disc, stack, k), _drop(smooth, stack, k)
                row = {
                    "stack": stack,
                    "method": method,
                    "k": k,
                    "uniform": du,
                    "disc": dd,
                    "smooth": ds,
                    "jaccard_disc_uniform": jaccard(dd, du),
                    "jaccard_smooth_uniform": jaccard(ds, du),
                    "jaccard_disc_smooth": jaccard(dd, ds),
                    "kendall_disc_uniform": tau_du,
                    "kendall_smooth_uniform": tau_su,
                    "kendall_disc_smooth": tau_ds,
                    "tag_uniform": _tag(stack, k, method, du),
                    "tag_disc": _tag(stack, k, method, dd),
                    "tag_smooth": _tag(stack, k, method, ds),
                }
                rows.append(row)
                print(
                    f"{stack:10s} {method:10s} k={k:2d} "
                    f"J(D,U)={row['jaccard_disc_uniform']:.2f} "
                    f"J(S,U)={row['jaccard_smooth_uniform']:.2f} "
                    f"J(D,S)={row['jaccard_disc_smooth']:.2f} "
                    f"D={dd} [{row['tag_disc']}]  "
                    f"S={ds} [{row['tag_smooth']}]",
                    flush=True,
                )

    payload = {
        "protocol": "care_v2_composition",
        "note": (
            "Same 480 episodes. U = v2 random timestamps, equal mean. "
            "D = max whitened-d frame. S = min d among frames above pool 10th percentile. "
            "No α. No closed loop."
        ),
        "rows": rows,
    }
    (COMPOSE_DIR / "composition.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# CARE v2 composition: discontinuous vs smooth vs uniform timestamps",
        "",
        payload["note"],
        "",
        "| Stack | Method | k | U | D | S | J(D,U) | J(S,U) | J(D,S) | U tag | D tag | S tag |",
        "|---|---|---:|---|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stack']} | {row['method']} | {row['k']} | `{row['uniform']}` | `{row['disc']}` | "
            f"`{row['smooth']}` | {row['jaccard_disc_uniform']:.2f} | {row['jaccard_smooth_uniform']:.2f} | "
            f"{row['jaccard_disc_smooth']:.2f} | {row['tag_uniform']} | {row['tag_disc']} | {row['tag_smooth']} |"
        )
    (COMPOSE_DIR / "composition.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {COMPOSE_DIR / 'composition.md'}", flush=True)


if __name__ == "__main__":
    main()

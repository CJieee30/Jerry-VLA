#!/usr/bin/env python3
"""E1: CARE v2 uniform / weighted drop sets vs Stage-1 operating points.

CPU only. Writes results/stage2_care_v2/e1.{md,json}. Does not open closed loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from scipy.stats import kendalltau

from protocol import (
    METHODS,
    OUT_DIR,
    STACKS,
    STAGE1,
    drop_set,
    jaccard,
    weighted_scores,
)


def _stage1_scores(stack: str, method: str) -> np.ndarray:
    spec = STAGE1[stack]
    blob = torch.load(spec["score_dir"] / f"{method}.pt", map_location="cpu", weights_only=False)
    key = spec["score_key"]
    val = blob[key]
    if torch.is_tensor(val):
        return val.detach().cpu().float().numpy()
    return np.asarray(val, dtype=np.float64)


def _load_frame(stack: str, method: str) -> torch.Tensor:
    path = OUT_DIR / "scores" / f"{stack}_{method}.pt"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["frame_scores"].float()


def _bootstrap_c(frame_scores: torch.Tensor, alpha: torch.Tensor | None, point_drop: list[int], stack: str, rng: np.random.Generator, b: int = 1000) -> float:
    n = int(frame_scores.shape[0])
    hits = 0
    for _ in range(b):
        idx = rng.integers(0, n, size=n)
        sub = frame_scores[idx]
        if alpha is None:
            scores = sub.mean(dim=0)
        else:
            scores = weighted_scores(sub, alpha[idx])
        drop = drop_set(scores, stack)
        if drop == point_drop:
            hits += 1
    return hits / b


def main() -> None:
    shared = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    alpha = shared["alpha"].float()
    rng = np.random.default_rng(0)
    rows = []
    for stack in STACKS:
        spec = STAGE1[stack]
        for method in METHODS:
            frame = _load_frame(stack, method)
            uni_scores = frame.mean(dim=0)
            care_scores = weighted_scores(frame, alpha)
            uni_drop = drop_set(uni_scores, stack)
            care_drop = drop_set(care_scores, stack)
            s1_drop = spec[method]
            s1_scores = _stage1_scores(stack, method)
            tau_u = float(kendalltau(uni_scores.numpy(), s1_scores).correlation)
            tau_c = float(kendalltau(care_scores.numpy(), s1_scores).correlation)
            c_uni = _bootstrap_c(frame, None, uni_drop, stack, rng)
            c_care = _bootstrap_c(frame, alpha, care_drop, stack, rng)
            row = {
                "stack": stack,
                "method": method,
                "k": spec["k"],
                "stage1_drop": s1_drop,
                "uniform_drop": uni_drop,
                "care_drop": care_drop,
                "jaccard_uniform_vs_stage1": jaccard(uni_drop, s1_drop),
                "jaccard_care_vs_stage1": jaccard(care_drop, s1_drop),
                "jaccard_care_vs_uniform": jaccard(care_drop, uni_drop),
                "kendall_uniform_vs_stage1": tau_u,
                "kendall_care_vs_stage1": tau_c,
                "bootstrap_c_uniform": c_uni,
                "bootstrap_c_care": c_care,
            }
            rows.append(row)
            print(
                f"{stack:10s} {method:10s} uni={uni_drop} care={care_drop} "
                f"J(u,s1)={row['jaccard_uniform_vs_stage1']:.2f} "
                f"J(c,s1)={row['jaccard_care_vs_stage1']:.2f}",
                flush=True,
            )

    payload = {
        "protocol": "care_v2_e1",
        "n_frames": int(alpha.numel()),
        "ess": float(shared["ess"]),
        "n_molmo_files": int(shared["n_molmo_files"]),
        "rows": rows,
        "note": (
            "Uniform uses the shared 480-frame set with equal weights. "
            "CARE uses whitened-action α. Per-frame Fisher/Taylor/GateProbe. "
            "π0.5 Stage-1 Fisher/GateProbe used orig-batch reduction; a Jaccard<1 "
            "can be estimator change, calib change, or both. No closed loop in E1."
        ),
    }
    (OUT_DIR / "e1.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# CARE v2 E1: shared 480-frame uniform / weighted vs Stage-1",
        "",
        f"N={payload['n_frames']}, ESS={payload['ess']:.1f}, Molmo parquet files={payload['n_molmo_files']}.",
        "α = per-dimension whitened adjacent L2, H=10, 7-d only. No 6-d branch.",
        "",
        "| Stack | Method | Stage-1 drop | v2 uniform | v2 CARE | J(uni,S1) | J(CARE,S1) | J(CARE,uni) | τ(uni,S1) | c_uni | c_CARE |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stack']} | {row['method']} | `{row['stage1_drop']}` | `{row['uniform_drop']}` | "
            f"`{row['care_drop']}` | {row['jaccard_uniform_vs_stage1']:.2f} | "
            f"{row['jaccard_care_vs_stage1']:.2f} | {row['jaccard_care_vs_uniform']:.2f} | "
            f"{row['kendall_uniform_vs_stage1']:.3f} | {row['bootstrap_c_uniform']:.2f} | "
            f"{row['bootstrap_c_care']:.2f} |"
        )
    lines += [
        "",
        payload["note"],
        "",
        "If a recommended operating-point set (π0.5 Fisher-6, Molmo Taylor-16, SmolVLA Taylor-4) "
        "changes under v2 uniform, that new set needs a closed loop before it replaces Stage-1. "
        "E1 itself does not launch evals.",
        "",
    ]
    (OUT_DIR / "e1.md").write_text("\n".join(lines))
    print(f"wrote {OUT_DIR / 'e1.md'}", flush=True)


if __name__ == "__main__":
    main()

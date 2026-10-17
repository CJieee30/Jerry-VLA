#!/usr/bin/env python3
"""CPU self-test: synthetic frame scores + mock Taylor groups → analyze + CL dedupe."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prospective_molmo_layer_common as common
from prospective_molmo_layer_common import (
    HISTORICAL_DROP16,
    N_LAYERS,
    N_OBS,
    recommend_closed_loop_candidates,
    taylor_group_path,
)


def _write_synthetic_scores(tmp: Path) -> None:
    score_dir = tmp / "scores"
    score_dir.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    for method in common.FRAME_METHODS:
        mat = torch.rand(N_OBS, N_LAYERS, generator=g) * 10
        # Bias layers so drops are deterministic but differ by method
        if method == "fisher":
            mat[:, 25:36] *= 0.01
        elif method == "igia":
            mat[:, 24:36] *= 0.02
        torch.save(
            {
                "protocol": common.PROTOCOL_ID,
                "method": method,
                "frame_scores": mat,
                "n_samples": N_OBS,
                "n_layers": N_LAYERS,
            },
            score_dir / f"molmoact2_{method}.pt",
        )


def _write_mock_taylor(tmp: Path) -> None:
    tg = tmp / "taylor_groups"
    tg.mkdir(parents=True)
    for strategy, seed in (
        ("random", 0),
        ("g_only", 0),
        ("alpha", None),
        ("care_event", None),
    ):
        scores = [float(i) for i in range(N_LAYERS)]
        if strategy == "care_event":
            scores = [1.0] * 20 + [0.01] * 16
        path = tg / taylor_group_path(strategy, seed).name
        torch.save(
            {
                "protocol": common.PROTOCOL_ID,
                "method": "taylor",
                "formula": "mock",
                "strategy": strategy,
                "seed": seed,
                "budget_frames": 80,
                "pool_indices": list(range(80)),
                "layer_scores": scores,
            },
            path,
        )


def main() -> int:
    orig_layer = common.MOLMO_LAYER_DIR
    orig_score = common.SCORE_DIR
    orig_taylor = common.TAYLOR_GROUP_DIR

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        common.MOLMO_LAYER_DIR = tmp
        common.SCORE_DIR = tmp / "scores"
        common.TAYLOR_GROUP_DIR = tmp / "taylor_groups"
        common.SEL_DIR = tmp / "selections"

        import analyze_layer_calibration_molmo_v1 as analyze

        analyze.MOLMO_LAYER_DIR = common.MOLMO_LAYER_DIR
        analyze.SCORE_DIR = common.SCORE_DIR

        _write_synthetic_scores(tmp)
        _write_mock_taylor(tmp)

        rc = analyze.run_analyze(require_scores=True)
        if rc != 0:
            print("analyze failed", flush=True)
            return 1

        summary = json.loads((tmp / "summary.json").read_text())
        cl = summary.get("closed_loop_recommendations", [])
        verified = {tuple(v[2]) for k, v in HISTORICAL_DROP16.items() if v[1] is not None}
        for rec in cl:
            drop = tuple(rec["drop_layers"])
            if drop in verified:
                print(f"CL wrongly recommended verified set {drop}", flush=True)
                return 1

        # Duplicate unknown drops should appear once
        drops = [tuple(r["drop_layers"]) for r in cl]
        if len(drops) != len(set(drops)):
            print("CL duplicate drop sets", flush=True)
            return 1
        if len(cl) > 2:
            print("CL more than 2", flush=True)
            return 1

        # selection_path smoke (no Path+str bug)
        p = common.selection_path("fisher", 80, "random", 3)
        if "_seed3.json" not in p.name:
            print("bad selection_path", p, flush=True)
            return 1

        details = (tmp / "details.csv").read_text()
        if "fisher" not in details:
            print("missing details", flush=True)
            return 1

    # recommend_closed_loop on synthetic rows: historical Taylor-16 must be skipped
    fake_rows = [
        {
            "budget_frames": 80,
            "method": "taylor",
            "strategy": "care_event",
            "drop_json": json.dumps(HISTORICAL_DROP16["S1_Taylor-16"][2]),
            "drop_layers": HISTORICAL_DROP16["S1_Taylor-16"][2],
            "comparability": "可比性已核实",
        },
        {
            "budget_frames": 80,
            "method": "fisher",
            "strategy": "care_event",
            "drop_json": "[9,10,11]",
            "drop_layers": [9, 10, 11],
            "comparability": "尚未评测",
        },
    ]
    cl2 = recommend_closed_loop_candidates(fake_rows, max_n=2)
    if any(tuple(r["drop_layers"]) == tuple(HISTORICAL_DROP16["S1_Taylor-16"][2]) for r in cl2):
        print("Taylor-16 should not be recommended", flush=True)
        return 1

    print("selftest OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

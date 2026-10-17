#!/usr/bin/env python3
"""Re-score b=80 IGIA subsets with original measure_grad; compare Drop-16 to rev2 analysis."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "molmoact2_libero"))
sys.path.insert(0, str(HERE.parents[1]))

import torch

from prospective_molmo_layer_common import CALIB_PATH, MOLMO_LAYER_DIR
from protocol import drop_from_redundancy, importance_ranking

DROP_K = 16
EXCLUDE_L0 = True


def drop_from_scores(scores: list[float]) -> list[int]:
    order = importance_ranking(scores)
    return drop_from_redundancy(order, DROP_K, exclude_l0=EXCLUDE_L0)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    all_samples = blob["samples"]
    sel_dir = MOLMO_LAYER_DIR / "selections"

    from load_policy import load_policy, vlm_blocks
    from molmo_score_runtime import prepare_policy_for_scoring
    from score import measure_grad

    policy = load_policy(str(device))
    blocks, final_norm = vlm_blocks(policy)
    prepare_policy_for_scoring(policy, blocks)

    groups = [
        ("care_event", sel_dir / "igia_b80_care_event.json"),
        ("random_seed0", sel_dir / "igia_b80_random_seed0.json"),
    ]
    details_path = MOLMO_LAYER_DIR / "details.csv"
    import csv

    rev2_drops = {}
    with details_path.open() as f:
        for row in csv.DictReader(f):
            if row["method"] != "igia" or int(row["budget_frames"]) != 80:
                continue
            if row["strategy"] == "care_event":
                rev2_drops["care_event"] = json.loads(row["drop_layers"])
            if row["strategy"] == "random" and row.get("seed") == "0":
                rev2_drops["random_seed0"] = json.loads(row["drop_layers"])

    out = {"device": str(device), "groups": []}
    for name, path in groups:
        sel = json.loads(path.read_text())
        indices = sel.get("pool_indices") or sel.get("frame_indices")
        subset = [all_samples[i] for i in indices]
        scores = measure_grad(policy, blocks, subset, device, "igia")
        drop = drop_from_scores(scores)
        rev2 = rev2_drops.get(name)
        out["groups"].append(
            {
                "name": name,
                "n_frames": len(subset),
                "ref_measure_grad_drop": drop,
                "rev2_analyzed_drop": rev2,
                "drop_match": drop == rev2,
                "score_sum": float(sum(scores)),
            }
        )

    (args.out_dir / "igia_b80_ref_selection.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

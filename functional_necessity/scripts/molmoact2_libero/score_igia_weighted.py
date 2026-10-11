#!/usr/bin/env python3
"""MolmoAct2 IGIA with action-discontinuity weights outside the loss.

Stage-1 IGIA is Σ_i ||g_i||² (no 1/N, no θ²). α_i multiplies that sample's
g² after the original one-step flow loss. α=1 matches Stage-1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage2_action_weight"))

import torch
from tqdm import tqdm

from continuity import drop_from_redundancy, jaccard
from load_policy import load_policy, vlm_blocks
from score import (
    _chunk_dots,
    _mean_flow,
    _scoring_mode,
    _set_vlm_grad,
    build_calib,
    importance_ranking,
)

FN_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_FILE = FN_ROOT / "results" / "stage2_action_weight" / "weights_molmoact2.pt"
S1_DROP16 = [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
TAYLOR_DROP16 = [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]


def _freeze_non_vlm(policy, blocks) -> None:
    allowed = {id(param) for block in blocks for param in block.parameters()}
    for param in policy.parameters():
        param.requires_grad_(id(param) in allowed)


def _backward_igia_weighted(policy, blocks, sample, device, acc, alpha: float) -> None:
    for block in blocks:
        for param in block.parameters():
            param.grad = None
    loss = _mean_flow(policy, [sample], device)
    loss.backward()
    scale = float(alpha)
    for i, block in enumerate(blocks):
        for param in block.parameters():
            if param.grad is None:
                continue
            acc[i] += scale * _chunk_dots(param.grad, param, "igia")
            param.grad = None
    del loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--weights-file", type=Path, default=WEIGHTS_FILE)
    parser.add_argument(
        "--save-file",
        type=Path,
        default=FN_ROOT / "results" / "stage2_action_weight" / "molmoact2_igia_action_discontinuity.pt",
    )
    args = parser.parse_args()
    weights_blob = torch.load(args.weights_file, map_location="cpu", weights_only=False)
    jobs = [("weights7", args.save_file)]
    if not bool(weights_blob.get("use_7d_on_gpu", True)):
        arm6_path = args.save_file.with_name(args.save_file.stem + "_arm6" + args.save_file.suffix)
        jobs.append(("weights6", arm6_path))
        print("Molmo 7d vs 6d Spearman < 0.9; scoring IGIA on both (no closed loop).", flush=True)

    device = torch.device(args.device)
    policy = load_policy(str(device))
    blocks, _ = vlm_blocks(policy)
    samples = build_calib(policy)
    _freeze_non_vlm(policy, blocks)
    _set_vlm_grad(blocks, True)
    _scoring_mode(policy, blocks, checkpoint=True)
    lm_head = getattr(getattr(policy, "model", None), "lm_head", None)
    if lm_head is not None:
        lm_head.to("cpu")
        for param in lm_head.parameters():
            param.requires_grad_(False)
    if hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()
    torch.cuda.empty_cache()

    for key, save_file in jobs:
        alphas = weights_blob[key].float()
        if abs(float(alphas.mean()) - 1.0) > 1e-5:
            raise SystemExit(f"{key} mean must be 1, got {float(alphas.mean())}")
        if len(samples) != int(alphas.numel()):
            raise SystemExit(f"calib {len(samples)} != {key} {int(alphas.numel())}")
        acc = [0.0] * len(blocks)
        for sample, alpha in tqdm(list(zip(samples, alphas.tolist())), desc=f"igia {key}"):
            _backward_igia_weighted(policy, blocks, sample, device, acc, alpha)
            torch.cuda.empty_cache()
        scores = list(acc)
        ranking = importance_ranking(scores, 16)
        drop = drop_from_redundancy(ranking["redundancy_order"], 16, exclude_l0=True)
        payload = {
            "method": f"igia_action_discontinuity_{key}",
            "weight_key": key,
            "scores": scores,
            "redundancy_order": ranking["redundancy_order"],
            "drop16_nol0": drop,
            "stage1_drop16_nol0": S1_DROP16,
            "jaccard_drop16": jaccard(drop, S1_DROP16),
            "jaccard_vs_taylor16": jaccard(drop, TAYLOR_DROP16),
            "n_samples": len(samples),
            "aggregation": "α outside loss; sum_i α_i ||g_i||²; no 1/N; no θ²",
            "weights_file": str(args.weights_file),
        }
        save_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, save_file)
        summary = {
            k: payload[k]
            for k in (
                "weight_key",
                "drop16_nol0",
                "stage1_drop16_nol0",
                "jaccard_drop16",
                "jaccard_vs_taylor16",
            )
        }
        print(json.dumps(summary, indent=2), flush=True)
        save_file.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"saved {save_file}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SmolVLA Fisher with action-discontinuity weights outside the loss.

α_i multiplies (g^2 θ^2). Loss still uses the stored 50-step chunk and frozen
(noise, time). Divide by N=32 so α=1 matches Stage-1 Fisher.
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
WEIGHTS_FILE = FN_ROOT / "results" / "stage2_action_weight" / "weights_smolvla.pt"
OP_DROP4 = [10, 11, 12, 14]


def _backward_fisher_weighted(policy, blocks, sample, device, acc, alpha: float) -> None:
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
            acc[i] += scale * _chunk_dots(param.grad, param, "fisher")
            param.grad = None
    del loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--weights-file", type=Path, default=WEIGHTS_FILE)
    parser.add_argument(
        "--save-file",
        type=Path,
        default=FN_ROOT / "results" / "stage2_action_weight" / "smolvla_fisher_action_discontinuity.pt",
    )
    args = parser.parse_args()
    weights_blob = torch.load(args.weights_file, map_location="cpu", weights_only=False)
    if not bool(weights_blob.get("use_7d_on_gpu", True)):
        raise SystemExit("7d vs 6d Spearman < 0.9; dump_weights asked to score both. Stop.")
    alphas = weights_blob["weights7"].float()
    if abs(float(alphas.mean()) - 1.0) > 1e-5:
        raise SystemExit(f"weights must have mean 1, got {float(alphas.mean())}")

    device = torch.device(args.device)
    policy = load_policy(str(device))
    blocks, _ = vlm_blocks(policy)
    samples = build_calib(policy)
    if len(samples) != int(alphas.numel()):
        raise SystemExit(f"calib {len(samples)} != weights {int(alphas.numel())}")

    _set_vlm_grad(blocks, True)
    _scoring_mode(policy, blocks, checkpoint=True)
    acc = [0.0] * len(blocks)
    for sample, alpha in tqdm(list(zip(samples, alphas.tolist())), desc="fisher weighted"):
        _backward_fisher_weighted(policy, blocks, sample, device, acc, alpha)
    scores = [v / len(samples) for v in acc]
    ranking = importance_ranking(scores, 4)
    drop = drop_from_redundancy(ranking["redundancy_order"], 4, exclude_l0=False)
    payload = {
        "method": "fisher_action_discontinuity",
        "scores": scores,
        "redundancy_order": ranking["redundancy_order"],
        "drop4": drop,
        "stage1_drop4": OP_DROP4,
        "jaccard_drop4": jaccard(drop, OP_DROP4),
        "n_samples": len(samples),
        "aggregation": "α outside loss; (1/N) sum_i α_i (g_i^2 θ^2); N=32",
        "weights_file": str(args.weights_file),
    }
    args.save_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.save_file)
    summary = {k: payload[k] for k in ("drop4", "stage1_drop4", "jaccard_drop4")}
    print(json.dumps(summary, indent=2), flush=True)
    args.save_file.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved {args.save_file}", flush=True)


if __name__ == "__main__":
    main()

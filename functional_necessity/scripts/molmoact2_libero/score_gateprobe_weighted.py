#!/usr/bin/env python3
"""MolmoAct2 GateProbe with action-discontinuity weights outside the loss.

Stage-1 GateProbe is (1/N) Σ_i |I_i| on the original one-step flow loss.
α_i multiplies |I_i| after that loss. Divide by N=32 so α=1 matches Stage-1.
Signed inners are computed once; 7-d and arm-6-d weights are applied after.
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
from score import _hooks, _mean_flow, _scoring_mode, _set_vlm_grad, build_calib, importance_ranking

FN_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_FILE = FN_ROOT / "results" / "stage2_action_weight" / "weights_molmoact2.pt"
S1_DROP17 = [2, 3, 4, 6, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]


def _freeze_non_vlm(policy, blocks) -> None:
    allowed = {id(param) for block in blocks for param in block.parameters()}
    for param in policy.parameters():
        param.requires_grad_(id(param) in allowed)


def _signed_row(policy, blocks, final_norm, sample, device) -> list[float]:
    states, handles = _hooks(blocks, final_norm)
    loss = _mean_flow(policy, [sample], device)
    for hidden in states.values():
        if hidden.requires_grad:
            hidden.retain_grad()
    loss.backward()
    row = []
    for i in range(len(blocks)):
        h_in, h_out = states.get(i), states.get(i + 1)
        if h_in is None or h_out is None or h_out.grad is None:
            row.append(0.0)
        else:
            row.append(float((h_out.grad.detach().float() * (h_out.detach().float() - h_in.detach().float())).sum()))
    for handle in handles:
        handle.remove()
    for block in blocks:
        for param in block.parameters():
            param.grad = None
    del loss
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--weights-file", type=Path, default=WEIGHTS_FILE)
    parser.add_argument(
        "--save-file",
        type=Path,
        default=FN_ROOT / "results" / "stage2_action_weight" / "molmoact2_gateprobe_action_discontinuity.pt",
    )
    args = parser.parse_args()
    weights_blob = torch.load(args.weights_file, map_location="cpu", weights_only=False)
    jobs = [("weights7", args.save_file)]
    if not bool(weights_blob.get("use_7d_on_gpu", True)):
        arm6_path = args.save_file.with_name(args.save_file.stem + "_arm6" + args.save_file.suffix)
        jobs.append(("weights6", arm6_path))
        print("Molmo 7d vs 6d Spearman < 0.9; applying both weights (one signed pass).", flush=True)

    device = torch.device(args.device)
    policy = load_policy(str(device))
    blocks, final_norm = vlm_blocks(policy)
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

    rows = []
    for sample in tqdm(samples, desc="gateprobe signed"):
        rows.append(_signed_row(policy, blocks, final_norm, sample, device))
        torch.cuda.empty_cache()
    signed = torch.tensor(rows, dtype=torch.float32)
    abs_rows = signed.abs()

    for key, save_file in jobs:
        alphas = weights_blob[key].float().reshape(-1)
        if abs(float(alphas.mean()) - 1.0) > 1e-5:
            raise SystemExit(f"{key} mean must be 1, got {float(alphas.mean())}")
        if signed.shape[0] != int(alphas.numel()):
            raise SystemExit(f"calib {signed.shape[0]} != {key} {int(alphas.numel())}")
        scores = (abs_rows * alphas.unsqueeze(1)).mean(dim=0).tolist()
        ranking = importance_ranking(scores, 17)
        drop = drop_from_redundancy(ranking["redundancy_order"], 17, exclude_l0=True)
        payload = {
            "method": f"gateprobe_action_discontinuity_{key}",
            "weight_key": key,
            "scores": scores,
            "redundancy_order": ranking["redundancy_order"],
            "drop17_nol0": drop,
            "stage1_drop17_nol0": S1_DROP17,
            "jaccard_drop17": jaccard(drop, S1_DROP17),
            "n_samples": int(signed.shape[0]),
            "aggregation": "α outside loss; (1/N) sum_i α_i |I_i|; N=32",
            "weights_file": str(args.weights_file),
        }
        save_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, save_file)
        summary = {
            k: payload[k]
            for k in ("weight_key", "drop17_nol0", "stage1_drop17_nol0", "jaccard_drop17")
        }
        print(json.dumps(summary, indent=2), flush=True)
        save_file.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"saved {save_file}", flush=True)


if __name__ == "__main__":
    main()

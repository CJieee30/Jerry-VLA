#!/usr/bin/env python3
"""SmolVLA GateProbe with action-discontinuity weights outside the loss.

Stage-1 GateProbe is (1/N) Σ_i |I_i| on the stored 50-step flow loss.
α_i multiplies |I_i| after that loss. Divide by N=32 so α=1 matches Stage-1.
7-d weights only (Spearman vs arm-6-d ≥ 0.9). Native Drop-4 includes L0.
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
WEIGHTS_FILE = FN_ROOT / "results" / "stage2_action_weight" / "weights_smolvla.pt"
S1_DROP4 = [7, 13, 14, 15]


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
            row.append(
                float(
                    (
                        h_out.grad.detach().float()
                        * (h_out.detach().float() - h_in.detach().float())
                    ).sum()
                )
            )
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
        default=FN_ROOT / "results" / "stage2_action_weight" / "smolvla_gateprobe_action_discontinuity.pt",
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
    blocks, final_norm = vlm_blocks(policy)
    samples = build_calib(policy)
    if len(samples) != int(alphas.numel()):
        raise SystemExit(f"calib {len(samples)} != weights {int(alphas.numel())}")

    _set_vlm_grad(blocks, True)
    _scoring_mode(policy, blocks, checkpoint=True)

    rows = []
    for sample in tqdm(samples, desc="gateprobe signed"):
        rows.append(_signed_row(policy, blocks, final_norm, sample, device))
        torch.cuda.empty_cache()
    signed = torch.tensor(rows, dtype=torch.float32)
    abs_rows = signed.abs()
    scores = (abs_rows * alphas.unsqueeze(1)).mean(dim=0).tolist()
    ranking = importance_ranking(scores, 4)
    drop = drop_from_redundancy(ranking["redundancy_order"], 4, exclude_l0=False)
    payload = {
        "method": "gateprobe_action_discontinuity",
        "scores": scores,
        "redundancy_order": ranking["redundancy_order"],
        "drop4": drop,
        "stage1_drop4": S1_DROP4,
        "jaccard_drop4": jaccard(drop, S1_DROP4),
        "n_samples": int(signed.shape[0]),
        "aggregation": "α outside loss; (1/N) sum_i α_i |I_i|; N=32",
        "weights_file": str(args.weights_file),
    }
    args.save_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.save_file)
    summary = {
        k: payload[k]
        for k in ("drop4", "stage1_drop4", "jaccard_drop4", "redundancy_order")
    }
    print(json.dumps(summary, indent=2), flush=True)
    args.save_file.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"saved {args.save_file}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CARE-weighted Fisher and Taylor on π0.5 SigLIP, same 4 batches as the uniform probe.

Taylor: |sum_t α_t g_t ⊙ w| / N_samples
Fisher: per original batch, ḡ = sum(α_i g_i) / sum(α_i), then (ḡ² ⊙ w²), mean over batches.
Weights are the stored 7-d CARE frame weights (mean 1). No closed-loop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
sys.path.insert(0, str(FN / "scripts" / "stage_nfe"))

from common import (  # noqa: E402
    DEFAULT_CALIB_FILE,
    DEFAULT_CHECKPOINT,
    load_calib_data,
    load_policy,
    move_to_device,
    slice_batch,
)
from probe_vision_layers import N_BATCHES, MB, vision_layers  # noqa: E402

OUT = FN / "results" / "stage_nfe" / "vision_probe_pi05_care.json"
UNIFORM = FN / "results" / "stage_nfe" / "vision_probe_pi05.json"
CARE7 = FN / "results" / "stage_nfe" / "vision_probe_pi05_care.json"
WEIGHTS = FN / "results" / "stage2_action_weight" / "weights_pi05.pt"


def per_sample_loss(policy, batch) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = policy(batch, reduction="none")
    return loss


def score_care(policy, calib, weights, device) -> tuple[list[float], list[float]]:
    layers = vision_layers(policy)
    for param in policy.parameters():
        param.requires_grad_(False)
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad_(True)
    policy.eval()
    if hasattr(policy.model, "gradient_checkpointing_enable"):
        policy.model.gradient_checkpointing_enable()
    taylor_sum: dict[tuple[int, str], torch.Tensor] = {}
    fisher = [0.0] * len(layers)
    n_samples = 0
    for batch_i in range(N_BATCHES):
        full = calib[batch_i]
        width = int(full["action"].shape[0])
        alphas = weights[batch_i].float()
        if int(alphas.numel()) != width:
            raise SystemExit(f"batch {batch_i} width {width} != weights {int(alphas.numel())}")
        fisher_g: dict[tuple[int, str], torch.Tensor] = {}
        stored_w: dict[tuple[int, str], torch.Tensor] = {}
        n_mb = 0
        for start in range(0, width, MB):
            batch = move_to_device(slice_batch(full, start, start + MB), device)
            a = alphas[start : start + MB].to(device)
            policy.zero_grad(set_to_none=True)
            loss = (per_sample_loss(policy, batch).float() * a).sum()
            loss.backward()
            for i, layer in enumerate(layers):
                for name, param in layer.named_parameters():
                    if param.grad is None:
                        continue
                    g = param.grad.detach().float().cpu()
                    key = (i, name)
                    if key not in stored_w:
                        stored_w[key] = param.detach().float().cpu()
                    prod = g * stored_w[key]
                    if key not in taylor_sum:
                        taylor_sum[key] = prod.clone()
                    else:
                        taylor_sum[key] += prod
                    if key not in fisher_g:
                        fisher_g[key] = g.clone()
                    else:
                        fisher_g[key] += g
            n_mb += 1
            n_samples += 1
            policy.zero_grad(set_to_none=True)
            del loss, batch
            torch.cuda.empty_cache()
            print(f"care batch {batch_i} step {start} done n={n_samples}", flush=True)
        denom = float(alphas.sum().clamp_min(1e-12))
        for i, layer in enumerate(layers):
            for name, _param in layer.named_parameters():
                key = (i, name)
                if key not in fisher_g:
                    continue
                g_bar = fisher_g[key] / denom
                fisher[i] += float((g_bar.square() * stored_w[key].square()).sum())
        print(f"care fisher batch {batch_i} closed denom={denom:.3f}", flush=True)
    taylor = []
    for i, layer in enumerate(layers):
        total = 0.0
        for name, _param in layer.named_parameters():
            key = (i, name)
            if key in taylor_sum:
                total += float(taylor_sum[key].abs().sum())
        taylor.append(total / max(n_samples, 1))
    fisher = [s / N_BATCHES for s in fisher]
    return taylor, fisher


def order_low(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i])


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0] * len(xs)
        for rank, i in enumerate(order):
            r[i] = rank
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db)


def jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-key", default="weights7_batched")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    device = torch.device("cuda")
    blob = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    weights = blob[args.weight_key][:N_BATCHES]
    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.compile_model = False
    policy.eval()
    calib = load_calib_data(DEFAULT_CALIB_FILE)
    print(f"siglip {len(vision_layers(policy))} layers, care weight mean {float(weights.mean()):.4f}", flush=True)
    taylor, fisher = score_care(policy, calib, weights, device)
    uniform = json.loads(UNIFORM.read_text())
    care7 = json.loads(CARE7.read_text()) if args.weight_key != "weights7_batched" and CARE7.is_file() else None
    compare = {}
    for name, care, base in (
        ("taylor", taylor, uniform["taylor"]),
        ("fisher", fisher, uniform["fisher"]),
    ):
        care_order = order_low(care)
        base_order = order_low(base)
        shifts = [abs(care_order.index(i) - base_order.index(i)) for i in range(len(care))]
        row = {
            "spearman_vs_uniform": spearman(base, care),
            "drop4_uniform": base_order[:4],
            "drop4_care": care_order[:4],
            "drop4_jaccard_vs_uniform": jaccard(base_order[:4], care_order[:4]),
            "mean_abs_rank_shift_vs_uniform": sum(shifts) / len(shifts),
            "max_abs_rank_shift_vs_uniform": max(shifts),
        }
        if care7 is not None:
            other = care7[name]
            other_order = order_low(other)
            shifts7 = [abs(care_order.index(i) - other_order.index(i)) for i in range(len(care))]
            row["spearman_vs_care7"] = spearman(other, care)
            row["drop4_care7"] = other_order[:4]
            row["drop4_jaccard_vs_care7"] = jaccard(other_order[:4], care_order[:4])
            row["mean_abs_rank_shift_vs_care7"] = sum(shifts7) / len(shifts7)
            row["max_abs_rank_shift_vs_care7"] = max(shifts7)
        compare[name] = row
        print(name, json.dumps(row), flush=True)
    payload = {
        "checkpoint": DEFAULT_CHECKPOINT,
        "device": torch.cuda.get_device_name(0),
        "n_batches": N_BATCHES,
        "microbatch": MB,
        "weights": args.weight_key,
        "weight_mean_used": float(weights.mean()),
        "taylor": taylor,
        "fisher": fisher,
        "taylor_order_low_to_high": order_low(taylor),
        "fisher_order_low_to_high": order_low(fisher),
        "vs_uniform": compare,
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()

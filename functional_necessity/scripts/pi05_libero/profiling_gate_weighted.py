#!/usr/bin/env python3
"""Weighted GateProbe for π0.5. Reduction stays at the original 64×8 batches.

Stage-1: each orig batch reconstructs the batch-8 signed inner product from
4× mb=2 mean-loss forwards, then abs; 64 batches equal-weighted.

Weighted: same 4× mb=2 forwards. Per-sample losses, L = sum_i α_i L_i, then
the batched (dL/dh * residual).sum() equals sum_i α_i I_i. Divide by sum(α),
then abs. When α=1 this matches Stage-1. 64 batches stay equal-weighted.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage2_action_weight"))

import numpy as np
import torch
from tqdm import tqdm

from common import (
    DEFAULT_GRAD_MICROBATCH,
    add_common_args,
    cuda_empty_cache,
    init_logging,
    load_calib_data,
    load_policy,
    paligemma_parts,
    prepare_policy_for_grad_scoring,
    print_ranking,
    save_scores,
    set_seed,
    slice_batch,
    _batch_size,
    move_to_device,
)
from continuity import drop_from_redundancy, jaccard
from profiling_gate import _signed_inners

FN_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_FILE = FN_ROOT / "results" / "stage2_action_weight" / "weights_pi05.pt"
OP_DROP6 = [2, 4, 14, 15, 16, 17]


def _compute_per_sample_loss(policy, batch) -> torch.Tensor:
    device_type = batch[next(k for k, v in batch.items() if torch.is_tensor(v))].device.type
    if device_type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch, reduction="none")
    else:
        loss, _ = policy(batch, reduction="none")
    return loss


def _snapshot_rng(device: torch.device) -> dict:
    snap = {
        "py": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        snap["cuda"] = torch.cuda.get_rng_state(device)
    return snap


def _restore_rng(snap: dict, device: torch.device) -> None:
    random.setstate(snap["py"])
    np.random.set_state(snap["numpy"])
    torch.set_rng_state(snap["cpu"])
    if device.type == "cuda" and snap.get("cuda") is not None:
        torch.cuda.set_rng_state(snap["cuda"], device)


def _signed_inners_weighted(policy, batch, layers, final_norm, alphas: torch.Tensor) -> list[float]:
    num_layers = len(layers)
    policy.zero_grad(set_to_none=True)
    states = {}
    hooks = []

    def make_pre_hook(idx):
        def hook(module, args, kwargs=None):
            hidden = args[0] if len(args) > 0 else (kwargs.get("hidden_states") if kwargs else None)
            if hidden is not None:
                hidden.retain_grad()
                states[idx] = hidden

        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.input_layernorm.register_forward_pre_hook(make_pre_hook(i), with_kwargs=True))
    hooks.append(final_norm.register_forward_pre_hook(make_pre_hook(num_layers), with_kwargs=True))

    row = [0.0] * num_layers
    try:
        with torch.enable_grad():
            per = _compute_per_sample_loss(policy, batch)
            a = alphas.to(device=per.device, dtype=torch.float32)
            loss = (per.float() * a).sum()
            loss.backward()
        for layer_idx in range(num_layers):
            h_in = states.get(layer_idx)
            h_out = states.get(layer_idx + 1)
            if h_in is None or h_out is None or h_out.grad is None:
                continue
            residual = h_out.float() - h_in.float()
            grad_h = h_out.grad.float()
            row[layer_idx] = float((grad_h * residual).sum().item())
    finally:
        for handle in hooks:
            handle.remove()
        policy.zero_grad(set_to_none=True)
        cuda_empty_cache()
    return row


def _orig_batch_signed(policy, batch, device, layers, final_norm, microbatch_size) -> list[float]:
    signed = [0.0] * len(layers)
    n_mb = 0
    step = min(microbatch_size, _batch_size(batch))
    bsz = _batch_size(batch)
    for start in range(0, bsz, step):
        mb = move_to_device(slice_batch(batch, start, min(start + step, bsz)), device)
        row, _ = _signed_inners(policy, mb, layers, final_norm)
        for i, value in enumerate(row):
            signed[i] += value
        n_mb += 1
    return [s / max(n_mb, 1) for s in signed]


def _orig_batch_signed_weighted(policy, batch, device, layers, final_norm, alphas, microbatch_size) -> list[float]:
    signed = [0.0] * len(layers)
    bsz = _batch_size(batch)
    step = min(microbatch_size, bsz)
    denom = float(alphas.float().sum().clamp_min(1e-12))
    for start in range(0, bsz, step):
        end = min(start + step, bsz)
        mb = move_to_device(slice_batch(batch, start, end), device)
        row = _signed_inners_weighted(policy, mb, layers, final_norm, alphas[start:end])
        for i, value in enumerate(row):
            signed[i] += value
    return [s / denom for s in signed]


def sanity_unweighted_matches(policy, batch, device, microbatch_size) -> dict:
    layers, final_norm = paligemma_parts(policy)
    bsz = _batch_size(batch)
    ones = torch.ones(bsz, dtype=torch.float32)
    snap = _snapshot_rng(device)
    orig = _orig_batch_signed(policy, batch, device, layers, final_norm, microbatch_size)
    _restore_rng(snap, device)
    weighted = _orig_batch_signed_weighted(
        policy, batch, device, layers, final_norm, ones, microbatch_size
    )
    diffs = [abs(a - b) for a, b in zip(orig, weighted)]
    scale = max(max(abs(x) for x in orig), 1e-12)
    rel = max(diffs) / scale
    return {
        "max_abs_diff_signed": max(diffs),
        "rel_to_max_orig": rel,
        "ok": bool(rel < 1e-3),
        "forwards": "4 x batch=2 on both paths",
    }


def measure_gate_weighted(policy, calib_data, device, batched_weights, microbatch_size) -> dict:
    layers, final_norm = paligemma_parts(policy)
    raw_rows = []
    policy.eval()
    if batched_weights.ndim != 2 or batched_weights.shape[0] != len(calib_data):
        raise ValueError(f"weights shape {tuple(batched_weights.shape)} != ({len(calib_data)}, ?)")
    for batch, alphas in tqdm(
        zip(calib_data, batched_weights),
        total=len(calib_data),
        desc="GateProbe weighted orig-batch",
    ):
        bsz = _batch_size(batch)
        if int(alphas.numel()) != bsz:
            raise ValueError(f"batch size {bsz} != weight row {int(alphas.numel())}")
        raw_rows.append(
            _orig_batch_signed_weighted(
                policy, batch, device, layers, final_norm, alphas.float(), microbatch_size
            )
        )
    raw = torch.tensor(raw_rows, dtype=torch.float32)
    abs_mean = raw.abs().mean(dim=0)
    return {
        "method": "gateprobe_action_discontinuity",
        "paligemma": abs_mean,
        "paligemma_raw": raw,
        "aggregation": "orig_batch_weighted_mean_signed_then_abs; 4x mb=2; 64 batches equal-weighted",
        "n_orig_batches": int(raw.shape[0]),
    }


def _drop6_nol0(scores: torch.Tensor) -> list[int]:
    from common import importance_ranking

    order = importance_ranking(scores, ks=(6,))["redundancy_order"]
    return drop_from_redundancy(order, 6, exclude_l0=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(microbatch_size=DEFAULT_GRAD_MICROBATCH)
    parser.add_argument("--weights-file", type=Path, default=WEIGHTS_FILE)
    parser.add_argument("--skip-sanity", action="store_true")
    args = parser.parse_args()
    init_logging()
    set_seed(args.seed)
    device = torch.device(args.device)
    weights_blob = torch.load(args.weights_file, map_location="cpu", weights_only=False)
    if not bool(weights_blob.get("use_7d_on_gpu", True)):
        raise SystemExit("7d vs 6d Spearman < 0.9; dump_weights asked to score both. Stop.")
    batched = weights_blob["weights7_batched"]
    if abs(float(batched.mean()) - 1.0) > 1e-5:
        raise SystemExit(f"weights must have mean 1, got {float(batched.mean())}")

    policy = load_policy(args.checkpoint, device)
    prepare_policy_for_grad_scoring(policy, recompute=False)
    policy.eval()
    calib = load_calib_data(args.calib_file)

    if not args.skip_sanity:
        print("sanity: α=1 GateProbe orig-batch vs weighted on batch 0", flush=True)
        check = sanity_unweighted_matches(policy, calib[0], device, args.microbatch_size)
        print(json.dumps(check, indent=2), flush=True)
        sanity_path = FN_ROOT / "results" / "stage2_action_weight" / "pi05_gateprobe_sanity.json"
        sanity_path.write_text(json.dumps(check, indent=2) + "\n")
        if not check["ok"]:
            raise SystemExit("α=1 GateProbe does not match orig-batch reduction; stop.")

    results = measure_gate_weighted(policy, calib, device, batched, args.microbatch_size)
    print_ranking("GateProbe weighted (action discontinuity)", results["paligemma"])
    drop = _drop6_nol0(results["paligemma"])
    results["drop6_nol0"] = drop
    results["stage1_drop6_nol0"] = OP_DROP6
    results["jaccard_drop6"] = jaccard(drop, OP_DROP6)
    out = args.save_file or (
        FN_ROOT / "results" / "stage2_action_weight" / "pi05_gateprobe_action_discontinuity.pt"
    )
    save_scores(out, results)
    sidecar = {
        "drop6_nol0": drop,
        "stage1_drop6_nol0": OP_DROP6,
        "jaccard_drop6": results["jaccard_drop6"],
        "aggregation": results["aggregation"],
    }
    Path(out).with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"Drop-6 no-L0 {drop} Jaccard vs Stage-1 {OP_DROP6} = {results['jaccard_drop6']:.4f}", flush=True)


if __name__ == "__main__":
    main()

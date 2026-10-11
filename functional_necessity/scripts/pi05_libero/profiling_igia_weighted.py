#!/usr/bin/env python3
"""Weighted IGIA for π0.5. Reduction stays at the original 64×8 batches.

Stage-1 IGIA is sum_t ||ḡ_t||² after reconstructing the batch-8 mean gradient
(DTR Eq. 10, no 1/N, no θ²). CARE: ḡ = (sum_i α_i g_i) / (sum_i α_i), then
square. Flattening 512 frames before squaring is forbidden. Forwards stay
4×(batch=2) so α=1 matches Stage-1.
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
    iter_orig_calib_microbatches,
    load_calib_data,
    load_policy,
    orig_batch_accumulate_grads,
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

FN_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_FILE = FN_ROOT / "results" / "stage2_action_weight" / "weights_pi05.pt"
S1_DROP6 = [2, 13, 14, 15, 16, 17]


def _layer0_igia_from_grad(layer, g_bar: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, param in layer.named_parameters():
        if name not in g_bar:
            continue
        g = g_bar[name].float().cpu()
        total += float(g.square().sum().item())
    return total


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


def _layer0_from_param_grad(layer) -> dict[str, torch.Tensor]:
    out = {}
    for name, param in layer.named_parameters():
        if param.grad is None:
            continue
        out[name] = param.grad.detach().float().cpu()
    return out


def _weighted_mean_grad_one_batch(policy, batch, device, alphas, microbatch_size) -> None:
    bsz = _batch_size(batch)
    step = min(microbatch_size, bsz)
    policy.zero_grad(set_to_none=True)
    for start in range(0, bsz, step):
        end = min(start + step, bsz)
        mb = move_to_device(slice_batch(batch, start, end), device)
        a = alphas[start:end].to(device=device, dtype=torch.float32)
        per = _compute_per_sample_loss(policy, mb)
        loss = (per.float() * a).sum()
        loss.backward()
        del loss, per


def sanity_unweighted_mean_matches(policy, batch, device, microbatch_size) -> dict:
    layers, _ = paligemma_parts(policy)
    bsz = _batch_size(batch)
    ones = torch.ones(bsz, dtype=torch.float32)
    snap = _snapshot_rng(device)

    policy.zero_grad(set_to_none=True)
    microbatches = next(iter_orig_calib_microbatches([batch], device, microbatch_size))
    n_mb = orig_batch_accumulate_grads(policy, microbatches)
    g_orig = _layer0_from_param_grad(layers[0])
    s_orig = _layer0_igia_from_grad(layers[0], {k: v / max(n_mb, 1) for k, v in g_orig.items()})
    policy.zero_grad(set_to_none=True)
    cuda_empty_cache()

    _restore_rng(snap, device)
    _weighted_mean_grad_one_batch(policy, batch, device, ones, microbatch_size)
    denom = float(ones.sum())
    g_w = _layer0_from_param_grad(layers[0])
    s_w = _layer0_igia_from_grad(layers[0], {k: v / denom for k, v in g_w.items()})
    policy.zero_grad(set_to_none=True)
    cuda_empty_cache()

    rel = abs(s_w - s_orig) / max(abs(s_orig), 1e-12)
    return {
        "layer0_igia_weighted_alpha1": s_w,
        "layer0_igia_orig_batch_mb": s_orig,
        "rel_abs_diff": rel,
        "ok": bool(rel < 1e-3),
        "forwards": "4 x batch=2 on both paths",
    }


def measure_igia_weighted(policy, calib_data, device, batched_weights, microbatch_size) -> dict:
    layers, _ = paligemma_parts(policy)
    scores = [0.0] * len(layers)
    n_orig = 0
    if batched_weights.ndim != 2 or batched_weights.shape[0] != len(calib_data):
        raise ValueError(
            f"weights must be (n_orig_batches, batch_size)=({len(calib_data)}, ?), "
            f"got {tuple(batched_weights.shape)}"
        )
    for batch, alphas in tqdm(
        zip(calib_data, batched_weights),
        total=len(calib_data),
        desc="IGIA weighted orig-batch",
    ):
        bsz = _batch_size(batch)
        if int(alphas.numel()) != bsz:
            raise ValueError(f"batch size {bsz} != weight row {int(alphas.numel())}")
        denom = float(alphas.float().sum().clamp_min(1e-12))
        try:
            _weighted_mean_grad_one_batch(policy, batch, device, alphas.float(), microbatch_size)
            for li, layer in enumerate(layers):
                for param in layer.parameters():
                    if param.grad is None:
                        continue
                    g_bar = param.grad.detach().float().cpu() / denom
                    scores[li] += float(g_bar.square().sum().item())
            n_orig += 1
        finally:
            policy.zero_grad(set_to_none=True)
            cuda_empty_cache()
    return {
        "method": "igia_action_discontinuity",
        "paligemma": torch.tensor(scores),
        "aggregation": "orig_batch_weighted_mean_grad_then_square; no θ²; no 1/N; 4x mb=2; 64 batches equal-weighted",
        "n_orig_batches": n_orig,
    }


def _drop6(scores: torch.Tensor) -> list[int]:
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
    prepare_policy_for_grad_scoring(policy, recompute=True)
    calib = load_calib_data(args.calib_file)

    if not args.skip_sanity:
        print("sanity: α=1, 4x batch=2 weighted-mean vs orig-batch IGIA on batch 0", flush=True)
        check = sanity_unweighted_mean_matches(policy, calib[0], device, args.microbatch_size)
        print(json.dumps(check, indent=2), flush=True)
        sanity_path = FN_ROOT / "results" / "stage2_action_weight" / "pi05_igia_sanity.json"
        sanity_path.parent.mkdir(parents=True, exist_ok=True)
        sanity_path.write_text(json.dumps(check, indent=2) + "\n")
        if not check["ok"]:
            raise SystemExit("α=1 weighted-mean does not match orig-batch IGIA reduction; stop.")

    results = measure_igia_weighted(policy, calib, device, batched, args.microbatch_size)
    print_ranking("IGIA weighted (action discontinuity)", results["paligemma"])
    drop = _drop6(results["paligemma"])
    results["drop6_nol0"] = drop
    results["stage1_drop6_nol0"] = S1_DROP6
    results["jaccard_drop6"] = jaccard(drop, S1_DROP6)
    results["weights_file"] = str(args.weights_file)
    out = args.save_file or (
        FN_ROOT / "results" / "stage2_action_weight" / "pi05_igia_action_discontinuity.pt"
    )
    save_scores(out, results)
    sidecar = {
        "drop6_nol0": drop,
        "stage1_drop6_nol0": S1_DROP6,
        "jaccard_drop6": results["jaccard_drop6"],
        "aggregation": results["aggregation"],
        "n_orig_batches": results["n_orig_batches"],
    }
    Path(out).with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"Drop-6 no-L0 {drop} Jaccard vs Stage-1 {S1_DROP6} = {results['jaccard_drop6']:.4f}", flush=True)


if __name__ == "__main__":
    main()

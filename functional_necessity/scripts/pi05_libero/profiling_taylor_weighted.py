#!/usr/bin/env python3
"""Weighted Taylor for π0.5. Aggregation matches Stage-1 Taylor.

Stage-1: 256× mb=2 mean-loss forwards, accumulate (g ⊙ θ), then
    S = (1/N) sum_w |sum_t (g_t ⊙ θ)|
with N = n_microbatches. The abs is on the global sum, not per orig-batch.

CARE: same 256× mb=2 forwards. Per-sample losses, L = mean_i(α_i L_i) inside
each microbatch so α=1 is identical to compute_loss. α multiplies the sample
contribution outside the stored 50-step chunk. Flattening 512 frames before
the abs does not change this linear sum; we still use mb=2 so the RNG stream
matches Stage-1 Taylor.
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
    compute_loss,
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

FN_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_FILE = FN_ROOT / "results" / "stage2_action_weight" / "weights_pi05.pt"
S1_DROP8 = [1, 2, 3, 13, 14, 15, 16, 17]


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


def _accumulate_salience(layers, salience) -> None:
    for i, layer in enumerate(layers):
        for name, param in layer.named_parameters():
            if param.grad is None:
                continue
            s = (param.grad.float() * param.data.float()).cpu()
            if name not in salience[i]:
                salience[i][name] = s
            else:
                salience[i][name] += s


def _layer0_score(salience) -> float:
    return float(sum(s.abs().sum().item() for s in salience[0].values()))


def _run_microbatches(policy, batch, device, layers, alphas, microbatch_size, salience) -> int:
    """Accumulate (g ⊙ θ) over mb=2 slices. alphas=None → Stage-1 mean loss."""
    bsz = _batch_size(batch)
    step = min(microbatch_size, bsz)
    n_mb = 0
    for start in range(0, bsz, step):
        end = min(start + step, bsz)
        mb = move_to_device(slice_batch(batch, start, end), device)
        policy.zero_grad(set_to_none=True)
        try:
            with torch.enable_grad():
                if alphas is None:
                    loss = compute_loss(policy, mb)
                else:
                    per = _compute_per_sample_loss(policy, mb)
                    a = alphas[start:end].to(device=per.device, dtype=torch.float32)
                    loss = (per.float() * a).mean()
                loss.backward()
            _accumulate_salience(layers, salience)
            n_mb += 1
        finally:
            del loss
            policy.zero_grad(set_to_none=True)
            cuda_empty_cache()
    return n_mb


def sanity_unweighted_matches(policy, batch, device, microbatch_size) -> dict:
    layers, _ = paligemma_parts(policy)
    bsz = _batch_size(batch)
    ones = torch.ones(bsz, dtype=torch.float32)
    snap = _snapshot_rng(device)
    sal_orig = [{} for _ in layers]
    n_orig = _run_microbatches(policy, batch, device, layers, None, microbatch_size, sal_orig)
    s_orig = _layer0_score(sal_orig) / max(n_orig, 1)
    _restore_rng(snap, device)
    sal_w = [{} for _ in layers]
    n_w = _run_microbatches(policy, batch, device, layers, ones, microbatch_size, sal_w)
    s_w = _layer0_score(sal_w) / max(n_w, 1)
    rel = abs(s_w - s_orig) / max(abs(s_orig), 1e-12)
    return {
        "layer0_taylor_weighted_alpha1": s_w,
        "layer0_taylor_stage1_mb": s_orig,
        "rel_abs_diff": rel,
        "ok": bool(rel < 1e-3),
        "forwards": "4 x batch=2 on both paths",
    }


def measure_taylor_weighted(policy, calib_data, device, batched_weights, microbatch_size) -> dict:
    layers, _ = paligemma_parts(policy)
    salience = [{} for _ in layers]
    n_batches = 0
    if batched_weights.ndim != 2 or batched_weights.shape[0] != len(calib_data):
        raise ValueError(
            f"weights must be (n_orig_batches, batch_size)=({len(calib_data)}, ?), "
            f"got {tuple(batched_weights.shape)}"
        )
    for batch, alphas in tqdm(
        zip(calib_data, batched_weights),
        total=len(calib_data),
        desc="Taylor weighted mb=2",
    ):
        bsz = _batch_size(batch)
        if int(alphas.numel()) != bsz:
            raise ValueError(f"batch size {bsz} != weight row {int(alphas.numel())}")
        n_batches += _run_microbatches(
            policy, batch, device, layers, alphas.float(), microbatch_size, salience
        )
    scores = []
    for i in range(len(layers)):
        total = sum(s.abs().sum().item() for s in salience[i].values())
        scores.append(total / max(n_batches, 1))
    return {
        "method": "taylor_action_discontinuity",
        "paligemma": torch.tensor(scores),
        "aggregation": "mb=2 mean(α L); accumulate (g⊙θ); abs of global sum / N_mb",
        "n_microbatches": n_batches,
    }


def _drop8(scores: torch.Tensor) -> list[int]:
    from common import importance_ranking

    order = importance_ranking(scores, ks=(8,))["redundancy_order"]
    return drop_from_redundancy(order, 8, exclude_l0=True)


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
        print("sanity: α=1, 4x batch=2 weighted-mean vs Stage-1 Taylor on batch 0", flush=True)
        check = sanity_unweighted_matches(policy, calib[0], device, args.microbatch_size)
        print(json.dumps(check, indent=2), flush=True)
        sanity_path = FN_ROOT / "results" / "stage2_action_weight" / "pi05_taylor_sanity.json"
        sanity_path.parent.mkdir(parents=True, exist_ok=True)
        sanity_path.write_text(json.dumps(check, indent=2) + "\n")
        if not check["ok"]:
            raise SystemExit("α=1 weighted Taylor does not match Stage-1 mb=2 reduction; stop.")

    results = measure_taylor_weighted(policy, calib, device, batched, args.microbatch_size)
    print_ranking("Taylor weighted (action discontinuity)", results["paligemma"])
    drop = _drop8(results["paligemma"])
    results["drop8_nol0"] = drop
    results["stage1_drop8_nol0"] = S1_DROP8
    results["jaccard_drop8"] = jaccard(drop, S1_DROP8)
    results["weights_file"] = str(args.weights_file)
    out = args.save_file or (
        FN_ROOT / "results" / "stage2_action_weight" / "pi05_taylor_action_discontinuity.pt"
    )
    save_scores(out, results)
    sidecar = {
        "drop8_nol0": drop,
        "stage1_drop8_nol0": S1_DROP8,
        "jaccard_drop8": results["jaccard_drop8"],
        "aggregation": results["aggregation"],
        "n_microbatches": results["n_microbatches"],
    }
    Path(out).with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(
        f"Drop-8 no-L0 {drop} Jaccard vs Stage-1 {S1_DROP8} = {results['jaccard_drop8']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

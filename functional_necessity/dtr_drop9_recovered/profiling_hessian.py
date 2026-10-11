#!/usr/bin/env python3
"""Hessian-trace via Hutchinson: S_l = |Tr(H_l)|  (DTR Eq. 12, K=1)."""

from __future__ import annotations

import sys
from pathlib import Path as _ScriptDir

sys.path.insert(0, str(_ScriptDir(__file__).resolve().parent))

import argparse
import contextlib

import torch
from tqdm import tqdm

from backend import (
    add_common_args,
    calib_num_steps,
    compute_loss,
    cuda_empty_cache,
    default_save_file,
    init_logging,
    iter_calib_batches,
    load_calib_data,
    load_model,
    paligemma_parts,
    prepare_model_for_grad_scoring,
    print_ranking,
    save_scores,
    set_seed,
)


def _rademacher_like(param: torch.Tensor) -> torch.Tensor:
    bits = torch.randint(0, 2, param.shape, device=param.device, dtype=torch.int8)
    return (bits * 2 - 1).to(dtype=param.dtype)


def _hvp_trace_for_params(loss, params, num_hutchinson: int, retain_loss_graph: bool) -> float:
    active = [p for p in params if p.requires_grad]
    if not active:
        return 0.0
    grads = torch.autograd.grad(
        loss, active, create_graph=True, retain_graph=True, allow_unused=True
    )
    paired = [(p, g) for p, g in zip(active, grads) if g is not None]
    del grads
    if not paired:
        return 0.0
    total = 0.0
    for k in range(num_hutchinson):
        probes = [_rademacher_like(p) for p, _ in paired]
        gvp = sum((g * z).sum() for (_, g), z in zip(paired, probes))
        keep_graph = (k != num_hutchinson - 1) or retain_loss_graph
        hvps = torch.autograd.grad(
            gvp,
            [p for p, _ in paired],
            retain_graph=keep_graph,
            allow_unused=True,
        )
        for (_, _), z, hvp in zip(paired, probes, hvps):
            if hvp is None:
                continue
            total += (z.float() * hvp.float()).sum().item() / num_hutchinson
        del gvp, hvps, probes
    del paired
    return total


def measure_hessian(model, calib_data, device, microbatch_size=None, num_hutchinson=1):
    layers, _ = paligemma_parts(model)
    num_layers = len(layers)
    traces = [0.0] * num_layers
    model.eval()
    per_layer_params = [[p for p in layer.parameters() if p.requires_grad] for layer in layers]
    n_batches = 0
    amp = device.type == "cuda"
    for batch in tqdm(
        iter_calib_batches(calib_data, device, microbatch_size),
        total=calib_num_steps(calib_data, microbatch_size),
        desc="Hessian",
    ):
        model.zero_grad(set_to_none=True)
        amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()
        with torch.enable_grad(), amp_ctx:
            loss = compute_loss(model, batch)
            for i, params in enumerate(per_layer_params):
                traces[i] += _hvp_trace_for_params(
                    loss, params, num_hutchinson, retain_loss_graph=i != num_layers - 1
                )
        del loss
        cuda_empty_cache()
        n_batches += 1
    traces = [t / max(n_batches, 1) for t in traces]
    raw = torch.tensor(traces)
    return {
        "method": "hessian",
        "paligemma": raw.abs(),
        "paligemma_raw_trace": raw,
        "microbatch_size": microbatch_size,
        "hvp": "per_layer",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--num-hutchinson", type=int, default=1)
    parser.set_defaults(microbatch_size=1)
    args = parser.parse_args()
    init_logging()
    set_seed(args.seed)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    prepare_model_for_grad_scoring(model, recompute=False)
    calib = load_calib_data(args.calib_file)
    results = measure_hessian(
        model, calib, device, args.microbatch_size, num_hutchinson=args.num_hutchinson
    )
    print_ranking("Hessian", results["paligemma"])
    save_scores(args.save_file or default_save_file("hessian"), results)


if __name__ == "__main__":
    main()

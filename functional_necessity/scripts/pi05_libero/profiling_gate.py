#!/usr/bin/env python3
"""GateProbe: S_l = E |<dL/dh_l, h_l - h_{l-1}>|  (DTR Appendix D).

Aggregation is over the original calibration batches (64 x batch=8). Microbatch
is only a memory split: signed inner products from the 4 x mb=2 forwards are
averaged to reconstruct the batch-8 inner product, then abs is taken.

If h_l.grad is None (typical for L17: language residual after the last layer is
not in the action-loss graph), the score is 0. Do not fall back to h_{l-1}.grad.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path as _ScriptDir

sys.path.insert(0, str(_ScriptDir(__file__).resolve().parent))

import argparse

import torch
from tqdm import tqdm

from common import (
    DEFAULT_GRAD_MICROBATCH,
    add_common_args,
    compute_loss,
    cuda_empty_cache,
    default_save_file,
    init_logging,
    iter_orig_calib_microbatches,
    load_calib_data,
    load_policy,
    paligemma_parts,
    prepare_policy_for_grad_scoring,
    print_ranking,
    save_scores,
    set_seed,
)


def _signed_inners(policy, batch, layers, final_norm) -> tuple[list[float], list[bool]]:
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

    loss = None
    row = [0.0] * num_layers
    had_hout_grad = [False] * num_layers
    try:
        with torch.enable_grad():
            loss = compute_loss(policy, batch)
            loss.backward()

        for layer_idx in range(num_layers):
            h_in = states.get(layer_idx)
            h_out = states.get(layer_idx + 1)
            if h_in is None or h_out is None:
                continue
            if h_out.grad is None:
                continue
            had_hout_grad[layer_idx] = True
            residual = h_out.float() - h_in.float()
            grad_h = h_out.grad.float()
            row[layer_idx] = float((grad_h * residual).sum().item())
    finally:
        for handle in hooks:
            handle.remove()
        del states, loss
        policy.zero_grad(set_to_none=True)
        cuda_empty_cache()
    return row, had_hout_grad


def measure_gate(policy, calib_data, device, microbatch_size=None):
    layers, final_norm = paligemma_parts(policy)
    num_layers = len(layers)
    raw_rows = []
    none_grad_counts = [0] * num_layers
    n_micro = 0
    policy.eval()

    for microbatches in tqdm(
        iter_orig_calib_microbatches(calib_data, device, microbatch_size),
        total=len(calib_data),
        desc="GateProbe",
    ):
        signed = [0.0] * num_layers
        n_mb = 0
        for batch in microbatches:
            row, had_grad = _signed_inners(policy, batch, layers, final_norm)
            for i, value in enumerate(row):
                signed[i] += value
                if not had_grad[i]:
                    none_grad_counts[i] += 1
            n_mb += 1
            n_micro += 1
        raw_rows.append([s / max(n_mb, 1) for s in signed])

    raw = torch.tensor(raw_rows, dtype=torch.float32)
    abs_mean = raw.abs().mean(dim=0)
    logging.info(
        "GateProbe none h_out.grad counts over %d microbatches: %s",
        n_micro,
        none_grad_counts,
    )
    return {
        "method": "gateprobe",
        "paligemma": abs_mean,
        "paligemma_signed": raw.mean(dim=0).abs(),
        "paligemma_square": (raw**2).mean(dim=0),
        "paligemma_raw": raw,
        "aggregation": "orig_batch_then_abs",
        "n_orig_batches": int(raw.shape[0]),
        "n_microbatches": n_micro,
        "none_hout_grad_counts": none_grad_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(microbatch_size=DEFAULT_GRAD_MICROBATCH)
    args = parser.parse_args()
    init_logging()
    set_seed(args.seed)
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device)
    prepare_policy_for_grad_scoring(policy, recompute=False)
    calib = load_calib_data(args.calib_file)
    results = measure_gate(policy, calib, device, args.microbatch_size)
    print_ranking("GateProbe", results["paligemma"])
    save_scores(args.save_file or default_save_file("gateprobe"), results)


if __name__ == "__main__":
    main()

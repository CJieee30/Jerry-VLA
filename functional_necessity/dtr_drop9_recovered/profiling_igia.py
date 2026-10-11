#!/usr/bin/env python3
"""IGIA: orig-batch mean grad, then square (no w^2). DTR Eq. 10."""

from __future__ import annotations

import sys
from pathlib import Path as _ScriptDir

sys.path.insert(0, str(_ScriptDir(__file__).resolve().parent))

import argparse

import torch
from tqdm import tqdm

from backend import (
    DEFAULT_GRAD_MICROBATCH,
    add_common_args,
    cuda_empty_cache,
    default_save_file,
    init_logging,
    iter_orig_calib_microbatches,
    load_calib_data,
    load_model,
    orig_batch_accumulate_grads,
    paligemma_parts,
    prepare_model_for_grad_scoring,
    print_ranking,
    save_scores,
    set_seed,
)


def measure_igia(model, calib_data, device, microbatch_size=None):
    layers, _ = paligemma_parts(model)
    scores = [0.0] * len(layers)
    n_orig = 0
    for microbatches in tqdm(
        iter_orig_calib_microbatches(calib_data, device, microbatch_size),
        total=len(calib_data),
        desc="IGIA",
    ):
        model.zero_grad(set_to_none=True)
        try:
            n_mb = orig_batch_accumulate_grads(model, microbatches)
            for i, layer in enumerate(layers):
                for param in layer.parameters():
                    if param.grad is None:
                        continue
                    g8 = param.grad.float() / max(n_mb, 1)
                    scores[i] += (g8**2).sum().item()
            n_orig += 1
        finally:
            model.zero_grad(set_to_none=True)
            cuda_empty_cache()
    return {
        "method": "igia",
        "paligemma": torch.tensor(scores),
        "aggregation": "orig_batch_mean_grad_then_square_sum",
        "n_orig_batches": n_orig,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(microbatch_size=DEFAULT_GRAD_MICROBATCH)
    args = parser.parse_args()
    init_logging()
    set_seed(args.seed)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    prepare_model_for_grad_scoring(model, recompute=True)
    calib = load_calib_data(args.calib_file)
    results = measure_igia(model, calib, device, args.microbatch_size)
    print_ranking("IGIA", results["paligemma"])
    save_scores(args.save_file or default_save_file("igia"), results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fisher OBD: S_l = (1/N) sum_t sum_w (dL_t/dw)^2 * w^2  (DTR Eq. 11).

N is the number of original calibration batches (64 x batch=8). Microbatch is
only a memory split: grads from the 4 x mb=2 backwards are accumulated, then
divided to reconstruct the batch-8 mean gradient, then squared. E[g^2] is not
(E g)^2, so squaring after mb=2 is not equivalent.
"""

from __future__ import annotations

import sys
from pathlib import Path as _ScriptDir

sys.path.insert(0, str(_ScriptDir(__file__).resolve().parent))

import argparse

import torch
from tqdm import tqdm

from common import (
    DEFAULT_GRAD_MICROBATCH,
    add_common_args,
    cuda_empty_cache,
    default_save_file,
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
)


def measure_fisher(policy, calib_data, device, microbatch_size=None):
    layers, _ = paligemma_parts(policy)
    scores = [0.0] * len(layers)
    n_orig = 0
    for microbatches in tqdm(
        iter_orig_calib_microbatches(calib_data, device, microbatch_size),
        total=len(calib_data),
        desc="Fisher",
    ):
        policy.zero_grad(set_to_none=True)
        try:
            n_mb = orig_batch_accumulate_grads(policy, microbatches)
            for i, layer in enumerate(layers):
                for param in layer.parameters():
                    if param.grad is None:
                        continue
                    g8 = param.grad.float() / max(n_mb, 1)
                    scores[i] += (g8**2 * param.data.float() ** 2).sum().item()
            n_orig += 1
        finally:
            policy.zero_grad(set_to_none=True)
            cuda_empty_cache()
    scores = [s / max(n_orig, 1) for s in scores]
    return {
        "method": "fisher",
        "paligemma": torch.tensor(scores),
        "aggregation": "orig_batch_mean_grad_then_square",
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
    policy = load_policy(args.checkpoint, device)
    prepare_policy_for_grad_scoring(policy, recompute=True)
    calib = load_calib_data(args.calib_file)
    results = measure_fisher(policy, calib, device, args.microbatch_size)
    print_ranking("Fisher", results["paligemma"])
    save_scores(args.save_file or default_save_file("fisher"), results)


if __name__ == "__main__":
    main()

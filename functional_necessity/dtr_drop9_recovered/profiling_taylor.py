#!/usr/bin/env python3
"""Taylor: S_l = (1/N) sum_w |sum_t (dL_t/dw) * w|  (DTR Eq. 9)."""

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


def measure_taylor(model, calib_data, device, microbatch_size=None):
    layers, _ = paligemma_parts(model)
    salience = [{} for _ in layers]
    n_batches = 0
    for batch in tqdm(
        iter_calib_batches(calib_data, device, microbatch_size),
        total=calib_num_steps(calib_data, microbatch_size),
        desc="Taylor",
    ):
        model.zero_grad(set_to_none=True)
        loss = None
        try:
            with torch.enable_grad():
                loss = compute_loss(model, batch)
                loss.backward()
            for i, layer in enumerate(layers):
                for name, param in layer.named_parameters():
                    if param.grad is None:
                        continue
                    s = (param.grad.float() * param.data.float()).cpu()
                    if name not in salience[i]:
                        salience[i][name] = s
                    else:
                        salience[i][name] += s
            n_batches += 1
        finally:
            del loss
            model.zero_grad(set_to_none=True)
            cuda_empty_cache()
    scores = []
    for i in range(len(layers)):
        total = sum(s.abs().sum().item() for s in salience[i].values())
        scores.append(total / max(n_batches, 1))
    return {"method": "taylor", "paligemma": torch.tensor(scores)}


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
    results = measure_taylor(model, calib, device, args.microbatch_size)
    print_ranking("Taylor", results["paligemma"])
    save_scores(args.save_file or default_save_file("taylor"), results)


if __name__ == "__main__":
    main()

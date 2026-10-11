#!/usr/bin/env python3
"""PPL leave-one-out on the OpenPI joint graph with frozen (noise, time).

Already-dropped Stage-1 layers stay in the drop list. Extra skip uses
`extra_llm_drop` (compute_layer_complete Case 3).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path as _ScriptDir

sys.path.insert(0, str(_ScriptDir(__file__).resolve().parent))

import argparse

import torch
from tqdm import tqdm

from backend import (
    N_LAYERS,
    FlowMatchingReplay,
    add_common_args,
    compute_loss,
    default_save_file,
    extra_llm_drop,
    init_logging,
    iter_calib_batches,
    load_calib_data,
    load_model,
    print_ranking,
    save_scores,
    set_seed,
)


def average_loss(model, calib_data, device, microbatch_size=None, replay=None) -> float:
    total = 0.0
    n = 0
    if replay is not None:
        replay.reset()
    with torch.no_grad():
        for batch in iter_calib_batches(calib_data, device, microbatch_size):
            kwargs = {}
            if replay is not None:
                noise, time = replay.current(device)
                kwargs["noise"] = noise
                kwargs["time"] = time
            total += float(compute_loss(model, batch, **kwargs).item())
            n += 1
            if replay is not None:
                replay.step()
    if replay is not None and replay.i != len(replay.noises):
        raise RuntimeError(f"PPL replay consumed {replay.i}/{len(replay.noises)} flow samples")
    return total / max(n, 1)


def measure_ppl(model, calib_data, device, microbatch_size=None):
    model.eval()
    replay = FlowMatchingReplay(model)
    replay.capture(calib_data, device, microbatch_size)
    logging.info("Computing baseline loss with frozen (noise, time)...")
    baseline = average_loss(model, calib_data, device, microbatch_size, replay=replay)
    logging.info("Baseline loss: %.6f", baseline)
    deltas = []
    print(f"\nPaliGemma PPL ({N_LAYERS} layers)")
    print(f"  {'Layer':>5}  {'DroppedLoss':>12}  {'Delta':>12}")
    for i in tqdm(range(N_LAYERS), desc="PPL"):
        with extra_llm_drop(model, i):
            dropped = average_loss(model, calib_data, device, microbatch_size, replay=replay)
        delta = dropped - baseline
        deltas.append(delta)
        print(f"  {i:>5}  {dropped:>12.6f}  {delta:>12.6f}")
    return {
        "method": "ppl",
        "baseline_loss": baseline,
        "paligemma": torch.tensor(deltas),
        "paired_flow_matching": True,
        "n_flow_pairs": len(replay.noises),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    init_logging()
    set_seed(args.seed)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    calib = load_calib_data(args.calib_file)
    results = measure_ppl(model, calib, device, args.microbatch_size)
    print_ranking("PPL", results["paligemma"])
    save_scores(args.save_file or default_save_file("ppl"), results)


if __name__ == "__main__":
    main()

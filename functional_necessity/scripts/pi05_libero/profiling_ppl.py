#!/usr/bin/env python3
"""PPL / leave-one-out: S_l = L(model w/o layer l) - L(full)  (DTR Eq. 14).

Flow-matching noise and time are sampled once per calibration batch and replayed
for the baseline and every dropped-layer evaluation. Otherwise L17's delta is
dominated by a different ξ rather than by dropping a layer whose language
residual is unused by the action loss.
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
    FlowMatchingReplay,
    add_common_args,
    compute_loss,
    default_save_file,
    init_logging,
    iter_calib_batches,
    load_calib_data,
    load_policy,
    paligemma_parts,
    print_ranking,
    save_scores,
    set_seed,
)
from layer_skip import skip_llm_layers


def average_loss(policy, calib_data, device, microbatch_size=None, replay=None) -> float:
    total = 0.0
    n = 0
    if replay is not None:
        replay.reset()
    with torch.no_grad():
        for batch in iter_calib_batches(calib_data, device, microbatch_size):
            total += float(compute_loss(policy, batch).item())
            n += 1
            if replay is not None:
                replay.step()
    if replay is not None and replay.i != len(replay.noises):
        raise RuntimeError(f"PPL replay consumed {replay.i}/{len(replay.noises)} flow samples")
    return total / max(n, 1)


def measure_ppl(policy, calib_data, device, microbatch_size=None):
    layers, _ = paligemma_parts(policy)
    num_layers = len(layers)
    policy.eval()
    replay = FlowMatchingReplay(policy)
    replay.capture(calib_data, device, microbatch_size)

    logging.info("Computing baseline loss with frozen (noise, time)...")
    with replay:
        baseline = average_loss(policy, calib_data, device, microbatch_size, replay=replay)
        logging.info("Baseline loss: %.6f", baseline)

        deltas = []
        print(f"\nPaliGemma PPL ({num_layers} layers)")
        print(f"  {'Layer':>5}  {'DroppedLoss':>12}  {'Delta':>12}")
        for i in tqdm(range(num_layers), desc="PPL"):
            with skip_llm_layers(policy, [i]):
                dropped = average_loss(policy, calib_data, device, microbatch_size, replay=replay)
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
    policy = load_policy(args.checkpoint, device)
    calib = load_calib_data(args.calib_file)
    results = measure_ppl(policy, calib, device, args.microbatch_size)
    print_ranking("PPL", results["paligemma"])
    save_scores(args.save_file or default_save_file("ppl"), results)


if __name__ == "__main__":
    main()

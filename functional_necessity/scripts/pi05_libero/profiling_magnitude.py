#!/usr/bin/env python3
"""Magnitude: S_l = sum_w |w|  (DTR Eq. 15, data-independent)."""

from __future__ import annotations

import sys
from pathlib import Path as _ScriptDir

sys.path.insert(0, str(_ScriptDir(__file__).resolve().parent))

import argparse

import torch

from common import (
    add_common_args,
    default_save_file,
    init_logging,
    load_policy,
    paligemma_parts,
    print_ranking,
    save_scores,
    set_seed,
)


def measure_magnitude(policy):
    layers, _ = paligemma_parts(policy)
    scores = []
    for layer in layers:
        mag = sum(p.data.float().abs().sum().item() for p in layer.parameters())
        scores.append(mag)
    return {"method": "magnitude", "paligemma": torch.tensor(scores)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, need_calib=False)
    args = parser.parse_args()
    init_logging()
    set_seed(args.seed)
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device)
    results = measure_magnitude(policy)
    print_ranking("Magnitude", results["paligemma"])
    save_scores(args.save_file or default_save_file("magnitude"), results)


if __name__ == "__main__":
    main()

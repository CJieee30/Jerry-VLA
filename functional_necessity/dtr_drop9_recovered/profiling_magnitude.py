#!/usr/bin/env python3
"""Magnitude: S_l = sum_w |w|  (DTR Eq. 15)."""

from __future__ import annotations

import sys
from pathlib import Path as _ScriptDir

sys.path.insert(0, str(_ScriptDir(__file__).resolve().parent))

import argparse

import torch

from backend import (
    add_common_args,
    default_save_file,
    init_logging,
    load_model,
    paligemma_parts,
    print_ranking,
    save_scores,
    set_seed,
)


def measure_magnitude(model):
    layers, _ = paligemma_parts(model)
    scores = [sum(p.data.float().abs().sum().item() for p in layer.parameters()) for layer in layers]
    return {"method": "magnitude", "paligemma": torch.tensor(scores)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, need_calib=False)
    args = parser.parse_args()
    init_logging()
    set_seed(args.seed)
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    results = measure_magnitude(model)
    print_ranking("Magnitude", results["paligemma"])
    save_scores(args.save_file or default_save_file("magnitude"), results)


if __name__ == "__main__":
    main()

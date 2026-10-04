#!/usr/bin/env python3
"""CosSim / Block Influence: S_l = 1 - E[cos(h_{l-1}, h_l)]  (DTR Eq. 13)."""

from __future__ import annotations

import sys
from pathlib import Path as _ScriptDir

sys.path.insert(0, str(_ScriptDir(__file__).resolve().parent))

import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import (
    add_common_args,
    calib_num_steps,
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


def measure_cossim(policy, calib_data, device, microbatch_size=None):
    layers, final_norm = paligemma_parts(policy)
    num_layers = len(layers)
    cos_sims = {i: [] for i in range(num_layers)}
    policy.eval()

    for batch in tqdm(
        iter_calib_batches(calib_data, device, microbatch_size),
        total=calib_num_steps(calib_data, microbatch_size),
        desc="CosSim",
    ):
        states = {}
        hooks = []

        def make_pre_hook(idx):
            def hook(module, args, kwargs=None):
                hidden = args[0] if len(args) > 0 else (kwargs.get("hidden_states") if kwargs else None)
                if hidden is not None:
                    states[idx] = hidden.detach()

            return hook

        for i, layer in enumerate(layers):
            hooks.append(layer.input_layernorm.register_forward_pre_hook(make_pre_hook(i), with_kwargs=True))
        hooks.append(final_norm.register_forward_pre_hook(make_pre_hook(num_layers), with_kwargs=True))

        with torch.no_grad():
            _ = compute_loss(policy, batch)

        for i in range(num_layers):
            if i in states and (i + 1) in states:
                sim = F.cosine_similarity(states[i].float(), states[i + 1].float(), dim=-1)
                cos_sims[i].append(sim.mean().item())
        for handle in hooks:
            handle.remove()

    bi = []
    avg_cos = []
    for i in range(num_layers):
        avg = sum(cos_sims[i]) / max(len(cos_sims[i]), 1)
        avg_cos.append(avg)
        bi.append(1.0 - avg)
    return {
        "method": "cossim",
        "paligemma": torch.tensor(bi),
        "paligemma_cos": torch.tensor(avg_cos),
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
    results = measure_cossim(policy, calib, device, args.microbatch_size)
    print_ranking("CosSim", results["paligemma"])
    save_scores(args.save_file or default_save_file("cossim"), results)


if __name__ == "__main__":
    main()

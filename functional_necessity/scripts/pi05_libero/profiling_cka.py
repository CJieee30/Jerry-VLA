#!/usr/bin/env python3
"""Adjacent linear CKA (CLP formula) on concatenated calibration tokens.

CKA(X, Y) = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F) with global centering
over all calibration tokens, not the mean of per-batch CKAs.

Deletion is the same as other methods: importance = 1 - CKA, drop high adjacent
CKA first. This does not use CLP plateau blocks / anchors / TopK.

Grams are accumulated in one pass (uncentered X^T X plus mean correction) so
activations are never concatenated in RAM.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path as _ScriptDir

sys.path.insert(0, str(_ScriptDir(__file__).resolve().parent))

import argparse

import torch
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


def _flatten_tokens(hidden: torch.Tensor, max_tokens: int) -> torch.Tensor:
    batch, seq, dim = hidden.shape
    if max_tokens and seq > max_tokens:
        step = int(math.ceil(seq / max_tokens))
        hidden = hidden[:, ::step, :][:, :max_tokens, :]
    return hidden.reshape(-1, dim).float()


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xtx = x.T @ x
    yty = y.T @ y
    ytx = y.T @ x
    numer = (ytx * ytx).sum()
    denom = xtx.norm() * yty.norm()
    if float(denom) <= 0:
        return 0.0
    return float((numer / denom).item())


def cka_from_moments(n: float, sum_x: torch.Tensor, sum_y: torch.Tensor, xtx, yty, ytx) -> float:
    mean_x = sum_x / n
    mean_y = sum_y / n
    xtx_c = xtx - n * torch.outer(mean_x, mean_x)
    yty_c = yty - n * torch.outer(mean_y, mean_y)
    ytx_c = ytx - n * torch.outer(mean_y, mean_x)
    numer = (ytx_c * ytx_c).sum()
    denom = xtx_c.norm() * yty_c.norm()
    if float(denom) <= 0:
        return 0.0
    return float((numer / denom).item())


def _self_check_moments() -> None:
    torch.manual_seed(0)
    x = torch.randn(128, 32)
    y = x + 0.1 * torch.randn(128, 32)
    ref = linear_cka(x, y)
    got = cka_from_moments(
        float(x.shape[0]),
        x.sum(dim=0),
        y.sum(dim=0),
        x.T @ x,
        y.T @ y,
        y.T @ x,
    )
    if abs(ref - got) > 1e-5:
        raise RuntimeError(f"CKA moment identity failed: concat={ref:.8f} moments={got:.8f}")


def measure_cka(policy, calib_data, device, microbatch_size=None, max_tokens=0):
    _self_check_moments()
    layers, final_norm = paligemma_parts(policy)
    num_layers = len(layers)
    n_tokens = [0.0] * num_layers
    sum_in = [None] * num_layers
    sum_out = [None] * num_layers
    xtx = [None] * num_layers
    yty = [None] * num_layers
    ytx = [None] * num_layers
    policy.eval()

    for batch in tqdm(
        iter_calib_batches(calib_data, device, microbatch_size),
        total=calib_num_steps(calib_data, microbatch_size),
        desc="CKA",
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
        for handle in hooks:
            handle.remove()

        for i in range(num_layers):
            if i not in states or (i + 1) not in states:
                continue
            h_in = _flatten_tokens(states[i], max_tokens)
            h_out = _flatten_tokens(states[i + 1], max_tokens)
            n = h_in.shape[0]
            n_tokens[i] += n
            if sum_in[i] is None:
                dim = h_in.shape[1]
                sum_in[i] = torch.zeros(dim, dtype=torch.float32, device=h_in.device)
                sum_out[i] = torch.zeros(dim, dtype=torch.float32, device=h_in.device)
                xtx[i] = torch.zeros(dim, dim, dtype=torch.float32, device=h_in.device)
                yty[i] = torch.zeros(dim, dim, dtype=torch.float32, device=h_in.device)
                ytx[i] = torch.zeros(dim, dim, dtype=torch.float32, device=h_in.device)
            sum_in[i] += h_in.sum(dim=0)
            sum_out[i] += h_out.sum(dim=0)
            xtx[i] += h_in.T @ h_in
            yty[i] += h_out.T @ h_out
            ytx[i] += h_out.T @ h_in
        del states

    adjacent = []
    for i in range(num_layers):
        if sum_in[i] is None or n_tokens[i] <= 1:
            adjacent.append(0.0)
            continue
        adjacent.append(
            cka_from_moments(n_tokens[i], sum_in[i], sum_out[i], xtx[i], yty[i], ytx[i])
        )
    adjacent = torch.tensor(adjacent, dtype=torch.float32)
    importance = 1.0 - adjacent
    logging.info("CKA token counts per layer: %s", [int(x) for x in n_tokens])
    return {
        "method": "cka",
        "paligemma": importance,
        "paligemma_cka": adjacent,
        "cka_reduction": "global_concat_moments",
        "n_tokens": n_tokens,
        "max_tokens": max_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Per-sequence subsample cap; 0 = use all tokens (CLP concat).",
    )
    args = parser.parse_args()
    init_logging()
    set_seed(args.seed)
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device)
    calib = load_calib_data(args.calib_file)
    results = measure_cka(policy, calib, device, args.microbatch_size, max_tokens=args.max_tokens)
    print_ranking("CKA", results["paligemma"])
    save_scores(args.save_file or default_save_file("cka"), results)


if __name__ == "__main__":
    main()

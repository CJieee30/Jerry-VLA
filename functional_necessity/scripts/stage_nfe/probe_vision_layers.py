#!/usr/bin/env python3
"""Offline probe: are π0.5 SigLIP blocks redundant for the action?

One backward records both scores on the flow-matching loss:
  Taylor  |sum_t g_t * w| / N_samples
  Fisher  mean over original batches of (g_batch^2 * w^2), g_batch = mean sample grad
Then identity-skip (not keep-KV) and measure action L2. No CARE, no LEDGE, no
closed-loop. Language and expert weights stay frozen; only SigLIP gets grads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))

from common import (  # noqa: E402
    DEFAULT_CALIB_FILE,
    DEFAULT_CHECKPOINT,
    compute_loss,
    load_calib_data,
    load_policy,
    move_to_device,
    slice_batch,
)

OUT = FN / "results" / "stage_nfe" / "vision_probe_pi05.json"
N_BATCHES = 4
MB = 1
N_L2_FRAMES = 4
EXEC = 10


def vision_layers(policy):
    tower = policy.model.paligemma_with_expert.paligemma.model.vision_tower
    return list(tower.vision_model.encoder.layers)


def score_both(policy, calib, device) -> tuple[list[float], list[float]]:
    """Taylor and Fisher from the same backward. Fisher squares the batch-mean grad."""
    layers = vision_layers(policy)
    for param in policy.parameters():
        param.requires_grad_(False)
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad_(True)
    policy.eval()
    if hasattr(policy.model, "gradient_checkpointing_enable"):
        policy.model.gradient_checkpointing_enable()
    taylor_sum: dict[tuple[int, str], torch.Tensor] = {}
    fisher = [0.0] * len(layers)
    n_samples = 0
    n_orig = 0
    for batch_i in range(N_BATCHES):
        full = calib[batch_i]
        width = int(full["action"].shape[0])
        fisher_g: dict[tuple[int, str], torch.Tensor] = {}
        weights: dict[tuple[int, str], torch.Tensor] = {}
        n_mb = 0
        for start in range(0, width, MB):
            batch = move_to_device(slice_batch(full, start, start + MB), device)
            policy.zero_grad(set_to_none=True)
            loss = compute_loss(policy, batch)
            loss.backward()
            for i, layer in enumerate(layers):
                for name, param in layer.named_parameters():
                    if param.grad is None:
                        continue
                    g = param.grad.detach().float().cpu()
                    key = (i, name)
                    if key not in weights:
                        weights[key] = param.detach().float().cpu()
                    prod = g * weights[key]
                    if key not in taylor_sum:
                        taylor_sum[key] = prod
                    else:
                        taylor_sum[key] += prod
                    if key not in fisher_g:
                        fisher_g[key] = g
                    else:
                        fisher_g[key] += g
            n_mb += 1
            n_samples += 1
            policy.zero_grad(set_to_none=True)
            del loss, batch
            torch.cuda.empty_cache()
            print(f"score batch {batch_i} step {start} done n={n_samples}", flush=True)
        for i, layer in enumerate(layers):
            for name, _param in layer.named_parameters():
                key = (i, name)
                if key not in fisher_g:
                    continue
                gmean = fisher_g[key] / max(n_mb, 1)
                fisher[i] += float((gmean**2 * weights[key] ** 2).sum())
        n_orig += 1
        del fisher_g
        print(f"fisher batch {batch_i} closed n_mb={n_mb}", flush=True)
    taylor = []
    for i, layer in enumerate(layers):
        total = 0.0
        for name, _param in layer.named_parameters():
            key = (i, name)
            if key in taylor_sum:
                total += float(taylor_sum[key].abs().sum())
        taylor.append(total / max(n_samples, 1))
    fisher = [s / max(n_orig, 1) for s in fisher]
    return taylor, fisher


def _identity(hidden_states, attention_mask=None, **kwargs):
    return hidden_states


def action_l2(policy, batch, noise, skip_ids: list[int]):
    layers = vision_layers(policy)
    originals = [(i, layers[i].forward) for i in skip_ids]
    for i, _orig in originals:
        layers[i].forward = _identity
    try:
        policy.reset()
        with torch.no_grad():
            return policy.predict_action_chunk(batch, noise=noise, num_steps=10)
    finally:
        for i, orig in originals:
            layers[i].forward = orig


def stats(pred, ref) -> dict:
    x = pred[:, :EXEC, :7].float()
    y = ref[:, :EXEC, :7].float()
    diff = x - y
    return {
        "max_abs": float(diff.abs().max()),
        "l2": float(diff.flatten(1).norm(dim=-1).mean()),
        "ref_l2": float(y.flatten(1).norm(dim=-1).mean()),
    }


def main() -> None:
    device = torch.device("cuda")
    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.n_action_steps = EXEC
    policy.config.compile_model = False
    policy.eval()
    calib = load_calib_data(DEFAULT_CALIB_FILE)
    layers = vision_layers(policy)
    print(f"siglip layers {len(layers)} type {type(layers[0]).__name__}", flush=True)

    taylor, fisher = score_both(policy, calib, device)
    taylor_order = sorted(range(len(taylor)), key=lambda i: taylor[i])
    fisher_order = sorted(range(len(fisher)), key=lambda i: fisher[i])
    print("taylor low-to-high", [(i, round(taylor[i], 4)) for i in taylor_order], flush=True)
    print("fisher low-to-high", [(i, round(fisher[i], 4)) for i in fisher_order], flush=True)

    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    policy.eval()
    for param in policy.parameters():
        param.requires_grad_(False)

    frames = []
    noises = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1000)
    for i in range(N_L2_FRAMES):
        batch = move_to_device(slice_batch(calib[0], i, i + 1), device)
        bsize = 1
        noise = torch.randn(
            bsize,
            int(policy.model.config.chunk_size),
            int(policy.model.config.max_action_dim),
            generator=gen,
        ).to(device)
        frames.append(batch)
        noises.append(noise)

    def mean_stats(skip_ids: list[int]) -> dict:
        rows = []
        for batch, noise in zip(frames, noises):
            policy.reset()
            with torch.no_grad():
                ref = policy.predict_action_chunk(batch, noise=noise, num_steps=10)
            pred = action_l2(policy, batch, noise, skip_ids)
            rows.append(stats(pred, ref))
        return {
            "skip": list(skip_ids),
            "l2": sum(r["l2"] for r in rows) / len(rows),
            "max_abs": max(r["max_abs"] for r in rows),
            "ref_l2": sum(r["ref_l2"] for r in rows) / len(rows),
        }

    # Fix action_l2: the lambda closure is fine. Verify replay is exact.
    replay = mean_stats([])
    print("replay", replay, flush=True)

    leave = []
    for i in range(len(layers)):
        row = mean_stats([i])
        leave.append(row)
        print(f"skip {i} l2={row['l2']:.4f} max_abs={row['max_abs']:.4f}", flush=True)

    groups = {
        "taylor_four_lowest": mean_stats(taylor_order[:4]),
        "taylor_four_highest": mean_stats(list(reversed(taylor_order[-4:]))),
        "fisher_four_lowest": mean_stats(fisher_order[:4]),
        "fisher_four_highest": mean_stats(list(reversed(fisher_order[-4:]))),
        "first_layer": mean_stats([0]),
        "last_layer": mean_stats([len(layers) - 1]),
    }
    print("groups", json.dumps(groups), flush=True)

    payload = {
        "checkpoint": DEFAULT_CHECKPOINT,
        "device": torch.cuda.get_device_name(0),
        "n_layers": len(layers),
        "n_batches": N_BATCHES,
        "microbatch": MB,
        "n_l2_frames": N_L2_FRAMES,
        "taylor": taylor,
        "fisher": fisher,
        "taylor_order_low_to_high": taylor_order,
        "fisher_order_low_to_high": fisher_order,
        "replay": replay,
        "leave_one_out": leave,
        "groups": groups,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

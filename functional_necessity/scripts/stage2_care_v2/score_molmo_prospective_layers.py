#!/usr/bin/env python3
"""One backward per prospective obs → IGIA / Fisher / Taylor / GateProbe layer scores."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "molmoact2_libero"))

import torch
from tqdm import tqdm

from prospective_molmo_layer_common import (
    CALIB_PATH,
    METHODS,
    N_LAYERS,
    N_OBS,
    SCORE_DIR,
    load_calibration_pool,
)


def _chunk_fisher_taylor(param) -> tuple[float, float]:
    if param.grad is None:
        return 0.0, 0.0
    g = param.grad.detach().reshape(-1)
    w = param.detach().reshape(-1)
    fisher = torch.zeros((), device=g.device, dtype=torch.float32)
    taylor = torch.zeros((), device=g.device, dtype=torch.float32)
    step = 262_144
    for start in range(0, g.numel(), step):
        gf = g[start : start + step].float()
        wf = w[start : start + step].float()
        fisher = fisher + (gf.square() * wf.square()).sum()
        taylor = taylor + (gf * wf).abs().sum()
    return float(fisher.item()), float(taylor.item())


def _chunk_igia(param) -> float:
    if param.grad is None:
        return 0.0
    g = param.grad.detach().reshape(-1)
    acc = torch.zeros((), device=g.device, dtype=torch.float32)
    step = 262_144
    for start in range(0, g.numel(), step):
        gf = g[start : start + step].float()
        acc = acc + gf.square().sum()
    return float(acc.item())


def _save(method: str, rows: list[list[float]]) -> Path:
    dest = SCORE_DIR / f"molmoact2_{method}.pt"
    mat = torch.tensor(rows, dtype=torch.float32)
    if tuple(mat.shape) != (len(rows), N_LAYERS):
        raise RuntimeError(f"{method} shape {tuple(mat.shape)}")
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": "prospective_layer_calib_v1",
            "stack": "molmoact2",
            "method": method,
            "frame_scores": mat,
            "n_samples": int(mat.shape[0]),
            "n_layers": N_LAYERS,
            "definitions": {
                "igia": "sum g^2 per layer",
                "fisher": "sum (g*theta)^2 per layer",
                "taylor": "sum |g*theta| per layer",
                "gateprobe": "|grad_h · (h_out - h_in)| at block boundary",
            },
        },
        dest,
    )
    dest.with_suffix(".json").write_text(
        json.dumps({"method": method, "shape": list(mat.shape)}, indent=2) + "\n"
    )
    return dest


def _done() -> bool:
    return all((SCORE_DIR / f"molmoact2_{m}.pt").is_file() for m in METHODS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if _done() and not args.force:
        print("[skip] all four score files exist", flush=True)
        return
    if not CALIB_PATH.is_file():
        raise SystemExit(f"missing {CALIB_PATH}; run dump_calib_molmo_prospective.py first")

    from load_policy import load_policy, vlm_blocks
    from score import _mean_flow, _scoring_mode, _set_vlm_grad

    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    samples = blob["samples"]
    if len(samples) != N_OBS:
        raise SystemExit(f"calib n={len(samples)} expected {N_OBS}")
    pool = load_calibration_pool()
    for i, (s, m) in enumerate(zip(samples, pool, strict=True)):
        if int(s["pool_index"]) != m.pool_index or int(s["global_index"]) != m.global_index:
            raise SystemExit(f"row {i} pool/global mismatch")

    device = torch.device(args.device)
    policy = load_policy(str(device))
    blocks, final_norm = vlm_blocks(policy)
    if len(blocks) != N_LAYERS:
        raise SystemExit(f"layers {len(blocks)} != {N_LAYERS}")
    allowed = {id(p) for b in blocks for p in b.parameters()}
    for p in policy.parameters():
        p.requires_grad_(id(p) in allowed)
    _set_vlm_grad(blocks, True)
    _scoring_mode(policy, blocks, checkpoint=True)
    lm_head = getattr(getattr(policy, "model", None), "lm_head", None)
    if lm_head is not None:
        lm_head.to("cpu")
        for p in lm_head.parameters():
            p.requires_grad_(False)

    rows = {m: [] for m in METHODS}
    log_path = SCORE_DIR.parent / "score_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as logf:
        logf.write(f"start device={device} n={N_OBS}\n")

    for sample in tqdm(samples, desc="molmo prospective score"):
        packed = {
            "batch": {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample["batch"].items()},
            "noise": sample["noise"].to(device),
            "timesteps": sample["timesteps"].to(device),
        }
        states = {}
        hooks = []

        def make(idx):
            def hook(module, args):
                if args:
                    states[idx] = args[0]

            return hook

        for i, block in enumerate(blocks):
            hooks.append(block.register_forward_pre_hook(make(i)))
        hooks.append(final_norm.register_forward_pre_hook(make(len(blocks))))
        for block in blocks:
            for p in block.parameters():
                p.grad = None
        try:
            loss = _mean_flow(policy, [packed], device)
            for h in states.values():
                if h.requires_grad:
                    h.retain_grad()
            loss.backward()
            igia, fisher, taylor, gate = [], [], [], []
            for i, block in enumerate(blocks):
                f_acc = t_acc = i_acc = 0.0
                for param in block.parameters():
                    f, t = _chunk_fisher_taylor(param)
                    f_acc += f
                    t_acc += t
                    i_acc += _chunk_igia(param)
                fisher.append(f_acc)
                taylor.append(t_acc)
                igia.append(i_acc)
                h_in, h_out = states.get(i), states.get(i + 1)
                if h_in is None or h_out is None or h_out.grad is None:
                    gate.append(0.0)
                else:
                    inner = float((h_out.grad.float() * (h_out.float() - h_in.float())).sum().item())
                    gate.append(abs(inner))
        finally:
            for h in hooks:
                h.remove()
            for block in blocks:
                for p in block.parameters():
                    p.grad = None
            del loss
            torch.cuda.empty_cache()

        rows["igia"].append(igia)
        rows["fisher"].append(fisher)
        rows["taylor"].append(taylor)
        rows["gateprobe"].append(gate)

    for method in METHODS:
        p = _save(method, rows[method])
        print(f"saved {p}", flush=True)
    with log_path.open("a") as logf:
        logf.write("done\n")


if __name__ == "__main__":
    main()

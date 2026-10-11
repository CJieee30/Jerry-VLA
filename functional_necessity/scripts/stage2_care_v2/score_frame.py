#!/usr/bin/env python3
"""Per-frame Fisher / Taylor / GateProbe on the CARE v2 480-frame set.

One backward per sample writes all three scores. GPU only. Does not overwrite
Stage-1 score files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch
from tqdm import tqdm

from protocol import METHODS, OUT_DIR, STACKS, STAGE1

CALIB_DIR = OUT_DIR
SCORE_DIR = OUT_DIR / "scores"


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


def _move(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=device.type == "cuda") if torch.is_tensor(value) else value
    return out


def _save_method(stack: str, method: str, rows: list[list[float]], n_layers: int) -> Path:
    dest = SCORE_DIR / f"{stack}_{method}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    mat = torch.tensor(rows, dtype=torch.float32)
    if tuple(mat.shape) != (len(rows), n_layers):
        raise RuntimeError(f"{stack} {method} shape {tuple(mat.shape)}")
    torch.save(
        {
            "protocol": "care_v2",
            "stack": stack,
            "method": method,
            "frame_scores": mat,
            "aggregation": "per_frame; fisher=g^2 θ^2; taylor=|g θ|; gateprobe=|I|",
            "n_samples": int(mat.shape[0]),
            "n_layers": n_layers,
        },
        dest,
    )
    dest.with_suffix(".json").write_text(
        json.dumps({"stack": stack, "method": method, "shape": list(mat.shape)}, indent=2) + "\n"
    )
    print(f"saved {dest} {tuple(mat.shape)}", flush=True)
    return dest


def _done(stack: str) -> bool:
    return all((SCORE_DIR / f"{stack}_{m}.pt").is_file() for m in METHODS)


class Pi05Stack:
    def __init__(self, device: torch.device):
        sys.path.insert(0, str(HERE.parents[0] / "pi05_libero"))
        from common import (
            load_policy,
            paligemma_parts,
            prepare_policy_for_grad_scoring,
            DEFAULT_CHECKPOINT,
        )

        self.device = device
        self.policy = load_policy(DEFAULT_CHECKPOINT, device)
        prepare_policy_for_grad_scoring(self.policy, recompute=True)
        self.layers, self.final_norm = paligemma_parts(self.policy)
        self.n_layers = len(self.layers)
        self.calib = torch.load(CALIB_DIR / "calib_pi05.pt", map_location="cpu", weights_only=False)["samples"]

    def _patch(self, noise, time):
        model = self.policy.model
        orig_n, orig_t = model.sample_noise, model.sample_time

        def sample_noise(_self, shape, device):
            n = noise.to(device)
            if tuple(n.shape) != tuple(shape):
                raise RuntimeError(f"pi05 noise {tuple(n.shape)} != {tuple(shape)}")
            return n

        def sample_time(_self, bsize, device):
            t = time.to(device)
            if int(t.shape[0]) != int(bsize):
                raise RuntimeError(f"pi05 time {int(t.shape[0])} != {bsize}")
            return t

        model.sample_noise = types.MethodType(sample_noise, model)
        model.sample_time = types.MethodType(sample_time, model)
        return orig_n, orig_t

    def _unpatch(self, orig_n, orig_t):
        self.policy.model.sample_noise = orig_n
        self.policy.model.sample_time = orig_t

    def score_one(self, sample) -> tuple[list[float], list[float], list[float]]:
        from common import compute_loss, cuda_empty_cache

        batch = _move(sample["batch"], self.device)
        orig = self._patch(sample["noise"], sample["time"])
        states = {}
        hooks = []

        def make(idx):
            def hook(module, args, kwargs=None):
                hidden = args[0] if len(args) > 0 else (kwargs.get("hidden_states") if kwargs else None)
                if hidden is not None:
                    hidden.retain_grad()
                    states[idx] = hidden

            return hook

        for i, layer in enumerate(self.layers):
            hooks.append(layer.input_layernorm.register_forward_pre_hook(make(i), with_kwargs=True))
        hooks.append(self.final_norm.register_forward_pre_hook(make(self.n_layers), with_kwargs=True))
        self.policy.zero_grad(set_to_none=True)
        try:
            with torch.enable_grad():
                loss = compute_loss(self.policy, batch)
                loss.backward()
            fisher, taylor, gate = [], [], []
            for i, layer in enumerate(self.layers):
                f_acc = t_acc = 0.0
                for param in layer.parameters():
                    f, t = _chunk_fisher_taylor(param)
                    f_acc += f
                    t_acc += t
                fisher.append(f_acc)
                taylor.append(t_acc)
                h_in, h_out = states.get(i), states.get(i + 1)
                if h_in is None or h_out is None or h_out.grad is None:
                    gate.append(0.0)
                else:
                    inner = float((h_out.grad.float() * (h_out.float() - h_in.float())).sum().item())
                    gate.append(abs(inner))
        finally:
            for handle in hooks:
                handle.remove()
            self.policy.zero_grad(set_to_none=True)
            self._unpatch(*orig)
            cuda_empty_cache()
        return fisher, taylor, gate


class SmolStack:
    def __init__(self, device: torch.device):
        sys.path.insert(0, str(HERE.parents[0] / "smolvla_libero"))
        from load_policy import load_policy, vlm_blocks
        from score import _scoring_mode, _set_vlm_grad

        self.device = device
        self.policy = load_policy(str(device))
        self.blocks, self.final_norm = vlm_blocks(self.policy)
        self.n_layers = len(self.blocks)
        _set_vlm_grad(self.blocks, True)
        _scoring_mode(self.policy, self.blocks, checkpoint=True)
        self.calib = torch.load(CALIB_DIR / "calib_smolvla.pt", map_location="cpu", weights_only=False)["samples"]

    def score_one(self, sample) -> tuple[list[float], list[float], list[float]]:
        from score import _hooks

        batch = _move(sample["batch"], self.device)
        noise = sample["noise"].to(self.device)
        time = sample["time"].to(self.device)
        states, handles = _hooks(self.blocks, self.final_norm)
        for block in self.blocks:
            for param in block.parameters():
                param.grad = None
        try:
            loss, _ = self.policy.forward(batch, noise=noise, time=time)
            for hidden in states.values():
                if hidden.requires_grad:
                    hidden.retain_grad()
            loss.backward()
            fisher, taylor, gate = [], [], []
            for i, block in enumerate(self.blocks):
                f_acc = t_acc = 0.0
                for param in block.parameters():
                    f, t = _chunk_fisher_taylor(param)
                    f_acc += f
                    t_acc += t
                fisher.append(f_acc)
                taylor.append(t_acc)
                h_in, h_out = states.get(i), states.get(i + 1)
                if h_in is None or h_out is None or h_out.grad is None:
                    gate.append(0.0)
                else:
                    inner = float((h_out.grad.float() * (h_out.float() - h_in.float())).sum().item())
                    gate.append(abs(inner))
        finally:
            for handle in handles:
                handle.remove()
            for block in self.blocks:
                for param in block.parameters():
                    param.grad = None
            del loss
            torch.cuda.empty_cache()
        return fisher, taylor, gate


class MolmoStack:
    def __init__(self, device: torch.device):
        sys.path.insert(0, str(HERE.parents[0] / "molmoact2_libero"))
        from load_policy import load_policy, vlm_blocks
        from score import _scoring_mode, _set_vlm_grad

        self.device = device
        self.policy = load_policy(str(device))
        self.blocks, self.final_norm = vlm_blocks(self.policy)
        self.n_layers = len(self.blocks)
        allowed = {id(param) for block in self.blocks for param in block.parameters()}
        for param in self.policy.parameters():
            param.requires_grad_(id(param) in allowed)
        _set_vlm_grad(self.blocks, True)
        _scoring_mode(self.policy, self.blocks, checkpoint=True)
        lm_head = getattr(getattr(self.policy, "model", None), "lm_head", None)
        if lm_head is not None:
            lm_head.to("cpu")
            for param in lm_head.parameters():
                param.requires_grad_(False)
        if hasattr(self.policy, "gradient_checkpointing_enable"):
            self.policy.gradient_checkpointing_enable()
        self.calib = torch.load(CALIB_DIR / "calib_molmoact2.pt", map_location="cpu", weights_only=False)["samples"]

    def score_one(self, sample) -> tuple[list[float], list[float], list[float]]:
        from score import _hooks, _mean_flow

        packed = {
            "batch": sample["batch"],
            "noise": sample["noise"],
            "timesteps": sample["timesteps"],
        }
        states, handles = _hooks(self.blocks, self.final_norm)
        for block in self.blocks:
            for param in block.parameters():
                param.grad = None
        try:
            loss = _mean_flow(self.policy, [packed], self.device)
            for hidden in states.values():
                if hidden.requires_grad:
                    hidden.retain_grad()
            loss.backward()
            fisher, taylor, gate = [], [], []
            for i, block in enumerate(self.blocks):
                f_acc = t_acc = 0.0
                for param in block.parameters():
                    f, t = _chunk_fisher_taylor(param)
                    f_acc += f
                    t_acc += t
                fisher.append(f_acc)
                taylor.append(t_acc)
                h_in, h_out = states.get(i), states.get(i + 1)
                if h_in is None or h_out is None or h_out.grad is None:
                    gate.append(0.0)
                else:
                    inner = float((h_out.grad.float() * (h_out.float() - h_in.float())).sum().item())
                    gate.append(abs(inner))
        finally:
            for handle in handles:
                handle.remove()
            for block in self.blocks:
                for param in block.parameters():
                    param.grad = None
            del loss
            torch.cuda.empty_cache()
        return fisher, taylor, gate


def run_stack(stack: str, device: torch.device) -> None:
    if _done(stack):
        print(f"[skip] {stack} scores exist", flush=True)
        return
    spec = STAGE1[stack]
    if stack == "pi05":
        worker = Pi05Stack(device)
    elif stack == "smolvla":
        worker = SmolStack(device)
    else:
        worker = MolmoStack(device)
    if worker.n_layers != spec["n_layers"]:
        raise SystemExit(f"{stack} layers {worker.n_layers} != {spec['n_layers']}")
    rows_f, rows_t, rows_g = [], [], []
    for sample in tqdm(worker.calib, desc=f"{stack} per-frame"):
        f, t, g = worker.score_one(sample)
        rows_f.append(f)
        rows_t.append(t)
        rows_g.append(g)
    _save_method(stack, "fisher", rows_f, worker.n_layers)
    _save_method(stack, "taylor", rows_t, worker.n_layers)
    _save_method(stack, "gateprobe", rows_g, worker.n_layers)


def main() -> None:
    global CALIB_DIR, SCORE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", choices=STACKS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--calib-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--score-dir", type=Path, default=None)
    args = parser.parse_args()
    CALIB_DIR = args.calib_dir
    SCORE_DIR = args.score_dir or (CALIB_DIR / "scores")
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    run_stack(args.stack, device)


if __name__ == "__main__":
    main()

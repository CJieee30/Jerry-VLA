#!/usr/bin/env python3
"""Molmo original Taylor on selected frame groups (budget=80, four strategies)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "molmoact2_libero"))

import torch

from dev_nfe_rules_cpu_v2 import g_only_group_k, random_group_k, select_per_task_k
from prospective_molmo_layer_common import (
    CALIB_PATH,
    N_LAYERS,
    PROTOCOL_ID,
    TAYLOR_BUDGET,
    TAYLOR_GROUP_DIR,
    TAYLOR_STOCHASTIC_SEED,
    load_calibration_pool,
    metas_for_selection,
    taylor_group_path,
)


def _pack_samples(blob_samples: list, indices: list[int]) -> list[dict]:
    out = []
    for i in indices:
        s = blob_samples[i]
        out.append(
            {
                "batch": s["batch"],
                "noise": s["noise"],
                "timesteps": s["timesteps"],
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    log_path = TAYLOR_GROUP_DIR.parent / "taylor_group_run.log"
    TAYLOR_GROUP_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ("random", TAYLOR_STOCHASTIC_SEED),
        ("g_only", TAYLOR_STOCHASTIC_SEED),
        ("alpha", None),
        ("care_event", None),
    ]

    try:
        if not CALIB_PATH.is_file():
            raise FileNotFoundError(CALIB_PATH)

        all_exist = all(taylor_group_path(s, seed).is_file() for s, seed in scenarios)
        if all_exist and not args.force:
            print("[skip] taylor group files exist", flush=True)
            return 0

        from load_policy import load_policy, vlm_blocks
        from molmo_score_runtime import mem_snapshot, prepare_policy_for_scoring
        from score import measure_grad

        blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
        pool = load_calibration_pool()
        wrapped = metas_for_selection(pool)
        device = torch.device(args.device)
        policy = load_policy(str(device))
        blocks, _ = vlm_blocks(policy)
        if len(blocks) != N_LAYERS:
            raise RuntimeError("layer count")
        ckpt = prepare_policy_for_scoring(policy, blocks)

        with log_path.open("a") as logf:
            logf.write(f"start taylor groups device={device}\n")
            logf.write(json.dumps({"checkpoint": ckpt, "mem": mem_snapshot()}) + "\n")

        for strategy, seed in scenarios:
            dest = taylor_group_path(strategy, seed)
            if dest.is_file() and not args.force:
                continue
            if strategy == "random":
                sel = sorted(random_group_k(wrapped, seed, 2))
            elif strategy == "g_only":
                sel = sorted(g_only_group_k(wrapped, seed, 2))
            elif strategy == "alpha":
                sel = sorted(select_per_task_k(wrapped, "alpha", 2))
            else:
                sel = sorted(select_per_task_k(wrapped, "alpha_event", 2))
            if len(sel) != TAYLOR_BUDGET:
                raise RuntimeError(f"{strategy} selected {len(sel)} != {TAYLOR_BUDGET}")
            samples = _pack_samples(blob["samples"], sel)
            # measure_grad expects list with batch/noise/timesteps on CPU; moves in _mean_flow
            scores = measure_grad(policy, blocks, samples, device, "taylor")
            torch.save(
                {
                    "protocol": PROTOCOL_ID,
                    "method": "taylor",
                    "formula": "sum_w |sum_i (g_i * theta_i)| / N over selected frames",
                    "not": "Taylor_absmean (per-frame |g*theta| mean)",
                    "strategy": strategy,
                    "seed": seed,
                    "budget_frames": TAYLOR_BUDGET,
                    "pool_indices": sel,
                    "layer_scores": scores,
                    "n_selected": len(sel),
                },
                dest,
            )
            print(f"saved {dest} drop preview", flush=True)

        with log_path.open("a") as logf:
            logf.write("done\n")
        return 0
    except Exception:
        with log_path.open("a") as logf:
            logf.write("FAILED\n")
            logf.write(traceback.format_exc())
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

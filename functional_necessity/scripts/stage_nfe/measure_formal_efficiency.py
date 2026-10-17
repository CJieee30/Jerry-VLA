#!/usr/bin/env python3
"""Formal wall-clock efficiency: 32 fixed obs, 4 configs, 3 rounds, sync GPU.

Primary metric: synchronous wall time for policy.predict_action_chunk (batch=1),
including vision/VLM prefix and action expert; excludes simulator/video I/O.
CUDA events recorded as auxiliary only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch

FN = Path(__file__).resolve().parents[2]
OUT_DIR = FN / "results" / "stage_nfe"
PI05_DROP = [1, 13, 14, 15, 16, 17]
MOLMO_DROP = [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

CONFIGS = (
    ("full_nfe10", 10, []),
    ("pruned_nfe10", 10, "drop"),
    ("full_nfe1", 1, []),
    ("pruned_nfe1", 1, "drop"),
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=30)
    except Exception as e:
        return f"error:{e}"


def capture_environment(gpu_index: int) -> dict:
    return {
        "utc": _utc(),
        "hostname": platform.node(),
        "loadavg": _run(["cat", "/proc/loadavg"]).strip(),
        "mem_free_kb": _run(["grep", "MemAvailable", "/proc/meminfo"]).strip(),
        "nvidia_smi": _run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,clocks.sm,power.draw",
                "--format=csv,noheader",
            ]
        ).strip(),
        "cpu_count": os.cpu_count(),
    }


def _p95(values: list[float]) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    i = min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))
    return float(xs[i])


def _summarize_ms(samples_ms: list[float]) -> dict:
    return {
        "n": len(samples_ms),
        "median_ms": float(statistics.median(samples_ms)),
        "mean_ms": float(statistics.mean(samples_ms)),
        "p95_ms": _p95(samples_ms),
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
        "stdev_ms": float(statistics.pstdev(samples_ms)) if len(samples_ms) > 1 else 0.0,
    }


def _wall_chunk_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _cuda_event_ms(fn, repeats: int = 1) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


def load_pi05_batches(device: torch.device, n_obs: int) -> list[dict]:
    sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
    from common import DEFAULT_CALIB_FILE, load_calib_data, move_to_device, slice_batch

    calib = load_calib_data(DEFAULT_CALIB_FILE)
    batches = []
    for i in range(min(n_obs, len(calib))):
        batches.append(move_to_device(slice_batch(calib[i], 0, 1), device))
    return batches


def load_molmo_batches(device: torch.device, n_obs: int) -> list[dict]:
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    from load_policy import CALIB_PATH

    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    samples = blob["samples"][:n_obs]
    out = []
    for s in samples:
        out.append({k: v.to(device) if torch.is_tensor(v) else v for k, v in s["batch"].items()})
    return out


def setup_pi05(device: torch.device):
    sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
    from common import DEFAULT_CHECKPOINT, load_policy
    from layer_skip import skip_llm_layers
    from measure_efficiency import count_params

    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.n_action_steps = 10
    policy.config.compile_model = False
    policy.eval()
    gen = torch.Generator(device=device)

    def run_chunk(batch, steps: int, drop: list[int]):
        ctx = skip_llm_layers(policy, drop) if drop else nullcontext()
        with torch.no_grad(), ctx:
            policy.reset()
            gen.manual_seed(1000)
            noise = torch.randn(
                1,
                int(policy.model.config.chunk_size),
                int(policy.model.config.max_action_dim),
                generator=gen,
                device=device,
            )
            policy.predict_action_chunk(batch, noise=noise, num_steps=steps)

    params_full = count_params(policy, [])
    params_drop = count_params(policy, PI05_DROP)
    return policy, run_chunk, params_full, params_drop, DEFAULT_CHECKPOINT


def setup_molmo(device: torch.device):
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    from layer_skip import skip_vlm_layers
    from load_policy import LEROBOT_CKPT, load_policy
    from measure_efficiency import count_params

    policy = load_policy(str(device))
    policy.eval()
    policy.config.compile_model = False
    policy.config.enable_inference_cuda_graph = False
    policy.config.gradient_checkpointing = False
    policy.config.n_action_steps = 10
    backbone = policy._backbone()
    backbone.gradient_checkpointing = False
    if getattr(backbone, "action_cuda_graph_manager", None) is not None:
        backbone.action_cuda_graph_manager = None

    def run_chunk(batch, steps: int, drop: list[int]):
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with torch.no_grad(), ctx, policy._autocast_context():
            policy.reset()
            policy.predict_action_chunk(batch, num_steps=steps)

    params_full = count_params(policy, [])
    params_drop = count_params(policy, MOLMO_DROP)
    return policy, run_chunk, params_full, params_drop, str(LEROBOT_CKPT)


def measure_config(
    name: str,
    steps: int,
    drop: list[int],
    run_chunk,
    batches: list[dict],
    warmup: int,
    rounds: int,
) -> dict:
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        run_chunk(batches[0], steps, drop)

    per_round: list[dict] = []
    all_ms: list[float] = []
    aux_cuda: list[float] = []
    for r in range(rounds):
        round_ms = []
        for bi, batch in enumerate(batches):
            ms = _wall_chunk_ms(lambda: run_chunk(batch, steps, drop))
            round_ms.append(ms)
            all_ms.append(ms)
            if bi == 0:
                aux_cuda.append(_cuda_event_ms(lambda: run_chunk(batch, steps, drop)))
        per_round.append({"round": r, "samples_ms": round_ms, **(_summarize_ms(round_ms))})

    peak_bytes = int(torch.cuda.max_memory_allocated())
    return {
        "config": name,
        "num_inference_steps": steps,
        "drop_layers": drop,
        "timing_boundary": (
            "torch.cuda.synchronize(); perf_counter around predict_action_chunk; "
            "batch already on GPU (no disk/video); output left on GPU"
        ),
        "warmup_chunks": warmup,
        "rounds": rounds,
        "n_obs_per_round": len(batches),
        "per_round": per_round,
        "all_samples": _summarize_ms(all_ms),
        "cuda_event_aux_ms_first_obs": _summarize_ms(aux_cuda),
        "peak_allocated_bytes": peak_bytes,
        "peak_allocated_gib": peak_bytes / (1024**3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", choices=("pi05", "molmoact2"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-obs", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--short-smoke", action="store_true", help="n_obs=4 rounds=1 warmup=3")
    args = parser.parse_args()
    if args.short_smoke:
        args.n_obs = 4
        args.rounds = 1
        args.warmup = 3

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    gpu_index = int(str(device).split(":")[-1]) if device.type == "cuda" else 0

    env_start = capture_environment(gpu_index)
    if args.stack == "pi05":
        policy, run_chunk, p_full, p_drop, ckpt = setup_pi05(device)
        drop_list = PI05_DROP
        skip_note = "dtr_appendix_b_prefix_kv (runtime skip_llm_layers)"
    else:
        policy, run_chunk, p_full, p_drop, ckpt = setup_molmo(device)
        drop_list = MOLMO_DROP
        skip_note = "keep_kv_identity_residual (runtime skip_vlm_layers)"

    batches = (
        load_pi05_batches(device, args.n_obs)
        if args.stack == "pi05"
        else load_molmo_batches(device, args.n_obs)
    )

    results = []
    for name, steps, drop_tag in CONFIGS:
        drop = drop_list if drop_tag == "drop" else []
        results.append(measure_config(name, steps, drop, run_chunk, batches, args.warmup, args.rounds))

    anchor_first = results[0]["all_samples"]["median_ms"]
    anchor_last = measure_config(
        "full_nfe10_recheck",
        10,
        [],
        run_chunk,
        batches,
        args.warmup,
        1,
    )
    anchor_recheck = anchor_last["all_samples"]["median_ms"]
    drift_pct = 100.0 * abs(anchor_recheck - anchor_first) / anchor_first if anchor_first else 0.0

    table = []
    for row in results:
        med = row["all_samples"]["median_ms"]
        table.append(
            {
                "config": row["config"],
                "median_ms": med,
                "p95_ms": row["all_samples"]["p95_ms"],
                "speedup_vs_full_nfe10": anchor_first / med if med > 0 else None,
                "peak_gib": row["peak_allocated_gib"],
            }
        )

    payload = {
        "protocol": "formal_efficiency_v1",
        "utc_start": env_start["utc"],
        "utc_end": _utc(),
        "stack": args.stack,
        "checkpoint": ckpt,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "compile_model": False,
        "enable_inference_cuda_graph": False,
        "batch_size": 1,
        "n_action_steps": 10,
        "skip_operator": skip_note,
        "pruned_layers": drop_list,
        "params": {
            "full_stored_m": p_full["stored_params"] / 1e6,
            "full_active_m": p_full["active_params"] / 1e6,
            "pruned_active_m": p_drop["active_params"] / 1e6,
        },
        "closed_loop_sr_note": (
            "SR cited from eval_*.md historical runs; not remeasured in this script"
        ),
        "n_obs": len(batches),
        "warmup": args.warmup,
        "rounds": args.rounds,
        "environment_start": env_start,
        "environment_end": capture_environment(gpu_index),
        "configs": results,
        "summary_table": table,
        "stability": {
            "full_nfe10_median_first_ms": anchor_first,
            "full_nfe10_median_recheck_ms": anchor_recheck,
            "recheck_round_median_drift_pct": drift_pct,
            "drift_target_practical_pct": 3.0,
            "drift_investigate_pct": 5.0,
        },
        "full_nfe10_recheck": anchor_last,
    }

    out = args.out or OUT_DIR / f"formal_efficiency_{args.stack}_20261017.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"wrote": str(out), "table": table, "drift_pct": drift_pct}, indent=2))


if __name__ == "__main__":
    main()

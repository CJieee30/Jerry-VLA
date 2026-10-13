#!/usr/bin/env python3
"""Full action-chunk latency: Drop-6 NFE=1 baseline vs CROP role / shared_mix.

Includes scout, token selection, the second prefix, and the action expert.
Formal mode refuses to start unless the target GPU is fully idle.
Diagnostic mode may share a card only when --gpu is set explicitly.
compile_model stays false. Do not use CROP_ALLOW_OCCUPIED as a bypass.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_CALIB_FILE,
    DEFAULT_CHECKPOINT,
    FN_ROOT,
    init_logging,
    load_calib_data,
    load_policy,
    move_to_device,
    slice_batch,
)
from crop_vla import (  # noqa: E402
    BUDGET_PRESETS,
    FISHER_DROP6,
    SCOUT_LAYER,
    bind_cuda_device,
    crop_visual_tokens,
    gpu_occupancy,
    require_idle_gpu,
    resolve_budgets,
    stats_dict,
)
from layer_skip import skip_llm_layers  # noqa: E402

OUT = FN_ROOT / "results" / "stage_nfe" / "crop_vla"
DROP = FISHER_DROP6
NFE = 1
GROUPS = ("baseline", "role", "shared_mix")
DIAGNOSTIC_NOTE = "共享GPU诊断，仅供参考，待独占复核"
MIN_FREE_MIB_DIAGNOSTIC = 12000
REPO_ROOT = FN_ROOT.parent


def _frames(calib, n: int) -> list[dict]:
    frames = []
    for batch in calib:
        bsz = int(batch["observation.images.image"].shape[0])
        for i in range(bsz):
            frames.append(slice_batch(batch, i, i + 1))
            if len(frames) >= n:
                return frames
    return frames


def _percentile(xs: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))


def _summarize(vals: list[float]) -> dict:
    return {
        "median_ms": _percentile(vals, 50),
        "p95_ms": _percentile(vals, 95),
        "mean_ms": float(np.mean(vals)),
        "n": len(vals),
        "raw_ms": vals,
    }


def _time_chunk(policy, batch, noise, repeats: int, device: torch.device) -> list[dict[str, float | int]]:
    """Synchronized wall-clock and CUDA-event times for a full chunk. profile=False path."""
    rows = []
    stream = torch.cuda.current_stream(device)
    for rep in range(repeats):
        policy.reset()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall0 = time.perf_counter()
        start.record(stream)
        policy.predict_action_chunk(batch, noise=noise, num_steps=NFE)
        end.record(stream)
        torch.cuda.synchronize(device)
        rows.append(
            {
                "repeat": rep,
                "cuda_ms": float(start.elapsed_time(end)),
                "wall_ms": (time.perf_counter() - wall0) * 1000.0,
            }
        )
    return rows


def _git_version() -> dict:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "root": str(REPO_ROOT),
        "head": _run(["git", "rev-parse", "HEAD"]),
        "head_short": _run(["git", "rev-parse", "--short", "HEAD"]),
        "status": _run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "functional_necessity/scripts/pi05_libero/crop_vla.py",
                "functional_necessity/scripts/pi05_libero/measure_crop_efficiency.py",
            ]
        ),
    }


def gpu_card_snapshot(gpu: int) -> dict:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=name,driver_version,memory.used,memory.total,memory.free,"
            "utilization.gpu,utilization.memory,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    parts = [p.strip() for p in raw.split(",")]
    return {
        "index": gpu,
        "name": parts[0],
        "driver": parts[1],
        "memory_used_mib": float(parts[2]),
        "memory_total_mib": float(parts[3]),
        "memory_free_mib": float(parts[4]),
        "util_gpu_pct": float(parts[5]),
        "util_mem_pct": float(parts[6]),
        "temp_c": float(parts[7]),
        "power_w": float(parts[8]),
        "compute_apps": gpu_occupancy(gpu),
    }


def snapshot_all(gpus: tuple[int, ...] = (0, 1, 2, 3)) -> list[dict]:
    return [gpu_card_snapshot(g) for g in gpus]


def group_order(frame_idx: int) -> tuple[str, ...]:
    n = len(GROUPS)
    return tuple(GROUPS[(frame_idx + k) % n] for k in range(n))


@contextmanager
def group_forward(policy, name: str, *, profile: bool):
    if name == "baseline":
        yield
        return
    with crop_visual_tokens(
        policy,
        keep_all=False,
        drop_layers=DROP,
        scout_layer=SCOUT_LAYER,
        profile=profile,
        budgets=name,
        budget_name=name,
    ):
        yield


def capture_layout(policy, batch, noise, name: str) -> dict:
    with group_forward(policy, name, profile=False):
        policy.reset()
        policy.predict_action_chunk(batch, noise=noise, num_steps=NFE)
        stats = getattr(policy.model, "_crop_stats", None)
    if name == "baseline":
        return {
            "budget_name": "baseline",
            "budgets": None,
            "visual_prune": False,
            "expected_prefix": "official 3*256 visual + language; not shortened",
        }
    if stats is None:
        raise RuntimeError(f"{name} did not write _crop_stats")
    payload = stats_dict(stats)
    payload["budgets"] = resolve_budgets(name)
    payload["visual_prune"] = True
    return payload


def breakdown_pass(policy, frames, noises, name: str) -> dict:
    parts = []
    with group_forward(policy, name, profile=True):
        for batch, noise in zip(frames, noises, strict=True):
            policy.reset()
            policy.predict_action_chunk(batch, noise=noise, num_steps=NFE)
            stats = getattr(policy.model, "_crop_stats", None)
            if stats is not None:
                parts.append(stats_dict(stats))
    timed = [p["timing_ms"] for p in parts if p.get("timing_ms")]
    out: dict = {
        "note": "separate profile=True pass; not mixed into full-chunk median/P95",
        "timer_keys": (
            "embed=vision+embed_prefix; scout; select=token choice; "
            "prefix=gather/pack + official prefix; expert"
        ),
        "n_frames": len(timed),
    }
    if timed:
        keys = sorted({k for t in timed for k in t})
        out["median_ms"] = {k: _percentile([t[k] for t in timed if k in t], 50) for k in keys}
        out["p95_ms"] = {k: _percentile([t[k] for t in timed if k in t], 95) for k in keys}
        out["raw"] = timed
    return out


def compare_to_baseline(group: dict, baseline: dict, *, diagnostic: bool) -> dict:
    note = DIAGNOSTIC_NOTE if diagnostic else "exclusive-GPU formal measurement"
    out = {}
    for clock in ("cuda", "wall"):
        g = group[clock]
        b = baseline[clock]
        out[clock] = {
            "delta_median_ms": g["median_ms"] - b["median_ms"],
            "delta_p95_ms": g["p95_ms"] - b["p95_ms"],
            "delta_median_pct": 100.0 * (g["median_ms"] / b["median_ms"] - 1.0),
            "relative_to_baseline": g["median_ms"] / b["median_ms"],
            "speedup_vs_baseline": b["median_ms"] / g["median_ms"],
            "note": note,
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("formal", "diagnostic"), default="formal")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def resolve_gpu(args: argparse.Namespace) -> tuple[int, dict]:
    os.environ.pop("CROP_ALLOW_OCCUPIED", None)
    gate = {
        "mode": args.mode,
        "env_bypass_popped": True,
        "CROP_ALLOW_OCCUPIED": os.environ.get("CROP_ALLOW_OCCUPIED"),
    }
    if args.mode == "formal":
        gpu = require_idle_gpu(args.gpu)
        gate["exclusive"] = True
        gate["gpu"] = gpu
        return gpu, gate
    if args.gpu is None:
        raise SystemExit("diagnostic mode requires an explicit --gpu N; refusing to auto-pick a shared card")
    snap = gpu_card_snapshot(args.gpu)
    gate["exclusive"] = False
    gate["gpu"] = args.gpu
    gate["precheck"] = snap
    if snap["memory_free_mib"] < MIN_FREE_MIB_DIAGNOSTIC:
        raise SystemExit(
            f"GPU{args.gpu} free {snap['memory_free_mib']:.0f} MiB < {MIN_FREE_MIB_DIAGNOSTIC} MiB; "
            "not enough headroom for π0.5 (~10 GiB). Not killing other processes."
        )
    return args.gpu, gate


def main() -> None:
    init_logging()
    args = parse_args()
    gpu, gate = resolve_gpu(args)
    occupancy_before = snapshot_all()
    device = bind_cuda_device(gpu)
    print(
        f"mode={args.mode} bound physical GPU{gpu} -> {device} "
        f"(current={torch.cuda.current_device()})",
        flush=True,
    )
    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.n_action_steps = 10
    policy.config.num_inference_steps = NFE
    policy.config.compile_model = False
    policy.eval()
    calib = load_calib_data(DEFAULT_CALIB_FILE)
    frames = [move_to_device(f, device) for f in _frames(calib, args.n_frames)]
    noises = []
    for i in range(len(frames)):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(1000 + i)
        noises.append(
            torch.randn(
                1,
                int(policy.config.chunk_size),
                int(policy.config.max_action_dim),
                generator=gen,
            ).to(device)
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.out is not None:
        out_path = args.out
    elif args.mode == "diagnostic":
        out_path = OUT / f"efficiency_diagnostic_{stamp}.json"
    else:
        out_path = OUT / "efficiency.json"

    summary: dict = {
        "measurement_mode": args.mode,
        "disclaimer": DIAGNOSTIC_NOTE if args.mode == "diagnostic" else None,
        "not_for_paper_table": args.mode == "diagnostic",
        "gpu": gpu,
        "device": str(device),
        "physical_gpu": gpu,
        "gpu_name": occupancy_before[gpu]["name"] if gpu < len(occupancy_before) else None,
        "checkpoint": DEFAULT_CHECKPOINT,
        "drop": DROP,
        "nfe": NFE,
        "scout_layer": SCOUT_LAYER,
        "compile_model": False,
        "batch_size": 1,
        "n_action_steps": 10,
        "n_frames": len(frames),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "group_names": list(GROUPS),
        "budgets": {k: BUDGET_PRESETS[k] for k in ("role", "shared_mix")},
        "interleave": "per-frame rotation of group order; not one group then the next",
        "flops": "unmeasured",
        "note": (
            "full predict_action_chunk with profile=False (includes vision encode, scout, "
            "select, pack, official prefix, expert). Segment breakdown is a separate "
            "profile=True pass. Model load and crop/baseline switching are outside the timer."
        ),
        "gate": gate,
        "code_version": _git_version(),
        "occupancy_before": occupancy_before,
        "started_utc": stamp,
    }

    logging.getLogger().setLevel(logging.WARNING)
    with torch.no_grad(), skip_llm_layers(policy, DROP, keep_kv=True):
        layouts = {}
        for name in GROUPS:
            with group_forward(policy, name, profile=False):
                for _ in range(args.warmup):
                    policy.reset()
                    policy.predict_action_chunk(frames[0], noise=noises[0], num_steps=NFE)
            layouts[name] = capture_layout(policy, frames[0], noises[0], name)

        collected: dict[str, list[dict]] = {name: [] for name in GROUPS}
        for fi, (batch, noise) in enumerate(zip(frames, noises, strict=True)):
            for name in group_order(fi):
                with group_forward(policy, name, profile=False):
                    timed = _time_chunk(policy, batch, noise, args.repeats, device)
                for row in timed:
                    collected[name].append(
                        {
                            "frame": fi,
                            "repeat": int(row["repeat"]),
                            "order": list(group_order(fi)),
                            "cuda_ms": row["cuda_ms"],
                            "wall_ms": row["wall_ms"],
                        }
                    )

        groups_out = {}
        for name in GROUPS:
            samples = collected[name]
            groups_out[name] = {
                "layout": layouts[name],
                "cuda": _summarize([s["cuda_ms"] for s in samples]),
                "wall": _summarize([s["wall_ms"] for s in samples]),
                "samples": samples,
            }
        base = groups_out["baseline"]
        for name in GROUPS:
            if name == "baseline":
                groups_out[name]["vs_baseline"] = None
            else:
                groups_out[name]["vs_baseline"] = compare_to_baseline(
                    groups_out[name], base, diagnostic=args.mode == "diagnostic"
                )
        summary["groups"] = groups_out

        summary["breakdown_profile_true"] = {
            name: breakdown_pass(policy, frames, noises, name) for name in ("role", "shared_mix")
        }

    occupancy_after = snapshot_all()
    summary["occupancy_after"] = occupancy_after
    summary["finished_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")

    def _line(name: str) -> str:
        g = summary["groups"][name]
        return (
            f"{name}: cuda median={g['cuda']['median_ms']:.2f} p95={g['cuda']['p95_ms']:.2f} ms  "
            f"wall median={g['wall']['median_ms']:.2f} p95={g['wall']['p95_ms']:.2f} ms"
        )

    extra = ""
    if args.mode == "diagnostic":
        extra = f"  [{DIAGNOSTIC_NOTE}]"
    print(_line("baseline"), flush=True)
    print(_line("role") + extra, flush=True)
    print(_line("shared_mix") + extra, flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

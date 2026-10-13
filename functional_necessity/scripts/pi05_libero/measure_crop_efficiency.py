#!/usr/bin/env python3
"""Full action-chunk latency: Drop-6 baseline vs CROP / CROP-Lite / uniform-grid.

Full-chunk and segment timing both rotate groups inside each frame (and each
repeat for the full pass). Formal mode refuses unless the target GPU is idle.
Diagnostic mode may share a card only when --gpu is set explicitly.
compile_model stays false. Do not use CROP_ALLOW_OCCUPIED as a bypass.
"""

from __future__ import annotations

import argparse
import hashlib
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
    PROFILE_ALIGN_MAX_ABS,
    SCOUT_LAYER,
    align_profiled_baseline,
    bind_cuda_device,
    count_official_velocity,
    crop_visual_tokens,
    gpu_occupancy,
    profile_official_actions,
    require_idle_gpu,
    resolve_budgets,
    resolve_selector,
    stats_dict,
)
from layer_skip import skip_llm_layers  # noqa: E402
from prefix_trim import prefix_trim  # noqa: E402

OUT = FN_ROOT / "results" / "stage_nfe" / "crop_vla"
DROP = FISHER_DROP6
GROUPS_GREEDY = ("baseline", "role", "shared_mix")
GROUPS_LITE = ("baseline", "role-lite", "shared-lite")
GROUPS_GRID = ("baseline", "uniform-grid", "role-lite")
GROUPS_TRIM_GRID = ("raw", "trim-only", "trim-grid")
DIAGNOSTIC_NOTE = "共享 GPU，仅供诊断，不进入论文正式效率表"
MIN_FREE_MIB_DIAGNOSTIC = 12000
REPO_ROOT = FN_ROOT.parent
HASHED_FILES = (
    "functional_necessity/scripts/pi05_libero/crop_vla.py",
    "functional_necessity/scripts/pi05_libero/check_crop_vla.py",
    "functional_necessity/scripts/pi05_libero/measure_crop_efficiency.py",
    "functional_necessity/scripts/pi05_libero/eval_drop.py",
    "functional_necessity/scripts/pi05_libero/prefix_trim.py",
)


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


def _time_one_chunk(policy, batch, noise, device: torch.device, nfe: int) -> dict[str, float]:
    """One synchronized full-chunk sample. profile=False path."""
    stream = torch.cuda.current_stream(device)
    policy.reset()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall0 = time.perf_counter()
    start.record(stream)
    policy.predict_action_chunk(batch, noise=noise, num_steps=nfe)
    end.record(stream)
    torch.cuda.synchronize(device)
    return {
        "cuda_ms": float(start.elapsed_time(end)),
        "wall_ms": (time.perf_counter() - wall0) * 1000.0,
    }


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
        "status": _run(["git", "status", "--porcelain", "--", *HASHED_FILES]),
        "file_hashes": {
            rel: hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()
            for rel in HASHED_FILES
            if (REPO_ROOT / rel).is_file()
        },
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


def parse_group(name: str) -> dict:
    key = name.strip().lower().replace("_", "-")
    if key in ("raw", "original"):
        return {"kind": "baseline", "budget": None, "selector": None, "prefix_trim": False}
    if key == "baseline":
        return {"kind": "baseline", "budget": None, "selector": None, "prefix_trim": False}
    if key in ("trim-only", "trim"):
        return {"kind": "trim", "budget": None, "selector": None, "prefix_trim": True}
    if key in ("trim-grid", "trim-uniform-grid"):
        return {"kind": "crop", "budget": None, "selector": "grid", "prefix_trim": True}
    if key in ("uniform-grid", "grid", "uniform"):
        return {"kind": "crop", "budget": None, "selector": "grid", "prefix_trim": False}
    selector = "lite" if key.endswith("-lite") else "greedy"
    core = key.replace("-lite", "")
    if core in ("shared", "shared-mix"):
        budget = "shared_mix"
    elif core in ("role", "crop", ""):
        budget = "role"
    else:
        budget = core
    return {"kind": "crop", "budget": budget, "selector": selector, "prefix_trim": False}


def group_order(frame_idx: int, groups: tuple[str, ...]) -> tuple[str, ...]:
    n = len(groups)
    return tuple(groups[(frame_idx + k) % n] for k in range(n))


@contextmanager
def group_forward(policy, name: str, *, profile: bool):
    spec = parse_group(name)
    if spec["kind"] == "baseline":
        if profile:
            with profile_official_actions(policy):
                yield
        else:
            yield
        return
    if spec["kind"] == "trim":
        with prefix_trim(policy):
            if profile:
                with profile_official_actions(policy):
                    yield
            else:
                yield
        return
    with crop_visual_tokens(
        policy,
        keep_all=False,
        drop_layers=DROP,
        scout_layer=SCOUT_LAYER,
        profile=profile,
        budgets=spec["budget"],
        budget_name=spec["budget"] or spec["selector"],
        selector=spec["selector"],
        prefix_trim=bool(spec.get("prefix_trim")),
    ):
        yield


def capture_layout(policy, batch, noise, name: str, nfe: int) -> dict:
    spec = parse_group(name)
    if spec["kind"] == "baseline":
        n_vel = count_official_velocity(policy, batch, noise, nfe)
        return {
            "budget_name": "baseline",
            "selector": None,
            "budgets": None,
            "prefix_trim": False,
            "visual_prune": False,
            "velocity_calls": n_vel,
            "num_steps": nfe,
            "expected_prefix": "official 3*256 visual + language; not shortened",
        }
    if spec["kind"] == "trim":
        with prefix_trim(policy):
            n_vel = count_official_velocity(policy, batch, noise, nfe)
            info = getattr(policy.model, "_trim_stats", None)
        payload = {
            "budget_name": "trim-only",
            "selector": None,
            "budgets": None,
            "prefix_trim": True,
            "visual_prune": False,
            "velocity_calls": n_vel,
            "num_steps": nfe,
            "expected_prefix": "empty cameras and language pad removed; live visual kept",
        }
        if info is not None:
            payload.update(info.to_dict())
        return payload
    with group_forward(policy, name, profile=False):
        policy.reset()
        policy.predict_action_chunk(batch, noise=noise, num_steps=nfe)
        stats = getattr(policy.model, "_crop_stats", None)
    if stats is None:
        raise RuntimeError(f"{name} did not write _crop_stats")
    payload = stats_dict(stats)
    payload["prefix_trim"] = bool(spec.get("prefix_trim"))
    if spec["selector"] == "grid":
        payload["budgets"] = None
        payload["keep_rule"] = "(row+col)%2==0; 128 unique tokens per live camera; no scout"
    else:
        payload["budgets"] = resolve_budgets(spec["budget"])
    payload["visual_prune"] = True
    return payload


def _timer_keys_note(name: str) -> str:
    spec = parse_group(name)
    if spec["kind"] == "baseline":
        return "embed; prefix=official full prefix; expert=all velocity steps"
    if spec["kind"] == "trim":
        return "embed=trimmed cameras+language; prefix=official trimmed prefix; expert=all velocity steps"
    if spec["selector"] == "grid":
        return "embed; prefix=fixed-index pack + official compressed prefix; expert=all velocity steps"
    return (
        "embed; scout; select=token choice; prefix=gather/pack + official compressed prefix; "
        "expert=all velocity steps"
    )


def _summarize_breakdown(name: str, parts: list[dict]) -> dict:
    timed = [p["timing_ms"] for p in parts if p.get("timing_ms")]
    out: dict = {
        "note": (
            "separate interleaved profile=True pass; analyze with mean_ms; do not add segment medians; "
            "do not subtract from a different full-chunk pass"
        ),
        "analysis_stat": "mean",
        "timer_keys": _timer_keys_note(name),
        "n_frames": len(timed),
        "velocity_calls": [p.get("velocity_calls") for p in parts],
        "num_steps": [p.get("num_steps") for p in parts],
        "scout_layer_calls": [p.get("scout_layer_calls") for p in parts][:1],
        "orders": [p.get("order") for p in parts],
    }
    if timed:
        keys = sorted({k for t in timed for k in t})
        out["mean_ms"] = {k: float(np.mean([t[k] for t in timed if k in t])) for k in keys}
        out["median_ms"] = {k: _percentile([t[k] for t in timed if k in t], 50) for k in keys}
        out["p95_ms"] = {k: _percentile([t[k] for t in timed if k in t], 95) for k in keys}
        out["raw"] = timed
    return out


def breakdown_interleaved(policy, frames, noises, groups: tuple[str, ...], nfe: int) -> dict:
    """One profiled chunk per group per frame; rotate group order each frame."""
    collected: dict[str, list[dict]] = {name: [] for name in groups}
    for fi, (batch, noise) in enumerate(zip(frames, noises, strict=True)):
        order = group_order(fi, groups)
        for name in order:
            with group_forward(policy, name, profile=True):
                policy.reset()
                policy.predict_action_chunk(batch, noise=noise, num_steps=nfe)
                stats = getattr(policy.model, "_crop_stats", None)
                if stats is None:
                    raise RuntimeError(f"{name} profile=True did not write _crop_stats")
                payload = stats_dict(stats)
                payload["frame"] = fi
                payload["order"] = list(order)
                collected[name].append(payload)
    return {name: _summarize_breakdown(name, collected[name]) for name in groups}


def compare_to_baseline(group: dict, baseline: dict, *, diagnostic: bool) -> dict:
    note = DIAGNOSTIC_NOTE if diagnostic else "exclusive-GPU formal measurement"
    out = {}
    for clock in ("cuda", "wall"):
        g = group[clock]
        b = baseline[clock]
        out[clock] = {
            "delta_median_ms": g["median_ms"] - b["median_ms"],
            "delta_mean_ms": g["mean_ms"] - b["mean_ms"],
            "delta_p95_ms": g["p95_ms"] - b["p95_ms"],
            "delta_median_pct": 100.0 * (g["median_ms"] / b["median_ms"] - 1.0),
            "delta_mean_pct": 100.0 * (g["mean_ms"] / b["mean_ms"] - 1.0),
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
    parser.add_argument("--nfe", type=int, default=1)
    parser.add_argument(
        "--selector",
        default="greedy",
        help="greedy, lite, or grid. Default groups follow the selector unless --groups is set.",
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="comma-separated group names. trim default: raw,trim-only,trim-grid",
    )
    parser.add_argument(
        "--prefix-trim",
        action="store_true",
        help="Default groups raw,trim-only,trim-grid. Requires explicit --nfe.",
    )
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
    nfe = int(args.nfe)
    selector = resolve_selector(args.selector)
    if args.prefix_trim and not any(a == "--nfe" or a.startswith("--nfe=") for a in sys.argv[1:]):
        raise SystemExit("--prefix-trim requires explicit --nfe")
    if args.groups:
        groups = tuple(part.strip() for part in args.groups.split(",") if part.strip())
    elif args.prefix_trim:
        groups = GROUPS_TRIM_GRID
        selector = "grid"
    else:
        if selector == "lite":
            groups = GROUPS_LITE
        elif selector == "grid":
            groups = GROUPS_GRID
        else:
            groups = GROUPS_GREEDY
    policy.config.n_action_steps = 10
    policy.config.num_inference_steps = nfe
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
        if any(parse_group(g).get("prefix_trim") for g in groups):
            tag = "efficiency_diagnostic_trim_grid"
        elif selector == "grid" or any(parse_group(g)["selector"] == "grid" for g in groups):
            tag = "efficiency_diagnostic_grid"
        elif selector == "lite":
            tag = "efficiency_diagnostic_lite"
        else:
            tag = "efficiency_diagnostic"
        out_path = OUT / f"{tag}_nfe{nfe}_{stamp}.json"
    elif any(parse_group(g).get("prefix_trim") for g in groups):
        out_path = OUT / f"efficiency_trim_grid_nfe{nfe}.json"
    elif selector == "grid":
        out_path = OUT / f"efficiency_grid_nfe{nfe}.json"
    elif selector == "lite":
        out_path = OUT / f"efficiency_lite_nfe{nfe}.json"
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
        "nfe": nfe,
        "prefix_trim": bool(args.prefix_trim) or any(parse_group(g).get("prefix_trim") for g in groups),
        "selector": selector,
        "scout_layer": SCOUT_LAYER,
        "compile_model": False,
        "batch_size": 1,
        "n_action_steps": 10,
        "n_frames": len(frames),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "group_names": list(groups),
        "budgets": {k: BUDGET_PRESETS[k] for k in ("role", "shared_mix")},
        "interleave": {
            "full_chunk": (
                "per (frame, repeat): rotate the three groups; not all repeats of one group, "
                "then the next group"
            ),
            "breakdown": (
                "per frame: one profiled chunk per group in rotated order; not all frames of "
                "one group, then the next group"
            ),
        },
        "flops": "unmeasured",
        "prefix_trim_note": "empty cameras dropped before vision encode; language right-trimmed by attention mask",
        "note": (
            "full predict_action_chunk with profile=False (vision encode, select/pack, official "
            "prefix, expert; role-lite also includes scout). Segment breakdown is a separate "
            "interleaved profile=True pass. Baseline segments wrap original embed_prefix / "
            "denoise_step / sample_actions. Model load and group switching are outside the timer."
        ),
        "gate": gate,
        "code_version": _git_version(),
        "occupancy_before": occupancy_before,
        "started_utc": stamp,
    }

    logging.getLogger().setLevel(logging.WARNING)
    with torch.no_grad(), skip_llm_layers(policy, DROP, keep_kv=True):
        for w in range(args.warmup):
            for name in group_order(w, groups):
                with group_forward(policy, name, profile=False):
                    policy.reset()
                    policy.predict_action_chunk(frames[0], noise=noises[0], num_steps=nfe)

        alignment = align_profiled_baseline(policy, frames[0], noises[0], nfe)
        summary["baseline_profile"] = alignment
        if not alignment["aligned"] or not alignment["finite"]:
            raise RuntimeError(
                f"profiled baseline drifted from official path "
                f"(max_abs={alignment['max_abs']} > {PROFILE_ALIGN_MAX_ABS})"
            )
        if int(alignment.get("velocity_calls") or 0) != nfe:
            raise RuntimeError(
                f"profiled baseline velocity_calls={alignment.get('velocity_calls')} != num_steps={nfe}"
            )
        print(
            f"baseline profile wrap aligned max_abs={alignment['max_abs']:.4g} "
            f"dtype={alignment['official_dtype']} attn={alignment['attn_implementation']} "
            f"nfe={alignment['velocity_calls']}",
            flush=True,
        )

        layouts = {}
        for name in groups:
            layouts[name] = capture_layout(policy, frames[0], noises[0], name, nfe)
            vel = layouts[name].get("velocity_calls")
            if vel is not None and int(vel) != nfe:
                raise RuntimeError(f"{name} velocity_calls={vel} != num_steps={nfe}")

        collected: dict[str, list[dict]] = {name: [] for name in groups}
        for fi, (batch, noise) in enumerate(zip(frames, noises, strict=True)):
            for rep in range(args.repeats):
                order = group_order(fi * args.repeats + rep, groups)
                for name in order:
                    with group_forward(policy, name, profile=False):
                        timed = _time_one_chunk(policy, batch, noise, device, nfe)
                    collected[name].append(
                        {
                            "frame": fi,
                            "repeat": rep,
                            "order": list(order),
                            "cuda_ms": timed["cuda_ms"],
                            "wall_ms": timed["wall_ms"],
                        }
                    )

        groups_out = {}
        for name in groups:
            samples = collected[name]
            groups_out[name] = {
                "layout": layouts[name],
                "cuda": _summarize([s["cuda_ms"] for s in samples]),
                "wall": _summarize([s["wall_ms"] for s in samples]),
                "samples": samples,
            }
        anchor = "raw" if "raw" in groups_out else "baseline"
        base = groups_out[anchor]
        for name in groups:
            if name == anchor:
                groups_out[name]["vs_baseline"] = None
            else:
                groups_out[name]["vs_baseline"] = compare_to_baseline(
                    groups_out[name], base, diagnostic=args.mode == "diagnostic"
                )
        summary["groups"] = groups_out
        if {"raw", "trim-only", "trim-grid"} <= set(groups_out):
            diag = args.mode == "diagnostic"
            summary["deltas"] = {
                "A_raw_to_B_trim_only": compare_to_baseline(
                    groups_out["trim-only"], groups_out["raw"], diagnostic=diag
                ),
                "B_trim_only_to_C_trim_grid": compare_to_baseline(
                    groups_out["trim-grid"], groups_out["trim-only"], diagnostic=diag
                ),
                "A_raw_to_C_trim_grid": compare_to_baseline(
                    groups_out["trim-grid"], groups_out["raw"], diagnostic=diag
                ),
            }
        summary["breakdown_profile_true"] = breakdown_interleaved(
            policy, frames, noises, groups, nfe
        )

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
    for name in groups:
        spec = parse_group(name)
        suffix = extra if spec["kind"] in ("crop", "trim") else ""
        print(_line(name) + suffix, flush=True)
    if "deltas" in summary:
        for key, val in summary["deltas"].items():
            d = val["cuda"]
            print(
                f"{key}: cuda median {d['delta_median_ms']:+.2f} ms ({d['delta_median_pct']:+.2f}%)"
                + extra,
                flush=True,
            )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

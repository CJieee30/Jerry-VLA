#!/usr/bin/env python3
"""Full action-chunk latency: Drop-6 NFE=1 baseline vs CROP.

Includes scout, token selection, the second prefix, and the action expert.
Refuses to start unless the target GPU is fully idle. Multi-frame, warmup,
median and P95. compile_model stays false.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    FISHER_DROP6,
    SCOUT_LAYER,
    bind_cuda_device,
    crop_visual_tokens,
    require_idle_gpu,
    stats_dict,
)
from layer_skip import skip_llm_layers  # noqa: E402

OUT = FN_ROOT / "results" / "stage_nfe" / "crop_vla"
DROP = FISHER_DROP6
NFE = 1


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


def _time_chunk(policy, batch, noise, repeats: int, device: torch.device) -> list[float]:
    times = []
    stream = torch.cuda.current_stream(device)
    for _ in range(repeats):
        policy.reset()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        policy.predict_action_chunk(batch, noise=noise, num_steps=NFE)
        end.record(stream)
        torch.cuda.synchronize(device)
        times.append(float(start.elapsed_time(end)))
    return times


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    gpu = require_idle_gpu(args.gpu)
    device = bind_cuda_device(gpu)
    print(f"bound physical GPU{gpu} -> {device} (current={torch.cuda.current_device()})", flush=True)
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

    summary = {
        "gpu": gpu,
        "drop": DROP,
        "nfe": NFE,
        "scout_layer": SCOUT_LAYER,
        "n_frames": len(frames),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "compile_model": False,
        "note": (
            "full predict_action_chunk with profile=False (includes scout+select+"
            "official prefix+expert). Segment breakdown is a separate profile=True pass."
        ),
        "device": str(device),
        "physical_gpu": gpu,
    }

    with torch.no_grad(), skip_llm_layers(policy, DROP, keep_kv=True):
        for _ in range(args.warmup):
            policy.reset()
            policy.predict_action_chunk(frames[0], noise=noises[0], num_steps=NFE)
        base_ms = []
        for batch, noise in zip(frames, noises, strict=True):
            base_ms.extend(_time_chunk(policy, batch, noise, args.repeats, device))
        summary["baseline"] = {
            "samples_ms": base_ms,
            "median_ms": _percentile(base_ms, 50),
            "p95_ms": _percentile(base_ms, 95),
            "mean_ms": float(np.mean(base_ms)),
        }

        with crop_visual_tokens(
            policy, keep_all=False, drop_layers=DROP, scout_layer=SCOUT_LAYER, profile=False
        ):
            for _ in range(args.warmup):
                policy.reset()
                policy.predict_action_chunk(frames[0], noise=noises[0], num_steps=NFE)
            crop_ms = []
            for batch, noise in zip(frames, noises, strict=True):
                crop_ms.extend(_time_chunk(policy, batch, noise, args.repeats, device))
            last = getattr(policy.model, "_crop_stats", None)
            summary["crop"] = {
                "samples_ms": crop_ms,
                "median_ms": _percentile(crop_ms, 50),
                "p95_ms": _percentile(crop_ms, 95),
                "mean_ms": float(np.mean(crop_ms)),
                "last_stats": None if last is None else stats_dict(last),
                "profile": False,
            }

        with crop_visual_tokens(
            policy, keep_all=False, drop_layers=DROP, scout_layer=SCOUT_LAYER, profile=True
        ):
            policy.reset()
            policy.predict_action_chunk(frames[0], noise=noises[0], num_steps=NFE)
            parts = []
            for batch, noise in zip(frames, noises, strict=True):
                policy.reset()
                policy.predict_action_chunk(batch, noise=noise, num_steps=NFE)
                stats = getattr(policy.model, "_crop_stats", None)
                if stats is not None:
                    parts.append(stats_dict(stats))
            timed = [p["timing_ms"] for p in parts if p.get("timing_ms")]
            if timed:
                keys = sorted({k for t in timed for k in t})
                summary["crop"]["breakdown_median_ms"] = {
                    k: _percentile([t[k] for t in timed if k in t], 50) for k in keys
                }
                summary["crop"]["breakdown_p95_ms"] = {
                    k: _percentile([t[k] for t in timed if k in t], 95) for k in keys
                }
                summary["crop"]["breakdown_note"] = (
                    "separate profile=True pass; not used for median/P95 full-chunk latency"
                )

    base_med = summary["baseline"]["median_ms"]
    crop_med = summary["crop"]["median_ms"]
    summary["delta_median_ms"] = crop_med - base_med
    summary["delta_median_pct"] = 100.0 * (crop_med / base_med - 1.0)
    summary["delta_p95_ms"] = summary["crop"]["p95_ms"] - summary["baseline"]["p95_ms"]
    out = OUT / "efficiency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"baseline median={base_med:.2f} p95={summary['baseline']['p95_ms']:.2f} ms  "
        f"crop median={crop_med:.2f} p95={summary['crop']['p95_ms']:.2f} ms  "
        f"Δmedian={summary['delta_median_ms']:+.2f} ({summary['delta_median_pct']:+.1f}%)  "
        f"wrote {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""π0.5 formal deployment latency — per-obs warmup, eager full predict_action_chunk."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

FN = Path(__file__).resolve().parents[2]
DEFAULT_OUT = FN / "results" / "stage_nfe" / "formal_deployment_pi05_steady_20261017"
PI05_DROP = [1, 13, 14, 15, 16, 17]
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

CONFIGS = [
    "pi05_full_nfe10_eager",
    "pi05_pruned_nfe10_materialized",
    "pi05_full_nfe1_eager",
    "pi05_pruned_nfe1_materialized",
]


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _p95(values: list[float]) -> float:
    xs = sorted(values)
    i = min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))
    return float(xs[i])


def _summarize(samples_ms: list[float]) -> dict:
    return {
        "n": len(samples_ms),
        "median_ms": float(statistics.median(samples_ms)),
        "mean_ms": float(statistics.mean(samples_ms)),
        "p95_ms": _p95(samples_ms),
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
        "stdev_ms": float(statistics.pstdev(samples_ms)) if len(samples_ms) > 1 else 0.0,
    }


def run_config(config_id: str, device: torch.device, n_obs: int, rounds: int) -> dict:
    sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
    sys.path.insert(0, str(FN / "scripts" / "stage_nfe"))
    from common import DEFAULT_CHECKPOINT, load_policy
    from layer_skip import skip_llm_layers
    from materialize import materialize_pi05
    from measure_formal_deployment_efficiency import (
        _load_pi05_obs,
        _pi05_noise,
        _release_policy,
        capture_environment,
    )

    materialized = "materialized" in config_id
    steps = 10 if "nfe10" in config_id else 1

    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.compile_model = False
    policy.config.n_action_steps = 10
    policy.eval()

    batches, obs_meta = _load_pi05_obs(device, n_obs)
    noise = _pi05_noise(device, policy)
    setup: dict = {}

    if materialized:
        batch0 = batches[0]
        with torch.no_grad(), skip_llm_layers(policy, PI05_DROP):
            policy.reset()
            hooked = policy.predict_action_chunk(batch0, noise=noise.clone(), num_steps=steps)
        setup["materialize"] = materialize_pi05(policy, PI05_DROP)
        with torch.no_grad():
            policy.reset()
            mat = policy.predict_action_chunk(batch0, noise=noise.clone(), num_steps=steps)
        setup["action_max_abs_vs_kvonly"] = float((mat - hooked).detach().float().abs().max().item())

    params_stored = int(sum(p.numel() for p in policy.parameters()))

    def _run_chunk(batch):
        with torch.no_grad():
            policy.reset()
            policy.predict_action_chunk(batch, noise=noise.clone(), num_steps=steps)

    def _timed_obs(batch) -> float:
        policy.reset()
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            policy.predict_action_chunk(batch, noise=noise.clone(), num_steps=steps)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0

    torch.cuda.reset_peak_memory_stats()
    per_round = []
    all_ms = []
    all_detail = []

    for r in range(rounds):
        round_ms = []
        for oi, batch in enumerate(batches):
            for _ in range(2):
                _run_chunk(batch)
            ms = _timed_obs(batch)
            round_ms.append(ms)
            all_ms.append(ms)
            all_detail.append(
                {
                    "round": r,
                    "obs_index": oi,
                    "pool_index": obs_meta[oi]["pool_index"],
                    "fingerprint": obs_meta[oi]["fingerprint"],
                    "latency_ms": ms,
                }
            )
        per_round.append({"round": r, "samples_ms": round_ms, **_summarize(round_ms)})

    peak = int(torch.cuda.max_memory_allocated())
    _release_policy(policy)
    return {
        "protocol": "pi05_formal_deployment_steady_v1",
        "config_id": config_id,
        "stack": "pi05",
        "mode": "steady_per_obs_eager_warmup",
        "num_inference_steps": steps,
        "observations": obs_meta,
        "n_obs": len(batches),
        "rounds": rounds,
        "setup_outside_timed_region": setup,
        "params_stored_m": params_stored / 1e6,
        "per_round": per_round,
        "all_samples": _summarize(all_ms),
        "all_samples_detail": all_detail,
        "peak_allocated_gib": peak / (1024**3),
        "timing_boundary": (
            "noise tensor prepared once outside; reset before each warmup/timed call; "
            "two warmup predict_action_chunk then one sync-timed call per obs"
        ),
        "script_sha256": SCRIPT_SHA,
    }


def merge_results(results: list[dict], anchor_id: str) -> dict:
    anchor = next(r for r in results if r["config_id"] == anchor_id)
    anchor_med = anchor["all_samples"]["median_ms"]
    table = []
    for r in results:
        med = r["all_samples"]["median_ms"]
        meds = [x["median_ms"] for x in r["per_round"]]
        table.append(
            {
                "config_id": r["config_id"],
                "median_ms": med,
                "p95_ms": r["all_samples"]["p95_ms"],
                "per_round_median_ms": meds,
                "speedup_vs_anchor": anchor_med / med if med > 0 else None,
                "params_stored_m": r["params_stored_m"],
                "peak_allocated_gib": r["peak_allocated_gib"],
            }
        )
    return {"summary_table": table, "anchor_config": anchor_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", default=None)
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--n-obs", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--anchor-recheck", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.physical_gpu))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    out_dir = args.out_dir
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    from measure_formal_deployment_efficiency import capture_environment

    if args.merge_only:
        results = [json.loads((cfg_dir / f"{cid}.json").read_text()) for cid in CONFIGS]
        merged = {
            "protocol": "pi05_formal_deployment_steady_v1",
            "utc": _utc(),
            "physical_gpu_index": args.physical_gpu,
            **merge_results(results, CONFIGS[0]),
        }
        recheck = cfg_dir / "pi05_full_nfe10_eager_anchor_recheck.json"
        if recheck.is_file():
            rec = json.loads(recheck.read_text())
            merged["anchor_recheck"] = {
                "first_median_ms": results[0]["all_samples"]["median_ms"],
                "recheck_median_ms": rec["all_samples"]["median_ms"],
                "drift_pct": abs(rec["all_samples"]["median_ms"] - results[0]["all_samples"]["median_ms"])
                / results[0]["all_samples"]["median_ms"]
                * 100,
            }
        (out_dir / "summary_pi05_steady.json").write_text(json.dumps(merged, indent=2) + "\n")
        print(json.dumps(merged, indent=2))
        return

    env0 = capture_environment(args.physical_gpu)
    if args.anchor_recheck:
        cid = "pi05_full_nfe10_eager"
        payload = run_config(cid, device, args.n_obs, 1)
        payload["environment_start"] = env0
        payload["environment_end"] = capture_environment(args.physical_gpu)
        payload["utc"] = _utc()
        path = cfg_dir / "pi05_full_nfe10_eager_anchor_recheck.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {path}", flush=True)
        return

    cid = args.config_id or CONFIGS[0]
    payload = run_config(cid, device, args.n_obs, args.rounds)
    payload["environment_start"] = env0
    payload["environment_end"] = capture_environment(args.physical_gpu)
    payload["utc"] = _utc()
    path = cfg_dir / f"{cid}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path} median={payload['all_samples']['median_ms']:.2f}ms", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Formal deployment efficiency: eager full vs materialized/graph pruned, 32 obs × 3 rounds."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
DEFAULT_OUT = FN / "results" / "stage_nfe" / "formal_deployment_20261017"
PI05_DROP = [1, 13, 14, 15, 16, 17]
MOLMO_DROP = [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


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


def capture_environment(physical_gpu_index: int) -> dict:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=30).strip()
        except Exception as exc:
            return f"error:{exc}"

    uuid_line = run(
        [
            "nvidia-smi",
            f"--id={physical_gpu_index}",
            "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu,temperature.gpu,clocks.sm",
            "--format=csv,noheader",
        ]
    )
    return {
        "utc": _utc(),
        "hostname": platform.node(),
        "physical_gpu_index": physical_gpu_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": "cuda:0",
        "nvidia_smi_row": uuid_line,
        "loadavg": run(["cat", "/proc/loadavg"]).strip(),
        "mem_available": run(["grep", "MemAvailable", "/proc/meminfo"]).strip(),
    }


def _tensor_fingerprint(batch: dict) -> str:
    parts = []
    for key in sorted(batch.keys()):
        val = batch[key]
        if torch.is_tensor(val):
            t = val.detach().cpu().float().view(-1)
            parts.append(f"{key}:{tuple(val.shape)}:{float(t[: min(8, t.numel())].sum()):.6f}")
        else:
            parts.append(f"{key}:{type(val).__name__}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _wall_chunk(run_fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _release_policy(policy) -> None:
    del policy
    gc.collect()
    torch.cuda.empty_cache()


def _load_pi05_obs(device: torch.device, n_obs: int) -> tuple[list[dict], list[dict]]:
    sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
    from common import DEFAULT_CALIB_FILE, load_calib_data, move_to_device, slice_batch

    calib = load_calib_data(DEFAULT_CALIB_FILE)
    meta = []
    batches = []
    for i in range(min(n_obs, len(calib))):
        b = move_to_device(slice_batch(calib[i], 0, 1), device)
        batches.append(b)
        meta.append({"pool_index": i, "fingerprint": _tensor_fingerprint(b)})
    return batches, meta


def _load_molmo_obs(device: torch.device, n_obs: int) -> tuple[list[dict], list[dict]]:
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    from load_policy import CALIB_PATH

    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    meta = []
    batches = []
    for i, sample in enumerate(blob["samples"][:n_obs]):
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample["batch"].items()}
        batches.append(b)
        meta.append(
            {
                "pool_index": i,
                "dataset_index": int(sample.get("index", i)),
                "fingerprint": _tensor_fingerprint(b),
            }
        )
    return batches, meta


def _pi05_noise(device, policy, seed: int = 1000) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(
        1,
        int(policy.model.config.chunk_size),
        int(policy.model.config.max_action_dim),
        generator=gen,
        device=device,
    )


def run_pi05_config(config_id: str, device: torch.device, n_obs: int, warmup: int, rounds: int) -> dict:
    sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
    sys.path.insert(0, str(FN / "scripts" / "stage_nfe"))
    from common import DEFAULT_CHECKPOINT, load_policy
    from layer_skip import skip_llm_layers
    from materialize import materialize_pi05
    from measure_efficiency import count_params

    materialized = "materialized" in config_id
    steps = 10 if "nfe10" in config_id else 1

    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.compile_model = False
    policy.config.n_action_steps = 10
    policy.eval()
    setup = {"materialize": None, "graph": None}

    batches, obs_meta = _load_pi05_obs(device, n_obs)
    noise = _pi05_noise(device, policy)

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
    if materialized:
        params_active_m = params_stored / 1e6
    else:
        params_active_m = count_params(policy, [])["active_params"] / 1e6

    def run_one(batch):
        with torch.no_grad():
            policy.reset()
            policy.predict_action_chunk(batch, noise=noise.clone(), num_steps=steps)

    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        run_one(batches[0])

    per_round = []
    all_ms = []
    for r in range(rounds):
        round_ms = [_wall_chunk(lambda b=batch: run_one(b)) for batch in batches]
        per_round.append({"round": r, "samples_ms": round_ms, **_summarize(round_ms)})
        all_ms.extend(round_ms)

    peak = int(torch.cuda.max_memory_allocated())
    _release_policy(policy)
    return {
        "config_id": config_id,
        "stack": "pi05",
        "mode": "materialized_fisher_drop6" if materialized else "eager_full",
        "num_inference_steps": steps,
        "observations": obs_meta,
        "n_obs": len(batches),
        "setup_outside_timed_region": setup,
        "params_stored_m": params_stored / 1e6,
        "params_active_m": params_active_m,
        "warmup": warmup,
        "rounds": rounds,
        "per_round": per_round,
        "all_samples": _summarize(all_ms),
        "peak_allocated_bytes": peak,
        "peak_allocated_gib": peak / (1024**3),
        "timing_boundary": "sync wall clock around predict_action_chunk only; load/materialize/noise clone outside",
    }


def _molmo_disable_graph(policy) -> None:
    policy.config.compile_model = False
    policy.config.gradient_checkpointing = False
    policy.config.enable_inference_cuda_graph = False
    backbone = policy._backbone()
    backbone.gradient_checkpointing = False
    if getattr(backbone, "transformer", None) is not None:
        backbone.transformer.gradient_checkpointing = False
    policy._set_inference_cuda_graph_enabled(False)


def _molmo_enable_graph(policy) -> dict:
    policy.config.enable_inference_cuda_graph = True
    policy._set_inference_cuda_graph_enabled(True)
    mgr = getattr(policy._backbone(), "action_cuda_graph_manager", None)
    return {"manager_present": mgr is not None, "config_flag": bool(policy.config.enable_inference_cuda_graph)}


def run_molmo_config(config_id: str, device: torch.device, n_obs: int, warmup: int, rounds: int) -> dict:
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    sys.path.insert(0, str(FN / "scripts" / "stage_nfe"))
    from layer_skip import skip_vlm_layers
    from load_policy import LEROBOT_CKPT, load_policy
    from materialize import materialize_molmo

    materialized = "materialized" in config_id
    use_graph = True
    steps = 10 if "nfe10" in config_id else 1

    policy = load_policy(str(device))
    policy.eval()
    policy.config.n_action_steps = 10
    _molmo_disable_graph(policy)

    batches, obs_meta = _load_molmo_obs(device, n_obs)
    setup: dict = {}

    def predict(batch, steps_: int, eager: bool):
        gen = torch.Generator(device=device)
        gen.manual_seed(1000)
        if eager:
            _molmo_disable_graph(policy)
        else:
            _molmo_enable_graph(policy)
        with torch.no_grad(), policy._autocast_context():
            policy.reset()
            return policy.predict_action_chunk(batch, num_steps=steps_, generator=gen)

    if materialized:
        b0 = batches[0]
        with torch.no_grad(), skip_vlm_layers(MOLMO_DROP), policy._autocast_context():
            policy.reset()
            gen = torch.Generator(device=device)
            gen.manual_seed(1000)
            hooked = policy.predict_action_chunk(b0, num_steps=steps, generator=gen)
        setup["materialize"] = materialize_molmo(policy, MOLMO_DROP)
        with torch.no_grad(), policy._autocast_context():
            policy.reset()
            gen = torch.Generator(device=device)
            gen.manual_seed(1000)
            mat = policy.predict_action_chunk(b0, num_steps=steps, generator=gen)
        setup["action_max_abs_vs_kvonly"] = float((mat - hooked).detach().float().abs().max().item())

    # Graph vs eager check on first 3 obs
    graph_check = {"use_graph": use_graph, "per_sample": []}
    if use_graph:
        for bi in range(min(3, len(batches))):
            eager_out = predict(batches[bi], steps, eager=True)
            graph_info = _molmo_enable_graph(policy)
            # warmup graph capture
            for _ in range(3):
                predict(batches[bi], steps, eager=False)
            graph_out = predict(batches[bi], steps, eager=False)
            diff = float((graph_out - eager_out).detach().float().abs().max().item())
            graph_check["per_sample"].append(
                {"obs_index": bi, "max_abs_vs_eager": diff, **graph_info}
            )
        setup["graph_check"] = graph_check
        failed = [s for s in graph_check["per_sample"] if s["max_abs_vs_eager"] > 1e-3]
        if failed and any(s.get("manager_present") for s in graph_check["per_sample"]):
            setup["graph_check"]["note"] = "large diff but manager present — graph may still alter numerics"
        if not any(s.get("manager_present") for s in graph_check["per_sample"]):
            setup["graph_fallback"] = "expert_cuda_graph_manager missing; timing uses eager expert path"
            use_graph = False

    params_stored = int(sum(p.numel() for p in policy.parameters()))

    def run_one(batch):
        predict(batch, steps, eager=not use_graph)

    # CUDA graph capture must happen outside the timed region (one graph per obs shape/content).
    if use_graph:
        setup["graph_precapture"] = {"n_obs": len(batches), "replays_per_obs": 3}
        for batch in batches:
            for _ in range(3):
                predict(batch, steps, eager=False)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        run_one(batches[0])

    per_round = []
    all_ms = []
    for r in range(rounds):
        round_ms = [_wall_chunk(lambda b=batch: run_one(b)) for batch in batches]
        per_round.append({"round": r, "samples_ms": round_ms, **_summarize(round_ms)})
        all_ms.extend(round_ms)

    peak = int(torch.cuda.max_memory_allocated())
    mode = "materialized_taylor_drop16_expert_graph" if materialized else "full_expert_graph"
    if not use_graph:
        mode += "_eager_fallback"
    _release_policy(policy)
    return {
        "config_id": config_id,
        "stack": "molmoact2",
        "mode": mode,
        "num_inference_steps": steps,
        "checkpoint": str(LEROBOT_CKPT),
        "observations": obs_meta,
        "n_obs": len(batches),
        "setup_outside_timed_region": setup,
        "params_stored_m": params_stored / 1e6,
        "params_active_m": params_stored / 1e6,
        "warmup": warmup,
        "rounds": rounds,
        "per_round": per_round,
        "all_samples": _summarize(all_ms),
        "peak_allocated_bytes": peak,
        "peak_allocated_gib": peak / (1024**3),
        "timing_boundary": "sync wall clock around predict_action_chunk; graph capture/materialize outside",
    }


PI05_CONFIGS = [
    "pi05_full_nfe10_eager",
    "pi05_pruned_nfe10_materialized",
    "pi05_full_nfe1_eager",
    "pi05_pruned_nfe1_materialized",
]
MOLMO_CONFIGS = [
    "molmo_full_nfe10_graph",
    "molmo_pruned_nfe10_materialized_graph",
    "molmo_full_nfe1_graph",
    "molmo_pruned_nfe1_materialized_graph",
]


def merge_summary(config_results: list[dict], anchor_id: str) -> dict:
    anchor = next(r for r in config_results if r["config_id"] == anchor_id)
    anchor_med = anchor["all_samples"]["median_ms"]
    table = []
    round_stability = {}
    for r in config_results:
        if "full_nfe10" in r["config_id"] and r["config_id"].endswith(("eager", "graph")):
            meds = [x["median_ms"] for x in r["per_round"]]
            if len(meds) >= 2:
                spread = (max(meds) - min(meds)) / min(meds) * 100 if min(meds) else 0
                round_stability[r["config_id"]] = {"round_medians_ms": meds, "spread_pct": spread}
        med = r["all_samples"]["median_ms"]
        table.append(
            {
                "config_id": r["config_id"],
                "median_ms": med,
                "p95_ms": r["all_samples"]["p95_ms"],
                "per_round_median_ms": [x["median_ms"] for x in r["per_round"]],
                "speedup_vs_anchor": anchor_med / med if med > 0 else None,
                "params_stored_m": r["params_stored_m"],
                "peak_allocated_gib": r["peak_allocated_gib"],
            }
        )
    return {"summary_table": table, "round_stability": round_stability, "anchor_config": anchor_id}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", choices=("pi05", "molmoact2"), required=True)
    parser.add_argument("--config-id", default=None, help="single config; default runs all four for stack")
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--n-obs", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.physical_gpu))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    out_dir = args.out_dir
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        ids = PI05_CONFIGS if args.stack == "pi05" else MOLMO_CONFIGS
        results = [json.loads((cfg_dir / f"{cid}.json").read_text()) for cid in ids]
        anchor = ids[0]
        merged = {
            "protocol": "formal_deployment_efficiency_v1",
            "stack": args.stack,
            "utc": _utc(),
            "physical_gpu_index": args.physical_gpu,
            **merge_summary(results, anchor),
        }
        if args.stack == "pi05":
            recheck = run_pi05_config("pi05_full_nfe10_eager", device, args.n_obs, args.warmup, 1)
            first_med = results[0]["all_samples"]["median_ms"]
            recheck_med = recheck["all_samples"]["median_ms"]
            merged["anchor_recheck"] = {
                "first_median_ms": first_med,
                "recheck_median_ms": recheck_med,
                "drift_pct": abs(recheck_med - first_med) / first_med * 100 if first_med else None,
            }
        else:
            recheck = run_molmo_config("molmo_full_nfe10_graph", device, args.n_obs, args.warmup, 1)
            first_med = results[0]["all_samples"]["median_ms"]
            recheck_med = recheck["all_samples"]["median_ms"]
            merged["anchor_recheck"] = {
                "first_median_ms": first_med,
                "recheck_median_ms": recheck_med,
                "drift_pct": abs(recheck_med - first_med) / first_med * 100 if first_med else None,
            }
        (out_dir / f"summary_{args.stack}.json").write_text(json.dumps(merged, indent=2) + "\n")
        print(json.dumps(merged, indent=2))
        return

    config_ids = [args.config_id] if args.config_id else (PI05_CONFIGS if args.stack == "pi05" else MOLMO_CONFIGS)

    env0 = capture_environment(args.physical_gpu)
    results = []
    for cid in config_ids:
        if args.stack == "pi05":
            payload = run_pi05_config(cid, device, args.n_obs, args.warmup, args.rounds)
        else:
            payload = run_molmo_config(cid, device, args.n_obs, args.warmup, args.rounds)
        payload["environment_start"] = env0
        payload["environment_end"] = capture_environment(args.physical_gpu)
        payload["script_sha256"] = SCRIPT_SHA
        path = cfg_dir / f"{cid}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        results.append(payload)
        print(f"wrote {path}", flush=True)

    if len(config_ids) == 4:
        anchor = PI05_CONFIGS[0] if args.stack == "pi05" else MOLMO_CONFIGS[0]
        # recheck anchor in fresh process would be separate invocation; optional here skip
        merged = {
            "protocol": "formal_deployment_efficiency_v1",
            "stack": args.stack,
            "utc": _utc(),
            "physical_gpu_index": args.physical_gpu,
            "configs": [r["config_id"] for r in results],
            **merge_summary(results, anchor),
        }
        if args.stack == "pi05":
            recheck = run_pi05_config("pi05_full_nfe10_eager", device, args.n_obs, args.warmup, 1)
            first_med = results[0]["all_samples"]["median_ms"]
            recheck_med = recheck["all_samples"]["median_ms"]
            merged["anchor_recheck"] = {
                "first_median_ms": first_med,
                "recheck_median_ms": recheck_med,
                "drift_pct": abs(recheck_med - first_med) / first_med * 100 if first_med else None,
            }
        (out_dir / f"summary_{args.stack}.json").write_text(json.dumps(merged, indent=2) + "\n")


if __name__ == "__main__":
    main()

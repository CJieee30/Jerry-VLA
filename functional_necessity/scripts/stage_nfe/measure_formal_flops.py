#!/usr/bin/env python3
"""Formal deployment FLOPs: 32 calib obs × 8 configs, eager full predict_action_chunk."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.flop_counter import FlopCounterMode

FN = Path(__file__).resolve().parents[2]
DEFAULT_OUT = FN / "results" / "stage_nfe" / "formal_flops_20261017"
PI05_DROP = [1, 13, 14, 15, 16, 17]
MOLMO_DROP = [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

PI05_CONFIGS = [
    "pi05_full_nfe10_eager",
    "pi05_pruned_nfe10_materialized",
    "pi05_full_nfe1_eager",
    "pi05_pruned_nfe1_materialized",
]
MOLMO_CONFIGS = [
    "molmo_full_nfe10_eager",
    "molmo_pruned_nfe10_materialized",
    "molmo_full_nfe1_eager",
    "molmo_pruned_nfe1_materialized",
]

COVERAGE_OPS = ("aten::mm", "aten::bmm", "aten::addmm", "aten::conv2d", "aten::scaled_dot_product_attention")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean_gflops": float(statistics.mean(values)),
        "median_gflops": float(statistics.median(values)),
        "min_gflops": float(min(values)),
        "max_gflops": float(max(values)),
        "stdev_gflops": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def _flop_breakdown(mode: FlopCounterMode) -> dict[str, float]:
    counts = mode.get_flop_counts()
    by_op: dict[str, int] = {}
    for _mod, ops in counts.items():
        for op, n in ops.items():
            by_op[op] = by_op.get(op, 0) + int(n)
    total = int(mode.get_total_flops())
    tracked = sum(by_op.get(op, 0) for op in COVERAGE_OPS)
    return {
        "total_flops": total,
        "tracked_major_ops_flops": tracked,
        "tracked_fraction_of_total": tracked / total if total else 0.0,
        "op_gflops": {op: by_op.get(op, 0) / 1e9 for op in COVERAGE_OPS},
        "other_ops_count": len(by_op) - sum(1 for op in COVERAGE_OPS if by_op.get(op)),
        "uncounted_note": (
            "Total is FlopCounterMode global get_total_flops() on one predict_action_chunk; "
            "not a sum of per-module wrappers (avoids nested double count)."
        ),
    }


def _load_pi05_obs(device: torch.device, n_obs: int) -> tuple[list[dict], list[dict]]:
    sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
    from common import DEFAULT_CALIB_FILE, load_calib_data, move_to_device, slice_batch

    calib = load_calib_data(DEFAULT_CALIB_FILE)
    meta, batches = [], []
    for i in range(min(n_obs, len(calib))):
        b = move_to_device(slice_batch(calib[i], 0, 1), device)
        batches.append(b)
        meta.append({"pool_index": i, "fingerprint": _tensor_fingerprint(b)})
    return batches, meta


def _load_molmo_obs(device: torch.device, n_obs: int) -> tuple[list[dict], list[dict]]:
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    from load_policy import CALIB_PATH

    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    meta, batches = [], []
    for i, sample in enumerate(blob["samples"][:n_obs]):
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample["batch"].items()}
        batches.append(b)
        meta.append(
            {
                "pool_index": i,
                "dataset_index": int(sample.get("index", i)),
                "fingerprint": _tensor_fingerprint(b),
                "input_ids_len": int(b["input_ids"].shape[1]),
            }
        )
    return batches, meta


def _release(policy) -> None:
    del policy
    gc.collect()
    torch.cuda.empty_cache()


def _measure_obs_flops(run_once: Callable[[], None]) -> dict:
    run_once()
    with FlopCounterMode(display=False) as mode:
        run_once()
    total = int(mode.get_total_flops())
    return {"gflops_per_chunk": total / 1e9, "flop_detail": _flop_breakdown(mode)}


def _precheck_memory(run_once: Callable[[], None], min_free_gib: float) -> dict:
    torch.cuda.reset_peak_memory_stats()
    try:
        run_once()
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        free, total = torch.cuda.mem_get_info()
        ok = free / (1024**3) >= min_free_gib
        return {
            "ok": ok,
            "peak_allocated_gib": peak,
            "free_gib_after": free / (1024**3),
            "total_gib": total / (1024**3),
        }
    except torch.cuda.OutOfMemoryError as exc:
        return {"ok": False, "oom": str(exc)}


def run_pi05_config(config_id: str, device: torch.device, n_obs: int) -> dict:
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

    batches, obs_meta = _load_pi05_obs(device, n_obs)
    gen = torch.Generator(device=device)
    gen.manual_seed(1000)
    noise = torch.randn(
        1,
        int(policy.model.config.chunk_size),
        int(policy.model.config.max_action_dim),
        generator=gen,
        device=device,
    )
    setup: dict[str, Any] = {}

    if materialized:
        b0 = batches[0]
        with torch.no_grad(), skip_llm_layers(policy, PI05_DROP):
            policy.reset()
            hooked = policy.predict_action_chunk(b0, noise=noise.clone(), num_steps=steps)
        setup["materialize"] = materialize_pi05(policy, PI05_DROP)
        with torch.no_grad():
            policy.reset()
            mat = policy.predict_action_chunk(b0, noise=noise.clone(), num_steps=steps)
        setup["action_max_abs_vs_kvonly"] = float((mat - hooked).detach().float().abs().max().item())

    params_stored = int(sum(p.numel() for p in policy.parameters()))
    params_active_m = (
        params_stored / 1e6 if materialized else count_params(policy, [])["active_params"] / 1e6
    )

    def run_one(batch):
        with torch.no_grad():
            policy.reset()
            policy.predict_action_chunk(batch, noise=noise.clone(), num_steps=steps)

    pre = _precheck_memory(lambda: run_one(batches[0]), min_free_gib=1.0)
    if not pre.get("ok"):
        _release(policy)
        return {
            "config_id": config_id,
            "stack": "pi05",
            "status": "skipped_oom_or_low_vram",
            "precheck": pre,
            "num_inference_steps": steps,
        }

    per_obs = []
    gflops_list = []
    for i, batch in enumerate(batches):
        row = _measure_obs_flops(lambda b=batch: run_one(b))
        row["obs"] = obs_meta[i]
        per_obs.append(row)
        gflops_list.append(row["gflops_per_chunk"])

    peak = int(torch.cuda.max_memory_allocated())
    _release(policy)
    return {
        "config_id": config_id,
        "stack": "pi05",
        "status": "ok",
        "mode": "materialized_fisher_drop6" if materialized else "eager_full",
        "num_inference_steps": steps,
        "compile_model": False,
        "cuda_graph": False,
        "timing_boundary": "FlopCounterMode on single predict_action_chunk; excludes load/materialize/warmup",
        "flop_count_rule": "1 mul-add = 2 FLOPs (PyTorch FlopCounterMode default)",
        "torch_version": torch.__version__,
        "params_stored_m": params_stored / 1e6,
        "params_active_m": params_active_m,
        "setup_outside": setup,
        "precheck": pre,
        "per_observation": per_obs,
        "summary_gflops": _summarize(gflops_list),
        "peak_allocated_gib": peak / (1024**3),
        "observations": obs_meta,
    }


def run_molmo_config(config_id: str, device: torch.device, n_obs: int) -> dict:
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    sys.path.insert(0, str(FN / "scripts" / "stage_nfe"))
    from layer_skip import skip_vlm_layers
    from load_policy import LEROBOT_CKPT, load_policy
    from materialize import materialize_molmo

    materialized = "materialized" in config_id
    steps = 10 if "nfe10" in config_id else 1
    policy = load_policy(str(device))
    policy.eval()
    policy.config.compile_model = False
    policy.config.enable_inference_cuda_graph = False
    policy.config.gradient_checkpointing = False
    policy.config.n_action_steps = 10
    backbone = policy._backbone()
    backbone.gradient_checkpointing = False
    policy._set_inference_cuda_graph_enabled(False)
    if getattr(backbone, "action_cuda_graph_manager", None) is not None:
        backbone.action_cuda_graph_manager = None

    batches, obs_meta = _load_molmo_obs(device, n_obs)
    setup: dict[str, Any] = {}
    gen = torch.Generator(device=device)

    if materialized:
        b0 = batches[0]
        gen.manual_seed(1000)
        with torch.no_grad(), skip_vlm_layers(MOLMO_DROP), policy._autocast_context():
            policy.reset()
            hooked = policy.predict_action_chunk(b0, num_steps=steps, generator=gen)
        setup["materialize"] = materialize_molmo(policy, MOLMO_DROP)
        gen.manual_seed(1000)
        with torch.no_grad(), policy._autocast_context():
            policy.reset()
            mat = policy.predict_action_chunk(b0, num_steps=steps, generator=gen)
        setup["action_max_abs_vs_kvonly"] = float((mat - hooked).detach().float().abs().max().item())

    params_stored = int(sum(p.numel() for p in policy.parameters()))

    def run_one(batch):
        gen.manual_seed(1000)
        with torch.no_grad(), policy._autocast_context():
            policy.reset()
            policy.predict_action_chunk(batch, num_steps=steps, generator=gen)

    pre = _precheck_memory(lambda: run_one(batches[0]), min_free_gib=2.0)
    if not pre.get("ok"):
        _release(policy)
        return {
            "config_id": config_id,
            "stack": "molmoact2",
            "status": "skipped_oom_or_low_vram",
            "precheck": pre,
            "num_inference_steps": steps,
        }

    per_obs = []
    gflops_list = []
    for i, batch in enumerate(batches):
        row = _measure_obs_flops(lambda b=batch: run_one(b))
        row["obs"] = obs_meta[i]
        per_obs.append(row)
        gflops_list.append(row["gflops_per_chunk"])

    peak = int(torch.cuda.max_memory_allocated())
    _release(policy)
    return {
        "config_id": config_id,
        "stack": "molmoact2",
        "status": "ok",
        "mode": "materialized_taylor_drop16" if materialized else "eager_full",
        "num_inference_steps": steps,
        "compile_model": False,
        "cuda_graph": False,
        "checkpoint": str(LEROBOT_CKPT),
        "timing_boundary": "FlopCounterMode on single predict_action_chunk; excludes load/materialize/warmup",
        "flop_count_rule": "1 mul-add = 2 FLOPs (PyTorch FlopCounterMode default)",
        "torch_version": torch.__version__,
        "params_stored_m": params_stored / 1e6,
        "params_active_m": params_stored / 1e6,
        "setup_outside": setup,
        "precheck": pre,
        "per_observation": per_obs,
        "summary_gflops": _summarize(gflops_list),
        "peak_allocated_gib": peak / (1024**3),
        "observations": obs_meta,
    }


def merge_summary(results: list[dict], anchor_id: str) -> dict:
    anchor = next(r for r in results if r.get("config_id") == anchor_id and r.get("status") == "ok")
    anchor_med = anchor["summary_gflops"]["median_gflops"]
    table = []
    for r in results:
        if r.get("status") != "ok":
            table.append({"config_id": r["config_id"], "status": r.get("status"), "precheck": r.get("precheck")})
            continue
        med = r["summary_gflops"]["median_gflops"]
        table.append(
            {
                "config_id": r["config_id"],
                "median_gflops": med,
                "mean_gflops": r["summary_gflops"]["mean_gflops"],
                "min_gflops": r["summary_gflops"]["min_gflops"],
                "max_gflops": r["summary_gflops"]["max_gflops"],
                "reduction_vs_anchor_pct": 100.0 * (1.0 - med / anchor_med) if anchor_med else None,
                "params_stored_m": r["params_stored_m"],
                "peak_allocated_gib": r["peak_allocated_gib"],
            }
        )
    return {"anchor_config": anchor_id, "summary_table": table}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", choices=("pi05", "molmoact2"), required=True)
    parser.add_argument("--config-id", default=None)
    parser.add_argument("--physical-gpu", type=int, default=3)
    parser.add_argument("--n-obs", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.physical_gpu))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    out_dir = args.out_dir
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    config_ids = (
        [args.config_id]
        if args.config_id
        else (PI05_CONFIGS if args.stack == "pi05" else MOLMO_CONFIGS)
    )

    if args.merge_only:
        results = [json.loads((cfg_dir / f"{cid}.json").read_text()) for cid in config_ids]
        anchor = config_ids[0]
        merged = {
            "protocol": "formal_flops_v1",
            "stack": args.stack,
            "utc": _utc(),
            "physical_gpu_index": args.physical_gpu,
            "hostname": platform.node(),
            "script_sha256": SCRIPT_SHA,
            **merge_summary(results, anchor),
        }
        (out_dir / f"summary_{args.stack}.json").write_text(json.dumps(merged, indent=2) + "\n")
        print(json.dumps(merged, indent=2))
        return

    results = []
    for cid in config_ids:
        payload = (
            run_pi05_config(cid, device, args.n_obs)
            if args.stack == "pi05"
            else run_molmo_config(cid, device, args.n_obs)
        )
        payload["utc"] = _utc()
        payload["physical_gpu_index"] = args.physical_gpu
        payload["script_sha256"] = SCRIPT_SHA
        path = cfg_dir / f"{cid}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        results.append(payload)
        print(f"wrote {path} status={payload.get('status')}", flush=True)

    if len(config_ids) == 4:
        anchor = config_ids[0]
        merged = {
            "protocol": "formal_flops_v1",
            "stack": args.stack,
            "utc": _utc(),
            "physical_gpu_index": args.physical_gpu,
            "hostname": platform.node(),
            "script_sha256": SCRIPT_SHA,
            **merge_summary(results, anchor),
        }
        (out_dir / f"summary_{args.stack}.json").write_text(json.dumps(merged, indent=2) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MolmoAct2 formal deployment latency — steady per-obs expert Graph protocol (GPU2)."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

FN = Path(__file__).resolve().parents[2]
DEFAULT_OUT = FN / "results" / "stage_nfe" / "formal_deployment_molmo_steady_20261017"
MOLMO_DROP = [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
GRAPH_DIAG_DEFAULT = FN / "results" / "stage_nfe" / "molmo_graph_diagnostic_20261017" / "diagnostic.json"

CONFIGS = [
    "molmo_full_nfe10_graph",
    "molmo_pruned_nfe10_materialized_graph",
    "molmo_full_nfe1_graph",
    "molmo_pruned_nfe1_materialized_graph",
]

_timed_capture_window = False
_timed_capture_events: list[bool] = []


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


def _install_graph_probe() -> dict:
    inference_mod = importlib.import_module("lerobot.policies.molmoact2.molmoact2_hf_model.inference")
    mgr_cls = inference_mod.ActionCudaGraphManager
    _cuda_graph_key = inference_mod._cuda_graph_key
    orig = mgr_cls.run_action_flow

    def patched(self, inputs, steps, run_loop):
        global _timed_capture_window, _timed_capture_events
        prev = self.action_flow_graph.key if self.action_flow_graph is not None else None
        key = _cuda_graph_key(inputs, steps)
        captured = self.action_flow_graph is None or self.action_flow_graph.key != key
        out = orig(self, inputs, steps, run_loop)
        if _timed_capture_window:
            _timed_capture_events.append(bool(captured))
        return out

    mgr_cls.run_action_flow = patched  # type: ignore[method-assign]
    return {"patched": True, "manager_class": f"{mgr_cls.__module__}.{mgr_cls.__name__}"}


@contextmanager
def _timed_capture_scope():
    global _timed_capture_window, _timed_capture_events
    _timed_capture_events = []
    _timed_capture_window = True
    try:
        yield _timed_capture_events
    finally:
        _timed_capture_window = False


def _wall_predict_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def run_config(
    config_id: str,
    device: torch.device,
    n_obs: int,
    rounds: int,
    graph_diag_path: Path,
) -> dict:
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    sys.path.insert(0, str(FN / "scripts" / "stage_nfe"))
    from layer_skip import skip_vlm_layers
    from load_policy import LEROBOT_CKPT, load_policy
    from materialize import materialize_molmo
    from measure_formal_deployment_efficiency import (
        _load_molmo_obs,
        _molmo_disable_graph,
        _molmo_enable_graph,
        _release_policy,
        capture_environment,
    )

    probe_meta = _install_graph_probe()
    materialized = "materialized" in config_id
    steps = 10 if "nfe10" in config_id else 1

    policy = load_policy(str(device))
    policy.eval()
    policy.config.compile_model = False
    policy.config.n_action_steps = 10
    policy.config.gradient_checkpointing = False
    _molmo_disable_graph(policy)

    batches, obs_meta = _load_molmo_obs(device, n_obs)
    setup: dict[str, Any] = {"graph_probe": probe_meta}

    def _warmup_obs(batch) -> None:
        for _ in range(2):
            gen = torch.Generator(device=device)
            gen.manual_seed(1000)
            _molmo_enable_graph(policy)
            with torch.no_grad(), policy._autocast_context():
                policy.reset()
                policy.predict_action_chunk(batch, num_steps=steps, generator=gen)

    def _timed_obs(batch) -> tuple[float, bool]:
        gen = torch.Generator(device=device)
        gen.manual_seed(1000)
        _molmo_enable_graph(policy)
        policy.reset()
        with torch.no_grad(), policy._autocast_context():
            with _timed_capture_scope() as cap_flags:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                policy.predict_action_chunk(batch, num_steps=steps, generator=gen)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000.0
        return ms, bool(cap_flags) and any(cap_flags)

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

    graph_check = {"per_sample": []}
    for bi in range(min(3, len(batches))):
        batch = batches[bi]
        gen = torch.Generator(device=device)
        gen.manual_seed(1000)
        _molmo_disable_graph(policy)
        with torch.no_grad(), policy._autocast_context():
            policy.reset()
            eager_out = policy.predict_action_chunk(batch, num_steps=steps, generator=gen)
        ginfo = _molmo_enable_graph(policy)
        for _ in range(2):
            gen.manual_seed(1000)
            policy.reset()
            with torch.no_grad(), policy._autocast_context():
                policy.predict_action_chunk(batch, num_steps=steps, generator=gen)
        gen.manual_seed(1000)
        policy.reset()
        with torch.no_grad(), policy._autocast_context():
            graph_out = policy.predict_action_chunk(batch, num_steps=steps, generator=gen)
        graph_check["per_sample"].append(
            {
                "obs_index": bi,
                "max_abs_vs_eager": float((graph_out - eager_out).detach().float().abs().max().item()),
                **ginfo,
            }
        )
    setup["graph_check"] = graph_check
    if not any(s.get("manager_present") for s in graph_check["per_sample"]):
        raise SystemExit("action_cuda_graph_manager missing — abort formal graph deployment timing")

    params_stored = int(sum(p.numel() for p in policy.parameters()))
    torch.cuda.reset_peak_memory_stats()

    per_round = []
    all_samples_detail = []
    all_ms = []
    capture_in_timed = 0
    n_timed = 0

    for r in range(rounds):
        round_ms = []
        round_details = []
        for oi, batch in enumerate(batches):
            _warmup_obs(batch)
            ms, captured = _timed_obs(batch)
            n_timed += 1
            if captured:
                capture_in_timed += 1
            round_ms.append(ms)
            all_ms.append(ms)
            detail = {
                "round": r,
                "obs_index": oi,
                "pool_index": obs_meta[oi]["pool_index"],
                "fingerprint": obs_meta[oi]["fingerprint"],
                "latency_ms": ms,
                "capture_in_timed_region": captured,
            }
            round_details.append(detail)
            all_samples_detail.append(detail)
        per_round.append(
            {
                "round": r,
                "samples_ms": round_ms,
                "samples_detail": round_details,
                **_summarize(round_ms),
            }
        )

    peak = int(torch.cuda.max_memory_allocated())
    _release_policy(policy)

    diag_ref = None
    if graph_diag_path.is_file():
        diag_ref = json.loads(graph_diag_path.read_text())

    return {
        "protocol": "molmo_formal_deployment_steady_v1",
        "config_id": config_id,
        "stack": "molmoact2",
        "mode": "steady_per_obs_graph_warmup",
        "num_inference_steps": steps,
        "checkpoint": str(LEROBOT_CKPT),
        "observations": obs_meta,
        "n_obs": len(batches),
        "rounds": rounds,
        "n_timed_samples": n_timed,
        "capture_events_in_timed_region": capture_in_timed,
        "capture_fraction_in_timed": capture_in_timed / n_timed if n_timed else 0.0,
        "setup_outside_timed_region": setup,
        "graph_diagnostic_reference": str(graph_diag_path),
        "graph_diagnostic_conclusion": (diag_ref or {}).get("conclusion"),
        "params_stored_m": params_stored / 1e6,
        "params_active_m": params_stored / 1e6,
        "per_round": per_round,
        "all_samples": _summarize(all_ms),
        "all_samples_detail": all_samples_detail,
        "peak_allocated_bytes": peak,
        "peak_allocated_gib": peak / (1024**3),
        "vram_note": "torch.cuda.max_memory_allocated after model+materialize; excludes host load",
        "timing_boundary": (
            "sync wall clock around predict_action_chunk only; per-obs graph warmup immediately "
            "before each timed sample; materialize/graph_check/reset+generator before warmup outside "
            "timed window; VLM prefix + action expert inside timed call"
        ),
        "script_sha256": SCRIPT_SHA,
    }


def merge_results(config_results: list[dict], anchor_id: str) -> dict:
    anchor = next(r for r in config_results if r["config_id"] == anchor_id)
    anchor_med = anchor["all_samples"]["median_ms"]
    table = []
    round_stability = {}
    for r in config_results:
        meds = [x["median_ms"] for x in r["per_round"]]
        if len(meds) >= 2:
            spread = (max(meds) - min(meds)) / min(meds) * 100 if min(meds) else 0
            round_stability[r["config_id"]] = {"round_medians_ms": meds, "spread_pct": spread}
        med = r["all_samples"]["median_ms"]
        table.append(
            {
                "config_id": r["config_id"],
                "median_ms": med,
                "mean_ms": r["all_samples"]["mean_ms"],
                "p95_ms": r["all_samples"]["p95_ms"],
                "per_round_median_ms": meds,
                "speedup_vs_anchor": anchor_med / med if med > 0 else None,
                "capture_fraction_in_timed": r["capture_fraction_in_timed"],
                "params_stored_m": r["params_stored_m"],
                "peak_allocated_gib": r["peak_allocated_gib"],
            }
        )
    return {
        "summary_table": table,
        "round_stability": round_stability,
        "anchor_config": anchor_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", default=None)
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--n-obs", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--graph-diagnostic", type=Path, default=GRAPH_DIAG_DEFAULT)
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

    if args.merge_only:
        results = [json.loads((cfg_dir / f"{cid}.json").read_text()) for cid in CONFIGS]
        merged = {
            "protocol": "molmo_formal_deployment_steady_v1",
            "utc": _utc(),
            "physical_gpu_index": args.physical_gpu,
            **merge_results(results, CONFIGS[0]),
        }
        recheck_path = cfg_dir / "molmo_full_nfe10_graph_anchor_recheck.json"
        if recheck_path.is_file():
            rec = json.loads(recheck_path.read_text())
            first_med = results[0]["all_samples"]["median_ms"]
            rec_med = rec["all_samples"]["median_ms"]
            merged["anchor_recheck"] = {
                "first_median_ms": first_med,
                "recheck_median_ms": rec_med,
                "drift_pct": abs(rec_med - first_med) / first_med * 100 if first_med else None,
            }
        (out_dir / "summary_molmoact2_steady.json").write_text(json.dumps(merged, indent=2) + "\n")
        print(json.dumps(merged, indent=2))
        return

    from measure_formal_deployment_efficiency import capture_environment

    env0 = capture_environment(args.physical_gpu)
    if args.anchor_recheck:
        cid = "molmo_full_nfe10_graph"
        payload = run_config(cid, device, args.n_obs, 1, args.graph_diagnostic)
        payload["environment_start"] = env0
        payload["environment_end"] = capture_environment(args.physical_gpu)
        payload["utc"] = _utc()
        payload["anchor_recheck_run"] = True
        path = cfg_dir / "molmo_full_nfe10_graph_anchor_recheck.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {path}", flush=True)
        return

    cid = args.config_id or CONFIGS[0]
    payload = run_config(cid, device, args.n_obs, args.rounds, args.graph_diagnostic)
    payload["environment_start"] = env0
    payload["environment_end"] = capture_environment(args.physical_gpu)
    payload["utc"] = _utc()
    path = cfg_dir / f"{cid}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path} median={payload['all_samples']['median_ms']:.2f}ms", flush=True)


if __name__ == "__main__":
    main()

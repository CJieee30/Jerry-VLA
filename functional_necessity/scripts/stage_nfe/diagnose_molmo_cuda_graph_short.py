#!/usr/bin/env python3
"""Short Molmo expert CUDA-graph diagnostic (shared GPU; not formal deployment latency)."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

FN = Path(__file__).resolve().parents[2]
DEFAULT_OUT = FN / "results" / "stage_nfe" / "molmo_graph_diagnostic_20261017"
STEPS_PATTERN = (10, 10, 11, 11, 10)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lerobot_root() -> Path:
    import lerobot

    return Path(lerobot.__file__).resolve().parent


def _source_record(module_name: str, qualname: str) -> dict[str, Any]:
    mod = importlib.import_module(module_name)
    obj = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    path = Path(inspect.getfile(obj)).resolve()
    lines, start = inspect.getsourcelines(obj)
    return {
        "module": module_name,
        "qualname": qualname,
        "path": str(path),
        "line_start": int(start),
        "line_count": len(lines),
        "sha256": _sha256_file(path),
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


def _wall_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def _key_digest(key: tuple[Any, ...]) -> str:
    return hashlib.sha256(repr(key).encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--obs-index", type=int, default=0, help="Primary calib obs for NFE step pattern")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.physical_gpu))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    from load_policy import CALIB_PATH, LEROBOT_CKPT, load_policy

    inference_mod = importlib.import_module("lerobot.policies.molmoact2.molmoact2_hf_model.inference")
    ActionCudaGraphManager = inference_mod.ActionCudaGraphManager
    _cuda_graph_key = inference_mod._cuda_graph_key

    lr_root = _lerobot_root()
    try:
        lerobot_git = (
            subprocess.check_output(["git", "-C", str(lr_root.parent.parent), "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception as exc:
        lerobot_git = f"error:{exc}"

    sources = {
        "policy_reset": _source_record(
            "lerobot.policies.molmoact2.modeling_molmoact2", "MolmoAct2Policy.reset"
        ),
        "action_cuda_graph_manager": _source_record(
            "lerobot.policies.molmoact2.molmoact2_hf_model.inference", "ActionCudaGraphManager"
        ),
        "action_cuda_graph_manager_run": _source_record(
            "lerobot.policies.molmoact2.molmoact2_hf_model.inference", "ActionCudaGraphManager.run_action_flow"
        ),
        "cuda_graph_key": _source_record(
            "lerobot.policies.molmoact2.molmoact2_hf_model.inference", "_cuda_graph_key"
        ),
        "depth_decode_cache_reset": _source_record(
            "lerobot.policies.molmoact2.molmoact2_hf_model.inference", "_DepthDecodeStaticCache.reset"
        ),
    }

    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    samples = blob["samples"]

    def batch_at(i: int) -> dict:
        s = samples[i]
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in s["batch"].items()}

    by_prefix_len: dict[int, int] = {}
    for i, s in enumerate(samples[:32]):
        plen = int(s["batch"]["input_ids"].shape[1])
        by_prefix_len.setdefault(plen, i)

    policy = load_policy(str(device))
    policy.eval()
    policy.config.compile_model = False
    policy.config.gradient_checkpointing = False
    policy.config.n_action_steps = 10
    policy.config.enable_inference_cuda_graph = True
    policy._set_inference_cuda_graph_enabled(True)
    mgr = getattr(policy._backbone(), "action_cuda_graph_manager", None)
    if mgr is None:
        raise SystemExit("action_cuda_graph_manager missing on backbone")

    events: list[dict[str, Any]] = []
    orig_run = ActionCudaGraphManager.run_action_flow

    def patched_run(self, inputs, steps, run_loop):
        prev = self.action_flow_graph.key if self.action_flow_graph is not None else None
        key = _cuda_graph_key(inputs, steps)
        captured = self.action_flow_graph is None or self.action_flow_graph.key != key
        t0 = time.perf_counter()
        out = orig_run(self, inputs, steps, run_loop)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0
        events.append(
            {
                "phase": current_phase,
                "step_index": step_counter,
                "num_inference_steps": int(steps),
                "graph_key_digest": _key_digest(key),
                "prev_graph_key_digest": _key_digest(prev) if prev is not None else None,
                "captured": bool(captured),
                "replayed": not captured,
                "run_action_flow_ms": elapsed,
                "manager_id": id(self),
            }
        )
        return out

    ActionCudaGraphManager.run_action_flow = patched_run  # type: ignore[method-assign]

    current_phase = "nfe_step_pattern"
    step_counter = 0
    primary = batch_at(args.obs_index)
    gen = torch.Generator(device=device)
    gen.manual_seed(1000)

    step_runs = []
    for steps in STEPS_PATTERN:
        policy.reset()
        gen.manual_seed(1000)
        ms = _wall_ms(
            lambda st=steps: policy.predict_action_chunk(primary, num_steps=st, generator=gen)
        )
        step_runs.append(
            {
                "num_inference_steps": steps,
                "predict_action_chunk_ms": ms,
                "obs_index": args.obs_index,
                "obs_fingerprint": _tensor_fingerprint(primary),
                "input_ids_len": int(primary["input_ids"].shape[1]),
            }
        )
        step_counter += 1

    current_phase = "prefix_len_switch_steps10"
    prefix_runs = []
    for plen in sorted(by_prefix_len):
        idx = by_prefix_len[plen]
        batch = batch_at(idx)
        policy.reset()
        gen.manual_seed(1000)
        ms = _wall_ms(lambda b=batch: policy.predict_action_chunk(b, num_steps=10, generator=gen))
        prefix_runs.append(
            {
                "obs_index": idx,
                "input_ids_len": plen,
                "predict_action_chunk_ms": ms,
                "obs_fingerprint": _tensor_fingerprint(batch),
            }
        )
        step_counter += 1

    ActionCudaGraphManager.run_action_flow = orig_run  # type: ignore[method-assign]

    nfe_captures = sum(1 for e in events if e["phase"] == "nfe_step_pattern" and e["captured"])
    prefix_captures = sum(1 for e in events if e["phase"] == "prefix_len_switch_steps10" and e["captured"])

    conclusion = []
    if nfe_captures >= 2:
        conclusion.append(
            "Single-slot ActionCudaGraphManager recaptures when num_inference_steps changes "
            f"(pattern {list(STEPS_PATTERN)}: {nfe_captures}/{len(STEPS_PATTERN)} captures)."
        )
    if prefix_captures >= 2:
        conclusion.append(
            f"KV/context shape change across prefix lengths recaptured graph "
            f"({prefix_captures}/{len(prefix_runs)} captures at NFE=10)."
        )
    if not conclusion:
        conclusion.append(
            "No repeated recapture in this short probe; latency spikes may need full 32-obs sweep."
        )
    policy_reset_clears_graph = (
        "MolmoAct2Policy.reset clears action queue only (see source); it does not null action_flow_graph."
    )

    payload = {
        "protocol": "molmo_cuda_graph_short_diagnostic_v1",
        "utc": _utc(),
        "hostname": platform.node(),
        "physical_gpu_index": args.physical_gpu,
        "lerobot_checkout": str(lr_root.parent.parent),
        "lerobot_git_head": lerobot_git,
        "checkpoint": str(LEROBOT_CKPT),
        "calib_path": str(CALIB_PATH),
        "steps_pattern": list(STEPS_PATTERN),
        "sources": sources,
        "policy_reset_note": policy_reset_clears_graph,
        "nfe_step_pattern_runs": step_runs,
        "prefix_len_runs": prefix_runs,
        "graph_events": events,
        "conclusion": conclusion,
        "recommended_timed_protocol_if_shape_switch": (
            "Per-observation: warmup predict_action_chunk in place, then measure immediately; "
            "report first capture / key-switch separately; keep all 32 calib obs; pin CPU cores."
        ),
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "diagnostic.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"wrote": str(out_json), "conclusion": conclusion}, indent=2))


if __name__ == "__main__":
    main()

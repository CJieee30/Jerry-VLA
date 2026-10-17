#!/usr/bin/env python3
"""Compare 1–2 calibration samples against molmoact2_libero/score.py (GPU)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "molmoact2_libero"))

import torch

from molmo_score_runtime import (
    checkpoint_status,
    mem_snapshot,
    prepare_policy_for_scoring,
    score_frame_three,
)
from prospective_molmo_layer_common import CALIB_PATH, MOLMO_LAYER_DIR, N_LAYERS


def _rel_err(a, b) -> float:
    ta = torch.tensor(a, dtype=torch.float64)
    tb = torch.tensor(b, dtype=torch.float64)
    denom = tb.abs().clamp_min(1e-12)
    return float(((ta - tb).abs() / denom).max().item())


def _max_abs(a, b) -> float:
    ta = torch.tensor(a, dtype=torch.float64)
    tb = torch.tensor(b, dtype=torch.float64)
    return float((ta - tb).abs().max().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--with-gate", action="store_true", help="include GateProbe retain_grad (higher VRAM)")
    args = parser.parse_args()
    out = MOLMO_LAYER_DIR / "score_smoke_verify.json"
    log = MOLMO_LAYER_DIR / "score_smoke.log"
    MOLMO_LAYER_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "device": args.device,
        "n": args.n,
        "rtol": args.rtol,
        "atol": args.atol,
        "mem_process_start": mem_snapshot(),
        "checks": [],
        "passed": False,
        "with_gate": bool(args.with_gate),
    }

    try:
        if not CALIB_PATH.is_file():
            raise FileNotFoundError(CALIB_PATH)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda not available")

        from load_policy import load_policy, vlm_blocks
        from score import measure_gate, measure_grad

        blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
        samples = blob["samples"][: args.n]
        policy = load_policy(str(device))
        blocks, final_norm = vlm_blocks(policy)
        ckpt = prepare_policy_for_scoring(policy, blocks)
        record["checkpoint"] = ckpt
        record["mem_after_load"] = mem_snapshot()
        if not ckpt["effective"]:
            raise RuntimeError(f"checkpointing ineffective: {ckpt}")

        ours = {m: [] for m in ("igia", "fisher", "gateprobe")}
        peak = mem_snapshot()
        for sample in samples:
            scored = score_frame_three(
                policy, blocks, final_norm, sample, device, with_gate=bool(args.with_gate)
            )
            if args.with_gate and not scored["gate_hidden_requires_grad"]:
                raise RuntimeError("GateProbe hidden states lack requires_grad")
            if scored["n_params_with_grad"] <= 0:
                raise RuntimeError("no VLM parameter grads")
            ours["igia"].append(scored["igia"])
            ours["fisher"].append(scored["fisher"])
            if args.with_gate:
                ours["gateprobe"].append(scored["gateprobe"])
            peak = mem_snapshot()
        record["mem_after_ours"] = peak
        record["n_params_with_grad"] = scored["n_params_with_grad"]

        # Original: IGIA is sum over samples; Fisher/GateProbe/Taylor are means.
        ref_igia = measure_grad(policy, blocks, samples, device, "igia")
        ref_fisher = measure_grad(policy, blocks, samples, device, "fisher")
        record["mem_after_ref_grad"] = mem_snapshot()
        comparisons = [
            ("igia", torch.tensor(ours["igia"]).sum(0).tolist(), ref_igia, "sum over frames of per-frame Σg²"),
            (
                "fisher",
                torch.tensor(ours["fisher"]).mean(0).tolist(),
                ref_fisher,
                "mean over frames of per-frame Σ(gθ)²",
            ),
        ]
        if args.with_gate:
            ref_gate = measure_gate(policy, blocks, final_norm, samples, device)
            comparisons.append(
                (
                    "gateprobe",
                    torch.tensor(ours["gateprobe"]).mean(0).tolist(),
                    ref_gate,
                    "mean over frames of |grad_h·Δh|",
                )
            )
        ref_taylor = measure_grad(policy, blocks, samples, device, "taylor")
        record["mem_after_ref"] = mem_snapshot()
        record["checkpoint_after_ref"] = checkpoint_status(policy)
        failed = []
        for name, ours_v, ref_v, how in comparisons:
            if len(ref_v) != N_LAYERS:
                raise RuntimeError(f"{name} layers {len(ref_v)}")
            rel = _rel_err(ours_v, ref_v)
            abs_e = _max_abs(ours_v, ref_v)
            ok = rel <= args.rtol or abs_e <= args.atol
            record["checks"].append(
                {
                    "method": name,
                    "how": how,
                    "max_rel_err": rel,
                    "max_abs_err": abs_e,
                    "ref_sum": float(sum(ref_v)),
                    "ours_sum": float(sum(ours_v)),
                    "ok": ok,
                }
            )
            if not ok:
                failed.append(name)

        record["taylor_ref_sum"] = float(sum(ref_taylor))
        record["taylor_note"] = "original grouped Taylor path exercised; not per-frame Taylor_absmean"
        record["failed_methods"] = failed
        record["passed"] = not failed
        out.write_text(json.dumps(record, indent=2) + "\n")
        with log.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps({"passed": record["passed"], "checks": record["checks"]}, indent=2), flush=True)
        return 0 if record["passed"] else 2
    except Exception as e:
        record["error"] = str(e)
        record["traceback"] = traceback.format_exc()
        record["mem_on_fail"] = mem_snapshot()
        out.write_text(json.dumps(record, indent=2) + "\n")
        with log.open("a") as f:
            f.write(record["traceback"])
            f.write("\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""IGIA path diagnostic: fixed calib samples, per-frame rev2 vs score.py measure_grad."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "molmoact2_libero"))

import torch

from molmo_score_runtime import mem_snapshot, prepare_policy_for_scoring, score_frame_three
from prospective_molmo_layer_common import CALIB_PATH, N_LAYERS


def _layer_stats(a: list[float], b: list[float]) -> dict:
    ta = torch.tensor(a, dtype=torch.float64)
    tb = torch.tensor(b, dtype=torch.float64)
    rel = ((ta - tb).abs() / tb.abs().clamp_min(1e-12)).tolist()
    return {
        "max_abs": float((ta - tb).abs().max().item()),
        "max_rel": float(max(rel)),
        "per_layer_abs": [float(x) for x in (ta - tb).abs().tolist()],
        "per_layer_rel": rel,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    samples = blob["samples"][: args.n_samples]

    from load_policy import load_policy, vlm_blocks
    from score import measure_grad

    policy = load_policy(str(device))
    blocks, final_norm = vlm_blocks(policy)
    ckpt = prepare_policy_for_scoring(policy, blocks)

    record = {
        "device": str(device),
        "n_samples": args.n_samples,
        "repeats": args.repeats,
        "sample_indices": [int(s.get("index", i)) for i, s in enumerate(samples)],
        "checkpoint": ckpt,
        "policy_reload_note": "single load; repeats are back-to-back on same weights (noise/timesteps fixed per sample dict)",
        "runs": [],
        "repeatability_ours": [],
        "aggregate": {},
        "smoke_cross_ref": "score_smoke_verify.json rtol=1e-3, atol=1e-4",
    }

    all_ours = []
    all_ref = []
    ours_r0: list[list[float]] = []

    for rep in range(args.repeats):
        run = {"repeat": rep, "per_sample": []}
        for si, sample in enumerate(samples):
            ours_vec = score_frame_three(policy, blocks, final_norm, sample, device, with_gate=False)["igia"]
            ref_vec = measure_grad(policy, blocks, [sample], device, "igia")
            st = _layer_stats(ours_vec, ref_vec)
            st.update(
                {
                    "sample_index": int(sample.get("index", si)),
                    "ours_sum": float(sum(ours_vec)),
                    "ref_sum": float(sum(ref_vec)),
                    "ours": ours_vec,
                    "ref": ref_vec,
                }
            )
            run["per_sample"].append(st)
            if rep == 0:
                ours_r0.append(ours_vec)
            if rep == 1 and si < len(ours_r0):
                record["repeatability_ours"].append(
                    {
                        "sample_index": st["sample_index"],
                        **_layer_stats(ours_r0[si], ours_vec),
                        "identical": ours_r0[si] == ours_vec,
                    }
                )
            all_ours.append(ours_vec)
            all_ref.append(ref_vec)
        run["mem_end"] = mem_snapshot()
        record["runs"].append(run)

    ours_sum = [0.0] * N_LAYERS
    for v in all_ours:
        for i, x in enumerate(v):
            ours_sum[i] += x
    ref_sum = [0.0] * N_LAYERS
    for v in all_ref:
        for i, x in enumerate(v):
            ref_sum[i] += x
    agg = _layer_stats(ours_sum, ref_sum)
    agg.update(
        {
            "how": f"sum over {args.n_samples} samples × {args.repeats} back-to-back passes",
            "ours_sum_total": float(sum(ours_sum)),
            "ref_sum_total": float(sum(ref_sum)),
            "rtol_1e-3_pass": agg["max_rel"] <= 1e-3,
        }
    )
    record["aggregate"] = agg
    record["mem_after"] = mem_snapshot()
    if device.type == "cuda":
        record["mem_peak_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated())

    out_json = args.out_dir / "igia_path_diagnostic.json"
    out_json.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"out": str(out_json), "aggregate": record["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

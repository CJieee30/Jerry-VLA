#!/usr/bin/env python3
"""One backward per obs → IGIA + Fisher + GateProbe (per-frame). Taylor is separate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "molmoact2_libero"))

import torch
from tqdm import tqdm

from molmo_score_runtime import (
    checkpoint_status,
    mem_snapshot,
    prepare_policy_for_scoring,
    score_frame_three,
)
from prospective_molmo_layer_common import (
    CALIB_PATH,
    FRAME_METHODS,
    N_LAYERS,
    N_OBS,
    PROTOCOL_ID,
    SCORE_DIR,
    load_calibration_pool,
)


def _save(method: str, rows: list[list[float]], extra: dict) -> Path:
    dest = SCORE_DIR / f"molmoact2_{method}.pt"
    mat = torch.tensor(rows, dtype=torch.float32)
    if tuple(mat.shape) != (len(rows), N_LAYERS):
        raise RuntimeError(f"{method} shape {tuple(mat.shape)}")
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol": PROTOCOL_ID,
            "stack": "molmoact2",
            "method": method,
            "frame_scores": mat,
            "n_samples": int(mat.shape[0]),
            "n_layers": N_LAYERS,
            **extra,
        },
        dest,
    )
    dest.with_suffix(".json").write_text(
        json.dumps({"method": method, "shape": list(mat.shape)}, indent=2) + "\n"
    )
    return dest


def _frame_done() -> bool:
    return all((SCORE_DIR / f"molmoact2_{m}.pt").is_file() for m in FRAME_METHODS)


def _load_partial() -> dict[str, list[list[float]]] | None:
    path = SCORE_DIR / "_partial_frames.pt"
    if not path.is_file():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    rows = blob.get("rows")
    if not rows:
        return None
    n = len(rows["igia"])
    if n <= 0:
        return None
    return rows


def run_score(device: torch.device, samples: list, max_samples: int | None, log_path: Path) -> None:
    from load_policy import load_policy, vlm_blocks

    if max_samples is not None:
        samples = samples[:max_samples]

    policy = load_policy(str(device))
    blocks, final_norm = vlm_blocks(policy)
    if len(blocks) != N_LAYERS:
        raise RuntimeError(f"layers {len(blocks)} != {N_LAYERS}")
    ckpt = prepare_policy_for_scoring(policy, blocks)
    with log_path.open("a") as logf:
        logf.write(json.dumps({"checkpoint": ckpt, "mem_after_load": mem_snapshot()}) + "\n")
        logf.flush()

    rows = _load_partial() or {m: [] for m in FRAME_METHODS}
    start = len(rows["igia"])
    extra = {
        "definitions": {
            "igia": "per-frame sum g^2 per layer (Molmo orig IGIA, no 1/N)",
            "fisher": "per-frame sum (g*theta)^2 per layer (Molmo orig Fisher term before 1/N)",
            "gateprobe": "per-frame |grad_h·(h_out-h_in)|",
            "taylor_note": "use score_molmo_prospective_taylor_groups.py (not Taylor_absmean)",
        },
        "checkpoint": ckpt,
    }

    for i, sample in enumerate(tqdm(samples[start:], desc="molmo frame scores", initial=start, total=len(samples))):
        scored = score_frame_three(policy, blocks, final_norm, sample, device, with_gate=True)
        if not scored["gate_hidden_requires_grad"]:
            raise RuntimeError("GateProbe hidden states lack requires_grad")
        if scored["n_params_with_grad"] <= 0:
            raise RuntimeError("no VLM parameter grads")
        rows["igia"].append(scored["igia"])
        rows["fisher"].append(scored["fisher"])
        rows["gateprobe"].append(scored["gateprobe"])
        done = start + i + 1
        if done % 10 == 0 or done == len(samples):
            SCORE_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"rows": rows, "n": done, "checkpoint": ckpt}, SCORE_DIR / "_partial_frames.pt")
            with log_path.open("a") as logf:
                logf.write(json.dumps({"progress": done, "mem": mem_snapshot()}) + "\n")
                logf.flush()

    if max_samples is None and len(rows["igia"]) != N_OBS:
        raise RuntimeError(f"expected {N_OBS} frames, got {len(rows['igia'])}")
    for method in FRAME_METHODS:
        _save(method, rows[method], extra)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    log_path = SCORE_DIR.parent / "score_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if _frame_done() and not args.force and args.max_samples is None:
            print("[skip] frame score files exist", flush=True)
            return 0
        if not CALIB_PATH.is_file():
            raise FileNotFoundError(f"missing {CALIB_PATH}")

        blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
        samples = blob["samples"]
        pool = load_calibration_pool()
        for i, (s, m) in enumerate(zip(samples, pool, strict=True)):
            if int(s["pool_index"]) != m.pool_index:
                raise RuntimeError(f"row {i} pool_index mismatch")

        device = torch.device(args.device)
        with log_path.open("a") as logf:
            logf.write(f"start device={device} n={len(samples)} max={args.max_samples}\n")
            logf.write(json.dumps({"mem_start": mem_snapshot()}) + "\n")

        run_score(device, samples, args.max_samples, log_path)

        if args.max_samples is None:
            for method in FRAME_METHODS:
                print(f"saved {SCORE_DIR / f'molmoact2_{method}.pt'}", flush=True)
        with log_path.open("a") as logf:
            logf.write("done\n")
        return 0
    except Exception:
        with log_path.open("a") as logf:
            logf.write("FAILED\n")
            logf.write(traceback.format_exc())
            logf.write(json.dumps({"mem_on_fail": mem_snapshot()}) + "\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

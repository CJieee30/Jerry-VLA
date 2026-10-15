#!/usr/bin/env python3
"""Compare manual native_to_physical vs official Molmo postprocessor (CPU, small batch)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "molmoact2_libero"))

from dev_metrics_common import MODEL_DIRS, native_to_physical
from protocol import OUT_DIR


def official_physical_batch(native: np.ndarray) -> np.ndarray:
    from load_policy import load_policy_and_processors

    policy, _pre, post, _cfg = load_policy_and_processors()
    # native: (B, 10, 7) policy-normalized actions
    t = torch.from_numpy(native.astype(np.float32))
    out = []
    for i in range(t.shape[0]):
        chunk = t[i : i + 1]
        phys = post(chunk)
        out.append(phys.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    z = np.load(MODEL_DIRS["molmoact2"] / "actions.npz")
    idx = [0, 1, 2, 47]
    batch = np.stack([z["nfe1"][i, 0] for i in idx], axis=0)
    manual = native_to_physical("molmoact2", batch)
    try:
        official = official_physical_batch(batch)
        max_abs = float(np.max(np.abs(manual - official)))
        status = "pass" if max_abs < 1e-4 else "warn"
    except Exception as e:
        max_abs = None
        status = f"error:{e}"
        official = None
    out = {
        "model": "molmoact2",
        "n_samples": len(idx),
        "batch_shape": list(batch.shape),
        "manual_vs_official_max_abs": max_abs,
        "status": status,
    }
    path = OUT_DIR / "dev_three_model_screening" / "molmo_official_postprocessor_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

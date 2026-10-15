#!/usr/bin/env python3
"""Official Molmo postprocessor vs manual native_to_physical on existing probe actions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MOLMO_LIB = HERE.parent / "molmoact2_libero"
if not (MOLMO_LIB / "load_policy.py").is_file():
    MOLMO_LIB = HERE.parents[1] / "molmoact2_libero"
sys.path.insert(0, str(MOLMO_LIB))

from dev_metrics_common import MODEL_DIRS, libero_mean_std, molmo_quantile_stats, native_to_physical
from protocol import OUT_DIR


def load_postprocessor_only():
    import lerobot.policies.molmoact2.modeling_molmoact2 as pol
    from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors_from_pretrained
    from load_policy import LEROBOT_CKPT, OUT_DIR as MOLMO_OUT, install_loader_patches

    install_loader_patches()
    cfg = pol.MolmoAct2Config.from_pretrained(MOLMO_OUT / "policy_config")
    _pre, post = make_molmoact2_pre_post_processors_from_pretrained(
        cfg, str(LEROBOT_CKPT), preprocessor_overrides={"device_processor": {"device": "cpu"}}
    )
    return post


def official_physical(native: np.ndarray, post) -> np.ndarray:
    """native (B,10,7) policy-normalized → physical via official postprocessor pipeline."""
    out = []
    for i in range(native.shape[0]):
        t = torch.from_numpy(native[i].astype(np.float32))
        act = post(t)
        if isinstance(act, torch.Tensor):
            out.append(act.detach().cpu().numpy())
        else:
            out.append(np.asarray(act))
    return np.stack(out, axis=0)


def main() -> None:
    if str(MOLMO_LIB) not in sys.path:
        sys.path.insert(0, str(MOLMO_LIB))
    z = np.load(MODEL_DIRS["molmoact2"] / "actions.npz")
    samples = [{"shared_index": si, "noise_id": 0} for si in (0, 1, 2, 47, 100)]
    batch = np.stack([z["nfe1"][s["shared_index"], s["noise_id"]] for s in samples], axis=0)
    manual = native_to_physical("molmoact2", batch)
    post = load_postprocessor_only()
    official = official_physical(batch, post)
    max_abs = float(np.max(np.abs(manual - official)))
    mean, std = libero_mean_std()
    q01, q99, mask = molmo_quantile_stats()
    out = {
        "model": "molmoact2",
        "samples": samples,
        "batch_shape": list(batch.shape),
        "manual_vs_official_max_abs": max_abs,
        "status": "pass" if max_abs < 1e-3 else "fail",
        "libero_mean": mean.tolist(),
        "libero_std": std.tolist(),
        "molmo_q01": q01.tolist(),
        "molmo_q99": q99.tolist(),
        "molmo_mask": mask.tolist(),
    }
    path = OUT_DIR / "dev_three_model_screening" / "molmo_official_postprocessor_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    if out["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

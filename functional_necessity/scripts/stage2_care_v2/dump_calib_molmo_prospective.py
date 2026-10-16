#!/usr/bin/env python3
"""Dump MolmoAct2 calibration tensors for prospective 400-obs pool (CPU)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

from dump_calib import _cpu_tensors, _rel_index
from prospective_molmo_layer_common import (
    CALIB_DRAW_SEED,
    CALIB_PATH,
    MOLMO_LAYER_DIR,
    N_OBS,
    load_calibration_pool,
    molmo_root_ready,
    verify_no_dev_overlap,
)
from protocol import MOLMO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not molmo_root_ready():
        raise SystemExit(f"molmo root not ready: {MOLMO_ROOT}")
    overlap = verify_no_dev_overlap()
    if not overlap["ok"]:
        raise SystemExit(f"episode overlap with dev pool: {overlap}")
    MOLMO_LAYER_DIR.mkdir(parents=True, exist_ok=True)
    if CALIB_PATH.is_file() and not args.force:
        blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
        if int(blob.get("n", 0)) == N_OBS:
            print(f"[skip] {CALIB_PATH}", flush=True)
            return

    pool = load_calibration_pool()
    sys.path.insert(0, str(HERE.parents[0] / "molmoact2_libero"))
    from load_policy import HF_CONFIG, LEROBOT_CKPT, OUT_DIR as MOLMO_OUT, install_loader_patches
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.molmoact2.processor_molmoact2 import (
        make_molmoact2_pre_post_processors_from_pretrained,
    )
    import lerobot.policies.molmoact2.modeling_molmoact2 as pol

    install_loader_patches()
    cfg = pol.MolmoAct2Config.from_pretrained(MOLMO_OUT / "policy_config")
    cfg.device = "cpu"
    cfg.checkpoint_path = str(HF_CONFIG)
    preprocessor, _ = make_molmoact2_pre_post_processors_from_pretrained(
        cfg,
        str(LEROBOT_CKPT),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    eps = sorted({m.episode_index for m in pool})
    dataset = LeRobotDataset(
        "allenai/MolmoAct2-LIBERO-Dataset",
        root=str(MOLMO_ROOT),
        episodes=eps,
        revision="main",
        download_videos=False,
    )
    mapping = dataset.absolute_to_relative_idx or {}
    missing = [m.global_index for m in pool if m.global_index not in mapping]
    if missing:
        raise SystemExit(
            f"molmo mapping missing {len(missing)}/{N_OBS} global indices "
            f"(e.g. {missing[:5]}); fetch shards per split_summary"
        )

    samples = []
    for m in pool:
        rel = _rel_index(dataset, m.global_index)
        frame = dataset[rel]
        batch = preprocessor(frame)
        action = batch["action"]
        gen = torch.Generator().manual_seed(CALIB_DRAW_SEED + m.pool_index)
        noise = torch.randn(action.shape[0], 1, action.shape[1], action.shape[2], generator=gen)
        timesteps = torch.empty(action.shape[0], 1).uniform_(0.05, 0.95, generator=gen)
        samples.append(
            {
                "pool_index": m.pool_index,
                "global_index": m.global_index,
                "episode_index": m.episode_index,
                "task_index": m.task_index,
                "batch": _cpu_tensors(batch),
                "noise": noise.cpu(),
                "timesteps": timesteps.cpu(),
            }
        )
        if (m.pool_index + 1) % 50 == 0 or m.pool_index == 0:
            print(f"molmo prospective calib {m.pool_index + 1}/{N_OBS}", flush=True)

    payload = {
        "n": len(samples),
        "stack": "molmoact2",
        "protocol": "prospective_layer_calib_v1",
        "run_name": "prospective_v1_20261015",
        "seed_base": CALIB_DRAW_SEED,
        "overlap_check": overlap,
        "samples": samples,
    }
    torch.save(payload, CALIB_PATH)
    (MOLMO_LAYER_DIR / "calib_manifest.json").write_text(
        __import__("json").dumps(
            {
                "path": str(CALIB_PATH),
                "n": N_OBS,
                "episodes": len(eps),
                "overlap_check": overlap,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {CALIB_PATH}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dump stack-specific CARE v2 calibration tensors for the shared 480 frames.

Does not load full policy weights. Frozen (noise, time) is seed=42 + frame index.
"""

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
import torch.nn.functional as F

from protocol import LIBERO_REPO, MOLMO_ROOT, N_FRAMES, OUT_DIR, SEED, STACKS


def _load_shared(path: Path | None = None) -> dict:
    path = path or (OUT_DIR / "shared.pt")
    if not path.is_file():
        raise SystemExit(f"missing shared {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _rel_index(dataset, global_index: int) -> int:
    mapping = getattr(dataset, "absolute_to_relative_idx", None)
    if mapping is None and hasattr(dataset, "reader"):
        mapping = dataset.reader.absolute_to_relative_idx
    if mapping:
        return int(mapping[int(global_index)])
    return int(global_index)


def _seeded_noise_time(shape, bsize: int, i: int, *, beta_a=1.5, beta_b=1.0, scale=0.999, offset=0.001):
    gen = torch.Generator().manual_seed(SEED + i)
    noise = torch.randn(shape, generator=gen)
    u = torch.rand((bsize,), generator=gen).clamp(1e-6, 1 - 1e-6)
    # Inverse-CDF is awkward for Beta(1.5,1). Draw Gamma mix with the same generator.
    # Beta(1.5,1) = Gamma(1.5) / (Gamma(1.5)+Gamma(1)).
    z = torch.randn((bsize,), generator=gen)
    u1 = torch.rand((bsize,), generator=gen).clamp_min(1e-12)
    u2 = torch.rand((bsize,), generator=gen).clamp_min(1e-12)
    gamma_a = z.square() * 0.5 + (-torch.log(u1))  # Gamma(1.5) ≈ 0.5*chi2(1) + Exp
    gamma_b = -torch.log(u2)
    time = (gamma_a / (gamma_a + gamma_b)) * scale + offset
    del u
    return noise, time.to(dtype=torch.float32)


def _cpu_tensors(batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu()
        else:
            out[key] = value
    return out


def _skip_if_done(dest: Path, n: int) -> bool:
    if not dest.is_file():
        return False
    blob = torch.load(dest, map_location="cpu", weights_only=False)
    if int(blob.get("n", 0)) == n:
        print(f"[skip] {dest}", flush=True)
        return True
    return False


def dump_pi05(shared: dict, dest_dir: Path) -> None:
    dest = dest_dir / "calib_pi05.pt"
    if _skip_if_done(dest, len(shared["records"])):
        return
    sys.path.insert(0, str(HERE.parents[0] / "pi05_libero"))
    from common import (
        DEFAULT_CHECKPOINT,
        load_policy_config,
        load_preprocessor,
        preprocess_raw_batch,
        resolve_dataset_root,
        tensor_to_cpu_store,
    )
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = resolve_dataset_root(LIBERO_REPO)
    config = load_policy_config(DEFAULT_CHECKPOINT)
    ds_meta = LeRobotDatasetMetadata(LIBERO_REPO, root=root)
    delta = resolve_delta_timestamps(config, ds_meta)
    dataset = LeRobotDataset(
        LIBERO_REPO,
        delta_timestamps=delta,
        return_uint8=True,
        root=root,
        download_videos=False,
    )
    preprocessor = load_preprocessor(DEFAULT_CHECKPOINT, device="cpu", dataset_stats=dataset.meta.stats)
    camera_keys = list(dataset.meta.camera_keys)
    max_dim = int(config.max_action_dim)
    samples = []
    for i, rec in enumerate(shared["records"]):
        gidx = int(rec["global_index"])
        frame = dataset[gidx]
        raw = {}
        for key, value in frame.items():
            if torch.is_tensor(value):
                raw[key] = value.unsqueeze(0)
            else:
                raw[key] = [value]
        processed = tensor_to_cpu_store(preprocess_raw_batch(raw, preprocessor, camera_keys))
        action = processed["action"]
        if action.shape[-1] < max_dim:
            action = F.pad(action, (0, max_dim - int(action.shape[-1])))
        noise, time = _seeded_noise_time(tuple(action.shape), int(action.shape[0]), i)
        samples.append({"global_index": gidx, "batch": processed, "noise": noise, "time": time})
        if (i + 1) % 40 == 0 or i == 0:
            print(f"pi05 calib {i+1}/{len(shared['records'])}", flush=True)
    torch.save({"n": len(samples), "stack": "pi05", "seed": SEED, "samples": samples}, dest)
    print(f"wrote {dest}", flush=True)


def dump_smolvla(shared: dict, dest_dir: Path) -> None:
    dest = dest_dir / "calib_smolvla.pt"
    if _skip_if_done(dest, len(shared["records"])):
        return
    sys.path.insert(0, str(HERE.parents[0] / "smolvla_libero"))
    sys.path.insert(0, str(HERE.parents[0] / "pi05_libero"))
    from common import resolve_dataset_root
    from load_policy import CKPT, DATASET_REPO, load_preprocessor
    from score import _ensure_policy_batch, _frozen_noise_time
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    root = resolve_dataset_root(DATASET_REPO)
    cfg = SmolVLAConfig.from_pretrained(CKPT)
    ds_meta = LeRobotDatasetMetadata(DATASET_REPO, root=root)
    delta = resolve_delta_timestamps(cfg, ds_meta)
    dataset = LeRobotDataset(
        DATASET_REPO,
        delta_timestamps=delta,
        download_videos=False,
        video_backend="pyav",
        return_uint8=False,
        root=root,
    )
    preprocessor = load_preprocessor("cpu")
    samples = []
    for i, rec in enumerate(shared["records"]):
        gidx = int(rec["global_index"])
        frame = dataset[gidx]
        batch = _ensure_policy_batch(preprocessor(frame))
        gen = torch.Generator().manual_seed(SEED + i)
        noise, time = _frozen_noise_time(batch["action"], gen, action_dim=int(cfg.max_action_dim))
        samples.append(
            {
                "global_index": gidx,
                "batch": _cpu_tensors(batch),
                "noise": noise.cpu(),
                "time": time.cpu(),
            }
        )
        if (i + 1) % 40 == 0 or i == 0:
            print(f"smolvla calib {i+1}/{len(shared['records'])}", flush=True)
    torch.save({"n": len(samples), "stack": "smolvla", "seed": SEED, "samples": samples}, dest)
    print(f"wrote {dest}", flush=True)


def dump_molmo(shared: dict, dest_dir: Path) -> None:
    dest = dest_dir / "calib_molmoact2.pt"
    if _skip_if_done(dest, len(shared["records"])):
        return
    if not (MOLMO_ROOT / "READY").is_file():
        raise SystemExit("run fetch_molmo.py first")
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
    eps = sorted({int(r["episode_index"]) for r in shared["records"]})
    dataset = LeRobotDataset(
        "allenai/MolmoAct2-LIBERO-Dataset",
        root=str(MOLMO_ROOT),
        episodes=eps,
        revision="main",
        download_videos=False,
    )
    mapping = dataset.absolute_to_relative_idx or {}
    missing = [
        int(row["global_index"])
        for row in shared["records"]
        if int(row["global_index"]) not in mapping
    ]
    if missing:
        raise SystemExit(
            f"molmo mapping missing {len(missing)}/{len(shared['records'])} "
            f"indices e.g. {missing[:5]}; fetch file_index+1 shards"
        )
    samples = []
    for i, rec in enumerate(shared["records"]):
        gidx = int(rec["global_index"])
        rel = _rel_index(dataset, gidx)
        frame = dataset[rel]
        batch = preprocessor(frame)
        action = batch["action"]
        gen = torch.Generator().manual_seed(SEED + i)
        noise = torch.randn(action.shape[0], 1, action.shape[1], action.shape[2], generator=gen)
        timesteps = torch.empty(action.shape[0], 1).uniform_(0.05, 0.95, generator=gen)
        samples.append(
            {
                "global_index": gidx,
                "batch": _cpu_tensors(batch),
                "noise": noise.cpu(),
                "timesteps": timesteps.cpu(),
            }
        )
        if (i + 1) % 40 == 0 or i == 0:
            print(f"molmo calib {i+1}/{len(shared['records'])}", flush=True)
    torch.save({"n": len(samples), "stack": "molmoact2", "seed": SEED, "samples": samples}, dest)
    print(f"wrote {dest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", choices=STACKS, required=True)
    parser.add_argument("--shared", type=Path, default=None)
    parser.add_argument("--dest-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    dest_dir = args.dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    shared = _load_shared(args.shared)
    if len(shared["records"]) != N_FRAMES:
        raise SystemExit(f"shared n={len(shared['records'])}")
    if args.stack == "pi05":
        dump_pi05(shared, dest_dir)
    elif args.stack == "smolvla":
        dump_smolvla(shared, dest_dir)
    else:
        dump_molmo(shared, dest_dir)


if __name__ == "__main__":
    main()

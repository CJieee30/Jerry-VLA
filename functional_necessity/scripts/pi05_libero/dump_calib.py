#!/usr/bin/env python3
"""Dump 64×8 calibration batches (seed 42) after the v044 preprocessor."""

from __future__ import annotations

import sys
from pathlib import Path as _ScriptDir

sys.path.insert(0, str(_ScriptDir(__file__).resolve().parent))

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CALIB_FILE,
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    DEFAULT_NUM_BATCHES,
    DEFAULT_SEED,
    checkpoint_needs_libero_adapt,
    init_logging,
    load_policy_config,
    load_preprocessor,
    preprocess_raw_batch,
    resolve_dataset_root,
    set_seed,
    tensor_to_cpu_store,
)


def _episode_index_column(episodes, name: str):
    column = episodes[name]
    if hasattr(column, "to_list"):
        return column.to_list()
    return list(column)


def build_dataset(
    checkpoint: str,
    dataset_repo_id: str,
    video_backend: str | None,
    dataset_root: str | None,
):
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    config = load_policy_config(checkpoint)
    root = resolve_dataset_root(dataset_repo_id, dataset_root)
    ds_meta = LeRobotDatasetMetadata(dataset_repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(config, ds_meta)
    kwargs: dict = {
        "delta_timestamps": delta_timestamps,
        "return_uint8": True,
        "root": root,
    }
    if video_backend:
        kwargs["video_backend"] = video_backend
    dataset = LeRobotDataset(dataset_repo_id, **kwargs)
    return dataset


def build_loader(
    dataset,
    batch_size: int,
    seed: int,
    num_workers: int,
    episode_indices_to_use: list[int] | None = None,
) -> DataLoader:
    from lerobot.datasets.sampler import EpisodeAwareSampler
    from lerobot.utils.collate import lerobot_collate_fn

    sampler = EpisodeAwareSampler(
        _episode_index_column(dataset.meta.episodes, "dataset_from_index"),
        _episode_index_column(dataset.meta.episodes, "dataset_to_index"),
        episode_indices_to_use=dataset.episodes if episode_indices_to_use is None else episode_indices_to_use,
        drop_n_last_frames=0,
        shuffle=True,
        seed=seed,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator,
        drop_last=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET)
    parser.add_argument("--num-batches", type=int, default=DEFAULT_NUM_BATCHES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--video-backend",
        default=None,
        help="Video decoder. Default: LeRobot auto (torchcodec if importable, else pyav).",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Local LeRobot dataset root. Default: resolve from HF_LEROBOT_HOME / hub snapshots.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_CALIB_FILE)
    args = parser.parse_args()

    init_logging()
    set_seed(args.seed)
    logging.info(
        "Dumping %dx%d calib from %s (seed=%d)",
        args.num_batches,
        args.batch_size,
        args.dataset_repo_id,
        args.seed,
    )

    dataset = build_dataset(
        args.checkpoint,
        args.dataset_repo_id,
        args.video_backend,
        args.dataset_root,
    )
    loader = build_loader(dataset, args.batch_size, args.seed, args.num_workers)
    preprocessor = load_preprocessor(
        args.checkpoint,
        device="cpu",
        dataset_stats=dataset.meta.stats,
    )
    camera_keys = list(dataset.meta.camera_keys)

    batches = []
    for i, raw in enumerate(loader):
        if i >= args.num_batches:
            break
        processed = preprocess_raw_batch(raw, preprocessor, camera_keys)
        batches.append(tensor_to_cpu_store(processed))
        if (i + 1) % 8 == 0 or i == 0:
            logging.info("dumped batch %d/%d", i + 1, args.num_batches)

    if len(batches) < args.num_batches:
        raise RuntimeError(
            f"Only dumped {len(batches)}/{args.num_batches} batches; dataset/sampler is too short"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "calib_data": batches,
            "checkpoint": args.checkpoint,
            "dataset_repo_id": args.dataset_repo_id,
            "seed": args.seed,
            "num_batches": len(batches),
            "batch_size": args.batch_size,
            "libero_adapt": checkpoint_needs_libero_adapt(args.checkpoint),
        },
        args.out,
    )
    logging.info("saved %d batches -> %s", len(batches), args.out)


if __name__ == "__main__":
    main()

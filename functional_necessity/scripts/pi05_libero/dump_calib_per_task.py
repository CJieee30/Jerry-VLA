#!/usr/bin/env python3
"""Dump 16×8 calibration batches per LIBERO task (seed 42) after the v044 preprocessor.

Writes one atomic `{short}.pt` per task under results/calib/per_task_16x8_seed42/
so Fisher / greedy PPL extra-3 jobs can start as soon as the first task lands.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import logging
from collections import defaultdict

import pandas as pd
import pyarrow.dataset as ds
import torch

from common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    DEFAULT_SEED,
    checkpoint_needs_libero_adapt,
    init_logging,
    load_preprocessor,
    preprocess_raw_batch,
    resolve_dataset_root,
    set_seed,
    tensor_to_cpu_store,
)
from drop6_extra3_protocol import (
    DEFAULT_CALIB_DIR,
    PER_TASK_BATCH_SIZE,
    PER_TASK_NUM_BATCHES,
    calib_path,
    dump_json,
    load_task_catalog,
)
from dump_calib import build_dataset, build_loader


def episode_task_index_map(dataset_root: str | Path) -> dict[int, int]:
    table = ds.dataset(str(Path(dataset_root) / "data"), format="parquet").to_table(
        columns=["episode_index", "task_index"]
    )
    mapping: dict[int, int] = {}
    for ep, ti in zip(table.column("episode_index").to_pylist(), table.column("task_index").to_pylist()):
        mapping[int(ep)] = int(ti)
    return mapping


def language_to_task_index(dataset_root: str | Path) -> dict[str, int]:
    tasks = pd.read_parquet(Path(dataset_root) / "meta" / "tasks.parquet")
    out: dict[str, int] = {}
    for idx, row in tasks.iterrows():
        out[str(idx)] = int(row["task_index"])
    return out


def dump_one_task(
    *,
    dataset,
    preprocessor,
    camera_keys: list[str],
    task: dict,
    episode_indices: list[int],
    out_path: Path,
    num_batches: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    checkpoint: str,
    dataset_repo_id: str,
) -> None:
    loader = build_loader(
        dataset,
        batch_size,
        seed,
        num_workers,
        episode_indices_to_use=episode_indices,
    )
    batches = []
    for i, raw in enumerate(loader):
        if i >= num_batches:
            break
        processed = preprocess_raw_batch(raw, preprocessor, camera_keys)
        batches.append(tensor_to_cpu_store(processed))
        if (i + 1) % 4 == 0 or i == 0:
            logging.info("%s dumped batch %d/%d", task["short"], i + 1, num_batches)
    if len(batches) < num_batches:
        raise RuntimeError(
            f"{task['short']}: only dumped {len(batches)}/{num_batches} batches "
            f"from {len(episode_indices)} episodes"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(
        {
            "calib_data": batches,
            "suite": task["suite"],
            "task_id": task["task_id"],
            "task": task["task"],
            "short": task["short"],
            "episode_indices": episode_indices,
            "n_episodes": len(episode_indices),
            "checkpoint": checkpoint,
            "dataset_repo_id": dataset_repo_id,
            "seed": seed,
            "num_batches": len(batches),
            "batch_size": batch_size,
            "libero_adapt": checkpoint_needs_libero_adapt(checkpoint),
        },
        tmp,
    )
    tmp.replace(out_path)
    logging.info("saved %s (%d batches, %d episodes) -> %s", task["short"], len(batches), len(episode_indices), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET)
    parser.add_argument("--num-batches", type=int, default=PER_TASK_NUM_BATCHES)
    parser.add_argument("--batch-size", type=int, default=PER_TASK_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CALIB_DIR)
    args = parser.parse_args()

    init_logging()
    set_seed(args.seed)
    catalog = load_task_catalog()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        args.checkpoint,
        args.dataset_repo_id,
        args.video_backend,
        args.dataset_root,
    )
    dataset_root = resolve_dataset_root(args.dataset_repo_id, args.dataset_root) or dataset.root
    ep_to_ti = episode_task_index_map(dataset_root)
    lang_to_ti = language_to_task_index(dataset_root)
    ti_to_eps: dict[int, list[int]] = defaultdict(list)
    for ep, ti in sorted(ep_to_ti.items()):
        ti_to_eps[ti].append(ep)

    preprocessor = load_preprocessor(
        args.checkpoint,
        device="cpu",
        dataset_stats=dataset.meta.stats,
    )
    camera_keys = list(dataset.meta.camera_keys)
    logging.info(
        "Dumping %d tasks × %dx%d from %s (seed=%d) -> %s",
        len(catalog),
        args.num_batches,
        args.batch_size,
        args.dataset_repo_id,
        args.seed,
        out_dir,
    )

    index: list[dict] = []
    for task in catalog:
        if task["task"] not in lang_to_ti:
            raise KeyError(f"eval task not in dataset: {task['task']!r}")
        ti = lang_to_ti[task["task"]]
        episodes = ti_to_eps.get(ti, [])
        if len(episodes) < 1:
            raise RuntimeError(f"{task['short']}: no episodes for dataset task_index={ti}")
        path = calib_path(out_dir, task["short"])
        rec = {
            "short": task["short"],
            "suite": task["suite"],
            "task_id": task["task_id"],
            "task": task["task"],
            "dataset_task_index": ti,
            "n_episodes": len(episodes),
            "path": str(path),
        }
        if path.is_file() and path.stat().st_size > 1024:
            logging.info("[skip] %s already dumped (%s)", task["short"], path)
        else:
            tmp = path.with_suffix(path.suffix + ".tmp")
            if tmp.exists():
                tmp.unlink()
            dump_one_task(
                dataset=dataset,
                preprocessor=preprocessor,
                camera_keys=camera_keys,
                task=task,
                episode_indices=episodes,
                out_path=path,
                num_batches=args.num_batches,
                batch_size=args.batch_size,
                seed=args.seed,
                num_workers=args.num_workers,
                checkpoint=args.checkpoint,
                dataset_repo_id=args.dataset_repo_id,
            )
        rec["dumped"] = True
        index.append(rec)
        dump_json({"tasks": index, "n_done": len(index), "n_total": len(catalog)}, out_dir / "index.json")

    dump_json(
        {
            "checkpoint": args.checkpoint,
            "dataset_repo_id": args.dataset_repo_id,
            "seed": args.seed,
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "n_tasks": len(index),
            "tasks": index,
        },
        out_dir / "manifest.json",
    )
    logging.info("done: %d/%d task calib files in %s", len(index), len(catalog), out_dir)


if __name__ == "__main__":
    main()

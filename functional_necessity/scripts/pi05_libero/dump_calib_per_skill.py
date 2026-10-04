#!/usr/bin/env python3
"""Dump 32×8 skill-pooled calib batches from AtomicVLA interior frames.

Samples only frames that survive the P0 boundary-margin filter. Does not reuse
per-task 16×8 dumps (those have no frame-level verb). Seed 42 is the primary
pool; seed 43 is the stability repeat.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    checkpoint_needs_libero_adapt,
    init_logging,
    load_preprocessor,
    preprocess_raw_batch,
    set_seed,
    tensor_to_cpu_store,
)
from drop6_extra3_protocol import (
    ATOMIC_SKILLS,
    DEFAULT_SKILL_INDEX_DIR,
    SKILL_BATCH_SIZE,
    SKILL_BOUNDARY_MARGIN,
    SKILL_NUM_BATCHES,
    dump_json,
    skill_calib_dir,
)
from dump_calib import build_dataset


class FiniteIndexSampler(Sampler[int]):
    def __init__(self, indices: list[int]):
        self.indices = [int(i) for i in indices]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def sample_indices(source: np.ndarray, n_samples: int, seed: int) -> tuple[list[int], bool]:
    source = [int(x) for x in source.tolist()]
    if not source:
        raise RuntimeError("skill index is empty")
    rng = random.Random(seed)
    if len(source) < n_samples:
        chosen = [source[rng.randrange(len(source))] for _ in range(n_samples)]
        return chosen, True
    return rng.sample(source, n_samples), False


def build_index_loader(dataset, indices: list[int], batch_size: int, seed: int, num_workers: int) -> DataLoader:
    from lerobot.utils.collate import lerobot_collate_fn

    sampler = FiniteIndexSampler(indices)
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


def dump_one_skill(
    *,
    dataset,
    preprocessor,
    camera_keys: list[str],
    verb: str,
    sampled_indices: list[int],
    oversampled: bool,
    n_source: int,
    out_path: Path,
    num_batches: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    checkpoint: str,
    dataset_repo_id: str,
    index_path: Path,
    margin: int,
) -> None:
    loader = build_index_loader(dataset, sampled_indices, batch_size, seed, num_workers)
    batches = []
    for i, raw in enumerate(loader):
        if i >= num_batches:
            break
        processed = preprocess_raw_batch(raw, preprocessor, camera_keys)
        batches.append(tensor_to_cpu_store(processed))
        if (i + 1) % 8 == 0 or i == 0:
            logging.info("%s dumped batch %d/%d", verb, i + 1, num_batches)
    if len(batches) < num_batches:
        raise RuntimeError(f"{verb}: only dumped {len(batches)}/{num_batches} batches from {len(sampled_indices)} indices")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(
        {
            "calib_data": batches,
            "skill": verb,
            "sampled_indices": sampled_indices,
            "n_source_frames": n_source,
            "oversampled": oversampled,
            "checkpoint": checkpoint,
            "dataset_repo_id": dataset_repo_id,
            "seed": seed,
            "num_batches": len(batches),
            "batch_size": batch_size,
            "boundary_margin": margin,
            "index_path": str(index_path),
            "libero_adapt": checkpoint_needs_libero_adapt(checkpoint),
        },
        tmp,
    )
    tmp.replace(out_path)
    logging.info(
        "saved %s (%d batches, source=%d, oversampled=%s) -> %s",
        verb,
        len(batches),
        n_source,
        oversampled,
        out_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET)
    parser.add_argument("--num-batches", type=int, default=SKILL_NUM_BATCHES)
    parser.add_argument("--batch-size", type=int, default=SKILL_BATCH_SIZE)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_SKILL_INDEX_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skills", nargs="+", default=list(ATOMIC_SKILLS))
    args = parser.parse_args()

    init_logging()
    set_seed(args.seed)
    out_dir = Path(args.out_dir) if args.out_dir is not None else skill_calib_dir(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = args.num_batches * args.batch_size

    dataset = build_dataset(
        args.checkpoint,
        args.dataset_repo_id,
        args.video_backend,
        args.dataset_root,
    )
    preprocessor = load_preprocessor(
        args.checkpoint,
        device="cpu",
        dataset_stats=dataset.meta.stats,
    )
    camera_keys = list(dataset.meta.camera_keys)
    coverage = json_coverage(args.index_dir)

    index_rows = []
    for verb in args.skills:
        if verb not in ATOMIC_SKILLS:
            raise ValueError(f"unknown skill {verb}; expected one of {ATOMIC_SKILLS}")
        index_path = Path(coverage["index_paths"][verb])
        source = np.load(index_path)
        sampled, oversampled = sample_indices(source, n_samples, args.seed)
        path = out_dir / f"{verb}.pt"
        rec = {
            "skill": verb,
            "n_source_frames": int(source.shape[0]),
            "n_sampled": len(sampled),
            "oversampled": oversampled,
            "path": str(path),
            "index_path": str(index_path),
        }
        if path.is_file() and path.stat().st_size > 1024:
            logging.info("[skip] %s already dumped (%s)", verb, path)
        else:
            tmp = path.with_suffix(path.suffix + ".tmp")
            if tmp.exists():
                tmp.unlink()
            dump_one_skill(
                dataset=dataset,
                preprocessor=preprocessor,
                camera_keys=camera_keys,
                verb=verb,
                sampled_indices=sampled,
                oversampled=oversampled,
                n_source=int(source.shape[0]),
                out_path=path,
                num_batches=args.num_batches,
                batch_size=args.batch_size,
                seed=args.seed,
                num_workers=args.num_workers,
                checkpoint=args.checkpoint,
                dataset_repo_id=args.dataset_repo_id,
                index_path=index_path,
                margin=int(coverage.get("margin", SKILL_BOUNDARY_MARGIN)),
            )
        rec["dumped"] = True
        index_rows.append(rec)
        dump_json({"seed": args.seed, "skills": index_rows}, out_dir / "index.json")

    dump_json(
        {
            "checkpoint": args.checkpoint,
            "dataset_repo_id": args.dataset_repo_id,
            "seed": args.seed,
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "boundary_margin": coverage.get("margin", SKILL_BOUNDARY_MARGIN),
            "index_dir": str(args.index_dir),
            "n_skills": len(index_rows),
            "skills": index_rows,
        },
        out_dir / "manifest.json",
    )
    logging.info("done: %d/%d skill calib files in %s", len(index_rows), len(args.skills), out_dir)


def json_coverage(index_dir: Path) -> dict:
    path = Path(index_dir) / "coverage.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run atomic_skill_index.py first")
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()

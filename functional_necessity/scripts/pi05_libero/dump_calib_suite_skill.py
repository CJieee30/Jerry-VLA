#!/usr/bin/env python3
"""Dump 32×8 suite×skill calib batches from AtomicVLA interior frames.

Cells are pick/place across four suites plus turn on Goal/Long. Goal-pick
excludes Goal5 (push mislabeled as pick). Seed 42 is primary; 43 is repeat.
Does not reuse per-task 16×8 dumps (those have no frame-level verb).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import logging

import numpy as np

from common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    init_logging,
    load_preprocessor,
    set_seed,
)
from drop6_extra3_protocol import (
    DEFAULT_SKILL_INDEX_DIR,
    SKILL_BATCH_SIZE,
    SKILL_BOUNDARY_MARGIN,
    SKILL_NUM_BATCHES,
    SUITE_SKILL_CELLS,
    dump_json,
    suite_skill_calib_dir,
    suite_skill_cell_name,
    suite_skill_stem,
)
from dump_calib import build_dataset
from dump_calib_per_skill import dump_one_skill, json_coverage, sample_indices


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
    args = parser.parse_args()

    init_logging()
    set_seed(args.seed)
    out_dir = Path(args.out_dir) if args.out_dir is not None else suite_skill_calib_dir(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = args.num_batches * args.batch_size
    coverage = json_coverage(args.index_dir)
    paths = coverage.get("suite_skill_index_paths")
    if not paths:
        raise FileNotFoundError(
            f"{args.index_dir}/coverage.json missing suite_skill_index_paths; "
            "re-run atomic_skill_index.py"
        )

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

    index_rows = []
    for suite_label, verb in SUITE_SKILL_CELLS:
        name = suite_skill_cell_name(suite_label, verb)
        stem = suite_skill_stem(suite_label, verb)
        index_path = Path(paths[name])
        source = np.load(index_path)
        sampled, oversampled = sample_indices(source, n_samples, args.seed + _cell_seed_offset(suite_label, verb))
        path = out_dir / f"{stem}.pt"
        rec = {
            "cell": name,
            "suite": suite_label,
            "skill": verb,
            "n_source_frames": int(source.shape[0]),
            "n_sampled": len(sampled),
            "oversampled": oversampled,
            "path": str(path),
            "index_path": str(index_path),
        }
        if path.is_file() and path.stat().st_size > 1024:
            logging.info("[skip] %s already dumped (%s)", name, path)
        else:
            tmp = path.with_suffix(path.suffix + ".tmp")
            if tmp.exists():
                tmp.unlink()
            dump_one_skill(
                dataset=dataset,
                preprocessor=preprocessor,
                camera_keys=camera_keys,
                verb=name,
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
        dump_json({"seed": args.seed, "cells": index_rows}, out_dir / "index.json")

    dump_json(
        {
            "checkpoint": args.checkpoint,
            "dataset_repo_id": args.dataset_repo_id,
            "seed": args.seed,
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "boundary_margin": coverage.get("margin", SKILL_BOUNDARY_MARGIN),
            "index_dir": str(args.index_dir),
            "n_cells": len(index_rows),
            "goal5_excluded_from_goal_pick": True,
            "cells": index_rows,
        },
        out_dir / "manifest.json",
    )
    logging.info("done: %d suite×skill calib files in %s", len(index_rows), out_dir)


def _cell_seed_offset(suite_label: str, verb: str) -> int:
    """Keep cells independently sampled while remaining deterministic per seed."""
    key = f"{suite_label}/{verb}"
    return sum((i + 1) * ord(ch) for i, ch in enumerate(key)) % 997


if __name__ == "__main__":
    main()

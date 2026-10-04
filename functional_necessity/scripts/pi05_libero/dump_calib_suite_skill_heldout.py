#!/usr/bin/env python3
"""Dump 32×8 held-out suite×skill calib, excluding seed42/43 sample indices.

Cells: Goal/Long turn and Object/Spatial/Long place. Seeds 44 and 45.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import gc
import logging

import numpy as np
import torch

from common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    init_logging,
    load_preprocessor,
    set_seed,
)
from drop6_extra3_protocol import (
    CORE2_EXCLUDE_CALIB_SEEDS,
    CORE2_HELDOUT_CELLS,
    DEFAULT_SKILL_INDEX_DIR,
    SKILL_BATCH_SIZE,
    SKILL_BOUNDARY_MARGIN,
    SKILL_NUM_BATCHES,
    dump_json,
    suite_skill_calib_dir,
    suite_skill_cell_name,
    suite_skill_heldout_calib_dir,
    suite_skill_stem,
)
from dump_calib import build_dataset
from dump_calib_per_skill import dump_one_skill, json_coverage, sample_indices
from dump_calib_suite_skill import _cell_seed_offset


def load_sampled_indices(path: Path) -> list[int]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    idx = [int(x) for x in blob["sampled_indices"]]
    del blob
    gc.collect()
    return idx


def load_used_indices(stem: str, cache: dict[str, set[int]]) -> set[int]:
    if stem in cache:
        return cache[stem]
    used: set[int] = set()
    for seed in CORE2_EXCLUDE_CALIB_SEEDS:
        path = suite_skill_calib_dir(seed) / f"{stem}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing discovery dump {path}")
        used.update(load_sampled_indices(path))
    cache[stem] = used
    return used


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
    out_dir = Path(args.out_dir) if args.out_dir is not None else suite_skill_heldout_calib_dir(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = args.num_batches * args.batch_size
    coverage = json_coverage(args.index_dir)
    paths = coverage.get("suite_skill_index_paths")
    if not paths:
        raise FileNotFoundError(
            f"{args.index_dir}/coverage.json missing suite_skill_index_paths; "
            "re-run atomic_skill_index.py"
        )

    used_cache: dict[str, set[int]] = {}
    plans = []
    for suite_label, verb in CORE2_HELDOUT_CELLS:
        name = suite_skill_cell_name(suite_label, verb)
        stem = suite_skill_stem(suite_label, verb)
        index_path = Path(paths[name])
        source = np.load(index_path)
        used = load_used_indices(stem, used_cache)
        remain = np.asarray([int(x) for x in source.tolist() if int(x) not in used], dtype=np.int64)
        sampled, oversampled = sample_indices(
            remain,
            n_samples,
            args.seed + _cell_seed_offset(suite_label, verb),
        )
        overlap = used.intersection(sampled)
        if overlap:
            raise RuntimeError(f"{name}: held-out sample leaked {len(overlap)} discovery indices")
        plans.append(
            {
                "cell": name,
                "suite": suite_label,
                "skill": verb,
                "stem": stem,
                "index_path": index_path,
                "n_source_frames": int(source.shape[0]),
                "n_used_discovery": len(used),
                "n_remain": int(remain.shape[0]),
                "sampled": sampled,
                "oversampled": oversampled,
                "path": out_dir / f"{stem}.pt",
            }
        )
        logging.info(
            "%s source=%d used=%d remain=%d sample=%d oversampled=%s",
            name,
            int(source.shape[0]),
            len(used),
            int(remain.shape[0]),
            len(sampled),
            oversampled,
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
    for rec in plans:
        path = rec["path"]
        row = {
            "cell": rec["cell"],
            "suite": rec["suite"],
            "skill": rec["skill"],
            "n_source_frames": rec["n_source_frames"],
            "n_used_discovery": rec["n_used_discovery"],
            "n_remain": rec["n_remain"],
            "n_sampled": len(rec["sampled"]),
            "oversampled": rec["oversampled"],
            "path": str(path),
            "index_path": str(rec["index_path"]),
            "sampled_indices": rec["sampled"],
        }
        if path.is_file() and path.stat().st_size > 1024:
            logging.info("[skip] %s already dumped (%s)", rec["cell"], path)
        else:
            tmp = path.with_suffix(path.suffix + ".tmp")
            if tmp.exists():
                tmp.unlink()
            dump_one_skill(
                dataset=dataset,
                preprocessor=preprocessor,
                camera_keys=camera_keys,
                verb=rec["cell"] + "/heldout",
                sampled_indices=rec["sampled"],
                oversampled=rec["oversampled"],
                n_source=rec["n_remain"],
                out_path=path,
                num_batches=args.num_batches,
                batch_size=args.batch_size,
                seed=args.seed,
                num_workers=args.num_workers,
                checkpoint=args.checkpoint,
                dataset_repo_id=args.dataset_repo_id,
                index_path=rec["index_path"],
                margin=int(coverage.get("margin", SKILL_BOUNDARY_MARGIN)),
            )
        row["dumped"] = True
        index_rows.append(row)
        dump_json({"seed": args.seed, "heldout": True, "cells": index_rows}, out_dir / "index.json")

    dump_json(
        {
            "checkpoint": args.checkpoint,
            "dataset_repo_id": args.dataset_repo_id,
            "seed": args.seed,
            "heldout": True,
            "exclude_calib_seeds": list(CORE2_EXCLUDE_CALIB_SEEDS),
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "boundary_margin": coverage.get("margin", SKILL_BOUNDARY_MARGIN),
            "index_dir": str(args.index_dir),
            "n_cells": len(index_rows),
            "cells": [{k: v for k, v in row.items() if k != "sampled_indices"} | {"n_sampled": row["n_sampled"]} for row in index_rows],
        },
        out_dir / "manifest.json",
    )
    logging.info("done held-out dump seed%s: %d cells in %s", args.seed, len(index_rows), out_dir)


if __name__ == "__main__":
    main()

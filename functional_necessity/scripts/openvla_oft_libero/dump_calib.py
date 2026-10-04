#!/usr/bin/env python3
"""Dump 128 calibration transitions from each modified LIBERO suite (seed 42).

Actions stay in the RLDS bounds_q99 normalization the OFT L1 head was trained on.
Image augmentation is off so the crop matches the eval center-crop path.
Exit code 75 means the RLDS download is still incomplete.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch

from paths import CALIB_PATH, CALIB_SEED, DATA_ROOT, N_PER_SUITE, SUITES, bootstrap

INCOMPLETE = 75


def dataset_ready(root: Path) -> bool:
    for name in SUITES:
        info_path = root / name / "1.0.0" / "dataset_info.json"
        if not info_path.is_file():
            print(f"missing {info_path}", flush=True)
            return False
        info = json.loads(info_path.read_text())
        splits = info.get("splits", [])
        if isinstance(splits, dict):
            train = splits.get("train", {})
            shards = train.get("shardLengths", [])
        else:
            train = next(s for s in splits if s.get("name") == "train")
            shards = train.get("shardLengths", [])
        n_have = len(list((root / name / "1.0.0").glob("*tfrecord*")))
        if n_have < len(shards):
            print(f"{name}: {n_have}/{len(shards)} tfrecords", flush=True)
            return False
    return True


def _store(batch: dict) -> dict:
    out = {}
    for key in ("pixel_values", "input_ids", "attention_mask", "labels", "actions", "proprio"):
        value = batch[key]
        if key == "pixel_values":
            value = value.detach().to(dtype=torch.bfloat16).cpu()
        elif torch.is_tensor(value):
            value = value.detach().cpu()
        out[key] = value
    return out


def main() -> int:
    if CALIB_PATH.is_file():
        blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
        print(f"calib exists: {CALIB_PATH} n={len(blob['samples'])}", flush=True)
        return 0
    if not dataset_ready(DATA_ROOT):
        print("RLDS not ready", flush=True)
        return INCOMPLETE

    bootstrap()
    import numpy as np
    import tensorflow as tf
    from torch.utils.data import DataLoader

    from load_policy import load_processor
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

    tf.random.set_seed(CALIB_SEED)
    np.random.seed(CALIB_SEED)
    torch.manual_seed(CALIB_SEED)

    processor = load_processor()
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    samples = []
    for suite in SUITES:
        print(f"==== suite {suite} ====", flush=True)
        dataset = RLDSDataset(
            DATA_ROOT,
            suite,
            batch_transform,
            resize_resolution=(224, 224),
            shuffle_buffer_size=1024,
            train=True,
            image_aug=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=collator,
            num_workers=0,
        )
        n = 0
        for batch in loader:
            samples.append(_store(batch))
            n += 1
            if n >= N_PER_SUITE:
                break
        if n != N_PER_SUITE:
            raise RuntimeError(f"{suite} yielded {n} samples, expected {N_PER_SUITE}")
        print(f"kept {n} from {suite}; total {len(samples)}", flush=True)

    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "seed": CALIB_SEED,
            "n_per_suite": N_PER_SUITE,
            "suites": list(SUITES),
            "image_aug": False,
            "samples": samples,
        },
        CALIB_PATH,
    )
    print(f"wrote {CALIB_PATH} n={len(samples)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

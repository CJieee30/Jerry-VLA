#!/usr/bin/env python3
"""Collect the nine score files into Drop-8 ranks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from paths import DROP_K, METHODS, RANKS_PATH, SCORE_DIR
from ranking import importance_ranking


def main() -> None:
    methods = {}
    missing = [m for m in METHODS if not (SCORE_DIR / f"{m}.pt").is_file()]
    if missing:
        raise SystemExit(f"missing scores: {missing}")
    for method in METHODS:
        blob = torch.load(SCORE_DIR / f"{method}.pt", map_location="cpu", weights_only=False)
        report = importance_ranking(blob["scores"], k=DROP_K)
        methods[method] = {
            "importance": report["importance"],
            "redundancy_order": report["redundancy_order"],
            "drop": report["drop"],
            "keep": report["keep"],
            "n_samples": blob.get("n_samples"),
        }
    unique = []
    aliases = {}
    seen = {}
    for method in METHODS:
        key = tuple(methods[method]["drop"])
        if key not in seen:
            seen[key] = method
            unique.append(method)
            aliases[method] = []
        else:
            aliases[seen[key]].append(method)
    payload = {
        "model": "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
        "k": DROP_K,
        "n_layers": 32,
        "skip": "llama_decoder_identity",
        "loss": "l1_action_head",
        "methods": methods,
        "unique_representatives": unique,
        "aliases": aliases,
    }
    RANKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RANKS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {RANKS_PATH}")
    for method in unique:
        extra = aliases[method]
        suffix = f"  aliases={extra}" if extra else ""
        print(f"  {method} drop {methods[method]['drop']}{suffix}")


if __name__ == "__main__":
    main()

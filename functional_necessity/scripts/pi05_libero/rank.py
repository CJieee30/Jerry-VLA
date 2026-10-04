#!/usr/bin/env python3
"""Aggregate per-method scores into redundancy rankings and k drop-sets."""

from __future__ import annotations

import sys
from pathlib import Path as _ScriptDir

sys.path.insert(0, str(_ScriptDir(__file__).resolve().parent))

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from common import (
    DEFAULT_DROP_KS,
    DEFAULT_RANKS_DIR,
    DEFAULT_SCORES_DIR,
    importance_ranking,
    init_logging,
)

METHODS = [
    "gateprobe",
    "taylor",
    "fisher",
    "igia",
    "hessian",
    "magnitude",
    "cossim",
    "cka",
    "ppl",
]


def load_importance(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(payload):
        return payload
    if "paligemma" in payload:
        return payload["paligemma"]
    raise KeyError(f"{path} has no paligemma scores")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RANKS_DIR)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_DROP_KS))
    parser.add_argument(
        "--checkpoint-label",
        default="lerobot/pi05_libero_finetuned_v044",
        help="Checkpoint id written into DROP_INDEX_LOOKUP.md",
    )
    args = parser.parse_args()
    init_logging()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    methods: dict[str, dict] = {}
    for name in METHODS:
        path = args.scores_dir / f"{name}.pt"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        scores = load_importance(path)
        methods[name] = importance_ranking(scores, ks=args.ks)

    unique_by_k: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for name, report in methods.items():
        for k, sets in report["drop"].items():
            key = tuple(sets["drop"])
            unique_by_k[int(k)][str(list(key))].append(name)

    payload = {
        "ks": list(args.ks),
        "methods": methods,
        "unique_drop_sets": {
            str(k): {drop: names for drop, names in groups.items()}
            for k, groups in unique_by_k.items()
        },
    }
    json_path = args.out_dir / "ranks.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {json_path}")

    md_lines = [
        "# Functional Necessity drop-index lookup",
        "",
        f"Checkpoint: `{args.checkpoint_label}`. PaliGemma 18 layers, indices 0–17.",
        "Decoding: highest importance = kept. CKA uses global concatenated linear CKA; drop high adjacent CKA first (`importance = 1 - CKA`).",
        "",
    ]
    for k in args.ks:
        md_lines.append(f"## Drop-{k}")
        md_lines.append("")
        md_lines.append("| Method | Drop | Keep |")
        md_lines.append("|---|---|---|")
        for name, report in methods.items():
            sets = report["drop"].get(k) or report["drop"].get(str(k))
            if not sets:
                continue
            md_lines.append(f"| {name} | {sets['drop']} | {sets['keep']} |")
        md_lines.append("")
        md_lines.append("Unique drop-sets:")
        for drop, names in unique_by_k[k].items():
            md_lines.append(f"- `{drop}` ← {', '.join(names)}")
        md_lines.append("")
        md_lines.append("Redundancy order (most redundant first):")
        md_lines.append("")
        for name, report in methods.items():
            md_lines.append(f"- **{name}**: {report['redundancy_order']}")
        md_lines.append("")

    md_path = args.out_dir / "DROP_INDEX_LOOKUP.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

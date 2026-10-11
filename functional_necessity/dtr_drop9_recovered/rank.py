#!/usr/bin/env python3
"""Rank remaining-9 layers; FN Drop-3 is the 3 least important among keep-set."""

from __future__ import annotations

import sys
from pathlib import Path as _ScriptDir

sys.path.insert(0, str(_ScriptDir(__file__).resolve().parent))

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from backend import (
    DEFAULT_RANKS_DIR,
    DEFAULT_SCORES_DIR,
    FN_DROP_K,
    REMAINING_LAYERS,
    STAGE1_DROP,
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
    parser.add_argument("--ks", type=int, nargs="+", default=[FN_DROP_K])
    parser.add_argument(
        "--checkpoint-label",
        default="s1ghhh/VLADrop_pi05_LIBERO_Drop9Block_GateProbe",
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
            unique_by_k[int(k)][str(list(sets["drop"]))].append(name)

    payload = {
        "checkpoint": args.checkpoint_label,
        "stage1_drop": STAGE1_DROP,
        "remaining_layers": REMAINING_LAYERS,
        "ks": list(args.ks),
        "methods": methods,
        "unique_drop_sets": {
            str(k): {drop: names for drop, names in groups.items()}
            for k, groups in unique_by_k.items()
        },
        "dtr_drop12_extra_vs_drop9": [3, 4, 9],
    }
    json_path = args.out_dir / "ranks.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {json_path}")

    md = [
        "# FN Drop-3 on DTR Drop-9 recovered π0.5",
        "",
        f"Checkpoint: `{args.checkpoint_label}`.",
        f"Stage-1 already dropped: `{STAGE1_DROP}`.",
        f"Remaining (scored for FN Drop-k): `{REMAINING_LAYERS}`.",
        "Drop-k is taken **only among remaining layers**. Indices stay 0–17.",
        "Official GateProbe Drop-12 extra vs this Drop-9 keep-set is `{3,4,9}`.",
        "",
    ]
    for k in args.ks:
        md += [
            f"## FN Drop-{k} (from remaining 9)",
            "",
            "| Method | Extra drop | Remaining keep | Union drop (Stage-1 ∪ extra) |",
            "|---|---|---|---|",
        ]
        for name, report in methods.items():
            sets = report["drop"].get(k) or report["drop"].get(str(k))
            if not sets:
                continue
            md.append(
                f"| {name} | {sets['drop']} | {sets['keep']} | {sets['union_drop']} |"
            )
        md += ["", "Unique extra-drop sets:"]
        for drop, names in unique_by_k[k].items():
            md.append(f"- `{drop}` ← {', '.join(names)}")
        md += ["", "Remaining redundancy order (most redundant first):", ""]
        for name, report in methods.items():
            md.append(f"- **{name}**: {report['remaining_redundancy_order']}")
        md.append("")
        md.append("Full 18-layer redundancy order (diagnostic; Stage-1 layers should rank as redundant):")
        md.append("")
        for name, report in methods.items():
            md.append(f"- **{name}**: {report['redundancy_order']}")
        md.append("")

    md_path = args.out_dir / "DROP_INDEX_LOOKUP.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()

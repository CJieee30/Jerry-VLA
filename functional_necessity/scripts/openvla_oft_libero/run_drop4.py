#!/usr/bin/env python3
"""Closed-loop Drop-4 for the eight unique layer sets, one GPU's share."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from paths import EVAL_SUITES, FN_ROOT, METHODS, PYTHON, SCORE_DIR
from ranking import importance_ranking

HERE = Path(__file__).resolve().parent
RESULT_ROOT = FN_ROOT / "results" / "openvla_oft_libero" / "drop4"
RANKS_PATH = RESULT_ROOT / "ranks" / "ranks.json"
EVAL_ROOT = RESULT_ROOT / "eval"
K = 4
ANCHOR = {
    "libero_spatial": 0.932,
    "libero_object": 0.992,
    "libero_goal": 0.972,
    "libero_10": 0.942,
}


def build_ranks() -> dict:
    methods = {}
    missing = [m for m in METHODS if not (SCORE_DIR / f"{m}.pt").is_file()]
    if missing:
        raise SystemExit(f"missing scores: {missing}")
    for method in METHODS:
        blob = torch.load(SCORE_DIR / f"{method}.pt", map_location="cpu", weights_only=False)
        report = importance_ranking(blob["scores"], k=K)
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
        "k": K,
        "n_layers": 32,
        "skip": "llama_decoder_identity",
        "loss": "l1_action_head",
        "methods": methods,
        "unique_representatives": unique,
        "aliases": aliases,
    }
    RANKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RANKS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {RANKS_PATH}", flush=True)
    for method in unique:
        extra = aliases[method]
        suffix = f" aliases={extra}" if extra else ""
        print(f"  {method} drop {methods[method]['drop']}{suffix}", flush=True)
    return payload


def write_summary(ranks: dict) -> None:
    lines = [
        "# OpenVLA-OFT Drop-4",
        "",
        "Checkpoint: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`.",
        "Skip: whole Llama decoder layer (no K/V). Scores are the same nine files as Drop-8, cut at k=4.",
        "Closed loop: seed 7, 50 trials, center crop. Anchor is this machine's seed-7 run (95.95%).",
        "",
        "| Method | Drop-4 | Spatial | Object | Goal | Long | Avg |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    pending = False
    for method in ranks["unique_representatives"]:
        cells = []
        rates = []
        for suite in EVAL_SUITES:
            info = EVAL_ROOT / f"k{K}_{method}" / suite / "eval_info.json"
            if not info.is_file():
                cells.append("")
                pending = True
                continue
            rate = json.loads(info.read_text())["rate"]
            rates.append(rate)
            cells.append(f"{rate * 100:.1f}%")
        avg = f"{(sum(rates) / len(rates)) * 100:.2f}%" if len(rates) == 4 else ""
        aliases = ranks["aliases"].get(method) or []
        label = method if not aliases else f"{method} ({', '.join(aliases)})"
        drop = " ".join(str(i) for i in ranks["methods"][method]["drop"])
        lines.append(f"| {label} | `{drop}` | " + " | ".join(cells) + f" | {avg} |")
    lines.append("")
    lines.append(
        f"Anchor seed-7: Spatial {ANCHOR['libero_spatial']*100:.1f}%, "
        f"Object {ANCHOR['libero_object']*100:.1f}%, Goal {ANCHOR['libero_goal']*100:.1f}%, "
        f"Long {ANCHOR['libero_10']*100:.1f}%, avg 95.95%."
    )
    if pending:
        lines.append("")
        lines.append("Some suites are still running.")
    path = RESULT_ROOT / "drop4.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}", flush=True)


def main() -> None:
    gpu = int(sys.argv[1])
    ranks = build_ranks()
    jobs = [(method, suite) for method in ranks["unique_representatives"] for suite in EVAL_SUITES]
    mine = [job for i, job in enumerate(jobs) if i % 2 == gpu % 2]
    print(f"GPU {gpu} jobs {len(mine)}/{len(jobs)}", flush=True)
    failed = 0
    for method, suite in mine:
        print(f"==== GPU {gpu} {method} {suite} ====", flush=True)
        rc = subprocess.call(
            [
                PYTHON,
                str(HERE / "eval_suite.py"),
                "--method",
                method,
                "--suite",
                suite,
                "--k",
                str(K),
                "--ranks",
                str(RANKS_PATH),
                "--eval-root",
                str(EVAL_ROOT),
            ]
        )
        if rc != 0:
            failed += 1
            print(f"FAIL {method} {suite} exit={rc}", flush=True)
        write_summary(ranks)
    write_summary(ranks)
    if failed:
        raise SystemExit(failed)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Suite-aware CosSim and Fisher for OpenVLA-OFT.

The 512-sample calibration file is 128 contiguous samples from each suite, in
SUITES order. Each suite is scored alone. Drop-4 and Drop-8 are compared with
the pooled global scores.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from load_policy import cap_cuda_memory, load_policy
from paths import CALIB_PATH, FN_ROOT, SCORE_DIR, SUITES
from ranking import importance_ranking
from score import measure_cossim, measure_fisher

OUT_DIR = FN_ROOT / "results" / "openvla_oft_libero" / "suite_aware"
LABELS = ("spatial", "object", "goal", "long")
METHODS = ("cossim", "fisher")
KS = (4, 8)
GLOBAL_DROP8 = {
    "cossim": [3, 4, 5, 6, 19, 20, 23, 28],
    "fisher": [0, 3, 4, 5, 28, 29, 30, 31],
}
GLOBAL_DROP4 = {
    "cossim": [3, 4, 5, 6],
    "fisher": [0, 3, 4, 30],
}


def split_samples(blob: dict) -> dict[str, list]:
    samples = blob["samples"]
    n = int(blob["n_per_suite"])
    suites = list(blob["suites"])
    if suites != list(SUITES) or len(samples) != n * len(suites):
        raise SystemExit(f"unexpected calib layout n={len(samples)} suites={suites}")
    return {LABELS[i]: samples[i * n : (i + 1) * n] for i in range(len(suites))}


def drops_from_scores(scores) -> dict[int, list[int]]:
    return {k: importance_ranking(scores, k=k)["drop"] for k in KS}


def load_global() -> dict[str, dict[int, list[int]]]:
    out = {}
    for method in METHODS:
        blob = torch.load(SCORE_DIR / f"{method}.pt", map_location="cpu", weights_only=False)
        out[method] = drops_from_scores(blob["scores"])
    return out


def write_report(per_suite: dict) -> None:
    global_drops = load_global()
    lines = [
        "# OpenVLA-OFT suite-aware CosSim / Fisher",
        "",
        "Each suite uses its own 128 calibration samples. Global uses all 512.",
        "High score = keep. Drop-k is the k lowest scores.",
        "",
        "| Method | Scope | Drop-4 | Drop-8 | vs global Drop-4 | vs global Drop-8 |",
        "|---|---|---|---|---|---|",
    ]
    for method in METHODS:
        g4, g8 = global_drops[method][4], global_drops[method][8]
        lines.append(f"| {method} | global | `{g4}` | `{g8}` | same | same |")
        for label in LABELS:
            if label not in per_suite or method not in per_suite[label]:
                lines.append(f"| {method} | {label} |  |  |  |  |")
                continue
            d4 = per_suite[label][method]["drop4"]
            d8 = per_suite[label][method]["drop8"]
            lines.append(
                f"| {method} | {label} | `{d4}` | `{d8}` | "
                f"{'same' if d4 == g4 else 'different'} | "
                f"{'same' if d8 == g8 else 'different'} |"
            )
    path = OUT_DIR / "suite_aware.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}", flush=True)


def main() -> None:
    cap_cuda_memory()
    device = torch.device("cuda:0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    groups = split_samples(blob)
    summary_path = OUT_DIR / "suite_aware.json"
    per_suite = {}
    if summary_path.is_file():
        per_suite = json.loads(summary_path.read_text()).get("per_suite", {})

    need_model = False
    for label in LABELS:
        done = per_suite.get(label, {})
        if any(m not in done for m in METHODS):
            need_model = True
            break
    vla = action_head = proprio = None
    if need_model:
        print("loading policy", flush=True)
        vla, action_head, proprio = load_policy("cuda:0")

    measure = {"cossim": measure_cossim, "fisher": measure_fisher}
    for label in LABELS:
        per_suite.setdefault(label, {})
        for method in METHODS:
            if method in per_suite[label]:
                print(f"skip {label} {method}", flush=True)
                continue
            print(f"==== {label} {method} n={len(groups[label])} ====", flush=True)
            result = measure[method](vla, action_head, proprio, groups[label], device)
            drops = drops_from_scores(result["scores"])
            per_suite[label][method] = {
                "scores": result["scores"],
                "drop4": drops[4],
                "drop8": drops[8],
                "n_samples": len(groups[label]),
            }
            summary_path.write_text(json.dumps({"per_suite": per_suite}, indent=2) + "\n")
            print(f"{label} {method} drop4={drops[4]} drop8={drops[8]}", flush=True)
            write_report(per_suite)
    write_report(per_suite)
    print("done", flush=True)


if __name__ == "__main__":
    main()

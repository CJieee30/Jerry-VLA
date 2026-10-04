#!/usr/bin/env python3
"""Drop-4 closed loop excluding layer 0, one method per GPU.

The four sets are the next four lowest scores after removing layer 0 from the
redundancy order. GateProbe's set equals the finished CosSim/PPL Drop-4, so
that GPU records the existing 500-episode logs instead of repeating them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import EVAL_SUITES, FN_ROOT, PYTHON

HERE = Path(__file__).resolve().parent
RESULT_ROOT = FN_ROOT / "results" / "openvla_oft_libero" / "drop4_nol0"
RANKS_PATH = RESULT_ROOT / "ranks" / "ranks.json"
EVAL_ROOT = RESULT_ROOT / "eval"
COSSIM_ROOT = FN_ROOT / "results" / "openvla_oft_libero" / "drop4" / "eval" / "k4_cossim"
K = 4
JOBS = {
    0: ("fisher", [3, 4, 30, 31]),
    1: ("igia", [3, 29, 30, 31]),
    2: ("gateprobe", [3, 4, 5, 6]),
    3: ("taylor", [1, 2, 3, 4]),
}
DROPS = {name: drop for _gpu, (name, drop) in JOBS.items()}
DROPS["magnitude"] = [1, 2, 3, 6]
ANCHOR = {
    "libero_spatial": 0.932,
    "libero_object": 0.992,
    "libero_goal": 0.972,
    "libero_10": 0.942,
}


def build_ranks() -> dict:
    methods = {
        name: {"drop": drop, "excluded": [0], "rule": "four lowest scores after removing layer 0"}
        for name, drop in DROPS.items()
    }
    payload = {
        "model": "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
        "k": K,
        "n_layers": 32,
        "skip": "llama_decoder_identity",
        "rule": "drop the four most redundant layers other than layer 0",
        "methods": methods,
        "unique_representatives": list(DROPS),
        "aliases": {"gateprobe": ["cossim_drop4", "ppl_drop4"]},
    }
    RANKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RANKS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(RANKS_PATH)
    print(f"wrote {RANKS_PATH}", flush=True)
    return payload


def reuse_gateprobe() -> None:
    """Same layers as the finished CosSim Drop-4 run."""
    for suite in EVAL_SUITES:
        src = COSSIM_ROOT / suite / "eval_info.json"
        if not src.is_file():
            raise SystemExit(f"missing finished CosSim eval {src}")
        info = json.loads(src.read_text())
        if info["drop"] != [3, 4, 5, 6] or info["episodes"] != 500 or info["seed"] != 7:
            raise SystemExit(f"CosSim {suite} is not the expected Drop-4 protocol")
        out = EVAL_ROOT / f"k{K}_gateprobe" / suite
        out.mkdir(parents=True, exist_ok=True)
        (out / "drop.json").write_text(
            json.dumps({"method": "gateprobe", "k": K, "drop": [3, 4, 5, 6], "same_as": "cossim"}) + "\n"
        )
        info["method"] = "gateprobe"
        info["same_intervention_as"] = "cossim Drop-4 [3, 4, 5, 6]"
        (out / "eval_info.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"reused cossim {suite} {info['successes']}/{info['episodes']}", flush=True)


def write_summary() -> None:
    lines = [
        "# OpenVLA-OFT Drop-4, layer 0 excluded",
        "",
        "Four lowest-scoring layers after removing layer 0 from the redundancy order.",
        "Protocol matches Drop-4: seed 7, 50 trials, center crop, whole-layer identity.",
        "GateProbe uses the finished CosSim/PPL run because the drop set is identical.",
        "",
        "| Method | Drop-4 | Spatial | Object | Goal | Long | Avg |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    pending = False
    for method, drop in DROPS.items():
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
        label = method if method != "gateprobe" else "gateprobe (same as cossim/ppl)"
        lines.append(f"| {label} | `{' '.join(str(i) for i in drop)}` | " + " | ".join(cells) + f" | {avg} |")
    lines.append("")
    lines.append(
        f"Anchor seed-7: Spatial {ANCHOR['libero_spatial']*100:.1f}%, "
        f"Object {ANCHOR['libero_object']*100:.1f}%, Goal {ANCHOR['libero_goal']*100:.1f}%, "
        f"Long {ANCHOR['libero_10']*100:.1f}%, avg 95.95%."
    )
    if pending:
        lines.append("")
        lines.append("Some suites are still running.")
    path = RESULT_ROOT / "drop4_nol0.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}", flush=True)


def main() -> None:
    arg = sys.argv[1]
    if arg in DROPS:
        gpu = -1
        method, drop = arg, DROPS[arg]
    else:
        gpu = int(arg)
        method, drop = JOBS[gpu]
    build_ranks()
    if method == "gateprobe":
        reuse_gateprobe()
        write_summary()
        print("gateprobe set already evaluated as cossim Drop-4", flush=True)
        return
    failed = 0
    for suite in EVAL_SUITES:
        print(f"==== {method} {suite} drop={drop} ====", flush=True)
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
        write_summary()
    if failed:
        raise SystemExit(failed)


if __name__ == "__main__":
    main()

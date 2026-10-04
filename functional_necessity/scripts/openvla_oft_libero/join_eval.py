#!/usr/bin/env python3
"""After all nine scores exist, run this GPU's share of unique Drop-8 suites."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import DROP_K, EVAL_ROOT, EVAL_SUITES, METHODS, PYTHON, RANKS_PATH, RESULT_ROOT, SCORE_DIR

HERE = Path(__file__).resolve().parent
ANCHOR = {
    "libero_spatial": 0.932,
    "libero_object": 0.992,
    "libero_goal": 0.972,
    "libero_10": 0.942,
}


def wait_scores() -> None:
    while True:
        missing = [m for m in METHODS if not (SCORE_DIR / f"{m}.pt").is_file()]
        if not missing:
            return
        print(f"waiting for scores: {missing}", flush=True)
        time.sleep(60)


def my_jobs(gpu: int) -> list[tuple[str, str]]:
    ranks = json.loads(RANKS_PATH.read_text())
    jobs = [(method, suite) for method in ranks["unique_representatives"] for suite in EVAL_SUITES]
    return [job for i, job in enumerate(jobs) if i % 2 == gpu % 2]


def write_summary() -> None:
    if not RANKS_PATH.is_file():
        return
    ranks = json.loads(RANKS_PATH.read_text())
    lines = [
        "# OpenVLA-OFT Drop-8",
        "",
        "Checkpoint: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`.",
        "Skip: whole Llama decoder layer (no K/V). Loss for scores: action L1.",
        "Closed loop: seed 7, 50 trials, center crop. Anchor is this machine's seed-7 run (95.95%).",
        "",
        "| Method | Drop-8 | Spatial | Object | Goal | Long | Avg |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    pending = False
    for method in ranks["unique_representatives"]:
        cells = []
        rates = []
        for suite in EVAL_SUITES:
            info = EVAL_ROOT / f"k{DROP_K}_{method}" / suite / "eval_info.json"
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
    path = RESULT_ROOT / "drop8.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}", flush=True)


def main() -> None:
    gpu = int(sys.argv[1])
    wait_scores()
    subprocess.check_call([PYTHON, str(HERE / "build_ranks.py")])
    failed = 0
    for method, suite in my_jobs(gpu):
        print(f"==== GPU {gpu} {method} {suite} ====", flush=True)
        rc = subprocess.call([PYTHON, str(HERE / "eval_suite.py"), "--method", method, "--suite", suite])
        if rc != 0:
            failed += 1
            print(f"FAIL {method} {suite} exit={rc}", flush=True)
        write_summary()
    write_summary()
    if failed:
        raise SystemExit(failed)


if __name__ == "__main__":
    main()

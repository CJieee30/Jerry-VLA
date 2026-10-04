#!/usr/bin/env python3
"""Closed-loop LIBERO for the finished OpenVLA-OFT greedy-PPL prefix."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import EVAL_SUITES, FN_ROOT, PYTHON

HERE = Path(__file__).resolve().parent
RESULT = FN_ROOT / "results" / "openvla_oft_libero" / "greedy_ppl_eval"
# Pick order from oft_drop8_greedy.json. Drop-k is the first k picks, sorted
# for the skip set. Drop-3/4/8 prefixes: [3,5,29], then 4, then 23,26,19,28.
PICK_ORDER = [3, 5, 29, 4, 23, 26, 19, 28]
DROPS = {
    3: [3, 5, 29],
    4: [3, 4, 5, 29],
    8: [3, 4, 5, 19, 23, 26, 28, 29],
}


def main() -> None:
    k = int(sys.argv[1])
    drop = DROPS[k]
    ranks = RESULT / "ranks" / f"k{k}.json"
    ranks.parent.mkdir(parents=True, exist_ok=True)
    ranks.write_text(
        json.dumps(
            {
                "k": k,
                "methods": {"greedy_ppl": {"drop": drop, "pick_order": PICK_ORDER[:k]}},
            },
            indent=2,
        )
        + "\n"
    )
    eval_root = RESULT / "eval"
    failed = 0
    for suite in EVAL_SUITES:
        print(f"==== greedy_ppl k={k} {suite} drop={drop} ====", flush=True)
        rc = subprocess.call(
            [
                PYTHON,
                str(HERE / "eval_suite.py"),
                "--method",
                "greedy_ppl",
                "--suite",
                suite,
                "--k",
                str(k),
                "--ranks",
                str(ranks),
                "--eval-root",
                str(eval_root),
            ]
        )
        if rc != 0:
            failed += 1
            print(f"FAIL k={k} {suite} exit={rc}", flush=True)
    if failed:
        raise SystemExit(failed)


if __name__ == "__main__":
    main()

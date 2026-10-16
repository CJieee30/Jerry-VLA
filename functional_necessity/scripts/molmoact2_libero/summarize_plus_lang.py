#!/usr/bin/env python3
"""Print MolmoAct2 Plus Language full-NFE10 aggregate from per-suite eval_info.json."""

from __future__ import annotations

import json
from pathlib import Path

FN = Path(__file__).resolve().parents[2]
BASE = FN / "results" / "eval_libero_plus"
SUITES = [
    ("spatial", "libero_spatial", 390),
    ("object", "libero_object", 354),
    ("goal", "libero_goal", 410),
    ("10", "libero_10", 383),
]
TAG = "molmoact2_full_nfe10_lang_{}"


def main() -> None:
    rows = []
    total_succ = 0
    total_n = 0
    for suffix, suite, expect_n in SUITES:
        tag = TAG.format(suffix)
        info_path = BASE / tag / "eval_info.json"
        if not info_path.is_file():
            rows.append((suffix, "pending", None, None, expect_n))
            continue
        info = json.loads(info_path.read_text())
        pg = info.get("per_group") or {}
        key = suite if suite in pg else next(iter(pg), suite)
        pc = float(pg.get(key, info.get("overall", {})).get("pc_success", 0))
        n = int((info.get("overall") or {}).get("n_episodes") or expect_n)
        succ = int(round(pc * n / 100.0))
        total_succ += succ
        total_n += n
        rows.append((suffix, "done", pc, f"{succ}/{n}", expect_n))

    print("MolmoAct2 Full NFE=10 Language Instructions")
    for suffix, status, pc, bits, exp in rows:
        if status == "pending":
            print(f"  {suffix}: pending (expect {exp} variants)")
        else:
            print(f"  {suffix}: {pc:.2f}%  {bits}")
    done_rates = [r[2] for r in rows if r[2] is not None]
    if len(done_rates) == 4:
        macro = sum(done_rates) / 4
        print(f"Macro Avg (4 suites): {macro:.2f}%")
    if total_n:
        print(f"Micro ({total_succ}/{total_n}): {100.0 * total_succ / total_n:.2f}%")


if __name__ == "__main__":
    main()

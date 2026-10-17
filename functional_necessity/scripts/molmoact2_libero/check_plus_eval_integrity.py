#!/usr/bin/env python3
"""Minimum completeness check for LIBERO-plus Molmo eval dirs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def expected_task_ids(classification_json: Path, suite: str, category: str) -> list[int]:
    payload = json.loads(classification_json.read_text())
    return [item["id"] - 1 for item in payload[suite] if item["category"] == category]


def check_eval_dir(
    out_dir: Path,
    *,
    suite: str,
    category: str,
    classification_json: Path,
    n_episodes: int = 1,
) -> dict:
    eval_path = out_dir / "eval_info.json"
    if not eval_path.is_file():
        return {"complete": False, "reason": "missing_eval_info"}

    expected = expected_task_ids(classification_json, suite, category)
    data = json.loads(eval_path.read_text())
    per_task = data.get("per_task") or data.get("per_episode") or []
    if not per_task and "aggregated" in data:
        per_task = data["aggregated"].get("per_task", [])

    seen_ids = []
    episode_counts: Counter[int] = Counter()
    success_n = 0
    for row in per_task:
        tid = row.get("task_id")
        if tid is None and "task_index" in row:
            tid = row["task_index"]
        if tid is None:
            continue
        tid = int(tid)
        seen_ids.append(tid)
        metrics = row.get("metrics") or {}
        succ = metrics.get("successes") or row.get("successes") or []
        episode_counts[tid] += len(succ) if isinstance(succ, list) else 1
        if isinstance(succ, list):
            success_n += sum(1 for x in succ if x)
        elif row.get("success") is not None:
            success_n += int(bool(row["success"]))

    counts = Counter(seen_ids)
    dup = [tid for tid, c in counts.items() if c > 1]
    missing = sorted(set(expected) - set(seen_ids))
    extra = sorted(set(seen_ids) - set(expected))
    wrong_eps = [tid for tid, c in episode_counts.items() if c != n_episodes]

    successes = success_n
    if "overall" in data and "success_rate" in data["overall"]:
        recomputed = successes / len(per_task) if per_task else 0.0
        reported = float(data["overall"]["success_rate"])
        sr_delta = abs(recomputed - reported)
    else:
        sr_delta = None

    complete = not missing and not extra and not dup and not wrong_eps
    return {
        "complete": complete,
        "expected_n_tasks": len(expected),
        "seen_n_tasks": len(seen_ids),
        "missing_task_ids": missing[:20],
        "missing_n": len(missing),
        "extra_task_ids": extra[:20],
        "extra_n": len(extra),
        "duplicate_task_ids": dup[:20],
        "wrong_episode_count_task_ids": wrong_eps[:20],
        "success_recompute_delta": sr_delta,
        "eval_info_path": str(eval_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--category", default="Robot Initial States")
    parser.add_argument("--classification-json", type=Path, required=True)
    parser.add_argument("--n-episodes", type=int, default=1)
    args = parser.parse_args()

    report = check_eval_dir(
        args.output_dir,
        suite=args.suite,
        category=args.category,
        classification_json=args.classification_json,
        n_episodes=args.n_episodes,
    )
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["complete"] else 2)


if __name__ == "__main__":
    main()

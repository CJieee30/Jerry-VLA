#!/usr/bin/env python3
"""Merge per-suite eval_info.json dirs. Missing suites are marked 0 if early=1."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    out = Path(sys.argv[1])
    early = sys.argv[2] == "1"
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    note = sys.argv[3] if len(sys.argv) > 3 else "Spatial pc_success==0; remaining marked 0."
    combined = {"per_task": [], "per_group": {}, "overall": {}}
    n_ep = 0
    weighted = 0.0
    for suite in suites:
        path = out / suite / "eval_info.json"
        if path.is_file():
            info = json.loads(path.read_text())
            combined["per_task"].extend(info.get("per_task", []))
            group = info.get("per_group", {}).get(suite, {})
            combined["per_group"][suite] = group
            n = int(group.get("n_episodes", 100))
            n_ep += n
            weighted += float(group.get("pc_success", 0.0)) * n
            continue
        combined["per_group"][suite] = {
            "avg_sum_reward": 0.0,
            "avg_max_reward": 0.0,
            "pc_success": 0.0,
            "n_episodes": 100,
            "marked_zero": True,
            "video_paths": [],
            "predicted_video_paths": [],
        }
        n_ep += 100
    meta_src = next(
        (out / s / name for s in suites for name in ("drop.json", "drop_meta.json") if (out / s / name).is_file()),
        None,
    )
    if meta_src is not None:
        (out / meta_src.name).write_text(meta_src.read_text())
    combined["overall"] = {
        "avg_sum_reward": weighted / max(n_ep, 1) / 100.0,
        "avg_max_reward": weighted / max(n_ep, 1) / 100.0,
        "pc_success": weighted / max(n_ep, 1),
        "n_episodes": n_ep,
        "early_stop_spatial_zero": early,
        "video_paths": [],
        "predicted_video_paths": [],
    }
    (out / "eval_info.json").write_text(json.dumps(combined, indent=2) + "\n")
    if early:
        stopped = {
            "stopped": True,
            "remaining_marked_zero": True,
            "drop": [0],
            "operator": "keep_kv_identity_residual",
            "completed": "libero_spatial",
            "suite_success_pct_remaining_zero": {
                suite: float(combined["per_group"][suite]["pc_success"]) for suite in suites
            },
            "avg": float(combined["overall"]["pc_success"]),
            "n_episodes_denom": 400,
            "note": note,
        }
        (out / "stopped.json").write_text(json.dumps(stopped, indent=2) + "\n")
    print(json.dumps(combined["overall"], indent=2))


if __name__ == "__main__":
    main()

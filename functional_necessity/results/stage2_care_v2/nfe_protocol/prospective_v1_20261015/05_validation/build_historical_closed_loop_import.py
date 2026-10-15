#!/usr/bin/env python3
"""Build closed-loop import JSON from historical eval_info.json (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path

FN = Path(__file__).resolve().parents[5]  # functional_necessity/
RUN = Path(__file__).resolve().parents[1]

GROUP = {
    "libero_spatial": "spatial",
    "libero_object": "object",
    "libero_goal": "goal",
    "libero_10": "long",
}


def load_entry(path: Path, *, label: str, nfe: int, notes: str) -> dict:
    info = json.loads(path.read_text())
    pg = info["per_group"]
    ov = info["overall"]
    suites = {GROUP[k]: float(pg[k]["pc_success"]) for k in GROUP}
    n_ep = int(ov["n_episodes"])
    n_ok = int(round(float(ov["pc_success"]) * n_ep / 100.0))
    return {
        "suites": suites,
        "meta": {
            "n_episodes": n_ep,
            "n_success": n_ok,
            "seed": 1000,
            "eval_protocol": "libero_four_suite",
            "setting": label,
            "nfe": nfe,
            "source_eval_info": str(path.relative_to(FN)),
            "notes": notes,
        },
    }


def main() -> None:
    sources = {
        "pi05": {
            "10": (FN / "results/eval_drop/baseline/eval_info.json", "full", "NFE=10 anchor; eval_drop baseline"),
            "1": (FN / "results/stage_nfe/pi05_full_nfe1/eval_info.json", "full", "NFE=1; stage_nfe pi05_full_nfe1"),
        },
        "smolvla": {
            "10": (
                FN / "results/smolvla_libero/lerobot_official_nas10/eval_info.json",
                "full",
                "NFE=10 anchor; eval_smolvla_libero seed-1000",
            ),
            "2": (FN / "results/stage_nfe/smolvla_full_nfe2/eval_info.json", "full", "NFE=2 recommended; stage_nfe smolvla_full_nfe2"),
        },
        "molmoact2": {
            "10": (
                FN / "results/molmoact2_libero/baseline_eval/eval_info.json",
                "full",
                "NFE=10 anchor; 50 ep/task × 2000",
            ),
            "1": (
                FN / "results/stage_nfe/molmoact2_full_nfe1_2000/eval_info.json",
                "full",
                "NFE=1 recommended; 50 ep/task × 2000",
            ),
        },
    }
    results: dict = {}
    manifest = []
    for model, per in sources.items():
        results[model] = {}
        for nfe, (path, setting, notes) in per.items():
            if not path.is_file():
                raise SystemExit(f"missing {path}")
            results[model][nfe] = load_entry(path, label=setting, nfe=int(nfe), notes=notes)
            manifest.append({"model": model, "nfe": int(nfe), "path": str(path.relative_to(FN))})
    out = {
        "success_rate_unit": "percent",
        "avg_margin_pp": 2.0,
        "suite_drop_pp": 5.0,
        "provenance": (
            "Historical LIBERO closed-loop runs predating prospective_v1_20261015 calibration. "
            "Reused for validate closed-loop column only; not new prospective closed-loop experiments."
        ),
        "manifest": manifest,
        "results": results,
    }
    dest = RUN / "05_validation" / "historical_closed_loop_import.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Equal-budget calibration selection on frozen 480-frame NFE probes (CPU only)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from event_care_nfe_diagnostic import (  # noqa: E402
    EVENT_DIR,
    GROUP_SEED,
    MODELS,
    N_FRAMES,
    TOP_K,
    build_metas,
    demo_gripper_switch,
    eval_selection,
    g_only_group,
    grip_deviation_set,
    load_probe_frames,
    random_group,
    select_per_task,
    tail_mask_nfe1,
    arm_top5_set,
)
from dump_shared import _load_actions  # noqa: E402
from protocol import LIBERO_REPO  # noqa: E402

OUT_DIR = EVENT_DIR.parent
COMPARE_DIR = OUT_DIR / "nfe_probe_event_care" / "budget_compare"
SELECTION_DIR = COMPARE_DIR / "selections"

# Fixed seeds for repeat random / G-only (includes legacy seed 42).
BUDGET_SEEDS: tuple[int, ...] = (
    42,
    0,
    1,
    7,
    13,
    17,
    23,
    29,
    31,
    37,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
)

LEGACY_KEYS = ("random", "care_alpha", "care_event")


def aggregate_metrics(rows: list[dict], key: str) -> dict:
    vals = [float(r[key]) for r in rows if key in r and np.isfinite(r[key])]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def merged_two_noise(stats0: dict, stats1: dict) -> dict:
    """Sum numerators/denominators for capture counts (not independent episodes)."""
    out = {}
    for key in (
        "n_grip_dev_nfe1_captured",
        "n_grip_dev_nfe1_total",
        "n_tail_captured",
        "n_tail_bad_total",
        "n_arm_top5_captured",
        "n_arm_top5_total",
        "n_grip_dev_nfe2_captured",
        "n_grip_dev_nfe2_total",
    ):
        out[key] = int(stats0.get(key, 0)) + int(stats1.get(key, 0))
    return out


def legacy_expected() -> dict:
    p = EVENT_DIR / "summary.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def verify_legacy(model: str, got: dict, legacy: dict) -> list[str]:
    mismatches: list[str] = []
    leg = legacy.get("models", {}).get(model, {}).get("by_noise", {})
    for gname in LEGACY_KEYS:
        for nid in ("0", "1"):
            if gname not in leg.get(nid, {}).get("groups", {}):
                continue
            ref = leg[nid]["groups"][gname]
            cur = got["by_noise"][nid]["groups"][gname]
            for field in (
                "n_tail_captured",
                "n_tail_bad_total",
                "n_grip_dev_nfe1_captured",
                "n_grip_dev_nfe1_total",
                "n_arm_top5_captured",
                "n_arm_top5_total",
            ):
                if int(ref[field]) != int(cur[field]):
                    mismatches.append(f"{model} {gname} noise{nid} {field}: legacy {ref[field]} got {cur[field]}")
    return mismatches


def model_block(model: str, df: pd.DataFrame, groups: dict[str, set[int]], g_all: set[int]) -> dict:
    by_noise = {}
    for nid in (0, 1):
        tail_bad, _, _ = tail_mask_nfe1(df, nid)
        g1 = grip_deviation_set(df, nid, 1)
        g2 = grip_deviation_set(df, nid, 2)
        arm5, _ = arm_top5_set(df, nid)
        by_noise[str(nid)] = {
            "groups": {
                gname: eval_selection(
                    gname, selected, g_all, tail_bad, g1, g2, arm5, df, nid
                )
                for gname, selected in groups.items()
            }
        }
    merged = {}
    for gname in groups:
        m = merged_two_noise(by_noise["0"]["groups"][gname], by_noise["1"]["groups"][gname])
        merged[gname] = m
    return {"model": model, "by_noise": by_noise, "merged_two_noises": merged}


def write_selection_index(
    metas, groups: dict[str, set[int]], seed_map: dict[str, int]
) -> None:
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    for gname, selected in groups.items():
        seed = seed_map.get(gname)
        payload = {
            "group": gname,
            "seed": seed,
            "n_selected": len(selected),
            "shared_indices": sorted(selected),
            "per_task": {},
        }
        by_task: dict[int, list[int]] = defaultdict(list)
        for si in selected:
            by_task[metas[si].task_index].append(si)
        for ti in sorted(by_task):
            payload["per_task"][str(ti)] = sorted(by_task[ti])
        fname = f"{gname}.json" if seed is None else f"{gname}_seed{seed}.json"
        (SELECTION_DIR / fname).write_text(json.dumps(payload, indent=2) + "\n")


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    blob = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    alpha = np.asarray(blob["alpha"], dtype=np.float64).reshape(-1)
    frames_df = _load_actions(LIBERO_REPO)
    actions = np.stack(frames_df["action"].to_numpy()).astype(np.float64)
    episode = frames_df["episode_index"].to_numpy()
    G = demo_gripper_switch(blob, actions, episode)
    probe_g = np.zeros(N_FRAMES, dtype=np.int32)
    metas = build_metas(blob, alpha, G, probe_g)
    g_all = set(np.where(G)[0])

    groups: dict[str, set[int]] = {
        "random": random_group(metas, GROUP_SEED),
        "care_alpha": select_per_task(metas, "alpha"),
        "care_event": select_per_task(metas, "alpha_event"),
        "g_only": g_only_group(metas, GROUP_SEED),
    }
    seed_map = {
        "random": GROUP_SEED,
        "care_alpha": None,
        "care_event": None,
        "g_only": GROUP_SEED,
    }
    write_selection_index(metas, {k: groups[k] for k in ("random", "care_alpha", "care_event", "g_only")}, seed_map)

    smol_df = load_probe_frames(MODELS["smolvla"])
    pi_df = load_probe_frames(MODELS["pi05"])
    legacy = legacy_expected()
    smol_main = model_block("smolvla", smol_df, groups, g_all)
    pi_main = model_block("pi05", pi_df, groups, g_all)
    repro = verify_legacy("smolvla", smol_main, legacy) + verify_legacy("pi05", pi_main, legacy)

    # Multi-seed random and G-only
    seed_stats: dict[str, dict[str, list[dict]]] = {"random": defaultdict(list), "g_only": defaultdict(list)}
    for seed in BUDGET_SEEDS:
        for label, fn in (("random", random_group), ("g_only", g_only_group)):
            sel = fn(metas, seed)
            write_selection_index(metas, {label: sel}, {label: seed})
            for model, df in (("smolvla", smol_df), ("pi05", pi_df)):
                blk = model_block(model, df, {label: sel}, g_all)
                m = blk["merged_two_noises"][label]
                seed_stats[label][model].append(
                    {
                        "seed": seed,
                        "grip_nfe1": f"{m['n_grip_dev_nfe1_captured']}/{m['n_grip_dev_nfe1_total']}",
                        "tail": f"{m['n_tail_captured']}/{m['n_tail_bad_total']}",
                        "arm_top5": f"{m['n_arm_top5_captured']}/{m['n_arm_top5_total']}",
                        "grip_nfe1_num": m["n_grip_dev_nfe1_captured"],
                        "tail_num": m["n_tail_captured"],
                        "arm_num": m["n_arm_top5_captured"],
                    }
                )

    agg = {}
    for label in ("random", "g_only"):
        agg[label] = {}
        for model in ("smolvla", "pi05"):
            rows = seed_stats[label][model]
            agg[label][model] = {
                "grip_nfe1_captured": aggregate_metrics(rows, "grip_nfe1_num"),
                "tail_captured": aggregate_metrics(rows, "tail_num"),
                "arm_top5_captured": aggregate_metrics(rows, "arm_num"),
                "per_seed": rows,
            }

    payload = {
        "protocol": {
            "pool_n": N_FRAMES,
            "per_task_k": TOP_K,
            "total_budget": 120,
            "budget_seeds": list(BUDGET_SEEDS),
            "legacy_random_seed": GROUP_SEED,
            "g_only_rule": "per task: shuffle G=1 then G=0 with same seed; no alpha tie-break",
        },
        "reproduce_legacy_ok": len(repro) == 0,
        "reproduce_mismatches": repro,
        "four_group": {
            "smolvla": smol_main,
            "pi05": pi_main,
        },
        "multi_seed_aggregate": agg,
    }
    (COMPARE_DIR / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    # Flat comparison table for paper appendix
    table_rows = []
    for model in ("smolvla", "pi05"):
        blk = payload["four_group"][model]["merged_two_noises"]
        for gname in ("random", "care_alpha", "g_only", "care_event"):
            m = blk[gname]
            table_rows.append(
                {
                    "model": model,
                    "group": gname,
                    "grip_nfe1": f"{m['n_grip_dev_nfe1_captured']}/{m['n_grip_dev_nfe1_total']}",
                    "tail_nfe1": f"{m['n_tail_captured']}/{m['n_tail_bad_total']}",
                    "arm_top5": f"{m['n_arm_top5_captured']}/{m['n_arm_top5_total']}",
                }
            )
    pd.DataFrame(table_rows).to_csv(COMPARE_DIR / "four_group_merged_two_noises.csv", index=False)

    md = [
        "# CARE_event 校准池等预算选样（480 帧，CPU 回放）",
        "",
        f"- 每任务 {TOP_K} 帧，共 120 帧；合并 noise0+noise1 的捕获数（非独立 episode）。",
        f"- 多随机 / G-only 种子：{', '.join(str(s) for s in BUDGET_SEEDS)}",
        f"- 与历史 `summary.json` 复现：{'通过' if not repro else '失败'}",
        "",
    ]
    if repro:
        md.append("复现差异：\n" + "\n".join(f"- {x}" for x in repro) + "\n")
    md.append("## 四组对照（合并两噪声）\n")
    md.append(markdown_table(table_rows))
    md.append("\n## 多随机种子波动（合并两噪声，捕获数）\n")
    for label in ("random", "g_only"):
        md.append(f"### {label}\n")
        for model in ("smolvla", "pi05"):
            a = agg[label][model]
            md.append(
                f"- **{model}** grip NFE1: mean={a['grip_nfe1_captured']['mean']:.2f} "
                f"std={a['grip_nfe1_captured']['std']:.2f} "
                f"[{a['grip_nfe1_captured']['min']:.0f},{a['grip_nfe1_captured']['max']:.0f}]; "
                f"tail: mean={a['tail_captured']['mean']:.2f} "
                f"std={a['tail_captured']['std']:.2f}; "
                f"arm top5%: mean={a['arm_top5_captured']['mean']:.2f} "
                f"std={a['arm_top5_captured']['std']:.2f}"
            )
        md.append("")
    (COMPARE_DIR / "REPORT.md").write_text("\n".join(md) + "\n")
    print(f"done -> {COMPARE_DIR}", flush=True)
    if repro:
        raise SystemExit(f"legacy reproduce failed: {repro}")


if __name__ == "__main__":
    main()

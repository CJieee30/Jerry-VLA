#!/usr/bin/env python3
"""CPU: four strategies × three models — per model×strategy×NFE (no cross-NFE averages)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dev_metrics_common import (
    MODEL_DIRS,
    POOL_CSV,
    SCREENING_DIR,
    build_conversion_check,
    grip_stats_for_selection,
    libero_mean_std,
    obs_arm_rmse_merged_two_noises,
    tail_stats_merged_noises,
    write_conversion_manifest,
)
from dump_shared import _load_actions
from event_care_nfe_diagnostic import (
    build_metas,
    demo_gripper_switch,
    load_probe_frames,
)
from protocol import LIBERO_REPO, OUT_DIR

SELECTION_DIR = OUT_DIR / "nfe_probe_event_care" / "budget_compare" / "selections"
STRATEGIES = {
    "random": SELECTION_DIR / "random_seed42.json",
    "alpha": SELECTION_DIR / "care_alpha.json",
    "g_only": SELECTION_DIR / "g_only_seed42.json",
    "alpha_event": SELECTION_DIR / "care_event.json",
}
ARCHIVE = SCREENING_DIR / "_archive_pre_nfe_split"
CLOSED_LOOP = {
    "pi05": {"nfe1": 97.00, "nfe2": None, "nfe10": 97.25, "n": 400, "source": "stage_nfe full"},
    "smolvla": {"nfe1": 79.25, "nfe2": 82.00, "nfe3": 81.00, "nfe10": 83.00, "n": 400, "source": "stage_nfe full"},
    "molmoact2": {
        "nfe1_2000": 97.80,
        "nfe10_2000": 97.50,
        "n": 2000,
        "source": "molmoact2_full_nfe1_2000 / baseline",
        "comparability": "2000-ep closed loop ≠ 400-ep Pi/Smol stage_nfe; label only",
    },
}


def load_selections() -> dict[str, set[int]]:
    out = {}
    for name, path in STRATEGIES.items():
        blob = json.loads(path.read_text())
        key = "shared_indices" if "shared_indices" in blob else "pool_index"
        out[name] = set(int(x) for x in blob[key])
    return out


def archive_old_reports() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for name in ("REPORT.md", "summary.json", "observation_level.csv"):
        p = SCREENING_DIR / name
        if p.is_file():
            shutil.copy2(p, ARCHIVE / f"{name}.v0_pre_nfe_split")


def aggregate_strategy_nfe(
    pool: pd.DataFrame,
    model: str,
    selected: set[int],
    z,
    df_native: pd.DataFrame | None,
) -> dict:
    sub = pool[(pool.model == model) & (pool.shared_index.isin(selected))]
    mean, std = libero_mean_std()
    by_nfe: dict[str, dict] = {}
    for nfe in sorted(sub.nfe.unique()):
        sn = sub[sub.nfe == nfe]
        arm_per_obs = [obs_arm_rmse_merged_two_noises(model, z, int(si), int(nfe), mean, std) for si in selected]
        block = {
            "n_selected_obs": len(selected),
            "n_comparison_rows": len(sn),
            "cosine_libero_mean": float(sn.cosine_libero.mean()),
            "rmse7_libero_mean": float(sn.rmse7_libero.mean()),
            "rmse6_arm_libero_mean": float(sn.rmse6_arm_libero.mean()),
            "rmse6_arm_libero_median": float(sn.rmse6_arm_libero.median()),
            "rmse7_libero_median": float(sn.rmse7_libero.median()),
            "obs_arm_p95_libero": float(np.percentile(arm_per_obs, 95)),
            "grip_libero": grip_stats_for_selection(sn, selected, int(nfe)),
        }
        if df_native is not None and int(nfe) == 1:
            block["tail_native_merged_two_noises"] = tail_stats_merged_noises(df_native, selected)
        by_nfe[str(int(nfe))] = block
    return by_nfe


def main() -> None:
    SCREENING_DIR.mkdir(parents=True, exist_ok=True)
    archive_old_reports()
    if not POOL_CSV.is_file():
        raise SystemExit(f"missing {POOL_CSV}; run export_pool_observation_metrics.py first")

    pool = pd.read_csv(POOL_CSV)
    write_conversion_manifest()
    build_conversion_check()

    shared = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    frames_df = _load_actions(LIBERO_REPO)
    actions = np.stack(frames_df["action"].to_numpy()).astype(np.float64)
    episode = frames_df["episode_index"].to_numpy()
    G_flags = demo_gripper_switch(shared, actions, episode)
    metas = build_metas(shared, np.asarray(shared["alpha"]), G_flags, G_flags)
    G_all = {m.shared_index for m in metas if m.G == 1}
    selections = load_selections()

    summary = {
        "note": "All aggregates are model×strategy×NFE; obs_arm_p95 pools 2 noises×10×6 arm per observation.",
        "closed_loop_labels": CLOSED_LOOP,
        "selections": {k: len(v) for k, v in selections.items()},
        "models": {},
    }
    strat_rows = []
    for model, mdir in MODEL_DIRS.items():
        if not (mdir / "actions.npz").is_file():
            summary["models"][model] = {"status": "missing_probe"}
            continue
        z = np.load(mdir / "actions.npz")
        df_native = load_probe_frames(mdir) if (mdir / "frames.csv").is_file() else None
        model_sum = {"strategies": {}}
        for strat, selected in selections.items():
            by_nfe = aggregate_strategy_nfe(pool, model, selected, z, df_native)
            model_sum["strategies"][strat] = {"G1": len(selected & G_all), "by_nfe": by_nfe}
            for nfe, block in by_nfe.items():
                g = block["grip_libero"]
                tail = block.get("tail_native_merged_two_noises", {})
                strat_rows.append(
                    {
                        "model": model,
                        "strategy": strat,
                        "nfe": int(nfe),
                        "G1": len(selected & G_all),
                        "cos_mean": block["cosine_libero_mean"],
                        "rmse7_mean": block["rmse7_libero_mean"],
                        "rmse6_median": block["rmse6_arm_libero_median"],
                        "arm_obs_p95": block["obs_arm_p95_libero"],
                        "grip_cmp": f"{g['grip_comparisons_with_disagree']}/{g['grip_comparisons_total']}",
                        "grip_obs": f"{g['grip_unique_obs_with_disagree']}/{g['grip_unique_obs_denominator']}",
                        "tail_cmp": (
                            f"{tail.get('tail_comparisons_captured', '—')}/{tail.get('tail_comparisons_total', '—')}"
                            if tail
                            else "—"
                        ),
                        "tail_obs": (
                            f"{tail.get('tail_unique_obs_captured', '—')}/{tail.get('tail_unique_obs_denominator_pool', '—')}"
                            if tail
                            else "—"
                        ),
                    }
                )
        summary["models"][model] = model_sum

    (SCREENING_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(strat_rows).to_csv(SCREENING_DIR / "strategy_nfe_table.csv", index=False)

    lines = [
        "# 三模型开发筛查（按 NFE 拆分，LIBERO 共同空间）",
        "",
        "阈值与推荐规则仍为**开发候选**，非已验证通用规则。",
        "手臂 P95：每观测先合并两噪声的 10×6 误差再算标量，再对策略内观测取 95 分位。",
        "夹爪/尾部：**比较数**（观测×噪声行）与**去重观测数**并列，分母见表头。",
        "",
    ]
    for model in summary["models"]:
        ms = summary["models"][model]
        if ms.get("status") == "missing_probe":
            lines.append(f"- **{model}**: 探针缺失")
            continue
        lines.append(f"## {model}")
        lines.append(
            "| 策略 | NFE | G=1 | cos↑ | rmse7 mean | rmse6 med | arm obs P95 | 夹爪(比较) | 夹爪(观测) | 尾部(比较) | 尾部(观测) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for row in strat_rows:
            if row["model"] != model:
                continue
            lines.append(
                f"| {row['strategy']} | {row['nfe']} | {row['G1']} | {row['cos_mean']:.5f} | "
                f"{row['rmse7_mean']:.4f} | {row['rmse6_median']:.4f} | {row['arm_obs_p95']:.4f} | "
                f"{row['grip_cmp']} | {row['grip_obs']} | {row['tail_cmp']} | {row['tail_obs']} |"
            )
        lines.append("")
    (SCREENING_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {SCREENING_DIR}/REPORT.md and summary.json")


if __name__ == "__main__":
    main()

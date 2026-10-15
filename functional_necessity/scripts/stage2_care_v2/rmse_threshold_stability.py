#!/usr/bin/env python3
"""20-seed stability for a few fixed RMSE rule candidates (dev only, not a grid search)."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dev_metrics_common import MODEL_DIRS, POOL_CSV, SCREENING_DIR
from event_care_nfe_diagnostic import grip_deviation_set
from protocol import OUT_DIR

SELECTION_DIR = OUT_DIR / "nfe_probe_event_care" / "budget_compare" / "selections"
CANDIDATES_NFE = [1, 2, 3]
REF_NFE = 10


def load_seed_selections(prefix: str) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for p in sorted(SELECTION_DIR.glob(f"{prefix}_seed*.json")):
        m = re.search(r"_seed(\d+)\.json$", p.name)
        if not m:
            continue
        blob = json.loads(p.read_text())
        out[int(m.group(1))] = set(int(x) for x in blob["shared_indices"])
    return out


def precompute_arm_obs_p95(pool: pd.DataFrame) -> pd.DataFrame:
    """Per (model, shared_index, nfe) merged-two-noise arm scalar."""
    rows = []
    for (model, si, nfe), g in pool.groupby(["model", "shared_index", "nfe"]):
        errs = []
        for _, r in g.iterrows():
            # approximate: use rmse6 per row; merge as pooled RMS of per-noise RMSEs
            errs.append(float(r.rmse6_arm_libero) ** 2)
        rows.append({"model": model, "shared_index": int(si), "nfe": int(nfe), "arm_obs_merged": float(np.sqrt(np.mean(errs)))})
    return pd.DataFrame(rows)


def grip_rate_delta(df_native: pd.DataFrame, selected: set[int], nfe: int, floor_nfe: int) -> float:
    def rate(n: int) -> float:
        bad = tot = 0
        for nid in (0, 1):
            g = grip_deviation_set(df_native, nid, n)
            for si in selected:
                tot += 1
                if si in g:
                    bad += 1
        return bad / tot if tot else 0.0

    return rate(nfe) - rate(floor_nfe)


def selection_block(pool: pd.DataFrame, arm_obs: pd.DataFrame, model: str, selected: set[int], nfe: int) -> dict[str, float]:
    sub = pool[(pool.model == model) & (pool.nfe == nfe) & (pool.shared_index.isin(selected))]
    ao = arm_obs[(arm_obs.model == model) & (arm_obs.nfe == nfe) & (arm_obs.shared_index.isin(selected))]
    return {
        "rmse7_median": float(sub.rmse7_libero.median()),
        "rmse6_median": float(sub.rmse6_arm_libero.median()),
        "rmse6_obs_p95": float(ao.arm_obs_merged.quantile(0.95)),
        "cos_median": float(sub.cosine_libero.median()),
    }


def recommend_fast(
    pool: pd.DataFrame,
    arm_obs: pd.DataFrame,
    df_native: pd.DataFrame,
    model: str,
    selected: set[int],
    rule: str,
    floors: dict[int, dict[str, float]],
    pool_p95_nfe1: float,
) -> int:
    passed = []
    for nfe in CANDIDATES_NFE:
        if not ((pool.model == model) & (pool.nfe == nfe)).any():
            continue
        sel = selection_block(pool, arm_obs, model, selected, nfe)
        fl = floors[nfe]
        if rule == "ratio_rmse6_median_le_1":
            ok = sel["rmse6_median"] / max(fl["rmse6_median"], 1e-12) <= 1.0
        elif rule == "ratio_rmse7_median_le_1":
            ok = sel["rmse7_median"] / max(fl["rmse7_median"], 1e-12) <= 1.0
        elif rule == "abs_rmse6_obs_p95_le_pool_p95_nfe1":
            ok = sel["rmse6_obs_p95"] <= pool_p95_nfe1
        else:
            raise ValueError(rule)
        if ok and grip_rate_delta(df_native, selected, nfe, REF_NFE) <= 0.0:
            passed.append(nfe)
    return min(passed) if passed else REF_NFE


def main() -> None:
    if not POOL_CSV.is_file():
        raise SystemExit("run export_pool_observation_metrics.py first")
    pool = pd.read_csv(POOL_CSV)
    arm_obs = precompute_arm_obs_p95(pool)
    random_seeds = load_seed_selections("random")
    gonly_seeds = load_seed_selections("g_only")
    rules = [
        ("ratio_rmse6_median_le_1", "六维手臂 RMSE 中位 / 全池同 NFE 中位 ≤ 1.0 + 夹爪率不差于 NFE10"),
        ("ratio_rmse7_median_le_1", "七维 RMSE 中位 / 全池同 NFE 中位 ≤ 1.0 + 夹爪率不差于 NFE10"),
        ("abs_rmse6_obs_p95_le_pool_p95_nfe1", "策略内观测级手臂 P95 ≤ 全池 NFE1 观测 P95 + 夹爪率不差于 NFE10"),
    ]
    out = {
        "status": "dev_candidate_only",
        "units": {
            "rmse7": "LIBERO-normalized 7-d vs NFE10 (median over comparison rows in selection)",
            "rmse6": "LIBERO-normalized 6-d arm vs NFE10 (median over comparison rows)",
            "rmse6_obs_p95": "merged per-obs arm scalar (2-noise RMS of rmse6 rows), P95 over selection",
        },
        "cosine": "auxiliary only",
        "models": {},
        "stability": {},
    }
    rows = []
    for model in MODEL_DIRS:
        mdir = MODEL_DIRS[model]
        if not (mdir / "frames.csv").is_file():
            continue
        df_native = pd.read_csv(mdir / "frames.csv")
        floors = {}
        for nfe in CANDIDATES_NFE:
            sub = pool[(pool.model == model) & (pool.nfe == nfe)]
            if sub.empty:
                continue
            floors[nfe] = {
                "rmse7_median": float(sub.rmse7_libero.median()),
                "rmse6_median": float(sub.rmse6_arm_libero.median()),
            }
        ao1 = arm_obs[(arm_obs.model == model) & (arm_obs.nfe == 1)]
        pool_p95_nfe1 = float(ao1.arm_obs_merged.quantile(0.95))
        out["models"][model] = {"floor_pool_medians": floors, "pool_arm_obs_p95_nfe1": pool_p95_nfe1}
        all_idx = set(pool[pool.model == model].shared_index.unique())
        pool_recs = {
            rid: recommend_fast(pool, arm_obs, df_native, model, all_idx, rid, floors, pool_p95_nfe1) for rid, _ in rules
        }
        for strat_name, seed_map in (("random", random_seeds), ("g_only", gonly_seeds)):
            hist: dict[str, list[int]] = defaultdict(list)
            for seed, selected in sorted(seed_map.items()):
                for rule_id, _ in rules:
                    r = recommend_fast(pool, arm_obs, df_native, model, selected, rule_id, floors, pool_p95_nfe1)
                    hist[rule_id].append(r)
                    pool_rec = pool_recs[rule_id]
                    rows.append(
                        {
                            "model": model,
                            "strategy": strat_name,
                            "seed": seed,
                            "rule": rule_id,
                            "recommended_nfe": r,
                            "pool_recommended_nfe": pool_rec,
                            "conservative_reject": r == REF_NFE and pool_rec < REF_NFE,
                            "false_accept": r < REF_NFE and pool_rec == REF_NFE,
                        }
                    )
            out["stability"][f"{model}_{strat_name}"] = {
                rule_id: {
                    "n_seeds": len(seed_map),
                    "recommendation_counts": {
                        str(nfe): int(sum(1 for x in hist[rule_id] if x == nfe)) for nfe in sorted(set(hist[rule_id]))
                    },
                }
                for rule_id, _ in rules
            }
    df = pd.DataFrame(rows)
    (SCREENING_DIR / "threshold_stability.json").write_text(json.dumps(out, indent=2) + "\n")
    df.to_csv(SCREENING_DIR / "threshold_stability_seeds.csv", index=False)

    md = [
        "# RMSE 阈值候选（开发候选，非冻结规则）",
        "",
        "仅比较三条固定规则；未做组合搜索以保证指定 NFE。",
        "",
    ]
    for rule_id, desc in rules:
        md.append(f"## `{rule_id}`")
        md.append(desc)
        sub = df[df.rule == rule_id]
        for model in sorted(sub.model.unique()):
            sm = sub[sub.model == model]
            md.append(
                f"- **{model}** random counts={sm[sm.strategy=='random'].recommended_nfe.value_counts().to_dict()}; "
                f"g_only counts={sm[sm.strategy=='g_only'].recommended_nfe.value_counts().to_dict()}; "
                f"conservative_reject={int(sm.conservative_reject.sum())}, false_accept={int(sm.false_accept.sum())}"
            )
        md.append("")
    md += [
        "## 局限",
        "- 观测级手臂合并用两噪声 rmse6 的 RMS 近似（与 screening 全量 10×6 池化略有数值差）。",
        "- 夹爪率基于 frames.csv 原生 `gripper_disagree_n`。",
        "- 闭环 episode 数不一致（Molmo 2000 vs Pi/Smol 400）。",
    ]
    (SCREENING_DIR / "THRESHOLD_CANDIDATES.md").write_text("\n".join(md) + "\n")
    print(f"wrote threshold_stability.json ({len(df)} seed rows)")


if __name__ == "__main__":
    main()

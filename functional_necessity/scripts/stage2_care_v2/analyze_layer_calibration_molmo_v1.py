#!/usr/bin/env python3
"""CPU: prospective Molmo layer-calibration screening (subset selection + equal weight)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dev_nfe_rules_cpu_v2 import g_only_group_k, random_group_k, select_per_task_k
from prospective_molmo_layer_common import (
    DROP_K,
    EXCLUDE_L0,
    HISTORICAL_DROP16,
    METHODS,
    MOLMO_LAYER_DIR,
    N_LAYERS,
    N_OBS,
    SCORE_DIR,
    SEL_DIR,
    load_calibration_pool,
    map_historical,
    metas_for_selection,
    verify_no_dev_overlap,
)
from protocol import drop_from_redundancy, importance_ranking, jaccard


BUDGETS = ((40, 1), (80, 2), (120, 3))
STRATEGIES = ("random", "alpha", "g_only", "care_event")
STOCHASTIC_SEEDS = tuple(range(20))
PRIMARY_BUDGET = 80


def drop_set_from_scores(scores, k: int = DROP_K) -> list[int]:
    order = importance_ranking(scores)
    return drop_from_redundancy(order, k, exclude_l0=EXCLUDE_L0)


def load_scores(method: str) -> torch.Tensor:
    path = SCORE_DIR / f"molmoact2_{method}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    mat = blob["frame_scores"].float()
    if tuple(mat.shape) != (N_OBS, N_LAYERS):
        raise RuntimeError(f"{method}: {tuple(mat.shape)}")
    return mat


def select(strategy: str, wrapped, k: int, seed: int | None) -> set[int]:
    if strategy == "random":
        return random_group_k(wrapped, seed, k)
    if strategy == "g_only":
        return g_only_group_k(wrapped, seed, k)
    if strategy == "alpha":
        return select_per_task_k(wrapped, "alpha", k)
    if strategy == "care_event":
        return select_per_task_k(wrapped, "alpha_event", k)
    raise ValueError(strategy)


def write_selection(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def recommend_closed_loop(details: pd.DataFrame) -> list[dict]:
    """At most two new CL configs; skip exact matches to verified historical sets."""
    verified_layers = {tuple(v[2]) for k, v in HISTORICAL_DROP16.items() if v[1] is not None}
    sub = details[
        (details["budget_frames"] == PRIMARY_BUDGET) & (details["method"].isin(["fisher", "taylor", "igia"]))
    ]
    candidates = []
    for method in ("fisher", "taylor", "igia"):
        for strat in ("care_event", "alpha", "random"):
            rows = sub[(sub["method"] == method) & (sub["strategy"] == strat)]
            if rows.empty:
                continue
            if strat == "random":
                # use modal drop across seeds
                mode = rows["drop_json"].mode().iloc[0]
                row = rows[rows["drop_json"] == mode].iloc[0]
            else:
                row = rows.iloc[0]
            drop = json.loads(row["drop_json"])
            if tuple(drop) in verified_layers and strat != "random":
                continue
            if row["comparability"] == "可比性已核实":
                continue
            candidates.append(
                {
                    "method": method,
                    "strategy": strat,
                    "budget": PRIMARY_BUDGET,
                    "drop": drop,
                    "jaccard_full": row["jaccard_vs_full400_equal"],
                    "jaccard_taylor_stage1": row.get("jaccard_vs_taylor_stage1"),
                    "rationale": row.get("note", ""),
                }
            )
    # Fisher 80 care_event vs random diversity
    fisher_ce = sub[(sub.method == "fisher") & (sub.strategy == "care_event")]
    fisher_r = sub[(sub.method == "fisher") & (sub.strategy == "random")]
    out = []
    if not fisher_ce.empty and not fisher_r.empty:
        ce_drop = fisher_ce.iloc[0]["drop_json"]
        r_uniq = fisher_r["drop_json"].nunique()
        if fisher_r["drop_json"].ne(ce_drop).any() or r_uniq > 1:
            d = json.loads(ce_drop)
            if tuple(d) not in verified_layers or d != HISTORICAL_DROP16["S1_Taylor-16"][2]:
                out.append(
                    {
                        "priority": 1,
                        "config": f"Fisher-16, budget={PRIMARY_BUDGET}, CARE_event",
                        "drop": d,
                        "why": "Primary budget; compare CARE_event modal drop vs random seed distribution (Fisher cliff).",
                        "comparability": fisher_ce.iloc[0]["comparability"],
                    }
                )
    # Second: IGIA if drop differs from Fisher/Taylor historical
    igia_ce = sub[(sub.method == "igia") & (sub.strategy == "care_event")]
    if not igia_ce.empty:
        d = json.loads(igia_ce.iloc[0]["drop_json"])
        if d != HISTORICAL_DROP16["S1_IGIA-16"][2]:
            out.append(
                {
                    "priority": 2,
                    "config": f"IGIA-16, budget={PRIMARY_BUDGET}, CARE_event",
                    "drop": d,
                    "why": "Test whether IGIA signal under event selection differs from known IGIA-16 cliff set.",
                    "comparability": igia_ce.iloc[0]["comparability"],
                }
            )
    return out[:2]


def run_analyze() -> None:
    MOLMO_LAYER_DIR.mkdir(parents=True, exist_ok=True)
    SEL_DIR.mkdir(parents=True, exist_ok=True)
    overlap = verify_no_dev_overlap()
    pool = load_calibration_pool()
    wrapped = metas_for_selection(pool)
    taylor_stage1 = HISTORICAL_DROP16["S1_Taylor-16"][2]

    score_mats = {m: load_scores(m) for m in METHODS}
    full_drops = {m: drop_set_from_scores(score_mats[m].mean(dim=0)) for m in METHODS}

    rows = []
    freq: dict[tuple, Counter] = defaultdict(Counter)

    for method in METHODS:
        frame = score_mats[method]
        ref_drop = full_drops[method]
        for budget, k in BUDGETS:
            scenarios: list[tuple[str, int | None]] = []
            for strat in STRATEGIES:
                if strat in ("random", "g_only"):
                    for seed in STOCHASTIC_SEEDS:
                        scenarios.append((strat, seed))
                else:
                    scenarios.append((strat, None))

            for strat, seed in scenarios:
                sel = select(strat, wrapped, k, seed)
                if len(sel) != budget:
                    raise RuntimeError(f"{method} {strat} seed={seed}: got {len(sel)}")
                idx = sorted(sel)
                drop = drop_set_from_scores(frame[idx].mean(dim=0))
                djson = json.dumps(drop)
                freq[(method, budget, strat)][djson] += 1
                hist = map_historical(drop)
                j_full = jaccard(drop, ref_drop)
                j_taylor = jaccard(drop, taylor_stage1)
                note = ""
                if method == "fisher" and budget == PRIMARY_BUDGET:
                    if drop == HISTORICAL_DROP16["S1_Taylor-16"][2]:
                        note = "Fisher scores → Taylor-16 layer set (known 98.10 mapping)"
                    elif drop == HISTORICAL_DROP16["S1_Fisher-16"][2]:
                        note = "Matches S1 Fisher-16 cliff (72.75)"

                sel_path = SEL_DIR / f"{method}_b{budget}_{strat}" + (
                    f"_seed{seed}.json" if seed is not None else ".json"
                )
                write_selection(
                    sel_path,
                    protocol="layer_calibration_molmo_v1",
                    method=method,
                    strategy=strat,
                    budget_frames=budget,
                    k_per_task=k,
                    seed=seed,
                    pool_indices=idx,
                    drop_layers=drop,
                )

                rows.append(
                    {
                        "method": method,
                        "strategy": strat,
                        "budget_frames": budget,
                        "k_per_task": k,
                        "seed": seed if seed is not None else "NA",
                        "drop_layers": drop,
                        "drop_json": djson,
                        "jaccard_vs_full400_equal": j_full,
                        "full400_equal_drop": ref_drop,
                        "jaccard_vs_taylor_stage1": j_taylor,
                        "note": note,
                        **hist,
                    }
                )

    details = pd.DataFrame(rows)
    details.to_csv(MOLMO_LAYER_DIR / "details.csv", index=False)

    freq_rows = []
    for key, ctr in sorted(freq.items()):
        method, budget, strat = key
        total = sum(ctr.values())
        for djson, count in ctr.most_common():
            freq_rows.append(
                {
                    "method": method,
                    "budget_frames": budget,
                    "strategy": strat,
                    "drop_json": djson,
                    "count": count,
                    "fraction": count / total,
                }
            )
    pd.DataFrame(freq_rows).to_csv(MOLMO_LAYER_DIR / "drop_frequency.csv", index=False)

    summary = {
        "protocol": "layer_calibration_molmo_v1",
        "status": "development_diagnostic",
        "run_name": "prospective_v1_20261015",
        "n_obs": N_OBS,
        "overlap_with_care_v2_dev": overlap,
        "full400_drops": full_drops,
        "primary_budget": PRIMARY_BUDGET,
        "closed_loop_recommendations": recommend_closed_loop(details),
        "scores_present": {m: (SCORE_DIR / f"molmoact2_{m}.pt").is_file() for m in METHODS},
    }
    (MOLMO_LAYER_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(details, summary, overlap)
    print(f"wrote {MOLMO_LAYER_DIR}", flush=True)


def write_report(details: pd.DataFrame, summary: dict, overlap: dict) -> None:
    b80 = details[details["budget_frames"] == PRIMARY_BUDGET]
    lines = [
        "# MolmoAct2 前瞻校准池层剪枝选样（v1）",
        "",
        "**开发诊断** · 子集选样后**等权**层分数（非旧 α 加权实验）。",
        "",
        "## 数据",
        "",
        f"- 池：`prospective_v1_20261015` **{N_OBS}** obs / **400** episodes / 40 任务 × 10 帧。",
        f"- 与 CARE v2 开发池 episode 重叠：**{overlap['overlap']}**（应为 0）。",
        f"- 分数文件：{summary['scores_present']}",
        "",
        "## 设置",
        "",
        "- 指标：IGIA (Σg²)、Fisher (Σ(wg)²)、Taylor (Σ|wg|)、GateProbe（与 V2/Molmo 现网一致）。",
        "- 策略：Random / α-only / G-only / CARE_event；种子 0–19（随机类）。",
        f"- 主预算 **{PRIMARY_BUDGET}** 帧（每任务 2）；补充 40 / 120。",
        "- Drop-16，排除 L0。",
        "",
        "## 问题 1：CARE_event vs Random（主预算 80）",
        "",
    ]

    for method in METHODS:
        ce = b80[(b80["method"] == method) & (b80["strategy"] == "care_event")]
        rnd = b80[(b80["method"] == method) & (b80["strategy"] == "random")]
        if ce.empty:
            continue
        ce_drop = ce.iloc[0]["drop_json"]
        r_uniq = rnd["drop_json"].nunique()
        diff_frac = float(rnd["drop_json"].ne(ce_drop).mean())
        j_ce = float(ce.iloc[0]["jaccard_vs_full400_equal"])
        lines.append(
            f"- **{method}**：CARE_event J(full400)={j_ce:.2f}；"
            f"random {r_uniq} 种 drop、{diff_frac*100:.0f}% 种子与 CARE 不同。"
        )

    lines += ["", "## 问题 2：能否被 α-only / G-only 替代？", ""]
    for method in METHODS:
        for budget in (40, 80, 120):
            sub = details[(details["method"] == method) & (details["budget_frames"] == budget)]
            ce = sub[sub["strategy"] == "care_event"].iloc[0]["drop_json"]
            al = sub[sub["strategy"] == "alpha"].iloc[0]["drop_json"]
            same_a = ce == al
            g = sub[sub["strategy"] == "g_only"]
            diff_g = float(g["drop_json"].ne(ce).mean())
            if not same_a or diff_g > 0:
                lines.append(
                    f"- {method} b={budget}: CARE vs α 相同={same_a}；"
                    f"G-only 与 CARE 不同比例={diff_g*100:.0f}%"
                )

    lines += ["", "## 问题 3：IGIA 新信号？", ""]
    for budget in (40, 80, 120):
        ig = details[(details["method"] == "igia") & (details["budget_frames"] == budget)]
        fi = details[(details["method"] == "fisher") & (details["budget_frames"] == budget)]
        if ig.empty:
            continue
        ig_ce = ig[ig["strategy"] == "care_event"].iloc[0]
        fi_ce = fi[fi["strategy"] == "care_event"].iloc[0]
        same_f = ig_ce["drop_json"] == fi_ce["drop_json"]
        lines.append(
            f"- b={budget}: IGIA vs Fisher（CARE_event）drop 相同={same_f}；"
            f"IGIA J(Taylor-S1)={ig_ce['jaccard_vs_taylor_stage1']:.2f}"
        )

    lines += [
        "",
        "## 问题 4：历史复用 vs 待评测",
        "",
        "见 `details.csv` 的 `comparability` / `historical_label`。",
        "Taylor-16 与已有 98.10% 闭环 **层集相同** → 不重跑。",
        "",
        "## 闭环候选（未启动）",
        "",
    ]
    for rec in summary.get("closed_loop_recommendations", []):
        lines.append(f"- **{rec.get('config', rec)}**：`{rec.get('drop')}` — {rec.get('why', '')}")

    lines += [
        "",
        "## 文件",
        "",
        "- `scores/molmoact2_{igia,fisher,taylor,gateprobe}.pt`",
        "- `details.csv`, `drop_frequency.csv`, `selections/`",
        "",
        "生成：`analyze_layer_calibration_molmo_v1.py`（需先 `score_molmo_prospective_layers.py`）。",
    ]
    (MOLMO_LAYER_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run_analyze()


if __name__ == "__main__":
    main()

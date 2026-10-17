#!/usr/bin/env python3
"""CPU: Molmo prospective layer-calibration v1 rev2 (subset + equal weight)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dev_nfe_rules_cpu_v2 import g_only_group_k, random_group_k, select_per_task_k
from prospective_molmo_layer_common import (
    BUDGETS,
    DROP_K,
    EXCLUDE_L0,
    FRAME_METHODS,
    HISTORICAL_DROP16,
    MOLMO_LAYER_DIR,
    N_LAYERS,
    N_OBS,
    PRIMARY_BUDGET,
    PROTOCOL_ID,
    PROTOCOL_NOTE,
    SCORE_DIR,
    STOCHASTIC_SEEDS,
    STRATEGIES,
    TAYLOR_BUDGET,
    TAYLOR_STOCHASTIC_SEED,
    load_calibration_pool,
    map_historical,
    metas_for_selection,
    recommend_closed_loop_candidates,
    selection_path,
    taylor_group_path,
    verify_no_dev_overlap,
)
from protocol import drop_from_redundancy, importance_ranking, jaccard


ANALYSIS_METHODS = ("igia", "fisher", "gateprobe", "taylor")


def drop_set_from_scores(scores, k: int = DROP_K) -> list[int]:
    order = importance_ranking(scores)
    return drop_from_redundancy(order, k, exclude_l0=EXCLUDE_L0)


def load_frame_scores(method: str) -> torch.Tensor:
    path = SCORE_DIR / f"molmoact2_{method}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    mat = blob["frame_scores"].float()
    if tuple(mat.shape) != (N_OBS, N_LAYERS):
        raise RuntimeError(f"{method}: {tuple(mat.shape)}")
    return mat


def load_taylor_group(strategy: str, seed: int | None) -> dict:
    path = taylor_group_path(strategy, seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


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


def append_frame_rows(
    rows: list,
    freq: dict,
    method: str,
    frame: torch.Tensor,
    ref_drop: list[int],
    taylor_stage1: list[int],
    wrapped,
) -> None:
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

            sel_path = selection_path(method, budget, strat, seed)
            write_selection(
                sel_path,
                protocol=PROTOCOL_ID,
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


def append_taylor_rows(
    rows: list,
    freq: dict,
    taylor_stage1: list[int],
    wrapped,
) -> None:
    scenarios = [
        ("random", TAYLOR_STOCHASTIC_SEED),
        ("g_only", TAYLOR_STOCHASTIC_SEED),
        ("alpha", None),
        ("care_event", None),
    ]
    ref_drop = HISTORICAL_DROP16["S1_Taylor-16"][2]
    for strat, seed in scenarios:
        blob = load_taylor_group(strat, seed)
        scores = blob["layer_scores"]
        drop = drop_set_from_scores(scores)
        djson = json.dumps(drop)
        freq[("taylor", TAYLOR_BUDGET, strat)][djson] += 1
        hist = map_historical(drop)
        j_taylor = jaccard(drop, taylor_stage1)
        j_full = jaccard(drop, ref_drop)
        idx = blob["pool_indices"]
        sel_path = selection_path("taylor", TAYLOR_BUDGET, strat, seed)
        write_selection(
            sel_path,
            protocol=PROTOCOL_ID,
            method="taylor",
            strategy=strat,
            budget_frames=TAYLOR_BUDGET,
            k_per_task=2,
            seed=seed,
            pool_indices=idx,
            drop_layers=drop,
            taylor_formula=blob.get("formula"),
        )
        rows.append(
            {
                "method": "taylor",
                "strategy": strat,
                "budget_frames": TAYLOR_BUDGET,
                "k_per_task": 2,
                "seed": seed if seed is not None else "NA",
                "drop_layers": drop,
                "drop_json": djson,
                "jaccard_vs_full400_equal": j_full,
                "full400_equal_drop": ref_drop,
                "jaccard_vs_taylor_stage1": j_taylor,
                "note": "Molmo orig Taylor on selected group (not Taylor_absmean)",
                **hist,
            }
        )


def scores_ready() -> dict[str, bool]:
    frame_ok = {m: (SCORE_DIR / f"molmoact2_{m}.pt").is_file() for m in FRAME_METHODS}
    taylor_ok = all(
        taylor_group_path(s, seed).is_file()
        for s, seed in (
            ("random", TAYLOR_STOCHASTIC_SEED),
            ("g_only", TAYLOR_STOCHASTIC_SEED),
            ("alpha", None),
            ("care_event", None),
        )
    )
    return {**frame_ok, "taylor": taylor_ok}


def run_analyze(require_scores: bool = True) -> int:
    MOLMO_LAYER_DIR.mkdir(parents=True, exist_ok=True)
    present = scores_ready()
    if require_scores and not all(present.values()):
        print(f"scores incomplete: {present}", flush=True)
        return 1

    overlap = verify_no_dev_overlap()
    pool = load_calibration_pool()
    wrapped = metas_for_selection(pool)
    taylor_stage1 = HISTORICAL_DROP16["S1_Taylor-16"][2]

    rows: list[dict] = []
    freq: dict[tuple, Counter] = defaultdict(Counter)
    full_drops: dict[str, list] = {}

    if all(present[m] for m in FRAME_METHODS):
        score_mats = {m: load_frame_scores(m) for m in FRAME_METHODS}
        full_drops = {m: drop_set_from_scores(score_mats[m].mean(dim=0)) for m in FRAME_METHODS}
        for method in FRAME_METHODS:
            append_frame_rows(
                rows,
                freq,
                method,
                score_mats[method],
                full_drops[method],
                taylor_stage1,
                wrapped,
            )

    if present["taylor"]:
        append_taylor_rows(rows, freq, taylor_stage1, wrapped)

    if not rows:
        print("no detail rows (scores missing?)", flush=True)
        return 1

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

    detail_dicts = details.to_dict(orient="records")
    cl = recommend_closed_loop_candidates(detail_dicts)

    summary = {
        "protocol": PROTOCOL_ID,
        "protocol_note": PROTOCOL_NOTE,
        "status": "development_diagnostic" if all(present.values()) else "partial_scores",
        "run_name": "prospective_v1_20261015",
        "n_obs": N_OBS,
        "overlap_with_care_v2_dev": overlap,
        "full400_drops": full_drops,
        "primary_budget": PRIMARY_BUDGET,
        "closed_loop_recommendations": cl,
        "scores_present": present,
        "taylor_reference_drop_s1": taylor_stage1,
    }
    (MOLMO_LAYER_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(details, summary, overlap)
    print(f"wrote {MOLMO_LAYER_DIR}", flush=True)
    return 0


def write_report(details: pd.DataFrame, summary: dict, overlap: dict) -> None:
    b80 = details[details["budget_frames"] == PRIMARY_BUDGET]
    lines = [
        "# MolmoAct2 前瞻校准池层剪枝选样（v1 rev2）",
        "",
        "**开发诊断** · 子集选样后**等权**层分数（非旧 α 加权实验）。",
        "",
        f"协议：`{PROTOCOL_ID}` — {summary.get('protocol_note', '')}",
        "",
        "## 数据",
        "",
        f"- 池：`prospective_v1_20261015` **{N_OBS}** obs / 40 任务 × 10 帧。",
        f"- 与 CARE v2 开发池 episode 重叠：**{overlap['overlap']}**（应为 0）。",
        f"- 分数文件：{summary['scores_present']}",
        "",
        "## 设置",
        "",
        "- 逐帧（一次反传）：IGIA (Σg²)、Fisher (Σ(g·θ)²)、GateProbe。",
        "- **Taylor**：原 Molmo 公式 `Σ_w |Σ_i (g_i·θ_i)| / N`（仅 b=80、四策略；Random/G-only seed=0）。",
        "- **Taylor_absmean**（逐帧先 |g·θ| 再平均）为旧 V2 口径，**不与 Taylor 混用**。",
        "- 策略：Random / α-only / G-only / CARE_event；Random/G-only 种子 0–19（逐帧指标）。",
        f"- 主预算 **{PRIMARY_BUDGET}** 帧（每任务 2）；补充 40 / 120。",
        "- Drop-16，排除 L0。",
        "",
        "## 问题 1：CARE_event vs Random（主预算 80，先看 Fisher 再看 IGIA）",
        "",
    ]

    for method in ("fisher", "igia", "gateprobe", "taylor"):
        ce = b80[(b80["method"] == method) & (b80["strategy"] == "care_event")]
        rnd = b80[(b80["method"] == method) & (b80["strategy"] == "random")]
        if ce.empty or rnd.empty:
            continue
        ce_drop = ce.iloc[0]["drop_json"]
        r_uniq = rnd["drop_json"].nunique()
        diff_frac = float(rnd["drop_json"].ne(ce_drop).mean()) if method != "taylor" else float(
            ce_drop != rnd.iloc[0]["drop_json"]
        )
        j_ce = float(ce.iloc[0]["jaccard_vs_full400_equal"])
        lines.append(
            f"- **{method}**：CARE_event J(ref)={j_ce:.2f}；"
            f"random 与 CARE drop 不同比例≈{diff_frac*100:.0f}%"
            + (f"（{r_uniq} 种 random drop）" if method != "taylor" else "（Taylor 仅 seed=0）")
        )

    lines += ["", "## 问题 2：能否被 α-only / G-only 替代？", ""]
    for method in ("fisher", "igia", "gateprobe"):
        for budget in (40, 80, 120):
            sub = details[(details["method"] == method) & (details["budget_frames"] == budget)]
            if sub.empty:
                continue
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

    lines += ["", "## 问题 3：原 Taylor 稳定性（b=80 分组）", ""]
    taylor_b80 = details[(details["method"] == "taylor") & (details["budget_frames"] == PRIMARY_BUDGET)]
    if not taylor_b80.empty:
        for _, row in taylor_b80.iterrows():
            lines.append(
                f"- {row['strategy']}: drop J(S1 Taylor-16)={row['jaccard_vs_taylor_stage1']:.2f} "
                f"historical={row.get('historical_label')}"
            )

    lines += [
        "",
        "## 问题 4：历史复用 vs 待评测",
        "",
        "见 `details.csv` 的 `comparability` / `historical_label`（未知集合不自动判低性能）。",
        "",
        "## 闭环候选（未启动，最多 2 个）",
        "",
    ]
    for rec in summary.get("closed_loop_recommendations", []):
        lines.append(
            f"- **{rec['method']}+{rec['strategy']}** drop={rec['drop_layers']} — {rec.get('why', '')}"
        )

    lines += [
        "",
        "## 文件",
        "",
        "- `scores/molmoact2_{igia,fisher,gateprobe}.pt`",
        "- `taylor_groups/taylor_*_b80*.pt`",
        "- `details.csv`, `drop_frequency.csv`, `selections/`",
        "",
        "生成：`analyze_layer_calibration_molmo_v1.py`（需先 GPU 打分脚本）。",
    ]
    (MOLMO_LAYER_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write outputs even if GPU scores missing (selftest uses synthetic scores).",
    )
    args = parser.parse_args()
    return run_analyze(require_scores=not args.allow_partial)


if __name__ == "__main__":
    raise SystemExit(main())

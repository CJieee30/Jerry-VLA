#!/usr/bin/env python3
"""Layer-pruning calibration screening: equal-budget frame selection (CPU only).

Uses frozen CARE v2 per-frame scores (480×L) and demo-action selection rules
(Random / α / G-only / CARE_event). Does not run GPU scoring or closed loop.

Development diagnostic — not a production recommendation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dump_shared import _load_actions  # noqa: E402
from event_care_nfe_diagnostic import (  # noqa: E402
    EVENT_BOOST,
    FrameMeta,
    build_metas,
    demo_gripper_switch,
)
from protocol import (  # noqa: E402
    FRAMES_PER_TASK,
    HORIZON,
    LIBERO_REPO,
    METHODS,
    N_FRAMES,
    N_TASKS,
    OUT_DIR,
    STACKS,
    STAGE1,
    drop_set,
    jaccard,
)

# Reuse k-per-task selectors from dev NFE CPU tooling (same tie-break rules).
from dev_nfe_rules_cpu_v2 import (  # noqa: E402
    g_only_group_k,
    random_group_k,
    select_per_task_k,
)

SCREEN_DIR = OUT_DIR / "layer_calibration_screening"
SEL_DIR = SCREEN_DIR / "selections"

BUDGETS: tuple[tuple[int, int], ...] = ((40, 1), (80, 2), (120, 3))
STRATEGIES = ("random", "alpha", "g_only", "care_event")
STOCHASTIC_SEEDS = tuple(range(20))

# Historical closed-loop labels (drop must match exactly). From analyze_ds / care_libero.
KNOWN_CLOSED_LOOP: dict[tuple[str, int, str], tuple[str, float, list[int]]] = {
    ("pi05", 6, "fisher"): ("S1 Fisher-6", 97.75, [1, 13, 14, 15, 16, 17]),
    ("pi05", 6, "taylor"): ("S1 Taylor-6", 97.25, [1, 2, 14, 15, 16, 17]),
    ("pi05", 6, "gateprobe"): ("CARE-v1 GP-6", 97.00, [2, 13, 14, 15, 16, 17]),
    ("pi05", 8, "fisher"): ("S1 Fisher-8", 85.50, [1, 2, 3, 13, 14, 15, 16, 17]),
    ("pi05", 8, "taylor"): ("S1 Taylor-8", 85.50, [1, 2, 3, 13, 14, 15, 16, 17]),
    ("pi05", 8, "gateprobe"): ("S1 GP-8", 85.75, [2, 4, 10, 13, 14, 15, 16, 17]),
    ("smolvla", 4, "fisher"): ("S1 Fisher-4", 71.25, [10, 12, 13, 14]),  # v2 uniform; S1 was [10,11,12,14] 69.75
    ("smolvla", 4, "taylor"): ("S1 Taylor-4", 78.50, [12, 13, 14, 15]),
    ("smolvla", 4, "gateprobe"): ("v2 uniform GP-4", float("nan"), [11, 13, 14, 15]),
    ("molmoact2", 16, "fisher"): ("S1 Fisher-16", 72.75, [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]),
    ("molmoact2", 16, "taylor"): ("S1 Taylor-16", 98.10, [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]),
    ("molmoact2", 16, "gateprobe"): ("S1 GP-16", 87.00, [2, 3, 4, 6, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]),
    ("molmoact2", 17, "fisher"): ("S1 Fisher-17", 70.60, [1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]),
    ("molmoact2", 17, "taylor"): ("S1 Taylor-17", 89.60, [1, 2, 3, 4, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]),
    ("molmoact2", 17, "gateprobe"): ("S1 GP-17", 82.65, [2, 3, 4, 6, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]),
}

# Sets that E1/D/S flagged as diverging from Stage-1 on full 480 uniform (reference only).
E1_UNIFORM_DROP: dict[tuple[str, str], list[int]] = {}


def _load_e1_uniform() -> None:
    global E1_UNIFORM_DROP
    p = OUT_DIR / "e1.json"
    if not p.is_file():
        return
    for row in json.loads(p.read_text())["rows"]:
        E1_UNIFORM_DROP[(row["stack"], row["method"])] = list(row["uniform_drop"])


def load_frame_scores(stack: str, method: str) -> torch.Tensor:
    path = OUT_DIR / "scores" / f"{stack}_{method}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    mat = blob["frame_scores"].float()
    n_layers = int(STAGE1[stack]["n_layers"])
    if tuple(mat.shape) != (N_FRAMES, n_layers):
        raise RuntimeError(f"{path}: expected ({N_FRAMES}, {n_layers}), got {tuple(mat.shape)}")
    return mat


def verify_alignment(shared: dict, frames_df: pd.DataFrame) -> dict:
    """Check records[i] matches frames_df row at global_index."""
    mismatches = []
    for i, rec in enumerate(shared["records"]):
        g = int(rec["global_index"])
        row = frames_df.iloc[g]
        if int(row["task_index"]) != int(rec["task_index"]):
            mismatches.append((i, "task_index", rec["task_index"], int(row["task_index"])))
        if int(row["episode_index"]) != int(rec["episode_index"]):
            mismatches.append((i, "episode_index", rec["episode_index"], int(row["episode_index"])))
        if len(mismatches) >= 5:
            break
    return {
        "n_records": len(shared["records"]),
        "n_frames_expected": N_FRAMES,
        "alignment_ok": len(mismatches) == 0,
        "sample_mismatches": mismatches[:5],
    }


def drop_for_indices(frame: torch.Tensor, indices: list[int], stack: str) -> list[int]:
    sub = frame[indices].mean(dim=0)
    return drop_set(sub, stack)


def format_drop(drop: list[int]) -> str:
    return json.dumps(drop)


def lookup_closed_loop(stack: str, method: str, drop: list[int]) -> dict:
    k = int(STAGE1[stack]["k"])
    stage1 = list(STAGE1[stack][method])
    out = {
        "stage1_operating_drop": stage1,
        "jaccard_vs_stage1": jaccard(drop, stage1),
        "historical_label": None,
        "historical_sr_pct": None,
        "historical_source": None,
    }
    for (st, kk, meth), (label, sr, layers) in KNOWN_CLOSED_LOOP.items():
        if st == stack and meth == method and drop == layers:
            out["historical_label"] = label
            out["historical_sr_pct"] = sr if np.isfinite(sr) else None
            out["historical_source"] = "care_libero.md / analyze_ds KNOWN"
            break
    e1 = E1_UNIFORM_DROP.get((stack, method))
    if e1 is not None:
        out["e1_full480_uniform_drop"] = e1
        out["jaccard_vs_e1_uniform"] = jaccard(drop, e1)
    else:
        out["e1_full480_uniform_drop"] = None
        out["jaccard_vs_e1_uniform"] = None
    return out


def write_selection(
    path: Path,
    *,
    stack: str,
    method: str,
    strategy: str,
    budget: int,
    k_per_task: int,
    seed: int | None,
    selected: set[int],
    metas: list[FrameMeta],
) -> None:
    by_task: dict[int, list[dict]] = defaultdict(list)
    for si in sorted(selected):
        m = metas[si]
        by_task[m.task_index].append(
            {
                "shared_index": si,
                "global_index": m.global_index,
                "episode_index": m.episode_index,
                "frame_in_episode": m.frame_in_episode,
                "alpha": m.alpha,
                "G": m.G,
                "alpha_event": m.alpha_event,
            }
        )
    payload = {
        "protocol": "layer_calibration_screening_v1",
        "stack": stack,
        "method": method,
        "strategy": strategy,
        "budget_frames": budget,
        "k_per_task": k_per_task,
        "seed": seed,
        "n_selected": len(selected),
        "shared_indices": sorted(selected),
        "per_task": {str(ti): by_task[ti] for ti in sorted(by_task)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def select_indices(
    strategy: str, metas: list[FrameMeta], k: int, seed: int | None
) -> set[int]:
    if strategy == "random":
        if seed is None:
            raise ValueError("random requires seed")
        return random_group_k(metas, seed, k)
    if strategy == "g_only":
        if seed is None:
            raise ValueError("g_only requires seed")
        return g_only_group_k(metas, seed, k)
    if strategy == "alpha":
        return select_per_task_k(metas, "alpha", k)
    if strategy == "care_event":
        return select_per_task_k(metas, "alpha_event", k)
    raise ValueError(strategy)


def run_screening() -> None:
    _load_e1_uniform()
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    SEL_DIR.mkdir(parents=True, exist_ok=True)

    shared = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    if int(shared.get("n", N_FRAMES)) != N_FRAMES:
        raise RuntimeError("shared.pt n != 480")
    alpha = np.asarray(shared["alpha"], dtype=np.float64).reshape(-1)
    if alpha.shape[0] != N_FRAMES:
        raise RuntimeError("alpha length mismatch")

    frames_df = _load_actions(LIBERO_REPO)
    align = verify_alignment(shared, frames_df)
    actions = np.stack(frames_df["action"].to_numpy()).astype(np.float64)
    episode = frames_df["episode_index"].to_numpy()
    G = demo_gripper_switch(shared, actions, episode)
    metas = build_metas(shared, alpha, G, np.zeros(N_FRAMES, dtype=np.int32))

    score_cache: dict[tuple[str, str], torch.Tensor] = {}
    full_pool_drop: dict[tuple[str, str], list[int]] = {}
    for stack in STACKS:
        for method in METHODS:
            frame = load_frame_scores(stack, method)
            score_cache[(stack, method)] = frame
            full_pool_drop[(stack, method)] = drop_set(frame.mean(dim=0), stack)

    detail_rows: list[dict] = []
    freq_counter: dict[tuple, Counter] = defaultdict(Counter)

    for stack in STACKS:
        for method in METHODS:
            frame = score_cache[(stack, method)]
            ref_drop = full_pool_drop[(stack, method)]
            for budget, k in BUDGETS:
                scenarios: list[tuple[str, int | None]] = []
                for strat in STRATEGIES:
                    if strat in ("random", "g_only"):
                        for seed in STOCHASTIC_SEEDS:
                            scenarios.append((strat, seed))
                    else:
                        scenarios.append((strat, None))

                for strategy, seed in scenarios:
                    selected = select_indices(strategy, metas, k, seed)
                    if len(selected) != budget:
                        raise RuntimeError(
                            f"{stack} {method} {strategy} seed={seed} k={k}: "
                            f"got {len(selected)} frames, expected {budget}"
                        )
                    idx_list = sorted(selected)
                    drop = drop_for_indices(frame, idx_list, stack)
                    j_full = jaccard(drop, ref_drop)
                    cl = lookup_closed_loop(stack, method, drop)
                    drop_key = format_drop(drop)
                    freq_counter[(stack, method, budget, strategy)][drop_key] += 1

                    sel_name = (
                        f"{stack}_{method}_b{budget}_{strategy}"
                        + (f"_seed{seed}" if seed is not None else "")
                        + ".json"
                    )
                    write_selection(
                        SEL_DIR / sel_name,
                        stack=stack,
                        method=method,
                        strategy=strategy,
                        budget=budget,
                        k_per_task=k,
                        seed=seed,
                        selected=selected,
                        metas=metas,
                    )

                    detail_rows.append(
                        {
                            "stack": stack,
                            "method": method,
                            "strategy": strategy,
                            "budget_frames": budget,
                            "k_per_task": k,
                            "seed": seed if seed is not None else "NA",
                            "n_selected": len(selected),
                            "drop_layers": drop,
                            "drop_json": drop_key,
                            "jaccard_vs_full480_equal": j_full,
                            "full480_equal_drop": ref_drop,
                            **{
                                k2: cl[k2]
                                for k2 in (
                                    "jaccard_vs_stage1",
                                    "jaccard_vs_e1_uniform",
                                    "historical_label",
                                    "historical_sr_pct",
                                    "historical_source",
                                )
                            },
                        }
                    )

    details = pd.DataFrame(detail_rows)
    details_path = SCREEN_DIR / "details.csv"
    details.to_csv(details_path, index=False)

    freq_rows = []
    for key, ctr in sorted(freq_counter.items()):
        stack, method, budget, strategy = key
        total = sum(ctr.values())
        for drop_json, count in ctr.most_common():
            freq_rows.append(
                {
                    "stack": stack,
                    "method": method,
                    "budget_frames": budget,
                    "strategy": strategy,
                    "drop_json": drop_json,
                    "count": count,
                    "fraction": count / total if total else float("nan"),
                }
            )
    freq_df = pd.DataFrame(freq_rows)
    freq_df.to_csv(SCREEN_DIR / "drop_frequency.csv", index=False)

    # Summaries for REPORT automation
    summaries = []
    for stack in STACKS:
        for method in METHODS:
            ref = full_pool_drop[(stack, method)]
            sub = details[(details.stack == stack) & (details.method == method)]
            for budget in (40, 80, 120):
                bsub = sub[sub.budget_frames == budget]
                for strat in STRATEGIES:
                    ssub = bsub[bsub.strategy == strat]
                    if ssub.empty:
                        continue
                    uniq = ssub["drop_json"].nunique()
                    j_mean = float(ssub["jaccard_vs_full480_equal"].mean())
                    j_stage1 = float(ssub["jaccard_vs_stage1"].mean())
                    hit_stage1 = float((ssub["drop_json"] == format_drop(STAGE1[stack][method])).mean())
                    hit_hist = ssub["historical_label"].notna().mean()
                    care_only = None
                    if strat == "care_event":
                        rand = bsub[bsub.strategy == "random"]
                        care_drop = ssub.iloc[0]["drop_json"]
                        care_only = float((rand["drop_json"] != care_drop).any())
                    summaries.append(
                        {
                            "stack": stack,
                            "method": method,
                            "budget": budget,
                            "strategy": strat,
                            "n_runs": len(ssub),
                            "unique_drop_sets": uniq,
                            "mean_jaccard_full480": j_mean,
                            "mean_jaccard_stage1": j_stage1,
                            "frac_exact_stage1_drop": hit_stage1,
                            "frac_known_historical_label": float(hit_hist),
                            "care_event_differs_from_any_random": care_only,
                        }
                    )

    summary_payload = {
        "protocol": "layer_calibration_screening_v1",
        "status": "development_diagnostic",
        "pool": {
            "n_frames": N_FRAMES,
            "n_tasks": N_TASKS,
            "frames_per_task": FRAMES_PER_TASK,
            "horizon": HORIZON,
            "event_boost": EVENT_BOOST,
            "score_dir": str(OUT_DIR / "scores"),
            "estimator": "care_v2 per-frame Fisher/Taylor/GateProbe (score_frame.py)",
            "aggregation_after_selection": "equal mean over selected frames only (no α weights)",
        },
        "alignment": align,
        "budgets": [{"frames": b, "k_per_task": k} for b, k in BUDGETS],
        "strategies": list(STRATEGIES),
        "stochastic_seeds": list(STOCHASTIC_SEEDS),
        "full_pool_drops": {f"{s}/{m}": full_pool_drop[(s, m)] for s in STACKS for m in METHODS},
        "summaries": summaries,
        "notes": [
            "Full-480 equal-weight drop is a reference, not ground truth.",
            "Historical SR labels apply only when drop set exactly matches a KNOWN row.",
            "Smol/Molmo legacy 32-frame calibrations are not comparable to budget=40 on this 480 pool.",
            "CPU screening does not prove closed-loop benefit.",
        ],
    }
    (SCREEN_DIR / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")

    write_report(summary_payload, details, freq_df)
    print(f"wrote {SCREEN_DIR}", flush=True)


def write_report(summary: dict, details: pd.DataFrame, freq_df: pd.DataFrame) -> None:
    lines = [
        "# CARE_event 层剪枝校准选样 — 第一轮 CPU 筛查",
        "",
        "**状态：开发诊断**（非论文结论）。未启动 GPU 打分或闭环。",
        "",
        "## 数据与对齐",
        "",
        f"- 候选池：CARE v2 **{N_FRAMES}** 帧（{N_TASKS} 任务 × {FRAMES_PER_TASK} 帧/任务），H={HORIZON}。",
        f"- 分数：`results/stage2_care_v2/scores/` 逐帧 Fisher/Taylor/GateProbe（`score_frame.py`）。",
        f"- 对齐检查：`alignment_ok={summary['alignment']['alignment_ok']}`（records[i].global_index ↔ libero 表）。",
        "",
        "选样后层分数：**等权平均**，不乘 α / α_event。全池参考：480 帧等权 Drop（同估计器）。",
        "",
        "## 实验设置",
        "",
        "| 策略 | 种子 | 预算（帧 / 每任务 k） |",
        "|---|---|---|",
        "| Random | 0–19 | 40/1，80/2，120/3 |",
        "| G-only | 0–19 | 同上 |",
        "| α | 确定性 | 同上 |",
        "| CARE_event | 确定性 | 同上 |",
        "",
        f"剪枝 k：π0.5 Drop-6，SmolVLA Drop-4，Molmo Drop-16；L0 规则见 `protocol.STAGE1`。",
        "",
        "## 核心发现（自动摘要，需人工审阅）",
        "",
    ]

    # CARE_event vs full pool
    care = details[details.strategy == "care_event"].copy()
    rand = details[details.strategy == "random"].copy()
    highlights = []
    for stack in STACKS:
        for method in METHODS:
            for budget in (40, 80, 120):
                c = care[(care.stack == stack) & (care.method == method) & (care.budget_frames == budget)]
                if c.empty:
                    continue
                row = c.iloc[0]
                r = rand[(rand.stack == stack) & (rand.method == method) & (rand.budget_frames == budget)]
                r_uniq = r["drop_json"].nunique()
                same_as_full = row["jaccard_vs_full480_equal"] == 1.0
                differs_rand = (r["drop_json"] != row["drop_json"]).any()
                if differs_rand or not same_as_full:
                    highlights.append(
                        f"- **{stack} / {method} / budget={budget}**: CARE_event J(full480)={row['jaccard_vs_full480_equal']:.2f}, "
                        f"J(stage1)={row['jaccard_vs_stage1']:.2f}, "
                        f"random 在 20 种子下 {r_uniq} 种 drop；CARE_event 与任一 random 不同={differs_rand}。"
                    )

    if highlights:
        lines.extend(highlights)
    else:
        lines.append("- （自动规则未抓到差异行；见 `details.csv`。）")

    lines += [
        "",
        "## 文件",
        "",
        "- `summary.json` — 汇总与全池 Drop 参考",
        "- `details.csv` — 模型×方法×策略×预算×种子",
        "- `drop_frequency.csv` — 各条件下 drop 集合频率",
        "- `selections/` — 选中帧清单",
        "",
        "## 与历史闭环的关系",
        "",
        "仅当 `drop_layers` **完全等于** `analyze_ds.KNOWN` 中记录集合时，`details.csv` 填写 `historical_label` / SR。",
        "更接近全池 Jaccard **不等于** 更准确；旧 CARE E1 在**固定 480 帧**上 α 加权不改变 Drop，本实验改变的是**子集帧**，故可产生差异。",
        "",
        "## 建议的最小 GPU 验证（未执行）",
        "",
        "审阅 `details.csv` 后，仅对 **CARE_event 与 random 中位数 J(stage1) 最高且 drop 分歧最大** 的 1–2 个 (stack, method, budget) 开闭环；",
        "优先候选通常落在 **π0.5 Fisher-6** 或 **Molmo Taylor-16** 经营点邻域，但必须以本表实际 drop 为准。",
        "",
        "---",
        "",
        f"生成脚本：`scripts/stage2_care_v2/layer_calibration_screening.py`",
    ]
    (SCREEN_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="CARE_event layer calibration CPU screening")
    parser.parse_args()
    run_screening()


if __name__ == "__main__":
    main()

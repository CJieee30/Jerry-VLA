#!/usr/bin/env python3
"""CPU: three NFE selection rules + strategy/budget comparison (dev results only)."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "molmoact2_libero"))

from dev_metrics_common import (
    MODEL_DIRS,
    POOL_CSV,
    build_conversion_check,
    compare_chunk,
    libero_mean_std,
    tail_stats_merged_noises,
    write_conversion_manifest,
)
from dump_shared import _load_actions
from event_care_nfe_diagnostic import (
    FrameMeta,
    build_metas,
    demo_gripper_switch,
    load_probe_frames,
    tail_mask_nfe1,
)
from protocol import OUT_DIR

OUT = OUT_DIR / "dev_nfe_rules_cpu_v2"
REF_NFE = 10
ABS_MAIN = 0.085
ABS_SENS = (0.08, 0.09)
ABS_IMP_THRESH = 0.03
REL_IMP_THRESH = 0.375
BUDGETS = (1, 2, 3)
SEED_DIR = OUT_DIR / "nfe_probe_event_care" / "budget_compare" / "selections"
RANDOM_SEEDS = sorted(
    int(m.group(1)) for p in SEED_DIR.glob("random_seed*.json") for m in [re.search(r"_seed(\d+)\.json$", p.name)] if m
)


def select_per_task_k(metas: list[FrameMeta], score_key: str, k: int) -> set[int]:
    by_task: dict[int, list[int]] = defaultdict(list)
    for m in metas:
        by_task[m.task_index].append(m.shared_index)
    chosen: set[int] = set()
    for ti in sorted(by_task):
        idxs = by_task[ti]
        if score_key == "alpha":
            ordered = sorted(idxs, key=lambda j: (-metas[j].alpha, metas[j].global_index))
        elif score_key == "alpha_event":
            ordered = sorted(idxs, key=lambda j: (-metas[j].alpha_event, metas[j].global_index))
        else:
            raise ValueError(score_key)
        chosen.update(ordered[:k])
    return chosen


def random_group_k(metas: list[FrameMeta], seed: int, k: int) -> set[int]:
    by_task: dict[int, list[int]] = defaultdict(list)
    for m in metas:
        by_task[m.task_index].append(m.shared_index)
    rng = np.random.default_rng(seed)
    out: set[int] = set()
    for ti in sorted(by_task):
        idxs = np.asarray(sorted(by_task[ti]), dtype=np.int64)
        pick = rng.choice(idxs, size=min(k, idxs.size), replace=False)
        out.update(int(x) for x in pick)
    return out


def g_only_group_k(metas: list[FrameMeta], seed: int, k: int) -> set[int]:
    by_task: dict[int, list[int]] = defaultdict(list)
    for m in metas:
        by_task[m.task_index].append(m.shared_index)
    rng = np.random.default_rng(seed)
    out: set[int] = set()
    for ti in sorted(by_task):
        idxs = by_task[ti]
        g1 = [j for j in idxs if metas[j].G == 1]
        g0 = [j for j in idxs if metas[j].G == 0]
        rng.shuffle(g1)
        rng.shuffle(g0)
        picked: list[int] = []
        picked.extend(g1[:k])
        if len(picked) < k:
            picked.extend(g0[: k - len(picked)])
        out.update(picked[:k])
    return out


def candidate_nfes(z) -> list[int]:
    return sorted(int(k.replace("nfe", "")) for k in z.files if k.startswith("nfe") and k != "nfe10")


def rmse7_vs_ref(model: str, z, si: int, nid: int, nfe: int, mean, std) -> float:
    return compare_chunk(model, z[f"nfe{nfe}"][si, nid], z[f"nfe10"][si, nid], mean, std)["rmse7_libero"]


def rmse7_between(model: str, z, si: int, nid: int, nfe_a: int, nfe_b: int, mean, std) -> float:
    return compare_chunk(model, z[f"nfe{nfe_a}"][si, nid], z[f"nfe{nfe_b}"][si, nid], mean, std)["rmse7_libero"]


def cos_vs_ref(model: str, z, si: int, nid: int, nfe: int, mean, std) -> float:
    return compare_chunk(model, z[f"nfe{nfe}"][si, nid], z[f"nfe10"][si, nid], mean, std)["cosine_libero"]


def precompute_model(model: str, z, pool: pd.DataFrame, mean, std) -> dict:
    cand = candidate_nfes(z)
    n_obs = z["nfe10"].shape[0]
    sub = pool[pool.model == model]
    vs_ref = {}
    vs_ref_cos = {}
    for r in sub.itertuples():
        vs_ref[(int(r.shared_index), int(r.noise_id), int(r.nfe))] = float(r.rmse7_libero)
        vs_ref_cos[(int(r.shared_index), int(r.noise_id), int(r.nfe))] = float(r.cosine_libero)
    between = {}
    for i, a in enumerate(cand):
        if i + 1 < len(cand):
            b = cand[i + 1]
            between[(a, b)] = {
                (si, nid): rmse7_between(model, z, si, nid, a, b, mean, std)
                for si in range(n_obs)
                for nid in (0, 1)
            }
    grip10 = {}
    for si in range(n_obs):
        m0 = compare_chunk(model, z["nfe10"][si, 0], z["nfe10"][si, 1], mean, std)
        grip10[si] = int(m0["grip_any_disagree"])
    return {"cand": cand, "vs_ref": vs_ref, "vs_ref_cos": vs_ref_cos, "between": between, "n_obs": n_obs, "grip10": grip10}


def compute_E(cache: dict, nfe: int, indices: set[int] | None) -> dict:
    sis = range(cache["n_obs"]) if indices is None else sorted(indices)
    rmses = [cache["vs_ref"][(si, nid, nfe)] for si in sis for nid in (0, 1)]
    coss = [cache["vs_ref_cos"][(si, nid, nfe)] for si in sis for nid in (0, 1)]
    return {"E": float(np.mean(rmses)), "cos_mean": float(np.mean(coss)), "n_cmp": len(rmses)}


def adjacent_pair_rmse(cache: dict, n_a: int, n_b: int, indices: set[int] | None) -> float:
    sis = range(cache["n_obs"]) if indices is None else sorted(indices)
    vals = [cache["between"][(n_a, n_b)][(si, nid)] for si in sis for nid in (0, 1)]
    return float(np.mean(vals))


def nfe10_grip_baseline(cache: dict, indices: set[int] | None) -> dict:
    sis = list(range(cache["n_obs"])) if indices is None else sorted(indices)
    obs_bad = sum(cache["grip10"][si] for si in sis)
    return {
        "comparisons_with_grip_disagree": obs_bad,
        "comparisons_total": len(sis),
        "unique_obs_with_disagree": obs_bad,
        "rate_per_obs": obs_bad / len(sis) if sis else float("nan"),
    }


def improvement_ok(E: dict[int, float], n: int, cand: list[int], mode: str) -> tuple[bool, str]:
    if n not in cand:
        return False, "not_candidate"
    nxt = n + 1
    if nxt not in cand:
        return True, "no_adjacent_higher_candidate"
    d = E[n] - E[nxt]
    if d < 0:
        return False, "non_monotonic_E"
    if mode == "abs":
        return d <= ABS_IMP_THRESH, f"delta={d:.4f}"
    rel = d / E[n] if E[n] > 0 else float("inf")
    return rel <= REL_IMP_THRESH, f"rel={rel:.4f}"


def pick_nfe(
    E: dict[int, float],
    cand: list[int],
    abs_tau: float,
    use_abs: bool,
    use_abs_imp: bool,
    use_rel_imp: bool,
) -> tuple[int, list[str]]:
    notes: list[str] = []
    for n in sorted(cand):
        ok = True
        if use_abs and E[n] > abs_tau:
            ok = False
        if use_abs_imp:
            pass_imp, why = improvement_ok(E, n, cand, "abs")
            if not pass_imp:
                ok = False
                notes.append(f"n={n}:{why}")
        if use_rel_imp:
            pass_rel, why = improvement_ok(E, n, cand, "rel")
            if not pass_rel:
                ok = False
                notes.append(f"n={n}:{why}")
        if ok:
            return n, notes
    return REF_NFE, notes + ["fallback_reference"]


def grip_diag_native(df: pd.DataFrame, selected: set[int], nfe: int) -> dict:
    sub = df[(df.nfe == nfe) & (df.shared_index.isin(selected))]
    n_cmp = len(sub)
    n_bad = int((sub.gripper_disagree_n > 0).sum())
    n_obs = int(sub.shared_index.nunique())
    n_obs_bad = int(sub.loc[sub.gripper_disagree_n > 0, "shared_index"].nunique())
    return {"grip_cmp": f"{n_bad}/{n_cmp}", "grip_obs": f"{n_obs_bad}/{n_obs}"}


def offline_budget(n_obs: int, nfes: list[int], n_noise: int = 2) -> dict:
    vel = n_noise * sum(nfes)
    return {"observations": n_obs, "velocity_evals": n_obs * vel, "chunk_rows": n_obs * n_noise * len(nfes)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_conversion_manifest()
    build_conversion_check()

    from protocol import LIBERO_REPO

    shared = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    frames_df = _load_actions(LIBERO_REPO)
    actions = np.stack(frames_df["action"].to_numpy()).astype(np.float64)
    episode = frames_df["episode_index"].to_numpy()
    G_flags = demo_gripper_switch(shared, actions, episode)
    alpha = np.asarray(shared["alpha"], dtype=np.float64).reshape(-1)
    metas = build_metas(shared, alpha, G_flags, G_flags)
    mean, std = libero_mean_std()
    if not POOL_CSV.is_file():
        raise SystemExit(f"missing {POOL_CSV}; run export_pool_observation_metrics.py first")
    pool = pd.read_csv(POOL_CSV)

    pool_metrics: dict[str, dict] = {}
    model_rows = []
    caches: dict[str, dict] = {}
    for model, mdir in MODEL_DIRS.items():
        z = np.load(mdir / "actions.npz")
        cache = precompute_model(model, z, pool, mean, std)
        caches[model] = cache
        cand = cache["cand"]
        E = {n: compute_E(cache, n, None)["E"] for n in cand}
        cos = {n: compute_E(cache, n, None)["cos_mean"] for n in cand}
        row = {"model": model, "candidates": cand}
        for n in cand:
            row[f"E_{n}"] = E[n]
            row[f"cos_{n}"] = cos[n]
        pairs = []
        for i, a in enumerate(cand):
            if i + 1 < len(cand):
                b = cand[i + 1]
                pairs.append(
                    {
                        "pair": f"{a}-{b}",
                        "E_delta": E[a] - E[b],
                        "non_monotonic_E": E[b] > E[a],
                        "mean_rmse_outputs": adjacent_pair_rmse(cache, a, b, None),
                    }
                )
        row["adjacent"] = pairs
        row["grip_nfe10_baseline"] = nfe10_grip_baseline(cache, None)
        recs = {}
        for tau in (ABS_MAIN, *ABS_SENS):
            recs[f"abs_{tau}"] = pick_nfe(E, cand, tau, True, False, False)[0]
        recs["abs_improve_only"] = pick_nfe(E, cand, 1.0, False, True, False)[0]
        recs["rel_improve_only"] = pick_nfe(E, cand, 1.0, False, False, True)[0]
        recs["all_three_abs085"] = pick_nfe(E, cand, ABS_MAIN, True, True, True)[0]
        row["recommendations_dev"] = recs
        pool_metrics[model] = {"E": E, "cos": cos, "pairs": pairs, "recs": recs}
        model_rows.append(row)

    # selections + budget
    sel_root = OUT / "selections"
    budget_rows = []
    stability: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for k in BUDGETS:
        kdir = sel_root / f"top{k}_per_task"
        kdir.mkdir(parents=True, exist_ok=True)
        det = {
            "alpha": select_per_task_k(metas, "alpha", k),
            "alpha_event": select_per_task_k(metas, "alpha_event", k),
        }
        for name, sset in det.items():
            payload = {"strategy": name, "k_per_task": k, "n_selected": len(sset), "shared_indices": sorted(sset)}
            (kdir / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
        for seed in RANDOM_SEEDS:
            for label, fn in (("random", random_group_k), ("g_only", g_only_group_k)):
                sset = fn(metas, seed, k)
                payload = {"strategy": label, "seed": seed, "k_per_task": k, "n_selected": len(sset), "shared_indices": sorted(sset)}
                (kdir / f"{label}_seed{seed}.json").write_text(json.dumps(payload, indent=2) + "\n")

        for model, mdir in MODEL_DIRS.items():
            cache = caches[model]
            cand = cache["cand"]
            nfes_probe = cand + [REF_NFE]
            z = np.load(mdir / "actions.npz")
            df_native = load_probe_frames(mdir)
            scenarios: list[tuple[str, int | None, set[int]]] = []
            for name, sset in det.items():
                scenarios.append((name, None, sset))
            for seed in RANDOM_SEEDS:
                scenarios.append(("random", seed, random_group_k(metas, seed, k)))
                scenarios.append(("g_only", seed, g_only_group_k(metas, seed, k)))
            for strat, seed, selected in scenarios:
                E_sel = {n: compute_E(cache, n, selected)["E"] for n in cand}
                rec = pick_nfe(E_sel, cand, ABS_MAIN, True, True, True)[0]
                key = f"k{k}_{strat}" + (f"_s{seed}" if seed is not None else "")
                stability[model][key].append(rec)
                tail = tail_stats_merged_noises(df_native, selected) if k == 3 else {}
                budget_rows.append(
                    {
                        "model": model,
                        "k_per_task": k,
                        "budget_obs": 40 * k,
                        "strategy": strat,
                        "seed": seed if seed is not None else "—",
                        "n_selected": len(selected),
                        "rec_all_three_abs085": rec,
                        "offline_budget": offline_budget(len(selected), nfes_probe),
                        "grip_nfe1": grip_diag_native(df_native, selected, 1),
                        "grip_nfe10_baseline_sel": nfe10_grip_baseline(cache, selected),
                        "tail_nfe1": tail,
                    }
                )

    (OUT / "pool_model_metrics.json").write_text(json.dumps(model_rows, indent=2) + "\n")
    rule_rows = []
    for r in model_rows:
        d = r["recommendations_dev"]
        rule_rows.append(
            {
                "model": r["model"],
                "abs_0.08": d[f"abs_{ABS_SENS[0]}"],
                "abs_0.085": d[f"abs_{ABS_MAIN}"],
                "abs_0.09": d[f"abs_{ABS_SENS[1]}"],
                "abs_improve_only": d["abs_improve_only"],
                "rel_improve_only": d["rel_improve_only"],
                "all_three_abs085": d["all_three_abs085"],
            }
        )
    pd.DataFrame(rule_rows).to_csv(OUT / "rule_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": r["model"],
                **{f"E_{n}": r.get(f"E_{n}") for n in (1, 2, 3)},
                **{f"rec_abs085": r["recommendations_dev"]["all_three_abs085"]},
                **{f"rec_abs_only": r["recommendations_dev"].get(f"abs_{ABS_MAIN}")},
            }
            for r in model_rows
        ]
    ).to_csv(OUT / "model_summary.csv", index=False)

    blockers = [
        "正式协议未冻结；本报告阈值均为开发结果",
        "Molmo dev_validation_probe=pending；闭环导入未接真实 JSON",
        "holdout 前瞻 split/freeze 未完成",
        "夹爪仅诊断，未设未经校准的硬门槛",
        "Pi 无 NFE3 探针，差分规则在 2→3 段为 NA",
    ]

    lines = [
        "# NFE 选步规则 CPU 分析（开发结果，非冻结协议）",
        "",
        "空间：LIBERO 归一化；10 步×7 维；E_N = 对观测与两噪声平均的 RMSE(vs 同噪声 NFE10)。",
        f"规则：绝对 E_N≤{ABS_MAIN}（敏感性 {ABS_SENS}）；绝对改善 E_N−E_{{N+1}}≤{ABS_IMP_THRESH}；"
        f"相对改善 (E_N−E_{{N+1}})/E_N≤{REL_IMP_THRESH}。仅相邻候选 NFE；误差不单调标 non_monotonic。",
        "",
        "## 1. 三模型全池指标与推荐",
        "",
    ]
    for r in model_rows:
        lines.append(f"### {r['model']}")
        lines.append("| NFE | E_N | cos mean |")
        lines.append("|---|---|---|")
        for n in r["candidates"]:
            lines.append(f"| {n} | {r[f'E_{n}']:.4f} | {r[f'cos_{n}']:.5f} |")
        lines.append("")
        lines.append("| 相邻 | E_N−E_{N+1} | 非单调? | 输出间 mean RMSE(N,N+1) |")
        lines.append("|---|---|---|---|")
        for p in r["adjacent"]:
            lines.append(
                f"| {p['pair']} | {p['E_delta']:.4f} | {p['non_monotonic_E']} | {p['mean_rmse_outputs']:.4f} |"
            )
        lines.append(f"NFE10 两噪声夹爪基准（观测级）：{r['grip_nfe10_baseline']}")
        lines.append("推荐（开发）：")
        for k, v in r["recommendations_dev"].items():
            lines.append(f"- `{k}` → **{v}**")
        lines.append("")

    lines += ["## 2. 规则对照（全池，独立应用）", "", "| model | abs≤0.08 | abs≤0.085 | abs≤0.09 | 仅绝对改善 | 仅相对改善 | 三者联合 |", "|---|---|---|---|---|---|---|"]
    for r in model_rows:
        d = r["recommendations_dev"]
        lines.append(
            f"| {r['model']} | {d[f'abs_{ABS_SENS[0]}']} | {d[f'abs_{ABS_MAIN}']} | {d[f'abs_{ABS_SENS[1]}']} | "
            f"{d['abs_improve_only']} | {d['rel_improve_only']} | {d['all_three_abs085']} |"
        )

    lines += [
        "",
        "## 3. 选样策略与预算（联合规则 all_three_abs085；非闭环准确率）",
        "",
        "Random/G-only：20 种子；α / CARE_event：确定性各 1 次。清单：`dev_nfe_rules_cpu_v2/selections/`。",
        "",
    ]
    # stability summary k=3 random only
    lines.append("### 推荐稳定性（k=3，all_three_abs085，random 20 种子）")
    lines.append("| model | counts |")
    lines.append("|---|---|")
    for model in MODEL_DIRS:
        counts: dict[int, int] = defaultdict(int)
        for key, vals in stability[model].items():
            if key.startswith("k3_random"):
                for v in vals:
                    counts[v] += 1
        lines.append(f"| {model} | {dict(counts)} |")
    lines.append("")
    lines.append("完整预算表见 `budget_comparison.json`。")
    lines += ["", "## 4. 仍阻止正式冻结", ""]
    for b in blockers:
        lines.append(f"- {b}")
    lines += [
        "",
        "## 复现",
        "",
        "```bash",
        "cd functional_necessity/scripts/stage2_care_v2",
        'export PYTHONPATH=".:../molmoact2_libero:../pi05_libero"',
        "python dev_nfe_rules_cpu_v2.py",
        "```",
        "",
        "数据：`nfe_probe_{pi05,smolvla,molmoact2}/actions.npz`；转换见 `dev_three_model_screening/conversion_manifest.json`。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    (OUT / "FREEZE_BLOCKERS.md").write_text("\n".join(f"- {b}" for b in blockers) + "\n")
    (OUT / "budget_comparison.json").write_text(
        json.dumps({"rows": budget_rows, "note": "rec is dev rule bundle, not closed-loop accuracy"}, indent=2) + "\n"
    )
    print(f"wrote {OUT}/REPORT.md")


if __name__ == "__main__":
    main()

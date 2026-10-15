#!/usr/bin/env python3
"""CPU open-loop holdout summary — does not append ledger ``validate`` (no closed-loop yet)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
SCR = RUN.parent.parent.parent.parent / "scripts" / "stage2_care_v2"
sys.path.insert(0, str(SCR))

from nfe_en_metrics import E_N, E_table, holdout_status_for_screening_nfe, recommend_three_rules
from nfe_protocol_v1 import SCREENING_RULES, floor_obs, load_sealed, per_obs, stats

CFG = json.loads((RUN / "config.frozen.json").read_text())
REC = json.loads((RUN / "04_recommend" / "recommendations.json").read_text())
RULE = {
    "arm_dims": [0, 1, 2, 3, 4, 5],
    "arm_quantile": 0.95,
    "arm_ratio_median": 1.0,
    "arm_ratio_tail": 1.0,
    "grip_rate_margin": 0.0,
    "grip_severe_steps": 3,
    "grip_severe_margin": 0.0,
    **CFG["rule"],
}
MODELS = CFG["models"]
NFES = sorted(set(RULE["candidates"]) | {RULE["reference_nfe"]})
VEL_PER_OBS_NOISE = sum(NFES)


def main() -> None:
    out: dict = {
        "run": RUN.name,
        "mode": "open_loop_holdout_only",
        "closed_loop": "pending — formal validate after closed-loop import",
        "n_observations": 480,
        "n_episodes": 160,
        "velocity_calls_per_model_expected": 480 * 2 * VEL_PER_OBS_NOISE,
        "models": {},
        "notes": [
            "Holdout pass/fail uses calibration-chosen NFE per rule; no threshold retuning.",
            "holdout_diagnostic_repick is diagnostic only (see per-model field).",
            "Bootstrap / episode clustering: one cluster per episode_index.",
        ],
    }
    lines = [
        "# 前瞻开环独立验证 — prospective_v1_20261015",
        "",
        "校准推荐已冻结；下表为 **验证集** holdout 规则检验（非闭环）。正式 `validate` 待闭环导入。",
        "",
        "## 验证集 E_N（相对 NFE=10，480 obs × 2 noise 均值）",
        "",
        "| model | E@1 | E@2 | E@3 |",
        "|---|---:|---:|---:|",
    ]
    rule_lines = [
        "",
        "## 校准选定 NFE 的 holdout 规则（各策略 screening NFE 相同则合并）",
        "",
        "| model | rule | screening NFE | holdout | E@N | detail |",
        "|---|---|---:|---|---:|---|",
    ]
    diag_lines = [
        "",
        "## 推荐 NFE 诊断（验证集，相对 NFE=10）",
        "",
        "| model | N* | grip disagree rate | arm median ratio | arm P95 ratio |",
        "|---|---:|---:|---:|---:|",
    ]

    for model, mc in MODELS.items():
        sealed = RUN / "05_validation" / model / "sealed_validation.npz"
        if not sealed.is_file():
            out["models"][model] = {"status": "missing", "path": str(sealed)}
            continue
        z = load_sealed(sealed)
        idx = np.arange(len(z["global_index"]))
        acts = {k: z[k] for k in z if k.startswith("nfe")}
        avail = sorted(n for n in RULE["candidates"] if f"nfe{n}" in acts)
        E = E_table(model, acts, avail, idx)
        lines.append(
            f"| {model} | {E.get(1, float('nan')):.4f} | {E.get(2, float('nan')):.4f} | {E.get(3, float('nan')):.4f} |"
        )
        repick = recommend_three_rules(model, acts, avail, idx, fallback=int(RULE["fallback_nfe"]))
        strat0 = next(iter(REC["models"][model]["strategies"].values()))
        n_primary = int(strat0["recommended_nfe"])
        fl = floor_obs(z, RULE, model, mc)
        po = per_obs(z, n_primary, RULE, model, mc)
        st = stats(po, fl, idx, RULE)
        diag_lines.append(
            f"| {model} | {n_primary} | {st['grip_rate']:.4f} | {st['ratio_median']:.3f} | {st['ratio_q']:.3f} |"
        )
        seen_rules: set[tuple[str, int]] = set()
        holdout_by_strat = {}
        for strat, r in REC["models"][model]["strategies"].items():
            by_rule = {}
            for label, kind, key in SCREENING_RULES:
                n_scr = int(r.get(key, r["recommended_nfe"]))
                sig = (label, n_scr)
                if sig not in seen_rules:
                    seen_rules.add(sig)
                    hs = holdout_status_for_screening_nfe(
                        model, acts, n_scr, avail, idx, int(RULE["fallback_nfe"]), kind
                    )
                    rule_lines.append(
                        f"| {model} | {label} | {n_scr} | {hs['status']} | {hs.get('E_n', float('nan')):.4f} | {hs.get('detail', '')} |"
                    )
                by_rule[label] = holdout_status_for_screening_nfe(
                    model, acts, n_scr, avail, idx, int(RULE["fallback_nfe"]), kind
                )
            holdout_by_strat[strat] = by_rule
        n_vel = int(len(idx) * 2 * sum(n for n in NFES if f"nfe{n}" in acts))
        out["models"][model] = {
            "status": "evaluated",
            "sealed": str(sealed.relative_to(RUN)),
            "n_observations": int(len(idx)),
            "n_episodes": int(np.unique(z["episode_index"]).size),
            "E_validation": {str(k): float(v) for k, v in E.items()},
            "holdout_by_strategy": holdout_by_strat,
            "holdout_diagnostic_repick_only": repick,
            "diagnostics_at_primary_nfe": {
                "nfe": n_primary,
                **{k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in st.items()},
            },
            "velocity_calls": n_vel,
        }

    md = "\n".join(lines + rule_lines + diag_lines) + "\n"
    out_path = RUN / "05_validation" / "OPEN_LOOP_VALIDATE.md"
    json_path = RUN / "05_validation" / "open_loop_validation.json"
    out_path.write_text(md)
    json_path.write_text(json.dumps(out, indent=2) + "\n")
    print(md)
    print(f"wrote {out_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rebuild 378-row budget table with three independent rule recommendations (shared nfe_en_metrics)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dev_metrics_common import MODEL_DIRS, tail_stats_merged_noises
from event_care_nfe_diagnostic import load_probe_frames
from nfe_en_metrics import E_table, REF_NFE, recommend_three_rules
from protocol import OUT_DIR

OUT = OUT_DIR / "dev_nfe_rules_cpu_v2"
SEL = OUT / "selections"


def load_sel(path: Path) -> set[int]:
    return set(int(x) for x in json.loads(path.read_text())["shared_indices"])


def scenarios(k: int) -> list[tuple[str, str | int, set[int]]]:
    kdir = SEL / f"top{k}_per_task"
    out: list[tuple[str, str | int, set[int]]] = []
    for name in ("alpha", "alpha_event"):
        p = kdir / f"{name}.json"
        if p.is_file():
            out.append((name, "—", load_sel(p)))
    for p in sorted(kdir.glob("random_seed*.json")):
        m = re.search(r"_seed(\d+)\.json$", p.name)
        if m:
            out.append(("random", int(m.group(1)), load_sel(p)))
    for p in sorted(kdir.glob("g_only_seed*.json")):
        m = re.search(r"_seed(\d+)\.json$", p.name)
        if m:
            out.append(("g_only", int(m.group(1)), load_sel(p)))
    return out


def grip_native(df, selected, nfe):
    sub = df[(df.nfe == nfe) & (df.shared_index.isin(selected))]
    return {
        "grip_cmp": f"{int((sub.gripper_disagree_n > 0).sum())}/{len(sub)}",
        "grip_obs": f"{int(sub.loc[sub.gripper_disagree_n > 0, 'shared_index'].nunique())}/{sub.shared_index.nunique()}",
    }


def main() -> None:
    rows = []
    for model, mdir in MODEL_DIRS.items():
        z = np.load(mdir / "actions.npz")
        acts = {k: z[k] for k in z.files if k.startswith("nfe")}
        cand = sorted(int(k.replace("nfe", "")) for k in acts if k != "nfe10")
        df = load_probe_frames(mdir)
        for k in (1, 2, 3):
            for strat, seed, selected in scenarios(k):
                idx = np.asarray(sorted(selected), dtype=np.int64)
                tr = recommend_three_rules(model, acts, cand, idx)
                tail = tail_stats_merged_noises(df, selected) if k >= 1 else {}
                rows.append(
                    {
                        "model": model,
                        "k_per_task": k,
                        "budget_obs": 40 * k,
                        "strategy": strat,
                        "seed": seed,
                        "n_selected": len(selected),
                        "rec_abs_0.085": tr["abs_0.085"]["recommended_nfe"],
                        "rec_abs_improve": tr["abs_improve"]["recommended_nfe"],
                        "rec_rel_improve": tr["rel_improve"]["recommended_nfe"],
                        "rec_joint_supplementary": tr["joint_supplementary"]["recommended_nfe"],
                        "E": tr["E_on_selection"],
                        "grip_nfe1": grip_native(df, selected, 1),
                        "tail_nfe1": tail,
                    }
                )
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "three_rule_budget_table.csv", index=False)
    (OUT / "three_rule_budget.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {len(rows)} rows to three_rule_budget_table.csv")


if __name__ == "__main__":
    main()

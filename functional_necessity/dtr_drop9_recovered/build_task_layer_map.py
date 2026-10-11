#!/usr/bin/env python3
"""Build 9×40 Drop-1 heatmap and extra_drop=f(task) maps from existing LOO evals.

Oracle k=1: keep L2 unless another remaining layer has strictly more successes.
That tie-break avoids calling 10/10 saturation 'skill-aware'. The oracle SR is
still 94.75% (leaky: same 10 trials). Do not use it as a fair method number.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from backend import REMAINING_LAYERS, STAGE1_DROP

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "task_layer"
LAYERS = list(REMAINING_LAYERS)
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_PREFIX = {
    "libero_spatial": "Spa",
    "libero_object": "Obj",
    "libero_goal": "Goal",
    "libero_10": "Long",
}
# Global extra Drop-1 cheapest-first (Avg. SR).
CHEAP_ORDER = [2, 1, 5, 6, 4, 9, 3, 8]
CHEAP_RANK = {layer: rank for rank, layer in enumerate(CHEAP_ORDER)}
SHUFFLE_SEED = 7


def _load_suite(path: Path) -> dict:
    return json.loads(path.read_text())


def load_loo() -> tuple[dict, dict, dict]:
    names: dict[str, list[str]] = {}
    base: dict[str, list[int]] = {}
    recovered = RESULTS / "eval_drop3" / "drop9_recovered"
    for suite in SUITES:
        payload = _load_suite(recovered / f"{suite}.json")
        names[suite] = [row["task"] for row in payload["per_task"]]
        base[suite] = [int(row["successes"]) for row in payload["per_task"]]
        if len(names[suite]) != 10:
            raise SystemExit(f"{suite} baseline has {len(names[suite])} tasks")
    successes: dict[int, dict[str, list[int]]] = {}
    for layer in LAYERS:
        successes[layer] = {}
        for suite in SUITES:
            payload = _load_suite(RESULTS / "eval_drop1" / f"l{layer}" / f"{suite}.json")
            got = [row["task"] for row in payload["per_task"]]
            if got != names[suite]:
                raise SystemExit(f"task order mismatch l{layer} {suite}")
            successes[layer][suite] = [int(row["successes"]) for row in payload["per_task"]]
    return names, base, successes


def pick_oracle_layer(scores: dict[int, int]) -> tuple[int, list[int]]:
    """Keep L2 unless another remaining layer strictly beats it."""
    rest = {layer: scores[layer] for layer in LAYERS if layer != 0}
    tied_best = [layer for layer, value in rest.items() if value == max(rest.values())]
    l2 = rest[2]
    better = [layer for layer in rest if layer != 2 and rest[layer] > l2]
    if not better:
        return 2, tied_best
    pick = max(better, key=lambda layer: (rest[layer], -CHEAP_RANK[layer]))
    return pick, tied_best


def build_tasks(names, base, successes) -> list[dict]:
    tasks = []
    for suite in SUITES:
        for task_id, prompt in enumerate(names[suite]):
            layer_successes = {str(layer): successes[layer][suite][task_id] for layer in LAYERS}
            deltas = {
                str(layer): successes[layer][suite][task_id] - base[suite][task_id] for layer in LAYERS
            }
            scores = {layer: successes[layer][suite][task_id] for layer in LAYERS}
            extra, tied_best = pick_oracle_layer(scores)
            tasks.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "task": prompt,
                    "short": f"{SUITE_PREFIX[suite]}{task_id}",
                    "baseline_successes": base[suite][task_id],
                    "successes": layer_successes,
                    "delta_successes": deltas,
                    "l2_successes": scores[2],
                    "oracle_extra": [extra],
                    "oracle_successes": scores[extra],
                    "tied_best_excluding_l0": sorted(tied_best),
                }
            )
    return tasks


def summarize(tasks: list[dict]) -> dict:
    oracle = sum(row["oracle_successes"] for row in tasks)
    l2 = sum(row["l2_successes"] for row in tasks)
    baseline = sum(row["baseline_successes"] for row in tasks)
    extras = [row["oracle_extra"][0] for row in tasks]
    switched = [row for row in tasks if row["oracle_extra"] != [2]]
    return {
        "n_tasks": len(tasks),
        "n_episodes": 10 * len(tasks),
        "baseline_successes": baseline,
        "baseline_pc": 100.0 * baseline / (10 * len(tasks)),
        "global_l2_successes": l2,
        "global_l2_pc": 100.0 * l2 / (10 * len(tasks)),
        "oracle_successes": oracle,
        "oracle_pc": 100.0 * oracle / (10 * len(tasks)),
        "n_switch_from_l2": len(switched),
        "oracle_layer_hist": dict(Counter(extras)),
        "leaky": True,
        "note": "Oracle uses the same seed=7 10-trial SR. Fair E4 must re-select extra_drop without this SR.",
    }


def map_payload(kind: str, tasks: list[dict], extras: list[list[int]], extra_meta: dict) -> dict:
    rows = []
    for row, extra in zip(tasks, extras):
        item = {
            "suite": row["suite"],
            "task_id": row["task_id"],
            "task": row["task"],
            "short": row["short"],
            "extra_drop": list(extra),
            "union_drop": sorted(set(STAGE1_DROP) | set(extra)),
        }
        rows.append(item)
    return {
        "kind": kind,
        "k": 1,
        "leaky": extra_meta.get("leaky", False),
        "stage1_drop": list(STAGE1_DROP),
        "remaining_layers": list(LAYERS),
        "tie_break": (
            "Keep L2 unless another remaining layer has strictly more successes "
            "on the same 10 trials; then max successes, then global cheap order "
            f"{CHEAP_ORDER}."
        ),
        **extra_meta,
        "tasks": rows,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def heatmap_md(tasks: list[dict], summary: dict) -> str:
    header = "| Task | " + " | ".join(f"L{layer}" for layer in LAYERS) + " | extra | vs L2 |"
    sep = "| --- | " + " | ".join("---:" for _ in LAYERS) + " | ---: | ---: |"
    lines = [
        "# Task × Layer extra Drop-1 heatmap (Drop-9 recovered π0.5)",
        "",
        "Cell = Δ successes vs recovered baseline (out of 10). Source: "
        "`results/eval_drop1/l{0,1,2,3,4,5,6,8,9}` and "
        "`results/eval_drop3/drop9_recovered`. Seed=7.",
        "",
        f"- Recovered baseline: **{summary['baseline_pc']:.2f}%** ({summary['baseline_successes']}/400)",
        f"- Global extra L2: **{summary['global_l2_pc']:.2f}%** ({summary['global_l2_successes']}/400)",
        f"- Task-oracle extra k=1 (leaky): **{summary['oracle_pc']:.2f}%** ({summary['oracle_successes']}/400)",
        f"- Tasks that leave L2: **{summary['n_switch_from_l2']}/40**  hist={summary['oracle_layer_hist']}",
        "",
        "Oracle mapping is leaky (selected on the same 10 trials). Use "
        "`map_oracle_k1.json` only as an upper-bound / wiring test. Fair E4 "
        "must pick extra_drop from calib loss, not this SR.",
        "",
        header,
        sep,
    ]
    for row in tasks:
        cells = " | ".join(
            f"{row['delta_successes'][str(layer)]:+d}" for layer in LAYERS
        )
        extra = row["oracle_extra"][0]
        gain = row["oracle_successes"] - row["l2_successes"]
        lines.append(
            f"| {row['short']} | {cells} | L{extra} | {gain:+d} |"
        )
    lines.extend(
        [
            "",
            "## Switches vs global L2",
            "",
            "| id | extra | oracle | L2 | base | task |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in tasks:
        if row["oracle_extra"] == [2]:
            continue
        extra = row["oracle_extra"][0]
        lines.append(
            f"| {row['short']} | {extra} | {row['oracle_successes']} | "
            f"{row['l2_successes']} | {row['baseline_successes']} | {row['task']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    names, base, successes = load_loo()
    tasks = build_tasks(names, base, successes)
    summary = summarize(tasks)
    OUT.mkdir(parents=True, exist_ok=True)

    heatmap = {
        "layers": LAYERS,
        "suites": SUITES,
        "protocol": {
            "checkpoint": "s1ghhh/VLADrop_pi05_LIBERO_Drop9Block_GateProbe",
            "stage1_drop": list(STAGE1_DROP),
            "seed": 7,
            "n_trials": 10,
            "skip": "dtr_appendix_b_prefix_kv",
            "recovery": False,
        },
        "summary": summary,
        "tasks": tasks,
        "delta_matrix": [[task["delta_successes"][str(layer)] for layer in LAYERS] for task in tasks],
        "oracle_extra": [task["oracle_extra"][0] for task in tasks],
        "short": [task["short"] for task in tasks],
    }
    write_json(OUT / "heatmap.json", heatmap)

    oracle_extras = [list(task["oracle_extra"]) for task in tasks]
    write_json(
        OUT / "map_oracle_k1.json",
        map_payload(
            "oracle_k1_loo_sr",
            tasks,
            oracle_extras,
            {
                "leaky": True,
                "default_extra": [2],
                "summary": summary,
            },
        ),
    )

    rng = random.Random(SHUFFLE_SEED)
    shuffled_layers = [extra[0] for extra in oracle_extras]
    rng.shuffle(shuffled_layers)
    n_same = sum(a[0] == b for a, b in zip(oracle_extras, shuffled_layers))
    write_json(
        OUT / "map_oracle_k1_shuffled.json",
        map_payload(
            "oracle_k1_shuffled",
            tasks,
            [[layer] for layer in shuffled_layers],
            {
                "leaky": True,
                "shuffle_seed": SHUFFLE_SEED,
                "n_same_as_oracle": n_same,
                "note": "Fisher-Yates permutation of the 40 oracle extras (same multiset, wrong tasks).",
            },
        ),
    )

    suite_extra = {
        "libero_spatial": [2],
        "libero_object": [5],
        "libero_goal": [2],
        "libero_10": [2],
    }
    write_json(
        OUT / "map_suite_k1.json",
        map_payload(
            "suite_k1",
            tasks,
            [suite_extra[task["suite"]] for task in tasks],
            {
                "leaky": False,
                "note": "Object→L5, other suites→L2. Not leaky as a rule, but the rule was read from LOO.",
            },
        ),
    )

    (OUT / "task_layer.md").write_text(heatmap_md(tasks, summary))
    print(f"wrote {OUT / 'task_layer.md'}")
    print(
        f"oracle {summary['oracle_pc']:.2f}%  L2 {summary['global_l2_pc']:.2f}%  "
        f"switch {summary['n_switch_from_l2']}/40  shuffle_same={n_same}/40"
    )


if __name__ == "__main__":
    main()

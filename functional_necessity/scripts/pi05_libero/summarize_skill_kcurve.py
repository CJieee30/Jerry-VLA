#!/usr/bin/env python3
"""Wave B1: calibration-only greedy extra-k Δloss curves (no closed-loop).

Reads existing skill greedy JSON. Does not rescore. Frozen τ ∈ {0.12, 0.20, 0.28}
are reported as a hypothesis table only — not used to pick closed-loop masks.
"""

from __future__ import annotations

import json
from pathlib import Path

from drop6_extra3_protocol import ATOMIC_SKILLS, GLOBAL_GREEDY_EXTRA3, UNIVERSAL_DROP

FN_ROOT = Path(__file__).resolve().parents[2]
SRC = FN_ROOT / "results" / "drop6_extra3" / "greedy_ppl_extra3_skill"
OUT_JSON = SRC / "k_curve.json"
OUT_MD = SRC / "k_curve.md"
SEEDS = (42, 43)
THRESHOLDS = (0.12, 0.20, 0.28)


def load_skill(seed: int, skill: str) -> dict:
    return json.loads((SRC / f"seed{seed}" / "per_skill" / f"{skill}.json").read_text())


def k_hat(deltas: list[float], tau: float) -> int:
    best = 0
    for k, d in enumerate(deltas, start=1):
        if d < tau:
            best = k
        else:
            break
    return best


def main() -> None:
    skills: dict[str, dict] = {}
    for skill in ATOMIC_SKILLS:
        by_seed = {}
        for seed in SEEDS:
            payload = load_skill(seed, skill)
            steps = payload["steps"]
            order = [int(s["picked"]) for s in steps]
            extras = [[int(x) for x in s["extra_after"]] for s in steps]
            deltas = [float(s["delta"]) for s in steps]
            losses = [float(s["loss"]) for s in steps]
            hats = {f"{tau:.2f}": k_hat(deltas, tau) for tau in THRESHOLDS}
            by_seed[str(seed)] = {
                "baseline_loss": float(payload["baseline_loss"]),
                "order": order,
                "extra_after": extras,
                "delta": deltas,
                "loss": losses,
                "extra3": [int(x) for x in payload["extra_drop"]],
                "k_hat": hats,
            }
        skills[skill] = by_seed

    doc = {
        "kind": "skill_greedy_k_curve",
        "note": (
            "Calibration-only. k_hat = max k with greedy prefix Δloss < τ. "
            "τ frozen at 0.12/0.20/0.28 before closed-loop. Not a closed-loop selector."
        ),
        "checkpoint": "lerobot/pi05_libero_finetuned_v044",
        "universal_drop": list(UNIVERSAL_DROP),
        "global_greedy_extra3": list(GLOBAL_GREEDY_EXTRA3),
        "thresholds": list(THRESHOLDS),
        "skills": skills,
    }
    OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n")

    lines = [
        "# Skill greedy extra-k calibration curves (Wave B1)",
        "",
        "Source: existing `per_skill/*.json` (no rescoring). Skip graph is already `skip(U)`.",
        f"U = `{list(UNIVERSAL_DROP)}`. Global greedy extra-3 control = `{list(GLOBAL_GREEDY_EXTRA3)}`.",
        "",
        "`k_hat(skill, τ) = max { k | greedy-prefix Δloss(k) < τ }`, with frozen τ ∈ {0.12, 0.20, 0.28}.",
        "This table is a **hypothesis**, not a closed-loop budget lock.",
        "",
        "## Greedy order and cumulative Δloss",
        "",
        "| Skill | seed | k=1 layer | k=2 extra | k=3 extra | Δ(k=1) | Δ(k=2) | Δ(k=3) | baseline loss |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for skill in ATOMIC_SKILLS:
        for seed in SEEDS:
            row = skills[skill][str(seed)]
            e1, e2, e3 = row["extra_after"]
            d1, d2, d3 = row["delta"]
            lines.append(
                f"| `{skill}` | {seed} | {row['order'][0]} | `{e2}` | `{e3}` | "
                f"{d1:.4f} | {d2:.4f} | {d3:.4f} | {row['baseline_loss']:.4f} |"
            )

    lines += [
        "",
        "## Frozen-τ k_hat (hypothesis only)",
        "",
        "| Skill | seed | k_hat(0.12) | k_hat(0.20) | k_hat(0.28) | vs global extra-3 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for skill in ATOMIC_SKILLS:
        for seed in SEEDS:
            row = skills[skill][str(seed)]
            hats = row["k_hat"]
            extra3 = row["extra3"]
            same = extra3 == list(GLOBAL_GREEDY_EXTRA3)
            lines.append(
                f"| `{skill}` | {seed} | {hats['0.12']} | {hats['0.20']} | {hats['0.28']} | "
                f"{'`{2,7,11}`' if same else extra3} |"
            )

    lines += [
        "",
        "## Readout (pre-registered, not tuned on closed-loop)",
        "",
        "- τ=0.12: almost all skills extra-1.",
        "- τ=0.20: extra-2 if Δ(k=2)<0.20 else extra-1.",
        "- τ=0.28: extra-3 unless Turn-like Δ(k=3) exceeds 0.28.",
        "- Compare later to Wave A Goal7 k=1/2/3. Do **not** export k_hat(turn) onto Long2.",
        "",
        f"JSON: `{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()

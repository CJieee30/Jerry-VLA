#!/usr/bin/env python3
"""Step 1: suite×skill stable core-2 audit from existing greedy JSON.

No rescoring. Reads seed42/43 per_cell greedy steps and writes whether the
extra-3 set-intersection equals both seeds' greedy prefix-2, plus k=2/k=3 gaps.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

from drop6_extra3_protocol import DEFAULT_OUT_DIR, SUITE_SKILL_CELLS, dump_json, suite_skill_stem


def _ranked(step: dict) -> list[dict]:
    return sorted(step["candidates"], key=lambda r: (r["loss"], r["layer"]))


def _gaps(ranked: list[dict], n: int = 3) -> list[float]:
    out = []
    for j in range(1, n + 1):
        if j < len(ranked):
            out.append(float(ranked[j]["delta"] - ranked[0]["delta"]))
    return out


def audit_cell(seed42: dict, seed43: dict) -> dict:
    rows = {}
    for seed, payload in ((42, seed42), (43, seed43)):
        steps = payload["steps"]
        rec = {
            "extra3": [int(x) for x in payload["extra_drop"]],
            "baseline_loss": float(payload["baseline_loss"]),
            "k1": int(steps[0]["picked"]),
            "prefix2_order": [int(steps[0]["picked"]), int(steps[1]["picked"])],
            "prefix2": [int(x) for x in steps[1]["extra_after"]],
            "prefix3": [int(x) for x in steps[2]["extra_after"]],
        }
        for i, st in enumerate(steps, start=1):
            ranked = _ranked(st)
            rec[f"k{i}_delta"] = float(st["delta"])
            rec[f"k{i}_top4"] = [
                {"layer": int(r["layer"]), "delta": float(r["delta"])} for r in ranked[:4]
            ]
            rec[f"k{i}_gap"] = _gaps(ranked)
        rows[seed] = rec
    shared3 = sorted(set(rows[42]["extra3"]) & set(rows[43]["extra3"]))
    p2a, p2b = set(rows[42]["prefix2"]), set(rows[43]["prefix2"])
    return {
        "seed42": rows[42],
        "seed43": rows[43],
        "shared_extra3": shared3,
        "n_shared_extra3": len(shared3),
        "same_extra3": rows[42]["extra3"] == rows[43]["extra3"],
        "same_prefix2": rows[42]["prefix2"] == rows[43]["prefix2"],
        "shared_equals_both_prefix2": len(shared3) == 2
        and set(shared3) == p2a
        and set(shared3) == p2b,
        "k1_l2_both": rows[42]["k1"] == 2 and rows[43]["k1"] == 2,
        "kind": (
            "stable_greedy_prefix2_plus_boundary"
            if rows[42]["prefix2"] == rows[43]["prefix2"] and rows[42]["extra3"] != rows[43]["extra3"]
            else "stable_greedy_prefix2"
            if rows[42]["prefix2"] == rows[43]["prefix2"]
            else "extra3_intersection_only"
        ),
    }


def write_md(payload: dict, path: Path) -> None:
    n_prefix = sum(1 for c in payload["cells"] if c["shared_equals_both_prefix2"])
    n_same_p2 = sum(1 for c in payload["cells"] if c["same_prefix2"])
    lines = [
        "# Suite×skill core-2 audit (Step 1, no rescoring)",
        "",
        "Source: existing greedy extra-3 `per_cell/*.json`, seeds 42/43. "
        "Skip graph is already `skip(U)`. Shared core is the extra-3 **set intersection**, "
        "which is not always the greedy prefix-2.",
        "",
        f"- Cells: **{len(payload['cells'])}**",
        f"- extra-3 identical across seeds: **{sum(1 for c in payload['cells'] if c['same_extra3'])}/10**",
        f"- greedy prefix-2 identical across seeds: **{n_same_p2}/10**",
        f"- extra-3 intersection equals **both** seeds' prefix-2: **{n_prefix}/10**",
        "",
        "| Cell | k=1 s42/s43 | prefix-2 s42 | prefix-2 s43 | extra-3 ∩ | ∩ = both prefix-2 | k=2 gap s42/s43 | k=3 gap s42/s43 | kind |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in payload["cells"]:
        a, b = c["seed42"], c["seed43"]
        k2_42 = a["k2_gap"][0] if a["k2_gap"] else float("nan")
        k2_43 = b["k2_gap"][0] if b["k2_gap"] else float("nan")
        k3_42 = a["k3_gap"][0] if a["k3_gap"] else float("nan")
        k3_43 = b["k3_gap"][0] if b["k3_gap"] else float("nan")
        lines.append(
            f"| `{c['cell']}` | {a['k1']}/{b['k1']} | `{a['prefix2']}` | `{b['prefix2']}` | "
            f"`{c['shared_extra3']}` | **{c['shared_equals_both_prefix2']}** | "
            f"{k2_42:.5f}/{k2_43:.5f} | {k3_42:.5f}/{k3_43:.5f} | {c['kind']} |"
        )
    lines += [
        "",
        "## Greedy top-4 at k=2 and k=3",
        "",
    ]
    for c in payload["cells"]:
        lines += [f"### `{c['cell']}`", ""]
        for seed in (42, 43):
            rec = c[f"seed{seed}"]
            t2 = ", ".join(f"L{x['layer']} Δ={x['delta']:.4f}" for x in rec["k2_top4"])
            t3 = ", ".join(f"L{x['layer']} Δ={x['delta']:.4f}" for x in rec["k3_top4"])
            lines += [
                f"- seed{seed} extra-3 `{rec['extra3']}` k1=L{rec['k1']}",
                f"  - k=2 top4: {t2}  gap={rec['k2_gap'][:3]}",
                f"  - k=3 top4: {t3}  gap={rec['k3_gap'][:3]}",
            ]
        lines.append("")
    lines += [
        "## Pre-registered readout",
        "",
    ]
    prefix2_cells = [c for c in payload["cells"] if c["shared_equals_both_prefix2"]]
    inter_only = [c for c in payload["cells"] if not c["shared_equals_both_prefix2"]]
    if prefix2_cells:
        lines.append("- **extra-3 ∩ equals both greedy prefix-2:**")
        for c in prefix2_cells:
            lines.append(
                f"  - `{c['cell']}` `{c['shared_extra3']}` ({c['kind']})"
            )
    if inter_only:
        lines.append("- **extra-3 ∩ is not both seeds' prefix-2:**")
        for c in inter_only:
            lines.append(
                f"  - `{c['cell']}` ∩ `{c['shared_extra3']}` "
                f"(s42 `{c['seed42']['prefix2']}` / s43 `{c['seed43']['prefix2']}`)"
            )
    lines += [
        "- L2 is in every extra-3, but is **not** always greedy k=1 (Goal-turn k1=L7 both seeds).",
        "- Step 2 tests Turn `{2,7}` vs `{2,9}` and Place `{2,8}` vs `{2,7}` as **pre-registered 2-sets**, not as proven greedy cores.",
        "",
        f"JSON: `{path.with_suffix('.json')}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=DEFAULT_OUT_DIR / "greedy_ppl_extra3_suite_skill",
    )
    args = parser.parse_args()
    cells = []
    for suite, verb in SUITE_SKILL_CELLS:
        stem = suite_skill_stem(suite, verb)
        a = json.loads((args.src_dir / "seed42" / "per_cell" / f"{stem}.json").read_text())
        b = json.loads((args.src_dir / "seed43" / "per_cell" / f"{stem}.json").read_text())
        rec = audit_cell(a, b)
        rec["cell"] = f"{suite}/{verb}"
        rec["suite"] = suite
        rec["skill"] = verb
        cells.append(rec)
    payload = {
        "source": str(args.src_dir),
        "n_cells": len(cells),
        "n_shared_equals_both_prefix2": sum(1 for c in cells if c["shared_equals_both_prefix2"]),
        "n_same_prefix2": sum(1 for c in cells if c["same_prefix2"]),
        "cells": cells,
    }
    out_json = args.src_dir / "core2_audit.json"
    out_md = args.src_dir / "core2_audit.md"
    dump_json(payload, out_json)
    write_md(payload, out_md)
    print(out_md.read_text())


if __name__ == "__main__":
    main()

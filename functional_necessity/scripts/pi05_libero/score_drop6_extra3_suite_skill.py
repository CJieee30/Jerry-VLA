#!/usr/bin/env python3
"""Drop-6 + greedy PPL extra-3 on suite×skill AtomicVLA pools.

Locks U={1,13,14,15,16,17}. Scores remaining-12 on skip(U) with frozen ξ.
Cells: pick/place × {Spatial,Object,Goal,Long} and turn × {Goal,Long}.
Goal-pick excludes Goal5. Control extra-3 stays {2,7,11}. No closed-loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging
import time

import torch

from common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SEED,
    FlowMatchingReplay,
    init_logging,
    load_calib_data,
    load_policy,
    paligemma_parts,
    set_seed,
)
from drop6_extra3_protocol import (
    CANDIDATES,
    DEFAULT_OUT_DIR,
    DEFAULT_SKILL_INDEX_DIR,
    EXTRA_K,
    GLOBAL_GREEDY_EXTRA3,
    N_LAYERS,
    SKILL_BATCH_SIZE,
    SKILL_NUM_BATCHES,
    SUITE_SKILL_CELLS,
    UNIVERSAL_DROP,
    dump_json,
    extra3_vs_global,
    protocol_dict,
    suite_skill_calib_dir,
    suite_skill_cell_name,
    suite_skill_stem,
    wait_for_file,
)
from score_drop6_extra3 import greedy_extra3, maybe_load_done


def write_seed_md(payload: dict, path: Path) -> None:
    lines = [
        f"# v044 Drop-6 + extra-3 (greedy_ppl, suite×skill, seed {payload['calib_seed']})",
        "",
        "Checkpoint: `lerobot/pi05_libero_finetuned_v044`. Universal skip "
        f"`U={UNIVERSAL_DROP}`. Candidates `{CANDIDATES}`. Greedy PPL on skip(U), "
        f"frozen ξ, extra-k={EXTRA_K}. Goal-pick excludes Goal5.",
        "",
        f"- Calib: `{payload['calib_dir']}` ({SKILL_NUM_BATCHES}×{SKILL_BATCH_SIZE})",
        f"- Global greedy control: `{GLOBAL_GREEDY_EXTRA3}`",
        "",
        "| Cell | extra-3 | vs `{2,7,11}` | Jaccard | n_source | Δloss | seconds |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for pool in payload["pools"]:
        vs = pool["vs_global_greedy"]
        lines.append(
            f"| `{pool['name']}` | `{pool['extra_drop']}` | `{vs['symdiff']}` | "
            f"{vs['jaccard']:.3f} | {pool.get('n_source_frames', '-')} | "
            f"{pool['delta']:.6f} | {pool['seconds']:.1f} |"
        )
    lines += ["", f"JSON: `{payload.get('out_json', path.with_suffix('.json'))}`", ""]
    path.write_text("\n".join(lines) + "\n")


def write_compare_md(payload: dict, path: Path) -> None:
    lines = [
        "# Suite×skill extra-3: seed 42 vs seed 43",
        "",
        f"- Global greedy control: `{GLOBAL_GREEDY_EXTRA3}`",
        "- Goal-pick excludes Goal5 (push mislabeled as pick).",
        "- This is calibration-only; no closed-loop.",
        "",
    ]
    for skill, block in payload["by_skill"].items():
        lines += [
            f"## `{skill}`",
            "",
            f"- Unique extra-3 seed42: **{block['n_unique_seed42']}** {block['uniques_seed42']}",
            f"- Unique extra-3 seed43: **{block['n_unique_seed43']}** {block['uniques_seed43']}",
            f"- Cross-suite split both seeds: **{block['split_both_seeds']}**",
            f"- Suites that differ from global on both seeds: `{block['suites_diff_global_both']}`",
            "",
            "| Suite | seed42 | seed43 | same across seeds | both ≠ `{2,7,11}` |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in block["suites"]:
            lines.append(
                f"| `{row['suite']}` | `{row['extra_seed42']}` | `{row['extra_seed43']}` | "
                f"{row['same_across_seeds']} | {row['differs_both_seeds']} |"
            )
        lines.append("")
        if block["pairwise"]:
            lines += ["| Pair | seed42 same | seed43 same | stable different |", "| --- | --- | --- | --- |"]
            for pair in block["pairwise"]:
                lines.append(
                    f"| `{pair['a']}` vs `{pair['b']}` | {pair['same_seed42']} | "
                    f"{pair['same_seed43']} | **{pair['stable_different']}** |"
                )
            lines.append("")
    lines += [
        "## Readout (pre-registered)",
        "",
        "- Stable Object-pick ≠ other-suite pick, with Object using L8 not L7: suite/scene, not verb.",
        "- Stable Goal-turn ≠ Long-turn: same as the already-known turn transfer failure.",
        "- All suites collapse to `{2,7,11}`: verb pooling was not hiding a suite×skill split.",
        "",
        f"JSON: `{path.with_suffix('.json')}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def score_seed(args, policy, calib_seed: int) -> dict:
    calib_dir = (
        suite_skill_calib_dir(calib_seed)
        if args.calib_root is None
        else Path(args.calib_root) / f"suite_skill_{SKILL_NUM_BATCHES}x{SKILL_BATCH_SIZE}_seed{calib_seed}"
    )
    out_dir = Path(args.out_dir) / f"seed{calib_seed}"
    per_dir = out_dir / "per_cell"
    per_dir.mkdir(parents=True, exist_ok=True)
    dump_json(protocol_dict(), out_dir / "protocol.json")
    pools = []
    for suite_label, verb in SUITE_SKILL_CELLS:
        name = suite_skill_cell_name(suite_label, verb)
        stem = suite_skill_stem(suite_label, verb)
        raw_path = per_dir / f"{stem}.json"
        done = maybe_load_done(raw_path)
        if done is not None and "extra_drop" in done:
            logging.info("[skip] seed%s %s already scored extra=%s", calib_seed, name, done["extra_drop"])
            raw = done
        else:
            calib_file = wait_for_file(calib_dir / f"{stem}.pt", timeout_s=args.wait_timeout)
            calib = load_calib_data(calib_file)
            logging.info("seed%s %s pool: %d batches from %s", calib_seed, name, len(calib), calib_file)
            t0 = time.time()
            replay = FlowMatchingReplay(policy)
            replay.capture(calib, args.device, args.microbatch_size)
            with replay:
                scored = greedy_extra3(policy, calib, args.device, args.microbatch_size, replay)
            extra = [int(x) for x in scored["extra_drop"]]
            n_source = int(
                (args.suite_keep or {}).get(suite_label, {}).get(verb, 0)
            ) or None
            raw = {
                "name": name,
                "suite": suite_label,
                "skill": verb,
                "calib_seed": calib_seed,
                "n_calib_batches": len(calib),
                "n_source_frames": n_source,
                "calib_file": str(calib_file),
                **scored,
                "vs_global_greedy": extra3_vs_global(extra),
                "seconds": round(time.time() - t0, 3),
            }
            dump_json(raw, raw_path)
            logging.info(
                "seed%s %s extra-3=%s vs_global_symdiff=%s Δ=%.6f (%.1fs)",
                calib_seed,
                name,
                raw["extra_drop"],
                raw["vs_global_greedy"]["symdiff"],
                raw["delta"],
                raw["seconds"],
            )
        pools.append(raw)
        dump_json(
            {"calib_seed": calib_seed, "n_done": len(pools), "n_total": len(SUITE_SKILL_CELLS), "pools": pools},
            out_dir / "partial.json",
        )

    payload = {
        "method": "greedy_ppl",
        "scope": "suite_skill",
        "kind": f"greedy_ppl_extra3_suite_skill_seed{calib_seed}",
        "checkpoint": args.checkpoint,
        "calib_dir": str(calib_dir),
        "calib_seed": calib_seed,
        "pooling": (
            f"AtomicVLA suite×skill pools, {SKILL_NUM_BATCHES}×{SKILL_BATCH_SIZE}, "
            "interior margin filter, Goal-pick excludes Goal5"
        ),
        "protocol": {
            **protocol_dict(),
            "global_greedy_extra3": list(GLOBAL_GREEDY_EXTRA3),
            "skill_boundary_margin": 10,
            "cells": [suite_skill_cell_name(s, v) for s, v in SUITE_SKILL_CELLS],
        },
        "pools": pools,
        "out_json": str(out_dir / "extra3.json"),
    }
    dump_json(payload, out_dir / "extra3.json")
    write_seed_md(payload, out_dir / "extra3.md")
    return payload


def compare_seeds(seed42: dict, seed43: dict) -> dict:
    by42 = {p["name"]: p for p in seed42["pools"]}
    by43 = {p["name"]: p for p in seed43["pools"]}
    glob = set(GLOBAL_GREEDY_EXTRA3)
    by_skill: dict[str, dict] = {}
    skills = sorted({v for _, v in SUITE_SKILL_CELLS})
    for skill in skills:
        suites = [s for s, v in SUITE_SKILL_CELLS if v == skill]
        rows = []
        extras42 = []
        extras43 = []
        both_global_diff = []
        for suite in suites:
            name = suite_skill_cell_name(suite, skill)
            a = [int(x) for x in by42[name]["extra_drop"]]
            b = [int(x) for x in by43[name]["extra_drop"]]
            extras42.append(tuple(a))
            extras43.append(tuple(b))
            differs_both = set(a) != glob and set(b) != glob
            if differs_both:
                both_global_diff.append(suite)
            rows.append(
                {
                    "suite": suite,
                    "extra_seed42": a,
                    "extra_seed43": b,
                    "same_across_seeds": a == b,
                    "differs_both_seeds": differs_both,
                    "vs_global_seed42": extra3_vs_global(a),
                    "vs_global_seed43": extra3_vs_global(b),
                }
            )
        pairwise = []
        for i, sa in enumerate(suites):
            for sb in suites[i + 1 :]:
                a42 = [int(x) for x in by42[suite_skill_cell_name(sa, skill)]["extra_drop"]]
                b42 = [int(x) for x in by42[suite_skill_cell_name(sb, skill)]["extra_drop"]]
                a43 = [int(x) for x in by43[suite_skill_cell_name(sa, skill)]["extra_drop"]]
                b43 = [int(x) for x in by43[suite_skill_cell_name(sb, skill)]["extra_drop"]]
                pairwise.append(
                    {
                        "a": sa,
                        "b": sb,
                        "same_seed42": a42 == b42,
                        "same_seed43": a43 == b43,
                        "stable_different": a42 != b42 and a43 != b43,
                    }
                )
        by_skill[skill] = {
            "suites": rows,
            "n_unique_seed42": len(set(extras42)),
            "n_unique_seed43": len(set(extras43)),
            "uniques_seed42": [list(x) for x in sorted(set(extras42))],
            "uniques_seed43": [list(x) for x in sorted(set(extras43))],
            "split_both_seeds": len(set(extras42)) >= 2 and len(set(extras43)) >= 2,
            "suites_diff_global_both": both_global_diff,
            "pairwise": pairwise,
            "n_stable_pairs": sum(1 for p in pairwise if p["stable_different"]),
        }
    return {
        "global_greedy_extra3": list(GLOBAL_GREEDY_EXTRA3),
        "goal5_excluded_from_goal_pick": True,
        "skills_with_split_both_seeds": [s for s, b in by_skill.items() if b["split_both_seeds"]],
        "by_skill": by_skill,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="scoring RNG; calib seeds are separate")
    parser.add_argument("--microbatch-size", type=int, default=None)
    parser.add_argument("--calib-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--wait-timeout", type=float, default=12 * 3600)
    parser.add_argument("--calib-seeds", type=int, nargs="+", default=[42, 43])
    args = parser.parse_args()

    init_logging()
    set_seed(args.seed)
    args.device = torch.device(args.device)
    args.out_dir = Path(args.out_dir) if args.out_dir is not None else DEFAULT_OUT_DIR / "greedy_ppl_extra3_suite_skill"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cov_path = DEFAULT_SKILL_INDEX_DIR / "coverage.json"
    if cov_path.is_file():
        args.suite_keep = json.loads(cov_path.read_text()).get("suite_frames_interior") or {}
    else:
        args.suite_keep = {}

    policy = load_policy(args.checkpoint, args.device)
    layers, _ = paligemma_parts(policy)
    if len(layers) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} PaliGemma layers, got {len(layers)}")
    policy.eval()

    scored = {}
    for calib_seed in args.calib_seeds:
        scored[int(calib_seed)] = score_seed(args, policy, int(calib_seed))
        print((Path(args.out_dir) / f"seed{calib_seed}" / "extra3.md").read_text())

    p42 = Path(args.out_dir) / "seed42" / "extra3.json"
    p43 = Path(args.out_dir) / "seed43" / "extra3.json"
    if p42.is_file() and p43.is_file():
        s42 = scored.get(42) or json.loads(p42.read_text())
        s43 = scored.get(43) or json.loads(p43.read_text())
        cmp_payload = compare_seeds(s42, s43)
        dump_json(cmp_payload, args.out_dir / "compare_seeds.json")
        write_compare_md(cmp_payload, args.out_dir / "compare_seeds.md")
        print((args.out_dir / "compare_seeds.md").read_text())
        logging.info("skills_with_split_both_seeds=%s", cmp_payload["skills_with_split_both_seeds"])
    else:
        logging.info("compare skipped until both seed42 and seed43 extra3.json exist")


if __name__ == "__main__":
    main()

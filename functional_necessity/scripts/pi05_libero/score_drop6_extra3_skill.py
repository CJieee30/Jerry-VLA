#!/usr/bin/env python3
"""Drop-6 + greedy PPL extra-3 on AtomicVLA skill-pooled calib.

Locks U={1,13,14,15,16,17}. Scores remaining-12 on skip(U) with frozen ξ.
Each of pick/place/open/close/turn is its own pool. Does not re-search the
global extra-3; the control stays {2,7,11} from the existing 40-task concat.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
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
    ATOMIC_SKILLS,
    CANDIDATES,
    DEFAULT_MAP_DIR,
    DEFAULT_OUT_DIR,
    EXTRA_K,
    GLOBAL_GREEDY_EXTRA3,
    N_LAYERS,
    SKILL_BATCH_SIZE,
    SKILL_NUM_BATCHES,
    UNIVERSAL_DROP,
    dump_json,
    extra3_vs_global,
    load_task_catalog,
    make_eval_map,
    protocol_dict,
    skill_calib_dir,
    wait_for_file,
)
from score_drop6_extra3 import greedy_extra3, maybe_load_done


def write_skill_md(payload: dict, path: Path) -> None:
    lines = [
        f"# v044 Drop-6 + extra-3 (greedy_ppl, skill pool, seed {payload['calib_seed']})",
        "",
        "Checkpoint: `lerobot/pi05_libero_finetuned_v044`. Universal skip "
        f"`U={UNIVERSAL_DROP}`. Candidates `{CANDIDATES}`. Greedy PPL on skip(U), "
        f"frozen ξ, extra-k={EXTRA_K}. Pools are AtomicVLA verbs after a "
        f"{payload['protocol'].get('skill_boundary_margin', 10)}-frame boundary filter.",
        "",
        f"- Calib: `{payload['calib_dir']}` ({SKILL_NUM_BATCHES}×{SKILL_BATCH_SIZE})",
        f"- Global greedy control (not re-searched): `{GLOBAL_GREEDY_EXTRA3}`",
        f"- Gate A this seed (≥2 skills differ): **{payload['gate_a_this_seed']}**",
        "",
        "| Skill | extra-3 | vs `{2,7,11}` symdiff | Jaccard | Δloss | seconds |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for pool in payload["pools"]:
        vs = pool["vs_global_greedy"]
        lines.append(
            f"| `{pool['name']}` | `{pool['extra_drop']}` | `{vs['symdiff']}` | "
            f"{vs['jaccard']:.3f} | {pool['delta']:.6f} | {pool['seconds']:.1f} |"
        )
    lines += ["", f"JSON: `{payload.get('out_json', path.with_suffix('.json'))}`", ""]
    path.write_text("\n".join(lines) + "\n")


def write_compare_md(payload: dict, path: Path) -> None:
    lines = [
        "# P0 skill extra-3: seed 42 vs seed 43",
        "",
        f"- Global greedy control: `{GLOBAL_GREEDY_EXTRA3}`",
        f"- Gate A (≥2 skills differ from global on **both** seeds): **{payload['gate_a']}**",
        f"- Skills differing on both seeds: `{payload['skills_diff_both']}`",
        "",
        "| Skill | seed42 | seed43 | same across seeds | both ≠ `{2,7,11}` |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["skills"]:
        lines.append(
            f"| `{row['skill']}` | `{row['extra_seed42']}` | `{row['extra_seed43']}` | "
            f"{row['same_across_seeds']} | {row['differs_both_seeds']} |"
        )
    lines += ["", "## Closed-loop implication", ""]
    if payload["gate_a"]:
        lines.append(
            "Gate A passed. Step 3 should run Goal0+Goal7 first. "
            "Add close / pick-place only if those masks differ from `{2,7,11}`."
        )
    else:
        lines.append(
            "Gate A failed. Stop Skill-aware as a method; do not start step-3 closed-loop."
        )
    lines += ["", f"JSON: `{path.with_suffix('.json')}`", ""]
    path.write_text("\n".join(lines) + "\n")


def score_seed(args, policy, catalog, calib_seed: int) -> dict:
    calib_dir = skill_calib_dir(calib_seed) if args.calib_root is None else Path(args.calib_root) / f"per_skill_{SKILL_NUM_BATCHES}x{SKILL_BATCH_SIZE}_seed{calib_seed}"
    out_dir = Path(args.out_dir) / f"seed{calib_seed}"
    per_dir = out_dir / "per_skill"
    per_dir.mkdir(parents=True, exist_ok=True)
    dump_json(protocol_dict(), out_dir / "protocol.json")
    pools = []
    for verb in ATOMIC_SKILLS:
        raw_path = per_dir / f"{verb}.json"
        done = maybe_load_done(raw_path)
        if done is not None and "extra_drop" in done:
            logging.info("[skip] seed%s %s already scored extra=%s", calib_seed, verb, done["extra_drop"])
            raw = done
        else:
            calib_file = wait_for_file(calib_dir / f"{verb}.pt", timeout_s=args.wait_timeout)
            calib = load_calib_data(calib_file)
            logging.info("seed%s %s pool: %d batches from %s", calib_seed, verb, len(calib), calib_file)
            t0 = time.time()
            replay = FlowMatchingReplay(policy)
            replay.capture(calib, args.device, args.microbatch_size)
            with replay:
                scored = greedy_extra3(policy, calib, args.device, args.microbatch_size, replay)
            extra = [int(x) for x in scored["extra_drop"]]
            raw = {
                "name": verb,
                "skill": verb,
                "calib_seed": calib_seed,
                "n_calib_batches": len(calib),
                "calib_file": str(calib_file),
                **scored,
                "vs_global_greedy": extra3_vs_global(extra),
                "seconds": round(time.time() - t0, 3),
            }
            dump_json(raw, raw_path)
            logging.info(
                "seed%s %s extra-3=%s vs_global_symdiff=%s Δ=%.6f (%.1fs)",
                calib_seed,
                verb,
                raw["extra_drop"],
                raw["vs_global_greedy"]["symdiff"],
                raw["delta"],
                raw["seconds"],
            )
        pools.append(raw)
        dump_json(
            {"calib_seed": calib_seed, "n_done": len(pools), "n_total": len(ATOMIC_SKILLS), "pools": pools},
            out_dir / "partial.json",
        )

    n_diff = sum(1 for p in pools if p["vs_global_greedy"]["differs"])
    payload = {
        "method": "greedy_ppl",
        "scope": "skill",
        "kind": f"greedy_ppl_extra3_skill_seed{calib_seed}",
        "checkpoint": args.checkpoint,
        "calib_dir": str(calib_dir),
        "calib_seed": calib_seed,
        "pooling": f"AtomicVLA verb pools, {SKILL_NUM_BATCHES}×{SKILL_BATCH_SIZE}, interior margin filter",
        "protocol": {
            **protocol_dict(),
            "global_greedy_extra3": list(GLOBAL_GREEDY_EXTRA3),
            "skill_boundary_margin": 10,
            "atomic_skills": list(ATOMIC_SKILLS),
        },
        "gate_a_this_seed": n_diff >= 2,
        "n_skills_differ": n_diff,
        "pools": pools,
        "out_json": str(out_dir / "extra3.json"),
    }
    dump_json(payload, out_dir / "extra3.json")
    write_skill_md(payload, out_dir / "extra3.md")
    for pool in pools:
        extra = [int(x) for x in pool["extra_drop"]]
        tasks = [{**task, "extra_drop": extra} for task in catalog]
        mapping = make_eval_map(
            kind=f"greedy_ppl_extra3_skill_{pool['name']}_seed{calib_seed}",
            method="greedy_ppl",
            tasks=tasks,
            extra={
                "pooling": "constant skill mask stamped on 40 tasks for bookkeeping",
                "skill": pool["name"],
                "calib_seed": calib_seed,
                "vs_global_greedy": pool["vs_global_greedy"],
                "note": "Closed-loop step 3 should use --drop-layers on a Goal0/Goal7 subset, not 40-task routing.",
            },
        )
        dump_json(mapping, Path(args.map_dir) / f"map_greedy_ppl_extra3_skill_{pool['name']}_seed{calib_seed}.json")
    return payload


def compare_seeds(seed42: dict, seed43: dict) -> dict:
    by42 = {p["name"]: [int(x) for x in p["extra_drop"]] for p in seed42["pools"]}
    by43 = {p["name"]: [int(x) for x in p["extra_drop"]] for p in seed43["pools"]}
    glob = set(GLOBAL_GREEDY_EXTRA3)
    rows = []
    both = []
    for verb in ATOMIC_SKILLS:
        a, b = by42[verb], by43[verb]
        differs_both = set(a) != glob and set(b) != glob
        if differs_both:
            both.append(verb)
        rows.append(
            {
                "skill": verb,
                "extra_seed42": a,
                "extra_seed43": b,
                "same_across_seeds": a == b,
                "differs_both_seeds": differs_both,
                "vs_global_seed42": extra3_vs_global(a),
                "vs_global_seed43": extra3_vs_global(b),
            }
        )
    return {
        "global_greedy_extra3": list(GLOBAL_GREEDY_EXTRA3),
        "skills_diff_seed42": [r["skill"] for r in rows if r["vs_global_seed42"]["differs"]],
        "skills_diff_seed43": [r["skill"] for r in rows if r["vs_global_seed43"]["differs"]],
        "skills_diff_both": both,
        "gate_a": len(both) >= 2,
        "skills": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="scoring RNG; calib seeds are separate")
    parser.add_argument("--microbatch-size", type=int, default=None)
    parser.add_argument("--calib-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--wait-timeout", type=float, default=12 * 3600)
    parser.add_argument("--calib-seeds", type=int, nargs="+", default=[42, 43])
    args = parser.parse_args()

    init_logging()
    set_seed(args.seed)
    args.device = torch.device(args.device)
    catalog = load_task_catalog()
    args.out_dir = Path(args.out_dir) if args.out_dir is not None else DEFAULT_OUT_DIR / "greedy_ppl_extra3_skill"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.map_dir).mkdir(parents=True, exist_ok=True)

    policy = load_policy(args.checkpoint, args.device)
    layers, _ = paligemma_parts(policy)
    if len(layers) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} PaliGemma layers, got {len(layers)}")
    policy.eval()

    scored = {}
    for calib_seed in args.calib_seeds:
        scored[int(calib_seed)] = score_seed(args, policy, catalog, int(calib_seed))
        print((Path(args.out_dir) / f"seed{calib_seed}" / "extra3.md").read_text())

    if 42 in scored and 43 in scored:
        cmp_payload = compare_seeds(scored[42], scored[43])
        dump_json(cmp_payload, args.out_dir / "compare_seeds.json")
        write_compare_md(cmp_payload, args.out_dir / "compare_seeds.md")
        print((args.out_dir / "compare_seeds.md").read_text())
        logging.info("Gate A=%s skills_diff_both=%s", cmp_payload["gate_a"], cmp_payload["skills_diff_both"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Held-out matched/swap Δloss for pre-registered suite×skill extra-2 cores.

Turn: {2,7} vs {2,9} on Goal-turn / Long-turn.
Place: {2,8} vs {2,7} on Object-place / Spatial-place / Long-place.
Anchors: extra=[] (U only) and extra={2}. skip(U), frozen ξ on held-out.
GO: matched Δloss + 0.01 <= swapped on both seeds 44 and 45.
No closed-loop. Does not search extra-3.
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
    CORE2_GAP_MIN,
    CORE2_HELDOUT_CELLS,
    CORE2_HELDOUT_SEEDS,
    CORE2_PLACE_CELLS,
    CORE2_PLACE_MASKS,
    CORE2_TURN_CELLS,
    CORE2_TURN_MASKS,
    DEFAULT_OUT_DIR,
    N_LAYERS,
    UNIVERSAL_DROP,
    dump_json,
    suite_skill_cell_name,
    suite_skill_heldout_calib_dir,
    suite_skill_stem,
    wait_for_file,
)
from score_drop6_extra3 import eval_extra_loss, maybe_load_done


def masks_for_cell(skill: str) -> list[tuple[str, list[int]]]:
    if skill == "turn":
        return [
            ("U_only", []),
            ("L2_only", [2]),
            ("core_27", [2, 7]),
            ("core_29", [2, 9]),
        ]
    if skill == "place":
        return [
            ("U_only", []),
            ("L2_only", [2]),
            ("core_27", [2, 7]),
            ("core_28", [2, 8]),
        ]
    raise ValueError(skill)


def score_cell(policy, calib, device, microbatch_size, skill: str) -> dict:
    replay = FlowMatchingReplay(policy)
    replay.capture(calib, device, microbatch_size)
    cache: dict[tuple[int, ...], float] = {}
    with replay:
        baseline = eval_extra_loss(policy, [], calib, device, microbatch_size, replay, cache)
        masks = {}
        for name, extra in masks_for_cell(skill):
            t0 = time.time()
            loss = eval_extra_loss(policy, extra, calib, device, microbatch_size, replay, cache)
            masks[name] = {
                "extra": list(extra),
                "loss": float(loss),
                "delta": float(loss - baseline),
                "seconds": round(time.time() - t0, 3),
            }
    return {"baseline_loss": float(baseline), "masks": masks}


def score_seed(args, policy, calib_seed: int) -> dict:
    calib_dir = suite_skill_heldout_calib_dir(calib_seed)
    out_dir = Path(args.out_dir) / f"seed{calib_seed}"
    per_dir = out_dir / "per_cell"
    per_dir.mkdir(parents=True, exist_ok=True)
    pools = []
    for suite_label, verb in CORE2_HELDOUT_CELLS:
        name = suite_skill_cell_name(suite_label, verb)
        stem = suite_skill_stem(suite_label, verb)
        raw_path = per_dir / f"{stem}.json"
        done = maybe_load_done(raw_path)
        if done is not None and "masks" in done:
            logging.info("[skip] seed%s %s", calib_seed, name)
            raw = done
        else:
            calib_file = wait_for_file(calib_dir / f"{stem}.pt", timeout_s=args.wait_timeout)
            calib = load_calib_data(calib_file)
            logging.info("seed%s %s held-out %d batches", calib_seed, name, len(calib))
            t0 = time.time()
            scored = score_cell(policy, calib, args.device, args.microbatch_size, verb)
            raw = {
                "name": name,
                "suite": suite_label,
                "skill": verb,
                "calib_seed": calib_seed,
                "n_calib_batches": len(calib),
                "calib_file": str(calib_file),
                **scored,
                "seconds": round(time.time() - t0, 3),
            }
            dump_json(raw, raw_path)
            logging.info(
                "seed%s %s baseline=%.4f masks=%s (%.1fs)",
                calib_seed,
                name,
                raw["baseline_loss"],
                {k: round(v["delta"], 4) for k, v in raw["masks"].items()},
                raw["seconds"],
            )
        pools.append(raw)
    payload = {
        "scope": "suite_skill_core2_heldout",
        "calib_seed": calib_seed,
        "calib_dir": str(calib_dir),
        "gap_min": CORE2_GAP_MIN,
        "pools": pools,
        "out_json": str(out_dir / "heldout.json"),
    }
    dump_json(payload, out_dir / "heldout.json")
    return payload


def _delta(pool: dict, mask: str) -> float:
    return float(pool["masks"][mask]["delta"])


def judge_turn(by_seed: dict[int, dict]) -> dict:
    rows = []
    go_flags = []
    for seed, payload in by_seed.items():
        pools = {p["name"]: p for p in payload["pools"]}
        g = pools["Goal/turn"]
        lng = pools["Long/turn"]
        g_m, g_s = _delta(g, "core_27"), _delta(g, "core_29")
        l_m, l_s = _delta(lng, "core_29"), _delta(lng, "core_27")
        goal_ok = g_m + CORE2_GAP_MIN <= g_s
        long_ok = l_m + CORE2_GAP_MIN <= l_s
        go_flags.append(goal_ok and long_ok)
        rows.append(
            {
                "seed": seed,
                "goal_matched_27": g_m,
                "goal_swapped_29": g_s,
                "goal_advantage": g_s - g_m,
                "goal_prefers_matched": goal_ok,
                "long_matched_29": l_m,
                "long_swapped_27": l_s,
                "long_advantage": l_s - l_m,
                "long_prefers_matched": long_ok,
                "goal_L2": _delta(g, "L2_only"),
                "long_L2": _delta(lng, "L2_only"),
                "diagonal": goal_ok and long_ok,
            }
        )
    return {
        "gap_min": CORE2_GAP_MIN,
        "go": all(go_flags) and len(go_flags) == 2,
        "seeds": rows,
        "matched_cores": CORE2_TURN_MASKS,
        "cells": [suite_skill_cell_name(s, v) for s, v in CORE2_TURN_CELLS],
    }


def judge_place(by_seed: dict[int, dict]) -> dict:
    rows = []
    go_flags = []
    for seed, payload in by_seed.items():
        pools = {p["name"]: p for p in payload["pools"]}
        obj = pools["Object/place"]
        spa = pools["Spatial/place"]
        lng = pools["Long/place"]
        o_m, o_s = _delta(obj, "core_28"), _delta(obj, "core_27")
        s_m, s_s = _delta(spa, "core_27"), _delta(spa, "core_28")
        l_m, l_s = _delta(lng, "core_27"), _delta(lng, "core_28")
        obj_ok = o_m + CORE2_GAP_MIN <= o_s
        spa_ok = s_m + CORE2_GAP_MIN <= s_s
        long_ok = l_m + CORE2_GAP_MIN <= l_s
        go_flags.append(obj_ok and spa_ok and long_ok)
        rows.append(
            {
                "seed": seed,
                "object_matched_28": o_m,
                "object_swapped_27": o_s,
                "object_advantage": o_s - o_m,
                "object_prefers_matched": obj_ok,
                "spatial_matched_27": s_m,
                "spatial_swapped_28": s_s,
                "spatial_advantage": s_s - s_m,
                "spatial_prefers_matched": spa_ok,
                "long_matched_27": l_m,
                "long_swapped_28": l_s,
                "long_advantage": l_s - l_m,
                "long_prefers_matched": long_ok,
                "diagonal": obj_ok and spa_ok and long_ok,
            }
        )
    return {
        "gap_min": CORE2_GAP_MIN,
        "go": all(go_flags) and len(go_flags) == 2,
        "seeds": rows,
        "matched_cores": CORE2_PLACE_MASKS,
        "cells": [suite_skill_cell_name(s, v) for s, v in CORE2_PLACE_CELLS],
        "note": "Place GO requires Object to prefer {2,8} and both Spatial and Long to prefer {2,7}.",
    }


def write_md(payload: dict, path: Path) -> None:
    turn, place = payload["turn"], payload["place"]
    lines = [
        "# Suite×skill core-2 held-out matched/swap Δloss (Step 2)",
        "",
        f"Protocol: skip(U={UNIVERSAL_DROP}), frozen ξ captured on held-out 32×8, "
        f"episode-disjoint from seed42/43 dumps. Val seeds {list(CORE2_HELDOUT_SEEDS)}. "
        f"GO if matched Δloss + {CORE2_GAP_MIN} ≤ swapped on **both** val seeds. "
        "No closed-loop, no extra-3 search.",
        "",
        f"- Turn GO: **{turn['go']}**",
        f"- Place GO: **{place['go']}**",
        "",
        "## Turn  `{2,7}` vs `{2,9}`",
        "",
        "| Seed | Goal matched `{2,7}` | Goal swap `{2,9}` | adv | Long matched `{2,9}` | Long swap `{2,7}` | adv | diagonal |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in turn["seeds"]:
        lines.append(
            f"| {r['seed']} | {r['goal_matched_27']:.4f} | {r['goal_swapped_29']:.4f} | "
            f"{r['goal_advantage']:.4f} | {r['long_matched_29']:.4f} | "
            f"{r['long_swapped_27']:.4f} | {r['long_advantage']:.4f} | **{r['diagonal']}** |"
        )
    lines += [
        "",
        "## Place  Object `{2,8}` vs Spatial/Long `{2,7}`",
        "",
        "| Seed | Obj `{2,8}` | Obj swap `{2,7}` | adv | Spa `{2,7}` | Spa swap `{2,8}` | adv | Long `{2,7}` | Long swap `{2,8}` | adv | diagonal |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in place["seeds"]:
        lines.append(
            f"| {r['seed']} | {r['object_matched_28']:.4f} | {r['object_swapped_27']:.4f} | "
            f"{r['object_advantage']:.4f} | {r['spatial_matched_27']:.4f} | "
            f"{r['spatial_swapped_28']:.4f} | {r['spatial_advantage']:.4f} | "
            f"{r['long_matched_27']:.4f} | {r['long_swapped_28']:.4f} | "
            f"{r['long_advantage']:.4f} | **{r['diagonal']}** |"
        )
    lines += [
        "",
        "## Per-cell masks",
        "",
        "| Cell | seed | U | L2 | `{2,7}` | `{2,8}`/`{2,9}` |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed in CORE2_HELDOUT_SEEDS:
        pools = {p["name"]: p for p in payload["by_seed"][str(seed)]["pools"]}
        for suite, verb in CORE2_HELDOUT_CELLS:
            name = suite_skill_cell_name(suite, verb)
            m = pools[name]["masks"]
            other = m["core_29"] if "core_29" in m else m["core_28"]
            lines.append(
                f"| `{name}` | {seed} | {m['U_only']['delta']:.4f} | {m['L2_only']['delta']:.4f} | "
                f"{m['core_27']['delta']:.4f} | {other['delta']:.4f} |"
            )
    lines += [
        "",
        "## Verdict",
        "",
    ]
    if turn["go"]:
        lines.append("- Turn: **GO**. Held-out Δloss prefers Goal `{2,7}` and Long `{2,9}` with gap ≥ 0.01 on both seeds.")
    else:
        lines.append("- Turn: **STOP**. No stable matched>swapped preference at the pre-registered 0.01 gap.")
    if place["go"]:
        lines.append("- Place: **GO**. Object prefers `{2,8}` and Spatial+Long prefer `{2,7}`.")
    else:
        lines.append("- Place: **STOP**. Object L8 is not a held-out place-core, or Spatial/Long do not prefer `{2,7}` over `{2,8}`.")
    lines.append("- Do not run extra-3 closed-loop or Drop-8 from this file.")
    lines += ["", f"JSON: `{path.with_suffix('.json')}`", ""]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--microbatch-size", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--wait-timeout", type=float, default=12 * 3600)
    parser.add_argument("--calib-seeds", type=int, nargs="+", default=list(CORE2_HELDOUT_SEEDS))
    args = parser.parse_args()

    init_logging()
    set_seed(args.seed)
    args.device = torch.device(args.device)
    args.out_dir = Path(args.out_dir) if args.out_dir is not None else DEFAULT_OUT_DIR / "suite_skill_core2_heldout"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    policy = load_policy(args.checkpoint, args.device)
    layers, _ = paligemma_parts(policy)
    if len(layers) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} PaliGemma layers, got {len(layers)}")
    policy.eval()

    scored = {}
    for calib_seed in args.calib_seeds:
        scored[int(calib_seed)] = score_seed(args, policy, int(calib_seed))

    by_seed = {}
    for seed in CORE2_HELDOUT_SEEDS:
        path = Path(args.out_dir) / f"seed{seed}" / "heldout.json"
        if path.is_file():
            by_seed[int(seed)] = scored.get(int(seed)) or json.loads(path.read_text())
    if set(by_seed) >= set(CORE2_HELDOUT_SEEDS):
        payload = {
            "gap_min": CORE2_GAP_MIN,
            "exclude_discovery_seeds": [42, 43],
            "val_seeds": list(CORE2_HELDOUT_SEEDS),
            "turn": judge_turn(by_seed),
            "place": judge_place(by_seed),
            "by_seed": {str(k): v for k, v in by_seed.items()},
        }
        dump_json(payload, args.out_dir / "compare_heldout.json")
        write_md(payload, args.out_dir / "compare_heldout.md")
        print((args.out_dir / "compare_heldout.md").read_text())
        logging.info("Turn GO=%s Place GO=%s", payload["turn"]["go"], payload["place"]["go"])
    else:
        logging.info("compare skipped; have seeds %s", sorted(by_seed))


if __name__ == "__main__":
    main()

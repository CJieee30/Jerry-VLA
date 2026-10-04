#!/usr/bin/env python3
"""Drop-6 + greedy PPL extra-3 with pooled calib: global or suite-aware.

Locks U={1,13,14,15,16,17}. Scores remaining-12 on skip(U) with frozen ξ.
Global concatenates all 40 per-task 16×8 dumps (equal task weight).
Suite concatenates the 10 dumps inside each LIBERO suite.
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
    DEFAULT_CALIB_DIR,
    DEFAULT_MAP_DIR,
    DEFAULT_OUT_DIR,
    EXTRA_K,
    N_LAYERS,
    PER_TASK_NUM_BATCHES,
    SUITE_LABEL,
    SUITE_ORDER,
    UNIVERSAL_DROP,
    calib_path,
    dump_json,
    load_task_catalog,
    make_eval_map,
    protocol_dict,
    summarize_extra_map,
    tasks_by_suite,
    wait_for_file,
    write_summary_md,
)
from score_drop6_extra3 import greedy_extra3, maybe_load_done


def concat_calib(calib_dir: Path, tasks: list[dict], wait_timeout: float) -> tuple[list, list[dict]]:
    merged: list = []
    sources: list[dict] = []
    for task in tasks:
        path = wait_for_file(calib_path(calib_dir, task["short"]), timeout_s=wait_timeout)
        data = load_calib_data(path)
        merged.extend(data)
        sources.append(
            {
                "short": task["short"],
                "suite": task["suite"],
                "task_id": task["task_id"],
                "n_batches": len(data),
                "path": str(path),
            }
        )
    if not merged:
        raise RuntimeError("pooled calib is empty")
    return merged, sources


def score_pool(policy, calib, device, microbatch_size) -> dict:
    replay = FlowMatchingReplay(policy)
    replay.capture(calib, device, microbatch_size)
    with replay:
        return greedy_extra3(policy, calib, device, microbatch_size, replay)


def write_maps(kind: str, tasks: list[dict], extra: dict, map_dir: Path) -> Path:
    payload = make_eval_map(kind=kind, method="greedy_ppl", tasks=tasks, extra=extra)
    path = map_dir / f"map_{kind}.json"
    dump_json(payload, path)
    return path


def write_pooled_md(scope: str, payload: dict, path: Path) -> None:
    summary = payload["summary"]
    lines = [
        f"# v044 Drop-6 + extra-3 (greedy_ppl, {scope} pool)",
        "",
        "Checkpoint: `lerobot/pi05_libero_finetuned_v044`. Universal skip "
        f"`U={UNIVERSAL_DROP}`. Candidates `{CANDIDATES}`. Greedy PPL on skip(U), "
        f"frozen ξ, extra-k={EXTRA_K}.",
        "",
        f"- Pooling: **{payload['pooling']}**",
        f"- Unique extra-3 triples: **{summary['n_unique_triples']}**",
        f"- Match global Fisher `{{2,3,10}}`: **{summary['n_match_global_fisher_extra3']}/40**",
        f"- extra-3 contains L0: **{summary.get('n_extra3_contains_l0', 0)}/40**",
        f"- extra-3 contains L7: **{summary.get('n_extra3_contains_l7', 0)}/40**",
        "",
        "| Triple | Count |",
        "| --- | ---: |",
    ]
    for triple, count in summary["histogram"].items():
        lines.append(f"| `{triple}` | {count} |")
    pools = payload.get("pools") or []
    if pools:
        lines += ["", "## Pools", "", "| Pool | n_batches | extra-3 | union | Δloss | seconds |", "| --- | ---: | --- | --- | ---: | ---: |"]
        for pool in pools:
            lines.append(
                f"| `{pool['name']}` | {pool['n_calib_batches']} | `{pool['extra_drop']}` | "
                f"`{pool['union_drop']}` | {pool['delta']:.6f} | {pool['seconds']:.1f} |"
            )
    lines += ["", "## Per-task extra-3", ""]
    for row in payload["tasks"]:
        lines.append(
            f"- `{row['short']}` ({row['suite']} {row['task_id']}): `{row['extra_drop']}`"
        )
    lines += ["", f"JSON: `{payload.get('out_json', path.with_suffix('.json'))}`", ""]
    path.write_text("\n".join(lines) + "\n")


def run_global(args, policy, catalog) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(protocol_dict(), out_dir / "protocol.json")
    raw_path = out_dir / "pool_global.json"
    done = maybe_load_done(raw_path)
    if done is not None and "extra_drop" in done:
        logging.info("[skip] global pool already scored extra=%s", done["extra_drop"])
        raw = done
    else:
        calib, sources = concat_calib(args.calib_dir, catalog, args.wait_timeout)
        logging.info("global pool: %d tasks, %d batches", len(catalog), len(calib))
        t0 = time.time()
        scored = score_pool(policy, calib, args.device, args.microbatch_size)
        raw = {
            "name": "global",
            "suite": None,
            "n_tasks": len(catalog),
            "n_calib_batches": len(calib),
            "expected_batches": len(catalog) * PER_TASK_NUM_BATCHES,
            "sources": sources,
            **scored,
            "seconds": round(time.time() - t0, 3),
        }
        dump_json(raw, raw_path)
        logging.info(
            "global extra-3=%s union=%s Δ=%.6f (%.1fs)",
            raw["extra_drop"],
            raw["union_drop"],
            raw["delta"],
            raw["seconds"],
        )
    extra = [int(x) for x in raw["extra_drop"]]
    tasks_out = []
    for task in catalog:
        rec = {
            **task,
            "extra_drop": extra,
            "union_drop": list(raw["union_drop"]),
            "pool": "global",
            "baseline_loss": raw["baseline_loss"],
            "loss": raw["loss"],
            "delta": raw["delta"],
        }
        tasks_out.append(rec)
    return {
        "pooling": f"concat {len(catalog)}×{PER_TASK_NUM_BATCHES}×8 per-task dumps, equal task weight",
        "pools": [raw],
        "tasks": tasks_out,
    }


def run_suite(args, policy, catalog) -> dict:
    out_dir = Path(args.out_dir)
    per_dir = out_dir / "per_suite"
    per_dir.mkdir(parents=True, exist_ok=True)
    dump_json(protocol_dict(), out_dir / "protocol.json")
    grouped = tasks_by_suite(catalog)
    pools = []
    extra_by_key: dict[tuple[str, int], list[int]] = {}
    for suite in SUITE_ORDER:
        label = SUITE_LABEL[suite]
        suite_tasks = grouped[suite]
        raw_path = per_dir / f"{label}.json"
        done = maybe_load_done(raw_path)
        if done is not None and "extra_drop" in done:
            logging.info("[skip] %s already scored extra=%s", label, done["extra_drop"])
            raw = done
        else:
            calib, sources = concat_calib(args.calib_dir, suite_tasks, args.wait_timeout)
            logging.info("%s pool: %d tasks, %d batches", label, len(suite_tasks), len(calib))
            t0 = time.time()
            scored = score_pool(policy, calib, args.device, args.microbatch_size)
            raw = {
                "name": label,
                "suite": suite,
                "n_tasks": len(suite_tasks),
                "n_calib_batches": len(calib),
                "expected_batches": len(suite_tasks) * PER_TASK_NUM_BATCHES,
                "sources": sources,
                **scored,
                "seconds": round(time.time() - t0, 3),
            }
            dump_json(raw, raw_path)
            logging.info(
                "%s extra-3=%s union=%s Δ=%.6f (%.1fs)",
                label,
                raw["extra_drop"],
                raw["union_drop"],
                raw["delta"],
                raw["seconds"],
            )
        pools.append(raw)
        extra = [int(x) for x in raw["extra_drop"]]
        for task in suite_tasks:
            extra_by_key[(task["suite"], int(task["task_id"]))] = extra
        dump_json(
            {"method": "greedy_ppl", "scope": "suite", "n_done": len(pools), "n_total": 4, "pools": pools},
            out_dir / "partial.json",
        )
    tasks_out = []
    for task in catalog:
        extra = extra_by_key[(task["suite"], int(task["task_id"]))]
        union = sorted(set(UNIVERSAL_DROP) | set(extra))
        pool = next(p for p in pools if p["suite"] == task["suite"])
        tasks_out.append(
            {
                **task,
                "extra_drop": extra,
                "union_drop": union,
                "pool": SUITE_LABEL[task["suite"]],
                "baseline_loss": pool["baseline_loss"],
                "loss": pool["loss"],
                "delta": pool["delta"],
            }
        )
    return {
        "pooling": f"concat 10×{PER_TASK_NUM_BATCHES}×8 per-task dumps inside each suite",
        "pools": pools,
        "tasks": tasks_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, choices=("global", "suite"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--microbatch-size", type=int, default=None)
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--map-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--wait-timeout", type=float, default=12 * 3600)
    args = parser.parse_args()

    init_logging()
    set_seed(args.seed)
    args.device = torch.device(args.device)
    catalog = load_task_catalog()
    kind = f"greedy_ppl_extra3_{args.scope}"
    args.out_dir = Path(args.out_dir) if args.out_dir is not None else DEFAULT_OUT_DIR / kind
    args.out_dir.mkdir(parents=True, exist_ok=True)

    policy = load_policy(args.checkpoint, args.device)
    layers, _ = paligemma_parts(policy)
    if len(layers) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} PaliGemma layers, got {len(layers)}")
    policy.eval()

    if args.scope == "global":
        packed = run_global(args, policy, catalog)
    else:
        packed = run_suite(args, policy, catalog)

    tasks_out = packed["tasks"]
    if len(tasks_out) != 40:
        raise RuntimeError(f"expected 40 tasks, got {len(tasks_out)}")
    summary = summarize_extra_map(tasks_out)
    payload = {
        "method": "greedy_ppl",
        "scope": args.scope,
        "kind": kind,
        "checkpoint": args.checkpoint,
        "calib_dir": str(args.calib_dir),
        "pooling": packed["pooling"],
        "protocol": protocol_dict(),
        "summary": summary,
        "pools": packed["pools"],
        "tasks": tasks_out,
        "out_json": str(args.out_dir / "extra3.json"),
    }
    dump_json(payload, args.out_dir / "extra3.json")
    write_pooled_md(args.scope, payload, args.out_dir / "extra3.md")
    map_path = write_maps(
        kind,
        tasks_out,
        extra={
            "pooling": packed["pooling"],
            "scope": args.scope,
            "n_pools": len(packed["pools"]),
        },
        map_dir=Path(args.map_dir),
    )
    logging.info("Wrote %s, %s, %s", args.out_dir / "extra3.json", args.out_dir / "extra3.md", map_path)
    logging.info("histogram=%s", summary["histogram"])
    print((args.out_dir / "extra3.md").read_text())


if __name__ == "__main__":
    main()

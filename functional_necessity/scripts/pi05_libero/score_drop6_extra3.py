#!/usr/bin/env python3
"""Conditional extra-3 on the v044 Drop-6 graph: Fisher or greedy PPL.

Locks U={1,13,14,15,16,17}. Scores the remaining 12 layers
{0,2,3,4,5,6,7,8,9,10,11,12} with skip(U) already on. L0/L7 may enter extra-3;
whether they are universal-critical is read from the histogram.
Per-task calib is 16×8; greedy PPL freezes flow-matching ξ.
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
    DEFAULT_GRAD_MICROBATCH,
    DEFAULT_SEED,
    FlowMatchingReplay,
    init_logging,
    load_calib_data,
    load_policy,
    paligemma_parts,
    prepare_policy_for_grad_scoring,
    set_seed,
)
from drop6_extra3_protocol import (
    CANDIDATES,
    DEFAULT_CALIB_DIR,
    DEFAULT_OUT_DIR,
    EXTRA_K,
    N_LAYERS,
    UNIVERSAL_DROP,
    calib_path,
    dump_json,
    extra3_from_fisher,
    load_task_catalog,
    protocol_dict,
    summarize_extra_map,
    wait_for_file,
    write_summary_md,
)
from layer_skip import skip_llm_layers
from profiling_fisher import measure_fisher
from profiling_ppl import average_loss


def eval_extra_loss(policy, extra, calib, device, microbatch_size, replay, cache: dict) -> float:
    key = tuple(sorted(int(x) for x in extra))
    if key in cache:
        return cache[key]
    drop = sorted(set(UNIVERSAL_DROP) | set(key))
    with skip_llm_layers(policy, drop):
        loss = average_loss(policy, calib, device, microbatch_size, replay=replay)
    cache[key] = float(loss)
    return cache[key]


def greedy_extra3(policy, calib, device, microbatch_size, replay) -> dict:
    cache: dict[tuple[int, ...], float] = {}
    baseline = eval_extra_loss(policy, [], calib, device, microbatch_size, replay, cache)
    remaining = list(CANDIDATES)
    chosen: list[int] = []
    steps = []
    for step in range(1, EXTRA_K + 1):
        ranked = []
        for layer in remaining:
            extras = chosen + [layer]
            t0 = time.time()
            loss = eval_extra_loss(policy, extras, calib, device, microbatch_size, replay, cache)
            ranked.append(
                {
                    "layer": layer,
                    "extra": sorted(extras),
                    "loss": loss,
                    "delta": loss - baseline,
                    "seconds": round(time.time() - t0, 3),
                }
            )
        ranked.sort(key=lambda r: (r["loss"], r["layer"]))
        best = ranked[0]
        picked = int(best["layer"])
        if picked in UNIVERSAL_DROP:
            raise RuntimeError(f"greedy picked universal-drop layer L{picked}")
        chosen.append(picked)
        remaining = [x for x in remaining if x != picked]
        steps.append(
            {
                "step": step,
                "picked": picked,
                "extra_after": sorted(chosen),
                "loss": best["loss"],
                "delta": best["delta"],
                "candidates": ranked,
            }
        )
        logging.info(
            "Greedy step %d/%d: pick L%d extra=%s loss=%.6f Δ=%.6f",
            step,
            EXTRA_K,
            picked,
            sorted(chosen),
            best["loss"],
            best["delta"],
        )
    extra = sorted(chosen)
    return {
        "extra_drop": extra,
        "keep_candidates": sorted(remaining),
        "union_drop": sorted(set(UNIVERSAL_DROP) | set(extra)),
        "baseline_loss": baseline,
        "loss": steps[-1]["loss"],
        "delta": steps[-1]["delta"],
        "matches_global_fisher_extra3": extra == [2, 3, 10],
        "symdiff_vs_global": sorted(set(extra) ^ {2, 3, 10}),
        "n_cache": len(cache),
        "steps": steps,
    }


def score_fisher_task(policy, calib, device, microbatch_size) -> dict:
    with skip_llm_layers(policy, UNIVERSAL_DROP):
        results = measure_fisher(policy, calib, device, microbatch_size)
    scores = [float(x) for x in results["paligemma"].detach().cpu().flatten().tolist()]
    if len(scores) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} Fisher scores, got {len(scores)}")
    extra = extra3_from_fisher(scores)
    extra.update(
        {
            "union_drop": sorted(set(UNIVERSAL_DROP) | set(extra["extra_drop"])),
            "all_layer_scores": scores,
            "n_orig_batches": int(results["n_orig_batches"]),
            "aggregation": results["aggregation"],
        }
    )
    return extra


def maybe_load_done(path: Path) -> dict | None:
    if path.is_file() and path.stat().st_size > 10:
        return json.loads(path.read_text())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("fisher", "greedy_ppl"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--microbatch-size", type=int, default=None)
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--wait-timeout", type=float, default=12 * 3600)
    args = parser.parse_args()
    if args.microbatch_size is None:
        args.microbatch_size = DEFAULT_GRAD_MICROBATCH if args.method == "fisher" else None

    init_logging()
    set_seed(args.seed)
    device = torch.device(args.device)
    catalog = load_task_catalog()
    out_dir = Path(args.out_dir) if args.out_dir is not None else DEFAULT_OUT_DIR / args.method
    per_dir = out_dir / "per_task"
    per_dir.mkdir(parents=True, exist_ok=True)
    dump_json(protocol_dict(), out_dir / "protocol.json")

    policy = load_policy(args.checkpoint, device)
    layers, _ = paligemma_parts(policy)
    if len(layers) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} PaliGemma layers, got {len(layers)}")
    if args.method == "fisher":
        prepare_policy_for_grad_scoring(policy, recompute=True)
    else:
        policy.eval()

    tasks_out: list[dict] = []
    for i, task in enumerate(catalog, start=1):
        task_path = per_dir / f"{task['short']}.json"
        done = maybe_load_done(task_path)
        if done is not None and "extra_drop" in done:
            logging.info("[skip] %s already scored extra=%s", task["short"], done["extra_drop"])
            tasks_out.append(done)
            continue

        calib_file = wait_for_file(calib_path(args.calib_dir, task["short"]), timeout_s=args.wait_timeout)
        calib = load_calib_data(calib_file)
        logging.info(
            "[%d/%d] %s %s %s  batches=%d",
            i,
            len(catalog),
            args.method,
            task["short"],
            task["task"],
            len(calib),
        )
        t0 = time.time()
        if args.method == "fisher":
            scored = score_fisher_task(policy, calib, device, args.microbatch_size)
        else:
            replay = FlowMatchingReplay(policy)
            replay.capture(calib, device, args.microbatch_size)
            with replay:
                scored = greedy_extra3(policy, calib, device, args.microbatch_size, replay)
        rec = {
            **task,
            **scored,
            "seconds": round(time.time() - t0, 3),
            "calib_file": str(calib_file),
            "n_calib_batches": len(calib),
        }
        dump_json(rec, task_path)
        tasks_out.append(rec)
        logging.info(
            "%s extra-3=%s union=%s (%.1fs)",
            task["short"],
            rec["extra_drop"],
            rec["union_drop"],
            rec["seconds"],
        )
        dump_json(
            {"method": args.method, "n_done": len(tasks_out), "n_total": len(catalog), "tasks": tasks_out},
            out_dir / "partial.json",
        )

    if len(tasks_out) != 40:
        raise RuntimeError(f"expected 40 tasks, got {len(tasks_out)}")
    summary = summarize_extra_map(tasks_out)
    payload = {
        "method": args.method,
        "checkpoint": args.checkpoint,
        "calib_dir": str(args.calib_dir),
        "protocol": protocol_dict(),
        "summary": summary,
        "tasks": tasks_out,
        "out_json": str(out_dir / "extra3.json"),
    }
    json_path = out_dir / "extra3.json"
    dump_json(payload, json_path)
    md_path = out_dir / "extra3.md"
    write_summary_md(args.method, payload, md_path)
    logging.info("Wrote %s and %s", json_path, md_path)
    logging.info(
        "histogram=%s match_global=%d/40 gate=%s",
        summary["histogram"],
        summary["n_match_global_fisher_extra3"],
        summary["should_run_closed_loop"],
    )
    print(md_path.read_text())


if __name__ == "__main__":
    main()

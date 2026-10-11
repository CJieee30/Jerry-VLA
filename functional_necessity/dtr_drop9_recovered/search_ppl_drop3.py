#!/usr/bin/env python3
"""Greedy PPL and exhaustive bypass-loss search for extra Drop-3.

Same protocol as profiling_ppl.py: Stage-1 drop stays on; extra skip via
`extra_llm_drop`; 64×8 calib; frozen (noise, time). Remaining layers only.

Greedy: 9+8+7 conditional single-layer bypasses.
Exhaustive: all C(9,3)=84 triples. Shared cache avoids double-evaluating
triples visited on the greedy path.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from itertools import combinations
from pathlib import Path

import torch
from tqdm import tqdm

from backend import (
    DEFAULT_CALIB_FILE,
    DEFAULT_CHECKPOINT,
    FN_DROP_K,
    REMAINING_LAYERS,
    STAGE1_DROP,
    FlowMatchingReplay,
    add_common_args,
    compute_loss,
    extra_llm_drop,
    init_logging,
    iter_calib_batches,
    load_calib_data,
    load_model,
    set_seed,
)

OUT_DIR = Path(__file__).resolve().parent / "results" / "ppl_subset_search"
INDEPENDENT_PPL_DROP3 = [2, 4, 9]
LOO_SR_CHEAPEST3 = [1, 2, 5]


def average_loss(model, calib_data, device, microbatch_size=None, replay=None) -> float:
    total = 0.0
    n = 0
    if replay is not None:
        replay.reset()
    with torch.no_grad():
        for batch in iter_calib_batches(calib_data, device, microbatch_size):
            kwargs = {}
            if replay is not None:
                noise, time_t = replay.current(device)
                kwargs["noise"] = noise
                kwargs["time"] = time_t
            total += float(compute_loss(model, batch, **kwargs).item())
            n += 1
            if replay is not None:
                replay.step()
    if replay is not None and replay.i != len(replay.noises):
        raise RuntimeError(f"PPL replay consumed {replay.i}/{len(replay.noises)} flow samples")
    return total / max(n, 1)


def eval_extra(model, extras, calib_data, device, microbatch_size, replay, cache: dict) -> float:
    key = tuple(sorted(int(x) for x in extras))
    if key in cache:
        return cache[key]
    ctx = extra_llm_drop(model, list(key)) if key else extra_llm_drop(model, [])
    with ctx:
        loss = average_loss(model, calib_data, device, microbatch_size, replay=replay)
    cache[key] = float(loss)
    return cache[key]


def dump_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def greedy_drop3(model, calib_data, device, microbatch_size, replay, cache, baseline: float) -> dict:
    remaining = list(REMAINING_LAYERS)
    chosen: list[int] = []
    steps = []
    for step in range(1, FN_DROP_K + 1):
        ranked = []
        for layer in remaining:
            extras = chosen + [layer]
            t0 = time.time()
            loss = eval_extra(model, extras, calib_data, device, microbatch_size, replay, cache)
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
        chosen.append(int(best["layer"]))
        remaining = [x for x in remaining if x != best["layer"]]
        steps.append(
            {
                "step": step,
                "picked": int(best["layer"]),
                "extra_after": sorted(chosen),
                "loss": best["loss"],
                "delta": best["delta"],
                "candidates": ranked,
            }
        )
        logging.info(
            "Greedy step %d: pick L%d  extra=%s  loss=%.6f  Δ=%.6f",
            step,
            best["layer"],
            sorted(chosen),
            best["loss"],
            best["delta"],
        )
    return {
        "extra_drop": sorted(chosen),
        "keep": sorted(remaining),
        "union_drop": sorted(STAGE1_DROP + chosen),
        "loss": steps[-1]["loss"],
        "delta": steps[-1]["delta"],
        "steps": steps,
    }


def exhaustive_drop3(model, calib_data, device, microbatch_size, replay, cache, baseline: float) -> dict:
    triples = list(combinations(REMAINING_LAYERS, FN_DROP_K))
    rows = []
    for comb in tqdm(triples, desc="Exhaustive Drop-3"):
        t0 = time.time()
        loss = eval_extra(model, comb, calib_data, device, microbatch_size, replay, cache)
        rows.append(
            {
                "extra": list(comb),
                "keep": sorted(set(REMAINING_LAYERS) - set(comb)),
                "union_drop": sorted(STAGE1_DROP + list(comb)),
                "loss": loss,
                "delta": loss - baseline,
                "seconds": round(time.time() - t0, 3),
                "cached": tuple(sorted(comb)) in cache and time.time() - t0 < 0.05,
            }
        )
    rows.sort(key=lambda r: (r["loss"], r["extra"]))
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    winner = rows[0]
    return {
        "n_triples": len(rows),
        "winner": {
            "extra_drop": winner["extra"],
            "keep": winner["keep"],
            "union_drop": winner["union_drop"],
            "loss": winner["loss"],
            "delta": winner["delta"],
            "rank": 1,
        },
        "rows": rows,
    }


def lookup_triple(rows: list[dict], extra: list[int]) -> dict | None:
    key = sorted(extra)
    for row in rows:
        if row["extra"] == key:
            return row
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    init_logging()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    model.eval()
    calib = load_calib_data(args.calib_file)
    replay = FlowMatchingReplay(model)
    replay.capture(calib, device, args.microbatch_size)

    cache: dict[tuple[int, ...], float] = {}
    logging.info("Baseline (Stage-1 drop only, frozen ξ)...")
    t0 = time.time()
    baseline = eval_extra(model, [], calib, device, args.microbatch_size, replay, cache)
    logging.info("Baseline loss: %.6f  (%.1fs)", baseline, time.time() - t0)

    logging.info("Greedy PPL extra Drop-%d on remaining %s", FN_DROP_K, REMAINING_LAYERS)
    greedy = greedy_drop3(model, calib, device, args.microbatch_size, replay, cache, baseline)
    dump_checkpoint(
        {
            "baseline_loss": baseline,
            "greedy": greedy,
            "n_cache": len(cache),
        },
        out_dir / "drop3_search_greedy_partial.json",
    )

    logging.info("Exhaustive C(%d,%d)=%d triples", len(REMAINING_LAYERS), FN_DROP_K, 84)
    exhaustive = exhaustive_drop3(model, calib, device, args.microbatch_size, replay, cache, baseline)

    ind = lookup_triple(exhaustive["rows"], INDEPENDENT_PPL_DROP3)
    loo = lookup_triple(exhaustive["rows"], LOO_SR_CHEAPEST3)
    greedy_row = lookup_triple(exhaustive["rows"], greedy["extra_drop"])

    payload = {
        "checkpoint": str(args.checkpoint),
        "calib_file": str(args.calib_file),
        "protocol": {
            "n_calib_batches": len(calib),
            "seed": args.seed,
            "paired_flow_matching": True,
            "n_flow_pairs": len(replay.noises),
            "k": FN_DROP_K,
            "stage1_drop": list(STAGE1_DROP),
            "remaining_layers": list(REMAINING_LAYERS),
            "skip": "extra_llm_drop on joint graph; keep prefix K/V",
        },
        "baseline_loss": baseline,
        "n_cache": len(cache),
        "greedy": greedy,
        "exhaustive": {
            "n_triples": exhaustive["n_triples"],
            "winner": exhaustive["winner"],
            "greedy_matches_winner": greedy["extra_drop"] == exhaustive["winner"]["extra_drop"],
            "independent_ppl_[2,4,9]": ind,
            "loo_sr_cheapest_[1,2,5]": loo,
            "greedy_triple": greedy_row,
            "top10": exhaustive["rows"][:10],
            "bottom5": exhaustive["rows"][-5:],
            "rows": exhaustive["rows"],
        },
    }
    json_path = out_dir / "drop3_search.json"
    dump_checkpoint(payload, json_path)

    md = [
        "# Extra Drop-3 PPL subset search (Drop-9 recovered π0.5)",
        "",
        "Protocol: same as `profiling_ppl.py` (64×8 calib, frozen noise/time, Stage-1 drop on).",
        f"Baseline loss: **{baseline:.6f}**.",
        "",
        "## Greedy PPL",
        "",
        f"- Extra drop: `{greedy['extra_drop']}`",
        f"- Keep: `{greedy['keep']}`",
        f"- Loss: {greedy['loss']:.6f} (Δ {greedy['delta']:+.6f})",
        "",
    ]
    for step in greedy["steps"]:
        md.append(
            f"- Step {step['step']}: pick L{step['picked']} → extra `{step['extra_after']}` "
            f"loss={step['loss']:.6f}"
        )
    md += [
        "",
        "## Exhaustive (84 triples)",
        "",
        f"- Winner: `{exhaustive['winner']['extra_drop']}` loss={exhaustive['winner']['loss']:.6f} "
        f"(Δ {exhaustive['winner']['delta']:+.6f})",
        f"- Greedy matches winner: **{payload['exhaustive']['greedy_matches_winner']}**",
        "",
        "| Rank | Extra | Loss | Δ vs baseline |",
        "| ---: | --- | ---: | ---: |",
    ]
    for row in exhaustive["rows"][:15]:
        md.append(
            f"| {row['rank']} | {row['extra']} | {row['loss']:.6f} | {row['delta']:+.6f} |"
        )
    md += ["", "Reference triples:", ""]
    for name, row in [
        ("Independent PPL / Fisher `[2,4,9]`", ind),
        ("LOO-SR cheapest `[1,2,5]`", loo),
        ("Greedy extra", greedy_row),
    ]:
        if row is None:
            md.append(f"- {name}: not found")
        else:
            md.append(
                f"- {name}: rank {row['rank']}/84, loss={row['loss']:.6f}, Δ={row['delta']:+.6f}"
            )
    md += [
        "",
        f"Full table: `{json_path}`",
        "",
    ]
    md_path = out_dir / "drop3_search.md"
    md_path.write_text("\n".join(md))
    logging.info("Wrote %s and %s", json_path, md_path)
    print("\n".join(md))


if __name__ == "__main__":
    main()

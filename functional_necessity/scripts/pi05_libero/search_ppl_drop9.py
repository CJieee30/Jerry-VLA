#!/usr/bin/env python3
"""Greedy PPL subset search for Drop-9 on lerobot/pi05_libero_finetuned_v044.

Same protocol as profiling_ppl.py: 64×8 calib, frozen (noise, time), DTR
Appendix B skip (keep prefix K/V). Does not enumerate C(18,9).

Greedy: at each step bypass the remaining layer whose drop increases calib
loss the least, then condition the next step on that drop. k=9 →
18+17+…+10 = 126 subset evals plus one full-model baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from common import (
    DEFAULT_CALIB_FILE,
    DEFAULT_CHECKPOINT,
    FlowMatchingReplay,
    add_common_args,
    init_logging,
    load_calib_data,
    load_policy,
    paligemma_parts,
    set_seed,
)
from layer_skip import skip_llm_layers
from profiling_ppl import average_loss

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "ppl_subset_search"
INDEPENDENT_PPL_DROP9 = [2, 4, 7, 9, 10, 14, 15, 16, 17]
N_LAYERS = 18
DROP_K = 9


def dump_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def eval_drop(policy, extras, calib, device, microbatch_size, replay, cache: dict) -> float:
    key = tuple(sorted(int(x) for x in extras))
    if key in cache:
        return cache[key]
    with skip_llm_layers(policy, list(key)):
        loss = average_loss(policy, calib, device, microbatch_size, replay=replay)
    cache[key] = float(loss)
    return cache[key]


def greedy_drop_k(policy, calib, device, microbatch_size, replay, cache, baseline: float, k: int) -> dict:
    remaining = list(range(N_LAYERS))
    chosen: list[int] = []
    steps = []
    for step in range(1, k + 1):
        ranked = []
        for layer in remaining:
            extras = chosen + [layer]
            t0 = time.time()
            loss = eval_drop(policy, extras, calib, device, microbatch_size, replay, cache)
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
        step_rec = {
            "step": step,
            "picked": int(best["layer"]),
            "drop_after": sorted(chosen),
            "keep_after": sorted(remaining),
            "loss": best["loss"],
            "delta": best["delta"],
            "candidates": ranked,
        }
        steps.append(step_rec)
        logging.info(
            "Greedy step %d/%d: pick L%d  drop=%s  loss=%.6f  Δ=%.6f",
            step,
            k,
            best["layer"],
            sorted(chosen),
            best["loss"],
            best["delta"],
        )
        dump_json(
            {
                "baseline_loss": baseline,
                "step": step,
                "drop_after": sorted(chosen),
                "keep_after": sorted(remaining),
                "loss": best["loss"],
                "delta": best["delta"],
            },
            OUT_DIR / "v044_drop9_greedy_partial.json",
        )
    return {
        "drop": sorted(chosen),
        "keep": sorted(remaining),
        "loss": steps[-1]["loss"],
        "delta": steps[-1]["delta"],
        "matches_independent_ppl": sorted(chosen) == sorted(INDEPENDENT_PPL_DROP9),
        "steps": steps,
    }


def write_markdown(payload: dict, path: Path) -> None:
    greedy = payload["greedy"]
    baseline = payload["baseline_loss"]
    lines = [
        "# v044 Drop-9 greedy PPL search",
        "",
        "Checkpoint: `lerobot/pi05_libero_finetuned_v044`. Protocol matches `profiling_ppl.py` (64×8, frozen ξ, Appendix B skip).",
        f"Baseline loss: **{baseline:.6f}**.",
        "",
        f"- Greedy Drop-9: `{greedy['drop']}`",
        f"- Keep: `{greedy['keep']}`",
        f"- Loss: {greedy['loss']:.6f} (Δ {greedy['delta']:+.6f})",
        f"- Matches independent PPL `{INDEPENDENT_PPL_DROP9}`: **{greedy['matches_independent_ppl']}**",
        "",
    ]
    for step in greedy["steps"]:
        lines.append(
            f"- Step {step['step']}: pick L{step['picked']} → drop `{step['drop_after']}` "
            f"loss={step['loss']:.6f}"
        )
    lines += ["", f"JSON: `{payload['out_json']}`", ""]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--k", type=int, default=DROP_K)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    init_logging()
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = load_policy(args.checkpoint, device)
    policy.eval()
    layers, _ = paligemma_parts(policy)
    if len(layers) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} PaliGemma layers, got {len(layers)}")
    calib = load_calib_data(args.calib_file)
    replay = FlowMatchingReplay(policy)
    replay.capture(calib, device, args.microbatch_size)

    cache: dict[tuple[int, ...], float] = {}
    logging.info("Baseline (full 18 layers, frozen ξ)...")
    t0 = time.time()
    with replay:
        baseline = eval_drop(policy, [], calib, device, args.microbatch_size, replay, cache)
        logging.info("Baseline loss: %.6f  (%.1fs)", baseline, time.time() - t0)
        greedy = greedy_drop_k(
            policy, calib, device, args.microbatch_size, replay, cache, baseline, args.k
        )

    payload = {
        "checkpoint": args.checkpoint,
        "calib_file": str(args.calib_file),
        "protocol": {
            "n_calib_batches": len(calib),
            "seed": args.seed,
            "paired_flow_matching": True,
            "n_flow_pairs": len(replay.noises),
            "k": args.k,
            "n_layers": N_LAYERS,
            "skip": "skip_llm_layers / DTR Appendix B",
        },
        "baseline_loss": baseline,
        "n_cache": len(cache),
        "independent_ppl_drop9": INDEPENDENT_PPL_DROP9,
        "greedy": greedy,
        "out_json": str(out_dir / "v044_drop9_greedy.json"),
    }
    json_path = out_dir / "v044_drop9_greedy.json"
    dump_json(payload, json_path)
    md_path = out_dir / "v044_drop9_greedy.md"
    write_markdown(payload, md_path)
    logging.info("Wrote %s and %s", json_path, md_path)
    print(md_path.read_text())


if __name__ == "__main__":
    main()

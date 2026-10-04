#!/usr/bin/env python3
"""Greedy PPL Drop-8 for the joint OpenVLA-OFT LIBERO checkpoint.

Same selection rule as pi05_libero/search_ppl_drop9.py: at each step, add the
remaining Llama layer whose removal increases action L1 the least, then
condition the next step on that set. k=8 over 32 layers is
32+31+...+25 = 228 subset evaluations plus the full-model baseline.

Loss is the finetune L1. A dropped layer is an identity block. There is no
flow-matching noise to freeze. Calibration is the existing 512-sample set.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from layer_skip import skip_llama_layers
from load_policy import cap_cuda_memory, llama_parts, load_policy, prepare_for_forward
from paths import CALIB_PATH, FN_ROOT
from score import _mean_loss

N_LAYERS = 32
DROP_K = 8
INDEPENDENT_PPL_DROP8 = [2, 3, 4, 5, 6, 23, 26, 29]
OUT_DIR = FN_ROOT / "results" / "openvla_oft_libero" / "ppl_subset_search"


def dump_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def cache_key(layers: list[int]) -> str:
    return ",".join(str(i) for i in sorted(int(x) for x in layers))


def eval_drop(vla, action_head, proprio, samples, device, layers: list[int], cache: dict) -> float:
    key = cache_key(layers)
    if key in cache:
        return float(cache[key])
    with skip_llama_layers(layers):
        loss = _mean_loss(vla, action_head, proprio, samples, device)
    cache[key] = float(loss)
    return cache[key]


def load_state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def main() -> None:
    cap_cuda_memory()
    device = torch.device("cuda:0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    partial_path = OUT_DIR / "oft_drop8_greedy_partial.json"
    state = load_state(partial_path) or {}
    cache = dict(state.get("cache") or {})
    chosen = [int(x) for x in state.get("chosen") or []]
    steps = list(state.get("steps") or [])

    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    samples = blob["samples"]
    print(f"calib n={len(samples)} seed={blob.get('seed')}", flush=True)
    print("loading policy", flush=True)
    vla, action_head, proprio = load_policy("cuda:0")
    prepare_for_forward(vla, action_head, proprio)
    layers, _ = llama_parts(vla)
    if len(layers) != N_LAYERS:
        raise SystemExit(f"expected {N_LAYERS} Llama layers, got {len(layers)}")

    if "baseline_loss" in state:
        baseline = float(state["baseline_loss"])
        print(f"resume baseline {baseline:.6f} chosen={chosen}", flush=True)
    else:
        print(f"baseline over {len(samples)} samples", flush=True)
        t0 = time.time()
        baseline = eval_drop(vla, action_head, proprio, samples, device, [], cache)
        print(f"baseline L1 {baseline:.6f} ({time.time() - t0:.1f}s)", flush=True)
        state = {"baseline_loss": baseline, "chosen": [], "steps": [], "cache": cache}
        dump_json(state, partial_path)

    remaining = [i for i in range(N_LAYERS) if i not in chosen]
    for step in range(len(chosen) + 1, DROP_K + 1):
        ranked = []
        for layer in remaining:
            extras = chosen + [layer]
            t0 = time.time()
            loss = eval_drop(vla, action_head, proprio, samples, device, extras, cache)
            elapsed = round(time.time() - t0, 3)
            row = {
                "layer": layer,
                "extra": sorted(extras),
                "loss": loss,
                "delta": loss - baseline,
                "seconds": elapsed,
            }
            ranked.append(row)
            print(
                f"  step {step}/{DROP_K} candidate L{layer} loss={loss:.6f} "
                f"delta={loss - baseline:+.6f} {elapsed:.1f}s",
                flush=True,
            )
            state["cache"] = cache
            dump_json(state, partial_path)
        ranked.sort(key=lambda r: (r["loss"], r["layer"]))
        best = ranked[0]
        chosen.append(int(best["layer"]))
        remaining = [i for i in remaining if i != best["layer"]]
        step_rec = {
            "step": step,
            "picked": int(best["layer"]),
            "drop_after": sorted(chosen),
            "keep_after": remaining,
            "loss": best["loss"],
            "delta": best["delta"],
            "candidates": ranked,
        }
        steps.append(step_rec)
        state.update({"chosen": chosen, "steps": steps, "cache": cache})
        dump_json(state, partial_path)
        print(
            f"Greedy step {step}/{DROP_K}: pick L{best['layer']} drop={sorted(chosen)} "
            f"loss={best['loss']:.6f} delta={best['delta']:+.6f}",
            flush=True,
        )

    payload = {
        "checkpoint": "openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
        "calib": str(CALIB_PATH),
        "n_samples": len(samples),
        "protocol": {
            "loss": "l1_action_head",
            "skip": "llama_decoder_identity",
            "selection": "greedy_min_conditional_delta",
            "tie_break": "lower layer index",
            "k": DROP_K,
            "n_layers": N_LAYERS,
        },
        "baseline_loss": baseline,
        "independent_ppl_drop8": INDEPENDENT_PPL_DROP8,
        "greedy": {
            "drop": sorted(chosen),
            "pick_order": chosen,
            "keep": remaining,
            "loss": steps[-1]["loss"],
            "delta": steps[-1]["delta"],
            "matches_independent_ppl": sorted(chosen) == sorted(INDEPENDENT_PPL_DROP8),
            "steps": steps,
        },
    }
    out_json = OUT_DIR / "oft_drop8_greedy.json"
    dump_json(payload, out_json)
    lines = [
        "# OpenVLA-OFT Drop-8 greedy PPL",
        "",
        "Same rule as pi0.5 `search_ppl_drop9.py`: each step adds the remaining layer with the smallest action-L1 increase.",
        f"Baseline L1: **{baseline:.6f}** over {len(samples)} calibration samples.",
        "",
        f"- Greedy Drop-8: `{sorted(chosen)}`",
        f"- Pick order: `{chosen}`",
        f"- Keep: `{remaining}`",
        f"- Loss: {steps[-1]['loss']:.6f} (Δ {steps[-1]['delta']:+.6f})",
        f"- Independent PPL Drop-8: `{INDEPENDENT_PPL_DROP8}`",
        f"- Same set: **{sorted(chosen) == sorted(INDEPENDENT_PPL_DROP8)}**",
        "",
    ]
    for step in steps:
        lines.append(
            f"- Step {step['step']}: pick L{step['picked']} → `{step['drop_after']}` "
            f"loss={step['loss']:.6f} Δ={step['delta']:+.6f}"
        )
    lines.append("")
    (OUT_DIR / "oft_drop8_greedy.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()

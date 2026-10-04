#!/usr/bin/env python3
"""Measure params / predict_action_chunk FLOPs / CUDA time: baseline vs Drop-k skip.

Uses the official cache path (prefix language forward + expert denoise). Does not
include simulator or video. Drop-k sets come from ranks.json; layer widths are
uniform so all Drop-6 methods have the same analytical prefix saving.

Params: stored counts every weight (Appendix B skip does not delete tensors).
Active subtracts unused language Q / O / MLP / post-attn LN on dropped layers;
K/V and input LN stay, matching the skip operator.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_CALIB_FILE,
    DEFAULT_CHECKPOINT,
    DEFAULT_RANKS_DIR,
    FN_ROOT,
    init_logging,
    load_calib_data,
    load_policy,
    move_to_device,
    slice_batch,
)
from eval_drop import drop_from_ranks, parse_layer_list  # noqa: E402
from layer_skip import _paligemma_layers, skip_llm_layers  # noqa: E402

LOGGER = logging.getLogger("measure_efficiency")
OUT_DIR = FN_ROOT / "results" / "eval_drop"


def _proj(seq: int, din: int, dout: int) -> int:
    return 2 * seq * din * dout


def _attn(q_heads: int, seq_q: int, seq_k: int, head_dim: int) -> int:
    return 2 * 2 * q_heads * seq_q * seq_k * head_dim


def _mlp_swiglu(seq: int, hidden: int, intermediate: int) -> int:
    return 2 * seq * hidden * intermediate * 3


def analytical_chunk_flops(*, llm_seq: int, expert_seq: int, n_drop: int, n_denoise: int) -> dict:
    """Official cache-path FLOPs: 1x language prefix (self-attn) + N expert denoise steps.

    Dropped language layer keeps K/V (Appendix B); Q/O/attn/MLP are skipped.
    Expert denoise is unchanged: cached language K/V stay the same shape.
    """
    llm_h, llm_i, llm_q, llm_kv, llm_hd = 2048, 16384, 8, 1, 256
    exp_h, exp_i, exp_q, exp_kv, exp_hd = 1024, 4096, 8, 1, 256
    n_layers = 18

    llm_qkv_o = (
        _proj(llm_seq, llm_h, llm_q * llm_hd)
        + _proj(llm_seq, llm_h, llm_kv * llm_hd)
        + _proj(llm_seq, llm_h, llm_kv * llm_hd)
        + _proj(llm_seq, llm_h, llm_q * llm_hd)
    )
    llm_full = llm_qkv_o + _attn(llm_q, llm_seq, llm_seq, llm_hd) + _mlp_swiglu(llm_seq, llm_h, llm_i)
    llm_dropped = _proj(llm_seq, llm_h, llm_kv * llm_hd) * 2

    exp_total_k = llm_seq + expert_seq
    exp_layer = (
        _proj(expert_seq, exp_h, exp_q * exp_hd)
        + _proj(expert_seq, exp_h, exp_kv * exp_hd)
        + _proj(expert_seq, exp_h, exp_kv * exp_hd)
        + _proj(expert_seq, exp_h, exp_q * exp_hd)
        + _attn(exp_q, expert_seq, exp_total_k, exp_hd)
        + _mlp_swiglu(expert_seq, exp_h, exp_i)
    )

    n_keep = n_layers - n_drop
    prefix = n_keep * llm_full + n_drop * llm_dropped
    denoise = n_denoise * n_layers * exp_layer
    vision = int(656.9e9)
    other = int(3.6e9)
    total = prefix + denoise + vision + other
    return {
        "llm_seq": llm_seq,
        "n_drop": n_drop,
        "n_denoise": n_denoise,
        "prefix_language_gflops": prefix / 1e9,
        "denoise_expert_gflops": denoise / 1e9,
        "vision_gflops": vision / 1e9,
        "other_gflops": other / 1e9,
        "total_gflops": total / 1e9,
        "dropped_layer_full_gflops": llm_full / 1e9,
        "dropped_layer_kept_kv_gflops": llm_dropped / 1e9,
    }


def _nparams(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def count_params(policy, drop: list[int]) -> dict:
    """Stored vs active params under Appendix B skip (keep K/V, drop Q/O/MLP)."""
    stored = _nparams(policy)
    layers, _ = _paligemma_layers(policy)
    language = _nparams(policy.model._paligemma_language_model_core())
    skipped = 0
    per_layer = []
    for idx in drop:
        layer = layers[idx]
        parts = {
            "q_proj": layer.self_attn.q_proj,
            "o_proj": layer.self_attn.o_proj,
            "mlp": layer.mlp,
            "post_attention_layernorm": layer.post_attention_layernorm,
        }
        used = {
            "input_layernorm": _nparams(layer.input_layernorm),
            "k_proj": _nparams(layer.self_attn.k_proj),
            "v_proj": _nparams(layer.self_attn.v_proj),
        }
        unused = {name: _nparams(mod) for name, mod in parts.items()}
        layer_skip = int(sum(unused.values()))
        skipped += layer_skip
        per_layer.append(
            {
                "layer": int(idx),
                "unused_numel": layer_skip,
                "unused": unused,
                "kept_kv_ln_numel": int(sum(used.values())),
            }
        )
    return {
        "stored_params": stored,
        "active_params": stored - skipped,
        "skipped_language_compute_params": skipped,
        "language_stored_params": language,
        "language_active_params": language - skipped,
        "per_dropped_layer": per_layer,
    }


def _chunk(policy, batch, drop: list[int]) -> torch.Tensor:
    policy.reset()
    ctx = skip_llm_layers(policy, drop) if drop else nullcontext()
    with torch.no_grad(), ctx:
        return policy.predict_action_chunk(batch)


def measure_flops(policy, batch, drop: list[int]) -> int:
    _chunk(policy, batch, drop)
    with FlopCounterMode(display=False) as mode:
        _chunk(policy, batch, drop)
    return int(mode.get_total_flops())


def measure_cuda_ms(policy, batch, drop: list[int], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        _chunk(policy, batch, drop)
    if batch[next(iter(batch))].is_cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            _chunk(policy, batch, drop)
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end) / repeats)
    t0 = time.perf_counter()
    for _ in range(repeats):
        _chunk(policy, batch, drop)
    return (time.perf_counter() - t0) * 1000.0 / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calib-file", type=Path, default=DEFAULT_CALIB_FILE)
    parser.add_argument("--from-ranks", type=Path, default=DEFAULT_RANKS_DIR / "ranks.json")
    parser.add_argument("--method", default="fisher")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--drop-layers", default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "efficiency_fisher_k6.json")
    args = parser.parse_args()
    init_logging()

    drop = parse_layer_list(args.drop_layers)
    if not drop:
        drop = drop_from_ranks(args.from_ranks, args.method, args.k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(args.checkpoint, device)
    policy.config.n_action_steps = 10
    calib = load_calib_data(args.calib_file)
    batch = move_to_device(slice_batch(calib[0], 0, 1), device)

    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS

    llm_seq = int(batch[OBS_LANGUAGE_TOKENS].shape[1]) if OBS_LANGUAGE_TOKENS in batch else 968
    n_denoise = int(policy.config.num_inference_steps)
    expert_seq = int(policy.config.chunk_size)

    rows = []
    for name, layers in (("baseline", []), (f"{args.method}_drop{args.k}", drop)):
        logging.info("Measuring %s drop=%s", name, layers)
        params = count_params(policy, layers)
        flops = measure_flops(policy, batch, layers)
        ms = measure_cuda_ms(policy, batch, layers, args.warmup, args.repeats)
        ana = analytical_chunk_flops(
            llm_seq=llm_seq, expert_seq=expert_seq, n_drop=len(layers), n_denoise=n_denoise
        )
        rows.append(
            {
                "name": name,
                "drop_layers": layers,
                "n_drop": len(layers),
                "params_stored_m": params["stored_params"] / 1e6,
                "params_active_m": params["active_params"] / 1e6,
                "params_language_stored_m": params["language_stored_params"] / 1e6,
                "params_language_active_m": params["language_active_params"] / 1e6,
                "params": params,
                "measured_gflops": flops / 1e9,
                "predict_action_chunk_ms": ms,
                "analytical": ana,
            }
        )
        logging.info(
            "%s stored=%.2fM active=%.2fM  measured=%.2f GFLOPs  wall=%.2f ms/chunk  analytical=%.2f GFLOPs",
            name,
            params["stored_params"] / 1e6,
            params["active_params"] / 1e6,
            flops / 1e9,
            ms,
            ana["total_gflops"],
        )

    base, skip = rows[0], rows[1]
    flop_ratio = skip["measured_gflops"] / base["measured_gflops"]
    time_ratio = skip["predict_action_chunk_ms"] / base["predict_action_chunk_ms"]
    stored_ratio = skip["params_stored_m"] / base["params_stored_m"]
    active_ratio = skip["params_active_m"] / base["params_active_m"]
    summary = {
        "checkpoint": args.checkpoint,
        "path": "official_prefix_cache_plus_expert_denoise",
        "skip": "dtr_appendix_b_prefix_kv",
        "llm_seq": llm_seq,
        "num_inference_steps": n_denoise,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "baseline": base,
        "skipped": skip,
        "measured_flops_ratio": flop_ratio,
        "measured_time_ratio": time_ratio,
        "measured_params_stored_ratio": stored_ratio,
        "measured_params_active_ratio": active_ratio,
        "measured_flops_speedup": 1.0 / flop_ratio,
        "measured_time_speedup": 1.0 / time_ratio,
        "delta_vs_baseline": {
            "params_stored_m": skip["params_stored_m"] - base["params_stored_m"],
            "params_active_m": skip["params_active_m"] - base["params_active_m"],
            "params_stored_pct": 100.0 * (stored_ratio - 1.0),
            "params_active_pct": 100.0 * (active_ratio - 1.0),
            "predict_action_chunk_ms": skip["predict_action_chunk_ms"] - base["predict_action_chunk_ms"],
            "predict_action_chunk_pct": 100.0 * (time_ratio - 1.0),
            "measured_gflops": skip["measured_gflops"] - base["measured_gflops"],
            "measured_flops_pct": 100.0 * (flop_ratio - 1.0),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    d = summary["delta_vs_baseline"]
    print(
        f"\nDrop-{args.k} ({args.method}) vs baseline\n"
        f"  params stored: {base['params_stored_m']:.2f} -> {skip['params_stored_m']:.2f} M  "
        f"Δ={d['params_stored_m']:+.2f} ({d['params_stored_pct']:+.2f}%)\n"
        f"  params active: {base['params_active_m']:.2f} -> {skip['params_active_m']:.2f} M  "
        f"Δ={d['params_active_m']:+.2f} ({d['params_active_pct']:+.2f}%)\n"
        f"  measured FLOPs: {base['measured_gflops']:.1f} -> {skip['measured_gflops']:.1f} GFLOPs  "
        f"Δ={d['measured_gflops']:+.1f} ({d['measured_flops_pct']:+.2f}%)  "
        f"speedup={1.0 / flop_ratio:.3f}x\n"
        f"  predict_action_chunk: {base['predict_action_chunk_ms']:.2f} -> "
        f"{skip['predict_action_chunk_ms']:.2f} ms  "
        f"Δ={d['predict_action_chunk_ms']:+.2f} ({d['predict_action_chunk_pct']:+.2f}%)  "
        f"speedup={1.0 / time_ratio:.3f}x\n"
        f"  analytical (prefix LLM / denoise expert / vision): "
        f"{skip['analytical']['prefix_language_gflops']:.0f} / "
        f"{skip['analytical']['denoise_expert_gflops']:.0f} / "
        f"{skip['analytical']['vision_gflops']:.0f} GFLOPs\n"
        f"wrote {args.out}"
    )
    if flop_ratio > 0.98 and time_ratio > 0.98:
        raise SystemExit(
            "Drop-k did not reduce FLOPs or wall time; skip may not be on the inference path."
        )


if __name__ == "__main__":
    main()

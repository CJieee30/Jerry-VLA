#!/usr/bin/env python3
"""Offline NFE check and efficiency for π0.5 and MolmoAct2.

Same observation and the same inference noise. NFE=10 run twice must match.
NFE=1 must move the action. Timing matches the existing module protocol:
CUDA events, warmup=5, repeats=20, compile off, one calibration frame.
Does not overwrite results/eval_drop or results/molmoact2_libero efficiency files.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
OUT = FN / "results" / "stage_nfe"
WARMUP = 5
REPEATS = 20
EXEC = 10
PI05_DROP = [1, 13, 14, 15, 16, 17]
MOLMO_DROP = [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]


def cuda_ms(fn, repeats: int = REPEATS) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


def flops(fn) -> float:
    fn()
    with FlopCounterMode(display=False) as mode:
        fn()
    return int(mode.get_total_flops()) / 1e9


def action_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    x = a[:, :EXEC, :7].float()
    y = b[:, :EXEC, :7].float()
    diff = x - y
    return {
        "max_abs": float(diff.abs().max()),
        "l2": float(diff.flatten(1).norm(dim=-1).mean()),
        "ref_l2": float(y.flatten(1).norm(dim=-1).mean()),
    }


def pi05(device: torch.device) -> dict:
    sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
    from common import DEFAULT_CALIB_FILE, DEFAULT_CHECKPOINT, load_calib_data, load_policy, move_to_device, slice_batch
    from layer_skip import skip_llm_layers
    from measure_module_efficiency import denoise_all, prefix_cache, prepare_inputs

    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.n_action_steps = EXEC
    policy.config.compile_model = False
    policy.eval()
    calib = load_calib_data(DEFAULT_CALIB_FILE)
    batch = move_to_device(slice_batch(calib[0], 0, 1), device)
    model = policy.model
    images, img_masks, tokens, masks, states, state_masks = prepare_inputs(policy, batch)
    bsize = int(tokens.shape[0])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1000)
    noise = torch.randn(
        bsize, int(model.config.chunk_size), int(model.config.max_action_dim), generator=gen
    ).to(device)

    def chunk(steps: int, drop: list[int]) -> torch.Tensor:
        ctx = skip_llm_layers(policy, drop) if drop else nullcontext()
        with torch.no_grad(), ctx:
            policy.reset()
            return policy.predict_action_chunk(batch, noise=noise, num_steps=steps).detach()

    a10 = chunk(10, [])
    a10b = chunk(10, [])
    a1 = chunk(1, [])
    d10 = chunk(10, PI05_DROP)
    d10b = chunk(10, PI05_DROP)
    d1 = chunk(1, PI05_DROP)
    offline = {
        "full_nfe10_replay": action_stats(a10b, a10),
        "full_nfe1_vs_nfe10": action_stats(a1, a10),
        "drop6_nfe10_replay": action_stats(d10b, d10),
        "drop6_nfe1_vs_nfe10": action_stats(d1, d10),
        "drop6_nfe10_vs_full_nfe10": action_stats(d10, a10),
    }
    print("pi05 offline", json.dumps(offline), flush=True)

    with torch.no_grad():
        prefix_embs, prefix_pad, past = prefix_cache(
            model, images, img_masks, tokens, masks, states, state_masks
        )
        del prefix_embs

    def denoise(steps: int):
        denoise_all(model, prefix_pad, past, noise.clone(), steps)

    def full(steps: int, drop: list[int]):
        ctx = skip_llm_layers(policy, drop) if drop else nullcontext()
        with ctx:
            policy.reset()
            policy.predict_action_chunk(batch, num_steps=steps)

    with torch.no_grad():
        for _ in range(WARMUP):
            full(10, [])
        timing = {
            "full_nfe10": {"ms": cuda_ms(lambda: full(10, [])), "flops_g": flops(lambda: full(10, []))},
            "full_nfe1": {"ms": cuda_ms(lambda: full(1, [])), "flops_g": flops(lambda: full(1, []))},
            "drop6_nfe10": {"ms": cuda_ms(lambda: full(10, PI05_DROP)), "flops_g": flops(lambda: full(10, PI05_DROP))},
            "drop6_nfe1": {"ms": cuda_ms(lambda: full(1, PI05_DROP)), "flops_g": flops(lambda: full(1, PI05_DROP))},
            "expert_nfe10": {"ms": cuda_ms(lambda: denoise(10)), "flops_g": flops(lambda: denoise(10))},
            "expert_nfe1": {"ms": cuda_ms(lambda: denoise(1)), "flops_g": flops(lambda: denoise(1))},
        }
    return {
        "stack": "pi05",
        "device": torch.cuda.get_device_name(0),
        "checkpoint": DEFAULT_CHECKPOINT,
        "drop6": PI05_DROP,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "offline": offline,
        "timing": timing,
    }


def molmo(device: torch.device) -> dict:
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    from layer_skip import skip_vlm_layers
    from load_policy import CALIB_PATH, LEROBOT_CKPT, load_policy

    policy = load_policy(str(device))
    policy.eval()
    policy.config.compile_model = False
    policy.config.enable_inference_cuda_graph = False
    policy.config.gradient_checkpointing = False
    policy.config.n_action_steps = EXEC
    backbone = policy._backbone()
    backbone.gradient_checkpointing = False
    if getattr(backbone, "action_cuda_graph_manager", None) is not None:
        backbone.action_cuda_graph_manager = None

    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    raw = blob["samples"][0]["batch"]
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}

    def chunk(steps: int, drop: list[int]) -> torch.Tensor:
        gen = torch.Generator(device=device)
        gen.manual_seed(1000)
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with torch.no_grad(), policy._autocast_context(), ctx:
            policy.reset()
            return policy.predict_action_chunk(batch, num_steps=steps, generator=gen).detach()

    a10 = chunk(10, [])
    a10b = chunk(10, [])
    a1 = chunk(1, [])
    d10 = chunk(10, MOLMO_DROP)
    d10b = chunk(10, MOLMO_DROP)
    d1 = chunk(1, MOLMO_DROP)
    offline = {
        "full_nfe10_replay": action_stats(a10b, a10),
        "full_nfe1_vs_nfe10": action_stats(a1, a10),
        "drop16_nfe10_replay": action_stats(d10b, d10),
        "drop16_nfe1_vs_nfe10": action_stats(d1, d10),
        "drop16_nfe10_vs_full_nfe10": action_stats(d10, a10),
    }
    print("molmo offline", json.dumps(offline), flush=True)

    model_inputs = policy._model_inputs(batch)
    with torch.no_grad(), policy._autocast_context():
        outputs = backbone(**model_inputs, use_cache=True, output_attentions=False, output_hidden_states=False)
        encoder_kv = backbone._extract_kv_states(outputs.past_key_values)
        encoder_attn = policy._encoder_attention_mask_for_action_expert(
            input_ids=model_inputs.get("input_ids"),
            attention_mask=model_inputs.get("attention_mask"),
        )
        depth_gate, depth_mask = backbone._depth_gate_from_condition(
            input_ids=model_inputs.get("input_ids"),
            encoder_attention_mask=encoder_attn,
            layer_kv_states=encoder_kv,
        )
        encoder_kv = backbone._apply_depth_gate_to_layer_kv_states(encoder_kv, depth_mask, depth_gate)
        action_expert = backbone._require_action_expert()
        horizon = policy._generation_action_horizon()
        bsize = int(encoder_kv[0][0].shape[0])
        dtype = action_expert.action_embed.weight.dtype
        context = action_expert.prepare_context(
            encoder_kv_states=encoder_kv,
            encoder_attention_mask=encoder_attn,
            state_embeddings=None,
            batch_size=bsize,
            seq_len=horizon,
            device=device,
            dtype=dtype,
        )
        pad = batch.get("action_dim_is_pad")
        shape = (bsize, horizon, int(backbone.config.max_action_dim))

    def expert(steps: int):
        times = [torch.full((bsize,), idx / steps, device=device, dtype=torch.float32) for idx in range(steps)]
        cache = action_expert.get_or_prepare_modulation_cache(
            times, cache_key=("nfe", steps, bsize, device, dtype)
        )
        trajectory = torch.randn(shape, device=device, dtype=dtype)
        if pad is not None:
            trajectory = backbone._mask_action_dim_tensor(trajectory, action_dim_is_pad=pad, enabled=True)
        dt = 1.0 / steps
        for idx in range(steps):
            velocity = action_expert.forward_with_context(
                trajectory, cache[idx].conditioning, context=context, modulation=cache[idx]
            )
            if pad is not None:
                velocity = backbone._mask_action_dim_tensor(velocity, action_dim_is_pad=pad, enabled=True)
            trajectory = trajectory + dt * velocity

    def full(steps: int, drop: list[int]):
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with ctx:
            policy.reset()
            policy.predict_action_chunk(batch, num_steps=steps)

    with torch.no_grad(), policy._autocast_context():
        for _ in range(WARMUP):
            full(10, [])
        timing = {
            "full_nfe10": {"ms": cuda_ms(lambda: full(10, [])), "flops_g": flops(lambda: full(10, []))},
            "full_nfe1": {"ms": cuda_ms(lambda: full(1, [])), "flops_g": flops(lambda: full(1, []))},
            "drop16_nfe10": {"ms": cuda_ms(lambda: full(10, MOLMO_DROP)), "flops_g": flops(lambda: full(10, MOLMO_DROP))},
            "drop16_nfe1": {"ms": cuda_ms(lambda: full(1, MOLMO_DROP)), "flops_g": flops(lambda: full(1, MOLMO_DROP))},
            "expert_nfe10": {"ms": cuda_ms(lambda: expert(10)), "flops_g": flops(lambda: expert(10))},
            "expert_nfe1": {"ms": cuda_ms(lambda: expert(1)), "flops_g": flops(lambda: expert(1))},
        }
    return {
        "stack": "molmoact2",
        "device": torch.cuda.get_device_name(0),
        "checkpoint": str(LEROBOT_CKPT),
        "drop16": MOLMO_DROP,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "offline": offline,
        "timing": timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", choices=("pi05", "molmoact2"), required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    payload = pi05(device) if args.stack == "pi05" else molmo(device)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.stack}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", path, flush=True)
    for name, row in payload["timing"].items():
        print(f"  {name}: {row['ms']:.1f} ms  {row['flops_g']:.1f} G", flush=True)


if __name__ == "__main__":
    main()

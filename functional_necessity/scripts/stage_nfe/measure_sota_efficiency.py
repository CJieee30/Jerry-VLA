#!/usr/bin/env python3
"""Paired efficiency for the paper tables in efficiency_analysis.md.

One process, one calibration frame, CUDA events warmup=5 / repeats=20.
Writes results/stage_nfe/sota_efficiency_{pi05,molmoact2,smolvla}.json.
Does not overwrite historical efficiency JSON.
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
PI05_DROP = [1, 13, 14, 15, 16, 17]
MOLMO_DROP = [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
SMOL_DROP = [12, 13, 14, 15]


def cuda_ms(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPEATS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / REPEATS)


def flops_g(fn) -> float:
    fn()
    with FlopCounterMode(display=False) as mode:
        fn()
    return int(mode.get_total_flops()) / 1e9


def cell(fn) -> dict:
    return {"ms": cuda_ms(fn), "flops_g": flops_g(fn)}


def pi05(device: torch.device) -> dict:
    sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
    from common import DEFAULT_CALIB_FILE, DEFAULT_CHECKPOINT, load_calib_data, load_policy, move_to_device, slice_batch
    from layer_skip import skip_llm_layers
    from measure_efficiency import count_params
    from measure_module_efficiency import denoise_all, prefix_cache, prepare_inputs
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d

    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.n_action_steps = 10
    policy.config.compile_model = False
    policy.eval()
    calib = load_calib_data(DEFAULT_CALIB_FILE)
    batch = move_to_device(slice_batch(calib[0], 0, 1), device)
    model = policy.model
    pal = model.paligemma_with_expert
    images, img_masks, tokens, masks, states, state_masks = prepare_inputs(policy, batch)
    bsize = int(tokens.shape[0])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1000)
    noise = torch.randn(
        bsize, int(model.config.chunk_size), int(model.config.max_action_dim), generator=gen
    ).to(device)

    def full(steps: int, drop: list[int]):
        ctx = skip_llm_layers(policy, drop) if drop else nullcontext()
        with ctx:
            policy.reset()
            policy.predict_action_chunk(batch, noise=noise, num_steps=steps)

    def vision():
        for img in images:
            pal.embed_image(img)

    with torch.no_grad():
        prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
            images, img_masks, tokens, masks, states, state_masks
        )
        attn = prepare_attention_masks_4d(make_att_2d_masks(prefix_pad, prefix_att))
        pos = torch.cumsum(prefix_pad, dim=1) - 1
        lang = pal._paligemma_language_model_core()
        lang.config._attn_implementation = "eager"

    def language(drop: list[int]):
        ctx = skip_llm_layers(policy, drop) if drop else nullcontext()
        with ctx:
            pal.forward(
                attention_mask=attn,
                position_ids=pos,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

    def build_past(drop: list[int]):
        ctx = skip_llm_layers(policy, drop) if drop else nullcontext()
        with ctx:
            return prefix_cache(model, images, img_masks, tokens, masks, states, state_masks)

    with torch.no_grad():
        _, pad_full, past_full = build_past([])
        _, pad_drop, past_drop = build_past(PI05_DROP)

    def action_from(pad, past, steps: int):
        denoise_all(model, pad, past, noise.clone(), steps)

    base_params = count_params(policy, [])
    drop_params = count_params(policy, PI05_DROP)
    with torch.no_grad():
        chunk = {
            "full_nfe10": cell(lambda: full(10, [])),
            "drop_nfe10": cell(lambda: full(10, PI05_DROP)),
            "full_nfe1": cell(lambda: full(1, [])),
            "drop_nfe1": cell(lambda: full(1, PI05_DROP)),
        }
        modules = {
            "vision": cell(vision),
            "language_full": cell(lambda: language([])),
            "language_drop": cell(lambda: language(PI05_DROP)),
            "action_nfe10": cell(lambda: action_from(pad_full, past_full, 10)),
            "action_drop_nfe1": cell(lambda: action_from(pad_drop, past_drop, 1)),
        }
    return {
        "stack": "pi05",
        "device": torch.cuda.get_device_name(0),
        "checkpoint": DEFAULT_CHECKPOINT,
        "drop": PI05_DROP,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "prefix_seq": int(prefix_embs.shape[1]),
        "n_views": len(images),
        "params_m": {
            "full": base_params["active_params"] / 1e6,
            "drop": drop_params["active_params"] / 1e6,
            "language_full": base_params["language_active_params"] / 1e6,
            "language_drop": drop_params["language_active_params"] / 1e6,
        },
        "chunk": chunk,
        "modules": modules,
    }


def molmo(device: torch.device) -> dict:
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    from layer_skip import skip_vlm_layers
    from load_policy import CALIB_PATH, LEROBOT_CKPT, load_policy
    from measure_efficiency import count_params

    policy = load_policy(str(device))
    policy.eval()
    policy.config.compile_model = False
    policy.config.enable_inference_cuda_graph = False
    policy.config.gradient_checkpointing = False
    policy.config.n_action_steps = 10
    backbone = policy._backbone()
    backbone.gradient_checkpointing = False
    if getattr(backbone, "action_cuda_graph_manager", None) is not None:
        backbone.action_cuda_graph_manager = None
    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    raw = blob["samples"][0]["batch"]
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
    model_inputs = policy._model_inputs(batch)

    def full(steps: int, drop: list[int]):
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with ctx:
            policy.reset()
            policy.predict_action_chunk(batch, num_steps=steps)

    with torch.no_grad(), policy._autocast_context():
        images, token_pooling = backbone.merge_visual_inputs(
            input_ids=model_inputs.get("input_ids"),
            pixel_values=model_inputs.get("pixel_values"),
            image_token_pooling=model_inputs.get("image_token_pooling"),
            image_grids=model_inputs.get("image_grids"),
            image_num_crops=model_inputs.get("image_num_crops"),
        )
        inputs_embeds, _image_features = backbone.build_input_embeddings(
            model_inputs["input_ids"], images, token_pooling
        )
        causal_mask = backbone._build_native_attention_bias(
            inputs_embeds=inputs_embeds,
            attention_mask=model_inputs.get("attention_mask"),
            token_type_ids=model_inputs.get("token_type_ids"),
            past_key_values=None,
        )
        n_views = int(model_inputs["image_grids"].shape[0]) if "image_grids" in model_inputs else 0
        prefix_seq = int(inputs_embeds.shape[1])

    def vision():
        imgs, pool = backbone.merge_visual_inputs(
            input_ids=model_inputs.get("input_ids"),
            pixel_values=model_inputs.get("pixel_values"),
            image_token_pooling=model_inputs.get("image_token_pooling"),
            image_grids=model_inputs.get("image_grids"),
            image_num_crops=model_inputs.get("image_num_crops"),
        )
        backbone.vision_backbone(imgs, pool)

    def language(drop: list[int]):
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with ctx:
            backbone.transformer(
                attention_mask=causal_mask,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=None,
                output_attentions=False,
                output_hidden_states=False,
            )

    def action_context(drop: list[int]):
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with ctx:
            outputs = backbone(
                **model_inputs, use_cache=True, output_attentions=False, output_hidden_states=False
            )
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
        expert = backbone._require_action_expert()
        horizon = policy._generation_action_horizon()
        bsize = int(encoder_kv[0][0].shape[0])
        dtype = expert.action_embed.weight.dtype
        context = expert.prepare_context(
            encoder_kv_states=encoder_kv,
            encoder_attention_mask=encoder_attn,
            state_embeddings=None,
            batch_size=bsize,
            seq_len=horizon,
            device=device,
            dtype=dtype,
        )
        return expert, context, bsize, horizon, dtype, batch.get("action_dim_is_pad")

    def expert_steps(built, steps: int):
        expert, context, bsize, horizon, dtype, pad = built
        times = [torch.full((bsize,), idx / steps, device=device, dtype=torch.float32) for idx in range(steps)]
        cache = expert.get_or_prepare_modulation_cache(times, cache_key=("sota", steps, bsize, device, dtype))
        trajectory = torch.randn((bsize, horizon, int(backbone.config.max_action_dim)), device=device, dtype=dtype)
        if pad is not None:
            trajectory = backbone._mask_action_dim_tensor(trajectory, action_dim_is_pad=pad, enabled=True)
        dt = 1.0 / steps
        for idx in range(steps):
            velocity = expert.forward_with_context(
                trajectory, cache[idx].conditioning, context=context, modulation=cache[idx]
            )
            if pad is not None:
                velocity = backbone._mask_action_dim_tensor(velocity, action_dim_is_pad=pad, enabled=True)
            trajectory = trajectory + dt * velocity

    base_params = count_params(policy, [])
    drop_params = count_params(policy, MOLMO_DROP)
    with torch.no_grad(), policy._autocast_context():
        built_full = action_context([])
        built_drop = action_context(MOLMO_DROP)
        chunk = {
            "full_nfe10": cell(lambda: full(10, [])),
            "drop_nfe10": cell(lambda: full(10, MOLMO_DROP)),
            "full_nfe1": cell(lambda: full(1, [])),
            "drop_nfe1": cell(lambda: full(1, MOLMO_DROP)),
        }
        modules = {
            "vision": cell(vision),
            "language_full": cell(lambda: language([])),
            "language_drop": cell(lambda: language(MOLMO_DROP)),
            "action_nfe10": cell(lambda: expert_steps(built_full, 10)),
            "action_drop_nfe1": cell(lambda: expert_steps(built_drop, 1)),
        }
    return {
        "stack": "molmoact2",
        "skip": "kv_only",
        "device": torch.cuda.get_device_name(0),
        "checkpoint": str(LEROBOT_CKPT),
        "drop": MOLMO_DROP,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "prefix_seq": prefix_seq,
        "n_views": n_views,
        "params_m": {
            "full": base_params["active_params"] / 1e6,
            "drop": drop_params["active_params"] / 1e6,
            "language_full": base_params["language_active_params"] / 1e6,
            "language_drop": drop_params["language_active_params"] / 1e6,
        },
        "chunk": chunk,
        "modules": modules,
    }


def smolvla(device: torch.device) -> dict:
    sys.path.insert(0, str(FN / "scripts" / "smolvla_libero"))
    from layer_skip import skip_vlm_layers
    from load_policy import CALIB_PATH, CKPT, load_policy
    from measure_efficiency import count_params
    from score import _ensure_policy_batch
    from lerobot.policies.common.flow_matching import euler_integrate
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    policy = load_policy(str(device))
    policy.config.compile_model = False
    policy.config.n_action_steps = 10
    policy.config.num_steps = 10
    policy.eval()
    model = policy.model
    vlm = model.vlm_with_expert
    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    raw = _ensure_policy_batch(blob["samples"][0]["batch"])
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in raw.items()}
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    bsize = int(state.shape[0])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1000)
    noise = torch.randn(
        bsize, int(model.config.chunk_size), int(model.config.max_action_dim), generator=gen
    ).to(device)

    def full(steps: int, drop: list[int]):
        policy.config.num_steps = steps
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with ctx:
            policy.reset()
            policy.predict_action_chunk(batch, noise=noise)

    def vision():
        for img in images:
            vlm.embed_image(img)

    with torch.no_grad():
        prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
            images, img_masks, tokens, masks, state=state
        )
        attn = make_att_2d_masks(prefix_pad, prefix_att)
        pos = torch.cumsum(prefix_pad, dim=1) - 1

    def language(drop: list[int]):
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with ctx:
            vlm.forward(
                attention_mask=attn,
                position_ids=pos,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

    def build_past(drop: list[int]):
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with ctx:
            _, past = vlm.forward(
                attention_mask=attn,
                position_ids=pos,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        return past

    def action_from(past, steps: int):
        euler_integrate(
            lambda input_x_t, current_timestep: model.denoise_step(
                x_t=input_x_t,
                prefix_pad_masks=prefix_pad,
                past_key_values=past,
                timestep=current_timestep,
            ),
            noise.clone(),
            steps,
            rtc_processor=model.rtc_processor,
            rtc_enabled=False,
        )

    base_params = count_params(policy, [])
    drop_params = count_params(policy, SMOL_DROP)
    with torch.no_grad():
        past_full = build_past([])
        past_drop = build_past(SMOL_DROP)
        chunk = {
            "full_nfe10": cell(lambda: full(10, [])),
            "drop_nfe10": cell(lambda: full(10, SMOL_DROP)),
            "full_nfe2": cell(lambda: full(2, [])),
            "drop_nfe2": cell(lambda: full(2, SMOL_DROP)),
        }
        modules = {
            "vision": cell(vision),
            "language_full": cell(lambda: language([])),
            "language_drop": cell(lambda: language(SMOL_DROP)),
            "action_nfe10": cell(lambda: action_from(past_full, 10)),
            "action_drop_nfe2": cell(lambda: action_from(past_drop, 2)),
        }
    return {
        "stack": "smolvla",
        "skip": "kv_only",
        "device": torch.cuda.get_device_name(0),
        "checkpoint": CKPT,
        "drop": SMOL_DROP,
        "operating_nfe": 2,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "prefix_seq": int(prefix_embs.shape[1]),
        "n_views": len(images),
        "chunk_size": int(model.config.chunk_size),
        "n_action_steps": 10,
        "params_m": {
            "full": base_params["active_params"] / 1e6,
            "drop": drop_params["active_params"] / 1e6,
            "language_full": base_params["language_active_params"] / 1e6,
            "language_drop": drop_params["language_active_params"] / 1e6,
        },
        "chunk": chunk,
        "modules": modules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", choices=("pi05", "molmoact2", "smolvla"), required=True)
    parser.add_argument("--tag", default="", help="Filename suffix. Empty keeps the historical json name.")
    args = parser.parse_args()
    device = torch.device("cuda")
    payload = {"pi05": pi05, "molmoact2": molmo, "smolvla": smolvla}[args.stack](device)
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"sota_efficiency_{args.stack}"
    if args.tag:
        name += f"_{args.tag}"
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", path, flush=True)
    for name, row in payload["chunk"].items():
        print(f"  chunk {name}: {row['ms']:.1f} ms  {row['flops_g']:.1f} G", flush=True)
    for name, row in payload["modules"].items():
        print(f"  module {name}: {row['ms']:.1f} ms  {row['flops_g']:.1f} G", flush=True)


if __name__ == "__main__":
    main()

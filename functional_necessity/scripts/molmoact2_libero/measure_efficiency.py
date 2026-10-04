#!/usr/bin/env python3
"""MolmoAct2 baseline module split + Taylor Drop-16 keep-KV efficiency.

Matches the π0.5 protocol: FlopCounterMode + CUDA events, warmup=5 / repeats=20,
one calib observation, compile_model=false, cuda_graph=false. Skip is keep-KV
identity residual (still runs attn for K/V; skips MLP + ff_norm).
"""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))

from layer_skip import skip_vlm_layers
from load_policy import CALIB_PATH, FN_ROOT, LEROBOT_CKPT, load_policy, vlm_blocks

OUT = FN_ROOT / "results" / "molmoact2_libero" / "efficiency_taylor_drop16.json"
DROP = [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
WARMUP = 5
REPEATS = 20
N_VLM = 36


def _nparams(mod) -> int:
    return int(sum(p.numel() for p in mod.parameters()))


def _cuda_ms(fn, repeats: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


def _flops(fn) -> int:
    fn()
    with FlopCounterMode(display=False) as mode:
        fn()
    return int(mode.get_total_flops())


def count_params(policy, drop: list[int]) -> dict:
    """Stored vs active: keep-KV still runs attn_norm + self_attn; skips mlp + ff_norm."""
    stored = _nparams(policy)
    blocks, _ = vlm_blocks(policy)
    language = _nparams(policy._backbone().transformer)
    skipped = 0
    per_layer = []
    for idx in drop:
        layer = blocks[idx]
        unused = {
            "mlp": _nparams(layer.mlp),
            "ff_norm": _nparams(layer.ff_norm),
        }
        kept = {
            "attn_norm": _nparams(layer.attn_norm),
            "self_attn": _nparams(layer.self_attn),
        }
        layer_skip = int(sum(unused.values()))
        skipped += layer_skip
        per_layer.append(
            {
                "layer": int(idx),
                "unused_numel": layer_skip,
                "unused": unused,
                "kept_attn_numel": int(sum(kept.values())),
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


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def main() -> None:
    device = torch.device("cuda")
    policy = load_policy(str(device))
    policy.eval()
    policy.config.compile_model = False
    policy.config.enable_inference_cuda_graph = False
    policy.config.gradient_checkpointing = False
    backbone = policy._backbone()
    backbone.gradient_checkpointing = False
    if getattr(backbone, "action_cuda_graph_manager", None) is not None:
        backbone.action_cuda_graph_manager = None

    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    raw_batch = blob["samples"][0]["batch"]
    batch = _to_device(raw_batch, device)
    model_inputs = policy._model_inputs(batch)
    autocast = policy._autocast_context()

    vision = backbone.vision_backbone
    transformer = backbone.transformer
    action_expert = backbone._require_action_expert()
    vis_params = _nparams(vision)
    lang_params = _nparams(transformer)
    act_params = _nparams(action_expert)

    with torch.no_grad(), autocast:
        images, token_pooling = backbone.merge_visual_inputs(
            input_ids=model_inputs.get("input_ids"),
            pixel_values=model_inputs.get("pixel_values"),
            image_token_pooling=model_inputs.get("image_token_pooling"),
            image_grids=model_inputs.get("image_grids"),
            image_num_crops=model_inputs.get("image_num_crops"),
        )
        inputs_embeds, image_features = backbone.build_input_embeddings(
            model_inputs["input_ids"], images, token_pooling
        )
        causal_mask = backbone._build_native_attention_bias(
            inputs_embeds=inputs_embeds,
            attention_mask=model_inputs.get("attention_mask"),
            token_type_ids=model_inputs.get("token_type_ids"),
            past_key_values=None,
        )
        n_vision_tokens = int(image_features.shape[0]) if image_features is not None else 0
        n_image_views = int(model_inputs["image_grids"].shape[0]) if "image_grids" in model_inputs else 0
        prefix_seq = int(inputs_embeds.shape[1])
        vision_tokens_per_view = n_vision_tokens // n_image_views if n_image_views else n_vision_tokens

        outputs = backbone(
            **model_inputs,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
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
        steps = int(backbone.config.flow_matching_num_steps)
        action_horizon = policy._generation_action_horizon()
        bsize = int(encoder_kv[0][0].shape[0])
        action_context = action_expert.prepare_context(
            encoder_kv_states=encoder_kv,
            encoder_attention_mask=encoder_attn,
            state_embeddings=None,
            batch_size=bsize,
            seq_len=action_horizon,
            device=device,
            dtype=action_expert.action_embed.weight.dtype,
        )
        flow_timesteps = [
            torch.full((bsize,), idx / steps, device=device, dtype=torch.float32) for idx in range(steps)
        ]
        modulation_cache = action_expert.get_or_prepare_modulation_cache(
            flow_timesteps,
            cache_key=(steps, bsize, device, action_expert.action_embed.weight.dtype),
        )
        action_dim_is_pad = batch.get("action_dim_is_pad")
        dt = 1.0 / steps
        traj_shape = (bsize, action_horizon, int(backbone.config.max_action_dim))
        traj_dtype = action_expert.action_embed.weight.dtype

    def vision_only():
        imgs, pool = backbone.merge_visual_inputs(
            input_ids=model_inputs.get("input_ids"),
            pixel_values=model_inputs.get("pixel_values"),
            image_token_pooling=model_inputs.get("image_token_pooling"),
            image_grids=model_inputs.get("image_grids"),
            image_num_crops=model_inputs.get("image_num_crops"),
        )
        backbone.vision_backbone(imgs, pool)

    def llm_prefix(drop: list[int]):
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with ctx:
            transformer(
                attention_mask=causal_mask,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=None,
                output_attentions=False,
                output_hidden_states=False,
            )

    def action_denoise():
        trajectory = torch.randn(traj_shape, device=device, dtype=traj_dtype)
        if action_dim_is_pad is not None:
            trajectory = backbone._mask_action_dim_tensor(
                trajectory, action_dim_is_pad=action_dim_is_pad, enabled=True
            )
        for idx in range(steps):
            velocity = action_expert.forward_with_context(
                trajectory,
                modulation_cache[idx].conditioning,
                context=action_context,
                modulation=modulation_cache[idx],
            )
            if action_dim_is_pad is not None:
                velocity = backbone._mask_action_dim_tensor(
                    velocity, action_dim_is_pad=action_dim_is_pad, enabled=True
                )
            trajectory = trajectory + dt * velocity

    def full_chunk(drop: list[int]):
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with ctx:
            policy.predict_action_chunk(batch)

    with torch.no_grad(), autocast:
        for _ in range(WARMUP):
            full_chunk([])

        vis_flops = _flops(vision_only)
        llm_flops = _flops(lambda: llm_prefix([]))
        act_flops = _flops(action_denoise)
        full_flops = _flops(lambda: full_chunk([]))
        drop_llm_flops = _flops(lambda: llm_prefix(DROP))
        drop_full_flops = _flops(lambda: full_chunk(DROP))

        vis_ms = _cuda_ms(vision_only, REPEATS)
        llm_ms = _cuda_ms(lambda: llm_prefix([]), REPEATS)
        act_ms = _cuda_ms(action_denoise, REPEATS)
        full_ms = _cuda_ms(lambda: full_chunk([]), REPEATS)
        drop_llm_ms = _cuda_ms(lambda: llm_prefix(DROP), REPEATS)
        drop_full_ms = _cuda_ms(lambda: full_chunk(DROP), REPEATS)

    base_params = count_params(policy, [])
    drop_params = count_params(policy, DROP)

    def delta(a, b):
        return {"abs": b - a, "pct": 100.0 * (b / a - 1.0) if a else 0.0}

    payload = {
        "checkpoint": str(LEROBOT_CKPT),
        "device": str(torch.cuda.get_device_name(0)),
        "compile_model": False,
        "enable_inference_cuda_graph": False,
        "n_action_steps": int(policy.config.n_action_steps),
        "chunk_size": int(policy.config.chunk_size),
        "num_inference_steps": steps,
        "skip": "keep_kv_identity_residual",
        "drop": DROP,
        "n_vlm_layers": N_VLM,
        "prefix_seq": prefix_seq,
        "n_image_views": n_image_views,
        "vision_tokens_per_view": vision_tokens_per_view,
        "vision_tokens_total": n_vision_tokens,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "params_m": {
            "vision": vis_params / 1e6,
            "language": lang_params / 1e6,
            "action": act_params / 1e6,
            "stored_total": base_params["stored_params"] / 1e6,
        },
        "flops_g": {
            "vision": vis_flops / 1e9,
            "language": llm_flops / 1e9,
            "action": act_flops / 1e9,
            "sum_modules": (vis_flops + llm_flops + act_flops) / 1e9,
            "full_chunk": full_flops / 1e9,
        },
        "inference_ms": {
            "vision": vis_ms,
            "language": llm_ms,
            "action": act_ms,
            "sum_modules": vis_ms + llm_ms + act_ms,
            "full_chunk": full_ms,
        },
        "baseline": {
            "drop": [],
            "params_stored_m": base_params["stored_params"] / 1e6,
            "params_active_m": base_params["active_params"] / 1e6,
            "params_language_stored_m": base_params["language_stored_params"] / 1e6,
            "params_language_active_m": base_params["language_active_params"] / 1e6,
            "measured_gflops": full_flops / 1e9,
            "predict_action_chunk_ms": full_ms,
            "language_flops_g": llm_flops / 1e9,
            "language_ms": llm_ms,
        },
        "taylor_drop16": {
            "drop": DROP,
            "params": drop_params,
            "params_stored_m": drop_params["stored_params"] / 1e6,
            "params_active_m": drop_params["active_params"] / 1e6,
            "params_language_stored_m": drop_params["language_stored_params"] / 1e6,
            "params_language_active_m": drop_params["language_active_params"] / 1e6,
            "measured_gflops": drop_full_flops / 1e9,
            "predict_action_chunk_ms": drop_full_ms,
            "language_flops_g": drop_llm_flops / 1e9,
            "language_ms": drop_llm_ms,
        },
        "delta_full": {
            "params_active_m": delta(base_params["active_params"] / 1e6, drop_params["active_params"] / 1e6),
            "predict_action_chunk_ms": delta(full_ms, drop_full_ms),
            "measured_gflops": delta(full_flops / 1e9, drop_full_flops / 1e9),
        },
        "delta_language": {
            "params_language_active_m": delta(
                base_params["language_active_params"] / 1e6, drop_params["language_active_params"] / 1e6
            ),
            "flops_g": delta(llm_flops / 1e9, drop_llm_flops / 1e9),
            "inference_ms": delta(llm_ms, drop_llm_ms),
        },
        "closed_loop": {
            "setting": "Taylor Drop-16 no L0",
            "avg": 98.10,
            "delta_vs_anchor": 0.60,
            "anchor": 97.50,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    if drop_llm_flops >= 0.98 * llm_flops:
        raise SystemExit("Drop-16 did not reduce language FLOPs; skip may not be on the prefix path.")


if __name__ == "__main__":
    main()

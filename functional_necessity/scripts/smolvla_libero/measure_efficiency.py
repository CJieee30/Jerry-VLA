#!/usr/bin/env python3
"""SmolVLA baseline module split + Taylor Drop-4 keep-KV efficiency.

Matches the π0.5 protocol: FlopCounterMode + CUDA events, warmup=5 / repeats=20,
one calib observation, compile_model=false. Skip is keep-KV on the VLM stream
(still writes K/V; skips VLM o_proj + MLP + post-attn LN). Expert is never dropped.
Eval protocol: n_action_steps=10, num_steps=10, chunk_size=50.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapt_lerobot_official import CKPT, N_ACTION_STEPS, NUM_STEPS
from layer_skip import skip_vlm_layers
from load_policy import CALIB_PATH, FN_ROOT, load_policy, vlm_blocks
from score import _ensure_policy_batch

OUT = FN_ROOT / "results" / "smolvla_libero" / "efficiency_taylor_drop4.json"
DROP = [12, 13, 14, 15]
WARMUP = 5
REPEATS = 20
N_VLM = 16


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
    """Stored vs active. Dropped VLM layers run input LN + K/V only, not Q/o_proj/MLP."""
    stored = _nparams(policy)
    blocks, _ = vlm_blocks(policy)
    text = policy.model.vlm_with_expert.get_vlm_model().text_model
    language = _nparams(text)
    skipped = 0
    per_layer = []
    for idx in drop:
        layer = blocks[idx]
        unused = {
            "q_proj": _nparams(layer.self_attn.q_proj),
            "o_proj": _nparams(layer.self_attn.o_proj),
            "mlp": _nparams(layer.mlp),
            "post_attention_layernorm": _nparams(layer.post_attention_layernorm),
        }
        kept = {
            "input_layernorm": _nparams(layer.input_layernorm),
            "k_proj": _nparams(layer.self_attn.k_proj),
            "v_proj": _nparams(layer.self_attn.v_proj),
        }
        layer_skip = int(sum(unused.values()))
        skipped += layer_skip
        per_layer.append(
            {
                "layer": int(idx),
                "unused_numel": layer_skip,
                "unused": unused,
                "kept_kv_ln_numel": int(sum(kept.values())),
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
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    device = torch.device("cuda")
    policy = load_policy(str(device))
    policy.config.compile_model = False
    policy.config.n_action_steps = N_ACTION_STEPS
    policy.config.num_steps = NUM_STEPS
    policy.eval()
    model = policy.model
    vlm = model.vlm_with_expert

    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    raw = _ensure_policy_batch(blob["samples"][0]["batch"])
    batch = _to_device(raw, device)

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    n_views = len(images)
    num_steps = int(policy.config.num_steps)

    vision_mod = vlm.get_vlm_model().vision_model
    connector = vlm.get_vlm_model().connector
    text = vlm.get_vlm_model().text_model
    expert = vlm.lm_expert
    vis_params = _nparams(vision_mod) + _nparams(connector)
    lang_params = _nparams(text)
    act_params = _nparams(expert) + _nparams(model.action_in_proj) + _nparams(model.action_out_proj)
    for name in ("action_time_mlp_in", "action_time_mlp_out"):
        if hasattr(model, name):
            act_params += _nparams(getattr(model, name))

    with torch.no_grad():
        img0 = vlm.embed_image(images[0])
        vision_tokens_per_view = int(img0.shape[1])
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, tokens, masks, state=state
        )
        prefix_len = int(prefix_embs.shape[1])
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        bsize = state.shape[0]
        noise0 = model.sample_noise(
            (bsize, model.config.chunk_size, model.config.max_action_dim), device
        )
        _, past_kv = vlm.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

    def vision_only():
        for img in images:
            vlm.embed_image(img)

    def llm_prefix(drop: list[int]):
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with ctx:
            vlm.forward(
                attention_mask=prefix_att_2d,
                position_ids=prefix_pos,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

    def action_denoise():
        from lerobot.policies.common.flow_matching import euler_integrate

        euler_integrate(
            lambda input_x_t, current_timestep: model.denoise_step(
                x_t=input_x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_kv,
                timestep=current_timestep,
            ),
            noise0.clone(),
            num_steps,
            rtc_processor=model.rtc_processor,
            rtc_enabled=False,
        )

    def full_chunk(drop: list[int]):
        policy.reset()
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with ctx:
            policy.predict_action_chunk(batch)

    with torch.no_grad():
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
        "checkpoint": CKPT,
        "device": str(torch.cuda.get_device_name(0)),
        "compile_model": False,
        "n_action_steps": N_ACTION_STEPS,
        "chunk_size": int(model.config.chunk_size),
        "num_inference_steps": num_steps,
        "skip": "keep_kv_identity_residual",
        "drop": DROP,
        "n_vlm_layers": N_VLM,
        "prefix_seq": prefix_len,
        "language_token_seq": int(tokens.shape[1]),
        "n_image_views": n_views,
        "vision_tokens_per_view": vision_tokens_per_view,
        "vision_tokens_total": vision_tokens_per_view * n_views,
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
        "taylor_drop4": {
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
            "setting": "Taylor Drop-4",
            "avg": 78.50,
            "delta_vs_anchor": -4.50,
            "anchor": 83.00,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    if drop_llm_flops >= 0.98 * llm_flops:
        raise SystemExit("Drop-4 did not reduce language FLOPs; skip may not be on the prefix path.")


if __name__ == "__main__":
    main()

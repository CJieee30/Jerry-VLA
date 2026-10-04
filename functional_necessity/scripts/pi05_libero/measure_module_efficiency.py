#!/usr/bin/env python3
"""Module-wise FLOPs and CUDA time for π0.5 v044 baseline (EfficientVLA Table 1 style).

Phases match official cache-path sample_actions:
  Vision  = SigLIP + projector over all camera views (inside embed_prefix)
  Language = PaliGemma prefix self-attn with KV cache
  Action  = 10 Euler denoise steps on the Gemma expert
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_CALIB_FILE,
    DEFAULT_CHECKPOINT,
    FN_ROOT,
    init_logging,
    load_calib_data,
    load_policy,
    move_to_device,
    slice_batch,
)

OUT = FN_ROOT / "results" / "eval_drop" / "efficiency_pi05_baseline_modules.json"
WARMUP = 5
REPEATS = 20


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


def prepare_inputs(policy, batch):
    policy.eval()
    images, img_masks = policy._preprocess_images(batch)
    states, state_masks = policy._prepare_memory_states(batch)
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    return images, img_masks, tokens, masks, states, state_masks


def prefix_cache(model, images, img_masks, tokens, masks, states, state_masks):
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, tokens, masks, states, state_masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
    model._paligemma_language_model_core().config._attn_implementation = "eager"
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return prefix_embs, prefix_pad_masks, past_key_values


def denoise_all(model, prefix_pad_masks, past_key_values, noise, num_steps):
    from lerobot.policies.pi05.modeling_pi05 import euler_integrate

    return euler_integrate(
        lambda input_x_t, current_timestep: model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=input_x_t,
            timestep=current_timestep,
        ),
        noise,
        num_steps,
        rtc_processor=model.rtc_processor,
        rtc_enabled=False,
    )


def main() -> None:
    init_logging()
    device = torch.device("cuda")
    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.n_action_steps = 10
    policy.config.compile_model = False
    calib = load_calib_data(DEFAULT_CALIB_FILE)
    batch = move_to_device(slice_batch(calib[0], 0, 1), device)
    model = policy.model
    num_steps = int(policy.config.num_inference_steps)

    pal = model.paligemma_with_expert
    vis = pal.paligemma.model.vision_tower
    proj = pal.paligemma.model.multi_modal_projector
    lang = pal._paligemma_language_model_core()
    expert = pal.gemma_expert

    vision_params = _nparams(vis) + _nparams(proj)
    language_params = _nparams(lang)
    action_params = _nparams(expert) + _nparams(model.action_in_proj) + _nparams(model.action_out_proj)
    for name in ("time_mlp_in", "time_mlp_out"):
        if hasattr(model, name):
            action_params += _nparams(getattr(model, name))

    images, img_masks, tokens, masks, states, state_masks = prepare_inputs(policy, batch)
    n_views = len(images)
    with torch.no_grad():
        img0 = pal.embed_image(images[0])
    vision_tokens_per_view = int(img0.shape[1])
    bsize = tokens.shape[0]
    noise0 = model.sample_noise((bsize, model.config.chunk_size, model.config.max_action_dim), device)

    def vision_only():
        for img, img_mask in zip(images, img_masks, strict=True):
            pal.embed_image(img)

    def language_embed_only():
        pal.embed_language_tokens(tokens)

    def language_prefix():
        prefix_cache(model, images, img_masks, tokens, masks, states, state_masks)

    # language_prefix includes vision+embed+LLM. Isolate LLM = prefix_cache after cached embeds.
    # Build embeds once, then only the language_model.forward.
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d

    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, tokens, masks, states, state_masks
        )
        prefix_len = int(prefix_embs.shape[1])
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(
            make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        lang.config._attn_implementation = "eager"

    def llm_prefix_only():
        pal.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

    with torch.no_grad():
        _, past_kv = pal.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        past_kv_keep = past_kv

    def action_denoise():
        denoise_all(model, prefix_pad_masks, past_kv_keep, noise0.clone(), num_steps)

    def full_chunk():
        policy.reset()
        policy.predict_action_chunk(batch)

    with torch.no_grad():
        # warmup
        for _ in range(WARMUP):
            full_chunk()

        vis_flops = _flops(vision_only)
        llm_flops = _flops(llm_prefix_only)
        act_flops = _flops(action_denoise)
        full_flops = _flops(full_chunk)

        vis_ms = _cuda_ms(vision_only, REPEATS)
        llm_ms = _cuda_ms(llm_prefix_only, REPEATS)
        act_ms = _cuda_ms(action_denoise, REPEATS)
        lang_emb_ms = _cuda_ms(language_embed_only, REPEATS)
        full_ms = _cuda_ms(full_chunk, REPEATS)

    payload = {
        "checkpoint": DEFAULT_CHECKPOINT,
        "device": str(torch.cuda.get_device_name(0)),
        "compile_model": False,
        "n_action_steps": 10,
        "chunk_size": int(model.config.chunk_size),
        "num_inference_steps": num_steps,
        "prefix_seq": prefix_len,
        "language_token_seq": int(tokens.shape[1]),
        "n_image_views": n_views,
        "vision_tokens_per_view": vision_tokens_per_view,
        "vision_tokens_total": vision_tokens_per_view * n_views,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "params_m": {
            "vision": vision_params / 1e6,
            "language": language_params / 1e6,
            "action": action_params / 1e6,
            "stored_total": _nparams(policy) / 1e6,
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
            "language_token_embed": lang_emb_ms,
            "sum_modules": vis_ms + llm_ms + act_ms,
            "full_chunk": full_ms,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

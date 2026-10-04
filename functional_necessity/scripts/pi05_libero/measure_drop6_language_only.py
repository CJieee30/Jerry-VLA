#!/usr/bin/env python3
"""Language-module only: baseline vs Fisher Drop-6 (Appendix B keep-KV)."""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
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
from layer_skip import skip_llm_layers  # noqa: E402
from measure_efficiency import count_params  # noqa: E402

OUT = FN_ROOT / "results" / "eval_drop" / "efficiency_pi05_drop6_language_only.json"
DROP = [1, 13, 14, 15, 16, 17]
WARMUP = 5
REPEATS = 20


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


def main() -> None:
    init_logging()
    device = torch.device("cuda")
    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.n_action_steps = 10
    calib = load_calib_data(DEFAULT_CALIB_FILE)
    batch = move_to_device(slice_batch(calib[0], 0, 1), device)
    model = policy.model
    pal = model.paligemma_with_expert

    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, img_masks = policy._preprocess_images(batch)
    states, state_masks = policy._prepare_memory_states(batch)
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]

    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, tokens, masks, states, state_masks
        )
        attn = prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        pal._paligemma_language_model_core().config._attn_implementation = "eager"

    def llm_forward(drop: list[int]) -> None:
        ctx = skip_llm_layers(policy, drop) if drop else nullcontext()
        with ctx:
            pal.forward(
                attention_mask=attn,
                position_ids=pos,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

    rows = {}
    with torch.no_grad():
        for name, drop in (("baseline", []), ("fisher_drop6", DROP)):
            fn = lambda d=drop: llm_forward(d)
            for _ in range(WARMUP):
                fn()
            flops = _flops(fn)
            ms = _cuda_ms(fn, REPEATS)
            params = count_params(policy, drop)
            rows[name] = {
                "drop": drop,
                "params_language_stored_m": params["language_stored_params"] / 1e6,
                "params_language_active_m": params["language_active_params"] / 1e6,
                "flops_g": flops / 1e9,
                "inference_ms": ms,
            }

    b, s = rows["baseline"], rows["fisher_drop6"]
    def delta(a, x):
        return {"abs": s[x] - b[x], "pct": 100.0 * (s[x] / b[x] - 1.0)}

    payload = {
        "checkpoint": DEFAULT_CHECKPOINT,
        "scope": "language_prefix_only",
        "prefix_seq": int(prefix_embs.shape[1]),
        "skip": "dtr_appendix_b_keep_kv",
        "drop": DROP,
        "device": str(torch.cuda.get_device_name(0)),
        "warmup": WARMUP,
        "repeats": REPEATS,
        "baseline": b,
        "fisher_drop6": s,
        "delta": {
            "params_language_stored_m": delta(b, "params_language_stored_m"),
            "params_language_active_m": delta(b, "params_language_active_m"),
            "flops_g": delta(b, "flops_g"),
            "inference_ms": delta(b, "inference_ms"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

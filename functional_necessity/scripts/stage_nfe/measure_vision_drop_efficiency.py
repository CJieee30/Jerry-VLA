#!/usr/bin/env python3
"""π0.5 SigLIP identity-skip efficiency (vision Drop-2 / Drop-4).

Same protocol as measure_sota_efficiency.py: one RTX 3090 process, one calib
frame, CUDA events warmup=5 / repeats=20, FlopCounterMode. Does not overwrite
sota_efficiency_pi05.json.

Vision skip is whole-block identity (eval_pi05_vision_drop.install). Language
skip, when used, is DTR Appendix B keep-KV.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))

from common import (  # noqa: E402
    DEFAULT_CALIB_FILE,
    DEFAULT_CHECKPOINT,
    load_calib_data,
    load_policy,
    move_to_device,
    slice_batch,
)
from layer_skip import skip_llm_layers  # noqa: E402
from measure_module_efficiency import prepare_inputs  # noqa: E402

OUT = FN / "results" / "stage_nfe" / "vision_drop4_efficiency.json"
WARMUP = 5
REPEATS = 20
ROUNDS = 3
LANG_DROP = [1, 13, 14, 15, 16, 17]
# Closed-loop §1: full-model Fisher vis Drop-4, SR 97.50 vs 97.25.
VIS4_FULL = [21, 22, 20, 17]
# Drop-6 graph Fisher 7d k=4 / k=2.
VIS4_D6 = [20, 18, 21, 17]
VIS2_D6 = [20, 18]


def _nparams(mod) -> int:
    return int(sum(p.numel() for p in mod.parameters()))


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


def _identity(hidden_states, attention_mask=None, **kwargs):
    return hidden_states


@contextmanager
def skip_vision(layers, drop: list[int]):
    if not drop:
        yield
        return
    originals = []
    hit = []
    for i, layer in enumerate(layers):
        if i in drop:
            originals.append((layer, layer.forward))
            layer.forward = _identity
            hit.append(i)
    if hit != sorted(drop):
        raise RuntimeError(f"vision skip {hit} != {sorted(drop)}")
    try:
        yield
    finally:
        for layer, fwd in originals:
            layer.forward = fwd


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs))


def main() -> None:
    device = torch.device("cuda")
    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.n_action_steps = 10
    policy.config.compile_model = False
    policy.eval()
    calib = load_calib_data(DEFAULT_CALIB_FILE)
    batch = move_to_device(slice_batch(calib[0], 0, 1), device)
    model = policy.model
    pal = model.paligemma_with_expert
    vis_layers = list(pal.paligemma.model.vision_tower.vision_model.encoder.layers)
    vis = pal.paligemma.model.vision_tower
    proj = pal.paligemma.model.multi_modal_projector
    images, img_masks, tokens, masks, states, state_masks = prepare_inputs(policy, batch)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1000)
    noise = torch.randn(
        1, int(model.config.chunk_size), int(model.config.max_action_dim), generator=gen
    ).to(device)

    per_layer = [_nparams(layer) for layer in vis_layers]
    vis_stored = _nparams(vis) + _nparams(proj)

    def vision_only():
        for img in images:
            pal.embed_image(img)

    def chunk(steps: int, vis_drop: list[int], lang_drop: list[int]):
        policy.config.num_inference_steps = steps
        lang_ctx = skip_llm_layers(policy, lang_drop) if lang_drop else nullcontext()
        with skip_vision(vis_layers, vis_drop), lang_ctx:
            policy.reset()
            policy.predict_action_chunk(batch, noise=noise, num_steps=steps)

    vis_cfgs = {
        "full": [],
        "k2_d6": VIS2_D6,
        "k4_d6": VIS4_D6,
        "k4_fullmodel": VIS4_FULL,
    }
    vision = {}
    with torch.no_grad():
        for name, drop in vis_cfgs.items():
            with skip_vision(vis_layers, drop):
                vision[name] = {"ms": cuda_ms(vision_only), "flops_g": flops_g(vision_only)}
            print(f"vision {name}: {vision[name]['ms']:.2f} ms  {vision[name]['flops_g']:.2f} G", flush=True)

        chunk_specs = [
            ("full_nfe10", 10, [], []),
            ("vis4_fullmodel_nfe10", 10, VIS4_FULL, []),
            ("vis4_d6_nfe10", 10, VIS4_D6, []),
            ("full_nfe1", 1, [], []),
            ("vis4_fullmodel_nfe1", 1, VIS4_FULL, []),
            ("d6_nfe1", 1, [], LANG_DROP),
            ("d6_vis2_nfe1", 1, VIS2_D6, LANG_DROP),
            ("d6_vis4_nfe1", 1, VIS4_D6, LANG_DROP),
        ]
        chunk_flops = {}
        chunk_rounds: dict[str, list[float]] = {name: [] for name, *_ in chunk_specs}
        for name, steps, vis_drop, lang_drop in chunk_specs:
            chunk_flops[name] = flops_g(lambda s=steps, v=vis_drop, l=lang_drop: chunk(s, v, l))
            print(f"flops {name}: {chunk_flops[name]:.2f} G", flush=True)
        for rnd in range(ROUNDS):
            for name, steps, vis_drop, lang_drop in chunk_specs:
                ms = cuda_ms(lambda s=steps, v=vis_drop, l=lang_drop: chunk(s, v, l))
                chunk_rounds[name].append(ms)
                print(f"round {rnd} {name}: {ms:.2f} ms", flush=True)

    payload = {
        "checkpoint": DEFAULT_CHECKPOINT,
        "device": torch.cuda.get_device_name(0),
        "warmup": WARMUP,
        "repeats": REPEATS,
        "rounds": ROUNDS,
        "n_vision_layers": len(vis_layers),
        "n_views": len(images),
        "lang_drop": LANG_DROP,
        "vis2_d6": VIS2_D6,
        "vis4_d6": VIS4_D6,
        "vis4_fullmodel": VIS4_FULL,
        "operator": "siglip_identity_skip",
        "params_m": {
            "vision_stored": vis_stored / 1e6,
            "per_encoder_layer": per_layer[0] / 1e6,
            "vis2_skipped": 2 * per_layer[0] / 1e6,
            "vis4_skipped": 4 * per_layer[0] / 1e6,
            "vision_active_k4": (vis_stored - 4 * per_layer[0]) / 1e6,
        },
        "vision_module": vision,
        "chunk_flops_g": chunk_flops,
        "chunk_ms_rounds": chunk_rounds,
        "chunk_ms_mean": {k: mean(v) for k, v in chunk_rounds.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", OUT, flush=True)
    for k, v in payload["chunk_ms_mean"].items():
        print(f"  mean {k}: {v:.2f} ms  {chunk_flops[k]:.2f} G", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()

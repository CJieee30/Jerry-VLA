#!/usr/bin/env python3
"""Vision-layer redundancy on π0.5 Fisher Drop-6, then the LEDGE bracket.

Scoring: one backward per calibration frame. Language layers [1, 13–17] are
identity-skipped (Appendix B). SigLIP grads are weighted by the stored 6-d and
7-d CARE weights. Taylor is |sum α g ⊙ w| / N. Fisher squares the
α-weighted mean gradient inside each original batch of 8.

LEDGE: r(k) is the CARE-weighted L2 of the first 10 action steps after
identity-skipping the k lowest vision layers of that order. σ is the same
model (Drop-6, full vision) under two frozen noises. The bracket is
k_lo = max {k: r(k) ≤ σ}, k_hi = argmax Δr − 1. k itself is not emitted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

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

OUT = FN / "results" / "stage_nfe" / "vision_drop6_ledge.json"
WEIGHTS = FN / "results" / "stage2_action_weight" / "weights_pi05.pt"
LANG_DROP = [1, 13, 14, 15, 16, 17]
EXEC = 10


def vision_layers(policy):
    tower = policy.model.paligemma_with_expert.paligemma.model.vision_tower
    return list(tower.vision_model.encoder.layers)


def _identity(hidden_states, attention_mask=None, **kwargs):
    return hidden_states


def per_sample_loss(policy, batch) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = policy(batch, reduction="none")
    return loss


def score(policy, calib, w6, w7, device) -> dict:
    layers = vision_layers(policy)
    for param in policy.parameters():
        param.requires_grad_(False)
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad_(True)
    policy.eval()
    if hasattr(policy.model, "gradient_checkpointing_enable"):
        policy.model.gradient_checkpointing_enable()
    taylor = {"6": {}, "7": {}}
    fisher = {"6": [0.0] * len(layers), "7": [0.0] * len(layers)}
    n_samples = 0
    for batch_i, full in enumerate(calib):
        width = int(full["action"].shape[0])
        alphas = {"6": w6[batch_i].float(), "7": w7[batch_i].float()}
        acc = {"6": {}, "7": {}}
        stored = {}
        for start in range(width):
            batch = move_to_device(slice_batch(full, start, start + 1), device)
            policy.zero_grad(set_to_none=True)
            with skip_llm_layers(policy, LANG_DROP):
                loss = per_sample_loss(policy, batch).float().sum()
                loss.backward()
            for i, layer in enumerate(layers):
                for name, param in layer.named_parameters():
                    if param.grad is None:
                        continue
                    g = param.grad.detach().float().cpu()
                    keyp = (i, name)
                    if keyp not in stored:
                        stored[keyp] = param.detach().float().cpu()
                    gw = g * stored[keyp]
                    for key in ("6", "7"):
                        scale = float(alphas[key][start])
                        prod = gw * scale
                        bucket = taylor[key]
                        if keyp not in bucket:
                            bucket[keyp] = prod.clone()
                        else:
                            bucket[keyp] += prod
                        gab = acc[key]
                        if keyp not in gab:
                            gab[keyp] = g * scale
                        else:
                            gab[keyp] += g * scale
            n_samples += 1
            policy.zero_grad(set_to_none=True)
            del loss, batch
            if n_samples % 8 == 0:
                print(f"score frames {n_samples}", flush=True)
                torch.cuda.empty_cache()
        for key in ("6", "7"):
            denom = float(alphas[key].sum().clamp_min(1e-12))
            for i, layer in enumerate(layers):
                for name, _param in layer.named_parameters():
                    keyp = (i, name)
                    if keyp not in acc[key]:
                        continue
                    g_bar = acc[key][keyp] / denom
                    fisher[key][i] += float((g_bar.square() * stored[keyp].square()).sum())
        print(f"score batch {batch_i} closed", flush=True)
    out = {}
    for key in ("6", "7"):
        scores_t = []
        for i, layer in enumerate(layers):
            total = 0.0
            for name, _param in layer.named_parameters():
                keyp = (i, name)
                if keyp in taylor[key]:
                    total += float(taylor[key][keyp].abs().sum())
            scores_t.append(total / max(n_samples, 1))
        scores_f = [s / max(len(calib), 1) for s in fisher[key]]
        out[f"taylor{key}"] = scores_t
        out[f"fisher{key}"] = scores_f
    return out


def order_low(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: scores[i])


def predict(policy, batch, noise, vision_ids: list[int]) -> torch.Tensor:
    layers = vision_layers(policy)
    saved = [(layers[i], layers[i].forward) for i in vision_ids]
    for layer, _fwd in saved:
        layer.forward = _identity
    try:
        with skip_llm_layers(policy, LANG_DROP):
            policy.reset()
            with torch.no_grad():
                pred = policy.predict_action_chunk(batch, noise=noise, num_steps=10)
    finally:
        for layer, fwd in saved:
            layer.forward = fwd
    return pred[:, :EXEC, :7].float().cpu()


def make_noise(batch, seed: int, device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    bsize = int(batch["action"].shape[0])
    return torch.randn(bsize, 50, 32, generator=gen).to(device)


def ledge_curve(policy, calib, alpha: torch.Tensor, order: list[int], device) -> dict:
    """α is (512,) aligned with flattened calib. Two noises, all 64×8 frames."""
    policy.eval()
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    for param in policy.parameters():
        param.requires_grad_(False)
    refs = []
    for seed in (1000, 2000):
        chunks = []
        for batch_i, full in enumerate(calib):
            batch = move_to_device(full, device)
            noise = make_noise(full, seed + batch_i * 17, device)
            chunks.append(predict(policy, batch, noise, []))
            if batch_i == 0 and seed == 1000:
                print("drift reference batch 0", chunks[-1].shape, flush=True)
        refs.append(torch.cat(chunks, dim=0))
    # self-noise: same observations, two global noise streams rebuilt per batch
    # Rebuild with a single seed stream so the two refs differ only by noise.
    # The loop above used seed+batch offset, so both refs are valid paired noises.
    per = (refs[0] - refs[1]).flatten(1).norm(dim=-1)
    sigma = float((per * alpha).sum() / alpha.sum())
    rs = [0.0]
    for k in range(1, len(order) + 1):
        ids = order[:k]
        drifts = []
        for seed, ref in zip((1000, 2000), refs):
            chunks = []
            for batch_i, full in enumerate(calib):
                batch = move_to_device(full, device)
                noise = make_noise(full, seed + batch_i * 17, device)
                chunks.append(predict(policy, batch, noise, ids))
            pred = torch.cat(chunks, dim=0)
            drifts.append((pred - ref).flatten(1).norm(dim=-1))
        frames = torch.stack(drifts).mean(0)
        r = float((frames * alpha).sum() / alpha.sum())
        rs.append(r)
        print(f"k={k} drop={ids} r={r:.4f} sigma={sigma:.4f}", flush=True)
    deltas = [rs[k] - rs[k - 1] for k in range(1, len(rs))]
    jump = max(range(len(deltas)), key=lambda i: deltas[i]) + 1
    below = [k for k in range(len(rs)) if rs[k] <= sigma]
    return {
        "sigma": sigma,
        "r": rs,
        "k_lo": max(below) if below else 0,
        "k_hi": jump - 1,
        "jump_k": jump,
        "jump_layer": order[jump - 1],
    }


def main() -> None:
    device = torch.device("cuda")
    blob = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    w6 = blob["weights6_batched"]
    w7 = blob["weights7_batched"]
    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.n_action_steps = EXEC
    policy.config.num_inference_steps = 10
    policy.config.compile_model = False
    policy.eval()
    calib = load_calib_data(DEFAULT_CALIB_FILE)
    print(f"layers {len(vision_layers(policy))} batches {len(calib)}", flush=True)
    scored = score(policy, calib, w6, w7, device)
    orders = {name: order_low(scores) for name, scores in scored.items()}
    for name, order in orders.items():
        print(name, "low-to-high", order, flush=True)
    payload = {"language_drop": LANG_DROP, "scores": scored, "orders_low_to_high": orders, "curves": {}}
    OUT.write_text(json.dumps(payload) + "\n")
    print("scores written", OUT, flush=True)
    curves = payload["curves"]
    for name, order in orders.items():
        alpha = (w7 if name.endswith("7") else w6).float().reshape(-1)
        print(f"ledge {name}", flush=True)
        curves[name] = {"order": order, **ledge_curve(policy, calib, alpha, order, device)}
        print(
            f"bracket {name}: k_lo={curves[name]['k_lo']} k_hi={curves[name]['k_hi']}",
            flush=True,
        )
        # checkpoint after each curve
        payload["curves"] = curves
        OUT.write_text(json.dumps(payload) + "\n")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

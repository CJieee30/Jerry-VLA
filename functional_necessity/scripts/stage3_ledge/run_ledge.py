#!/usr/bin/env python3
"""LEDGE offline curves: CARE-weighted action drift along the CARE redundancy order.

r(k) is the CARE-weighted L2 between the executed 10-step action of the
cumulative drop set and the full model. The same frozen inference noise is
reused at every k. Delta L is recorded on the frozen flow-matching pair and
is not used to pick k.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
sys.path.insert(0, str(FN / "scripts" / "smolvla_libero"))
sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))

import importlib.util

import torch

OUT = FN / "results" / "stage3_ledge"
AW = FN / "results" / "stage2_action_weight"
EXEC_STEPS = 10
DENOISE_STEPS = 10


_MODULES: dict[str, object] = {}


def _load(name: str, path: Path):
    key = str(path)
    cached = _MODULES.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _MODULES[key] = module
    return module


def redundancy_order(scores) -> list[int]:
    values = [float(x) for x in scores]
    keep = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return list(reversed(keep))


def drop_prefix(order: list[int], k: int) -> list[int]:
    seq = [int(i) for i in order if int(i) != 0]
    return sorted(seq[:k])


def entering_layer(order: list[int], k: int) -> int | None:
    seq = [int(i) for i in order if int(i) != 0]
    if k <= 0 or k > len(seq):
        return None
    return int(seq[k - 1])


def decide(r: list[float]) -> dict:
    deltas = [float(r[k] - r[k - 1]) for k in range(1, len(r))]
    if not deltas:
        return {"k_cliff": None, "k_safe": None, "has_cliff": False, "reason": "empty"}
    k_cliff = 1 + max(range(len(deltas)), key=lambda i: deltas[i])
    jump = float(deltas[k_cliff - 1])
    prefix = deltas[: k_cliff - 1]
    median = float(statistics.median(prefix)) if prefix else None
    has_cliff = median is not None and jump >= 3.0 * median and jump > 0.0
    return {
        "k_cliff": int(k_cliff),
        "k_safe": int(k_cliff - 1) if has_cliff else None,
        "has_cliff": bool(has_cliff),
        "jump": jump,
        "plateau_median": median,
        "threshold": None if median is None else 3.0 * median,
        "deltas": deltas,
        "entering": None,
    }


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    w = weights.float()
    v = values.float().reshape(-1)
    if v.numel() != w.numel():
        raise RuntimeError(f"values {v.numel()} != weights {w.numel()}")
    return float((v * w).sum().item() / w.sum().item())


def action_l2(pred: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    a = pred[:, :EXEC_STEPS].float()
    b = base[:, :EXEC_STEPS].float()
    n = min(a.shape[-1], b.shape[-1])
    return (a[..., :n] - b[..., :n]).reshape(a.shape[0], -1).norm(dim=-1)


def save_curve(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_order(stack: str) -> tuple[list[int], torch.Tensor]:
    weights = torch.load(AW / f"weights_{stack}.pt", map_location="cpu", weights_only=False)
    alpha = weights["weights7"].float().reshape(-1)
    if stack == "pi05":
        blob = torch.load(AW / "pi05_fisher_action_discontinuity.pt", map_location="cpu", weights_only=False)
        scores = blob["paligemma"]
    elif stack == "molmoact2":
        blob = torch.load(AW / "molmoact2_taylor_action_discontinuity.pt", map_location="cpu", weights_only=False)
        scores = blob["scores"]
    elif stack == "smolvla":
        blob = torch.load(AW / "smolvla_taylor_action_discontinuity.pt", map_location="cpu", weights_only=False)
        scores = blob["scores"]
    else:
        raise SystemExit(stack)
    return redundancy_order(scores), alpha


def run_pi05(device: torch.device) -> None:
    common = _load("pi05_common", FN / "scripts" / "pi05_libero" / "common.py")
    layer_skip = _load("pi05_layer_skip", FN / "scripts" / "pi05_libero" / "layer_skip.py")
    skip_llm_layers = layer_skip.skip_llm_layers
    FlowMatchingReplay = common.FlowMatchingReplay
    load_calib_data = common.load_calib_data
    load_policy = common.load_policy
    move_to_device = common.move_to_device

    order, alpha = load_order("pi05")
    calib = load_calib_data(FN / "results" / "calib" / "pi05_v044_libero_64x8_seed42.pt")
    policy = load_policy("lerobot/pi05_libero_finetuned_v044", device)
    policy.config.n_action_steps = EXEC_STEPS
    policy.config.num_inference_steps = DENOISE_STEPS
    policy.config.compile_model = False
    policy.eval()
    n = int(alpha.numel())
    if n != 512:
        raise SystemExit(f"expected 512 pi05 weights, got {n}")
    k_max = sum(1 for i in order if i != 0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1000)
    noises = []
    for batch in calib:
        bsz = int(batch["action"].shape[0])
        noises.append(
            torch.randn(
                bsz,
                int(policy.config.chunk_size),
                int(policy.config.max_action_dim),
                generator=gen,
                dtype=torch.float32,
            )
        )
    replay = FlowMatchingReplay(policy)
    replay.capture(calib, device, microbatch_size=None)
    out = OUT / "pi05_fisher.json"
    payload = _fresh("pi05", "fisher", order, alpha, k_max)
    base_actions: list[torch.Tensor] = []
    base_losses: list[torch.Tensor] = []
    _pi05_pass(
        policy,
        calib,
        noises,
        alpha,
        replay,
        device,
        [],
        [],
        [],
        True,
        base_actions=base_actions,
        base_losses=base_losses,
    )
    drift0 = action_l2(base_actions[0], base_actions[0])
    if float(drift0.max()) > 1e-5:
        raise SystemExit("pi05 self-drift failed")
    # second call checked inside by comparing stored base to itself only.
    # Verify replayed inference is deterministic on batch 0.
    again = _pi05_actions(policy, calib[0], noises[0], device, [])
    if float(action_l2(again, base_actions[0]).max()) > 1e-4:
        raise SystemExit(f"pi05 noise replay drift {float(action_l2(again, base_actions[0]).max())}")
    payload["r"].append(0.0)
    payload["loss"].append(0.0)
    payload["drops"].append([])
    save_curve(out, payload)
    print("pi05 baseline ok", flush=True)
    for k in range(1, k_max + 1):
        drop = drop_prefix(order, k)
        drifts, losses = [], []
        _pi05_pass(policy, calib, noises, alpha, replay, device, drop, drifts, losses, collect_base=False, base_actions=base_actions, base_losses=base_losses)
        r = weighted_mean(torch.cat(drifts), alpha)
        dloss = weighted_mean(torch.cat(losses), alpha)
        payload["r"].append(r)
        payload["loss"].append(dloss)
        payload["drops"].append(drop)
        decision = decide(payload["r"])
        decision["entering"] = entering_layer(order, decision["k_cliff"]) if decision["k_cliff"] else None
        payload["decision_so_far"] = decision
        save_curve(out, payload)
        print(f"pi05 k={k} drop={drop} r={r:.6f} dL={dloss:.6f}", flush=True)
    payload["decision"] = payload["decision_so_far"]
    save_curve(out, payload)
    print(json.dumps(payload["decision"], indent=2), flush=True)


def _pi05_actions(policy, batch, noise, device, drop):
    common = _load("pi05_common_actions", FN / "scripts" / "pi05_libero" / "common.py")
    layer_skip = _load("pi05_layer_skip_actions", FN / "scripts" / "pi05_libero" / "layer_skip.py")
    move_to_device = common.move_to_device
    skip_llm_layers = layer_skip.skip_llm_layers

    batch = move_to_device(batch, device)
    noise = noise.to(device)
    policy.reset()
    ctx = skip_llm_layers(policy, drop) if drop else nullcontext()
    with torch.no_grad(), ctx:
        return policy.predict_action_chunk(batch, noise=noise).detach().float().cpu()


def _pi05_pass(policy, calib, noises, alpha, replay, device, drop, drifts, losses, collect_base, base_actions=None, base_losses=None):
    layer_skip = _load("pi05_layer_skip_pass", FN / "scripts" / "pi05_libero" / "layer_skip.py")
    common = _load("pi05_common_pass", FN / "scripts" / "pi05_libero" / "common.py")
    skip_llm_layers = layer_skip.skip_llm_layers
    move_to_device = common.move_to_device

    ctx = skip_llm_layers(policy, drop) if drop else nullcontext()
    offset = 0
    with replay, ctx:
        for batch, noise in zip(calib, noises):
            batch = move_to_device(batch, device)
            bsz = int(batch["action"].shape[0])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                per, _ = policy(batch, reduction="none")
            replay.step()
            policy.reset()
            with torch.no_grad():
                pred = policy.predict_action_chunk(batch, noise=noise.to(device)).detach().float().cpu()
            if collect_base:
                base_actions.append(pred)
                base_losses.append(per.detach().float().cpu())
                drifts.append(torch.zeros(bsz))
                losses.append(torch.zeros(bsz))
            else:
                base = base_actions[len(drifts)]
                drifts.append(action_l2(pred, base))
                losses.append(per.detach().float().cpu() - base_losses[len(losses)])
            offset += bsz
            del batch
    if offset != int(alpha.numel()):
        raise SystemExit(f"pi05 frames {offset} != weights {int(alpha.numel())}")


def run_smolvla(device: torch.device) -> None:
    layer_skip = _load("smol_layer_skip", FN / "scripts" / "smolvla_libero" / "layer_skip.py")
    load_mod = _load("smol_load_policy", FN / "scripts" / "smolvla_libero" / "load_policy.py")
    score = _load("smol_score", FN / "scripts" / "smolvla_libero" / "score.py")
    skip_vlm_layers = layer_skip.skip_vlm_layers
    load_policy = load_mod.load_policy
    build_calib = score.build_calib

    order, alpha = load_order("smolvla")
    policy = load_policy(str(device))
    policy.config.n_action_steps = EXEC_STEPS
    if hasattr(policy.config, "num_steps"):
        policy.config.num_steps = DENOISE_STEPS
    samples = build_calib(policy)
    if len(samples) != int(alpha.numel()):
        raise SystemExit(f"smol n {len(samples)} != {int(alpha.numel())}")
    k_max = sum(1 for i in order if i != 0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1000)
    noises = [
        torch.randn(
            1,
            int(policy.config.chunk_size),
            int(policy.config.max_action_dim),
            generator=gen,
        )
        for _ in samples
    ]
    out = OUT / "smolvla_taylor.json"
    payload = _fresh("smolvla", "taylor", order, alpha, k_max)
    base_actions, base_losses = [], []
    for sample, noise in zip(samples, noises):
        pred, loss = _smol_one(policy, sample, noise, device, [])
        base_actions.append(pred)
        base_losses.append(loss)
    again, _ = _smol_one(policy, samples[0], noises[0], device, [])
    err = float(action_l2(again, base_actions[0]).max())
    if err > 1e-4:
        raise SystemExit(f"smol noise replay drift {err}")
    payload["r"].append(0.0)
    payload["loss"].append(0.0)
    payload["drops"].append([])
    save_curve(out, payload)
    print("smol baseline ok", flush=True)
    for k in range(1, k_max + 1):
        drop = drop_prefix(order, k)
        drifts, dlosses = [], []
        for i, (sample, noise) in enumerate(zip(samples, noises)):
            pred, loss = _smol_one(policy, sample, noise, device, drop)
            drifts.append(action_l2(pred, base_actions[i]))
            dlosses.append(loss - base_losses[i])
        r = weighted_mean(torch.cat(drifts), alpha)
        dloss = weighted_mean(torch.stack(dlosses), alpha)
        payload["r"].append(r)
        payload["loss"].append(dloss)
        payload["drops"].append(drop)
        decision = decide(payload["r"])
        decision["entering"] = entering_layer(order, decision["k_cliff"]) if decision["k_cliff"] else None
        payload["decision_so_far"] = decision
        save_curve(out, payload)
        print(f"smol k={k} drop={drop} r={r:.6f} dL={dloss:.6f}", flush=True)
    payload["decision"] = payload["decision_so_far"]
    save_curve(out, payload)
    print(json.dumps(payload["decision"], indent=2), flush=True)


def _smol_one(policy, sample, noise, device, drop):
    layer_skip = _load("smol_layer_skip_one", FN / "scripts" / "smolvla_libero" / "layer_skip.py")
    skip_vlm_layers = layer_skip.skip_vlm_layers

    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample["batch"].items()}
    policy.reset()
    with torch.no_grad(), skip_vlm_layers(drop):
        pred = policy.predict_action_chunk(batch, noise=noise.to(device)).detach().float().cpu()
        flow_noise = sample["noise"].to(device)
        time = sample["time"].to(device)
        loss, _ = policy.forward(batch, noise=flow_noise, time=time)
    return pred, loss.detach().float().cpu()


def run_molmo(device: torch.device) -> None:
    layer_skip = _load("molmo_layer_skip", FN / "scripts" / "molmoact2_libero" / "layer_skip.py")
    load_mod = _load("molmo_load_policy", FN / "scripts" / "molmoact2_libero" / "load_policy.py")
    score = _load("molmo_score", FN / "scripts" / "molmoact2_libero" / "score.py")
    skip_vlm_layers = layer_skip.skip_vlm_layers
    load_policy = load_mod.load_policy
    build_calib = score.build_calib

    order, alpha = load_order("molmoact2")
    policy = load_policy(str(device))
    policy.config.gradient_checkpointing = False
    policy.config.enable_inference_cuda_graph = False
    policy.config.n_action_steps = EXEC_STEPS
    if hasattr(policy.config, "num_inference_steps"):
        policy.config.num_inference_steps = DENOISE_STEPS
    policy.eval()
    samples = build_calib(policy)
    if len(samples) != int(alpha.numel()):
        raise SystemExit(f"molmo n {len(samples)} != {int(alpha.numel())}")
    k_max = sum(1 for i in order if i != 0)
    out = OUT / "molmoact2_taylor.json"
    payload = _fresh("molmoact2", "taylor", order, alpha, k_max)
    base_actions, base_losses = [], []
    for i, sample in enumerate(samples):
        pred, loss = _molmo_one(policy, sample, device, [], seed=10_000 + i)
        base_actions.append(pred)
        base_losses.append(loss)
    again, _ = _molmo_one(policy, samples[0], device, [], seed=10_000)
    err = float(action_l2(again, base_actions[0]).max())
    if err > 1e-3:
        raise SystemExit(f"molmo noise replay drift {err}")
    payload["r"].append(0.0)
    payload["loss"].append(0.0)
    payload["drops"].append([])
    save_curve(out, payload)
    print("molmo baseline ok", flush=True)
    for k in range(1, k_max + 1):
        drop = drop_prefix(order, k)
        drifts, dlosses = [], []
        for i, sample in enumerate(samples):
            pred, loss = _molmo_one(policy, sample, device, drop, seed=10_000 + i)
            drifts.append(action_l2(pred, base_actions[i]))
            dlosses.append(loss - base_losses[i])
        r = weighted_mean(torch.cat(drifts), alpha)
        dloss = weighted_mean(torch.stack(dlosses), alpha)
        payload["r"].append(r)
        payload["loss"].append(dloss)
        payload["drops"].append(drop)
        decision = decide(payload["r"])
        decision["entering"] = entering_layer(order, decision["k_cliff"]) if decision["k_cliff"] else None
        payload["decision_so_far"] = decision
        save_curve(out, payload)
        print(f"molmo k={k} drop={drop} r={r:.6f} dL={dloss:.6f}", flush=True)
        torch.cuda.empty_cache()
    payload["decision"] = payload["decision_so_far"]
    save_curve(out, payload)
    print(json.dumps(payload["decision"], indent=2), flush=True)


def _molmo_one(policy, sample, device, drop, seed: int):
    layer_skip = _load("molmo_layer_skip_one", FN / "scripts" / "molmoact2_libero" / "layer_skip.py")
    skip_vlm_layers = layer_skip.skip_vlm_layers

    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample["batch"].items()}
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    if hasattr(policy, "reset"):
        policy.reset()
    with torch.no_grad(), skip_vlm_layers(drop):
        pred = policy.predict_action_chunk(
            batch,
            generator=gen,
            num_steps=DENOISE_STEPS,
            inference_action_mode="continuous",
        ).detach().float().cpu()
        model_inputs = policy._model_inputs(batch)
        with policy._autocast_context():
            loss, _ = policy._compute_flow_matching_loss_joint_per_layer(
                batch=batch,
                model_inputs=model_inputs,
                timesteps=sample["timesteps"].to(device),
                noise=sample["noise"].to(device),
            )
    return pred, loss.detach().float().cpu().reshape(())


def _fresh(stack, method, order, alpha, k_max) -> dict:
    return {
        "method_name": "LEDGE",
        "stack": stack,
        "rank_method": method,
        "order_excl_l0": [int(i) for i in order if int(i) != 0],
        "k_max": int(k_max),
        "exec_steps": EXEC_STEPS,
        "denoise_steps": DENOISE_STEPS,
        "r": [],
        "loss": [],
        "drops": [],
    }


def nullcontext():
    from contextlib import nullcontext as _nc

    return _nc()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", choices=("pi05", "smolvla", "molmoact2"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.stack == "pi05":
        run_pi05(device)
    elif args.stack == "smolvla":
        run_smolvla(device)
    else:
        run_molmo(device)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialized pruning: action check, peak memory, deployment latency.

Does not overwrite the eager efficiency JSON. TF32 stays off. compile_model
stays false; the deployment column is torch.compile(mode="reduce-overhead")
applied after the dropped layers are real modules.
"""

from __future__ import annotations

import gc
import json
import sys
import traceback
from pathlib import Path

import torch

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
OUT = FN / "results" / "stage_nfe"
WARMUP = 5
REPEATS = 20
PI05_DROP = [1, 13, 14, 15, 16, 17]
MOLMO_DROP = [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
SMOL_DROP = [12, 13, 14, 15]

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _mib(n: int) -> float:
    return round(n / (1024 ** 2), 1)


def _sync_alloc() -> dict:
    torch.cuda.synchronize()
    return {
        "allocated_mib": _mib(torch.cuda.memory_allocated()),
        "reserved_mib": _mib(torch.cuda.memory_reserved()),
    }


def _peak_of(fn) -> dict:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return {
        "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_mib": _mib(torch.cuda.max_memory_reserved()),
    }


def _time_ms(fn) -> float:
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


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).detach().float().abs().max().item())


def _release(obj) -> None:
    del obj
    gc.collect()
    torch.cuda.empty_cache()


def _try_compile_time(fn, reference: torch.Tensor) -> dict:
    """reduce-overhead replays a CUDA graph. Clone its output before the next replay."""
    try:
        compiled = torch.compile(fn, mode="reduce-overhead")

        def once():
            torch.compiler.cudagraph_mark_step_begin()
            return compiled().clone()

        once()
        action = once()
        diff = _max_abs(action, reference)

        def step():
            torch.compiler.cudagraph_mark_step_begin()
            compiled()

        return {"ok": True, "ms": _time_ms(step), "action_max_abs": diff}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000], "trace": traceback.format_exc()[-2000:]}


def _dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", path, flush=True)


def _pi05_batch(policy, device):
    sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
    from common import DEFAULT_CALIB_FILE, load_calib_data, move_to_device, slice_batch

    calib = load_calib_data(DEFAULT_CALIB_FILE)
    batch = move_to_device(slice_batch(calib[0], 0, 1), device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1000)
    noise = torch.randn(
        1, int(policy.config.chunk_size), int(policy.config.max_action_dim), generator=gen
    ).to(device)
    return batch, noise


def _run_pi05(device) -> dict:
    sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
    sys.path.insert(0, str(FN / "scripts" / "stage_nfe"))
    from common import DEFAULT_CHECKPOINT, load_policy
    from layer_skip import skip_llm_layers
    from materialize import materialize_pi05

    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.compile_model = False
    policy.config.n_action_steps = 10
    policy.eval()
    batch, noise = _pi05_batch(policy, device)

    def act(steps, drop):
        ctx = skip_llm_layers(policy, drop) if drop else nullcontext()
        with torch.no_grad(), ctx:
            policy.reset()
            return policy.predict_action_chunk(batch, noise=noise.clone(), num_steps=steps)

    from contextlib import nullcontext

    out = {"stack": "pi05", "drop": PI05_DROP, "tf32": False}
    out["full_load"] = _sync_alloc()
    full_ref = act(1, []).cpu()
    out["eager_full"] = {
        "nfe10_ms": _time_ms(lambda: act(10, [])),
        "nfe1_ms": _time_ms(lambda: act(1, [])),
        "peak": _peak_of(lambda: act(1, [])),
    }
    _dump(OUT / "deployment_pi05.json", out)
    out["compile_full"] = _try_compile_time(lambda: act(1, []), full_ref.to(device))
    _dump(OUT / "deployment_pi05.json", out)
    _release(policy)

    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.compile_model = False
    policy.config.n_action_steps = 10
    policy.eval()
    batch, noise = _pi05_batch(policy, device)

    def act2(steps):
        with torch.no_grad():
            policy.reset()
            return policy.predict_action_chunk(batch, noise=noise.clone(), num_steps=steps)

    with torch.no_grad(), skip_llm_layers(policy, PI05_DROP):
        policy.reset()
        hooked1 = policy.predict_action_chunk(batch, noise=noise.clone(), num_steps=1)
        policy.reset()
        hooked10 = policy.predict_action_chunk(batch, noise=noise.clone(), num_steps=10)
    info = materialize_pi05(policy, PI05_DROP)
    mat = act2(1)
    diff1 = _max_abs(mat, hooked1)
    diff10 = _max_abs(act2(10), hooked10)
    out["materialize"] = {**info, "freed_mib": round(info["freed"] * 2 / (1024 ** 2), 1)}
    out["action_vs_kvonly"] = {"nfe1": diff1, "nfe10": diff10}
    out["pruned_load"] = _sync_alloc()
    if diff1 > 1e-4 or diff10 > 1e-4:
        out["stopped"] = "materialized actions differ from K/V-only eager"
        _dump(OUT / "deployment_pi05.json", out)
        raise SystemExit(out["stopped"])
    out["eager_pruned"] = {
        "nfe10_ms": _time_ms(lambda: act2(10)),
        "nfe1_ms": _time_ms(lambda: act2(1)),
        "peak": _peak_of(lambda: act2(1)),
    }
    _dump(OUT / "deployment_pi05.json", out)
    out["compile_pruned"] = _try_compile_time(lambda: act2(1), mat)
    _dump(OUT / "deployment_pi05.json", out)
    return out


def _disable_molmo_overhead(policy) -> None:
    policy.config.compile_model = False
    policy.config.gradient_checkpointing = False
    policy.config.enable_inference_cuda_graph = False
    backbone = policy._backbone()
    backbone.gradient_checkpointing = False
    if getattr(backbone, "transformer", None) is not None:
        backbone.transformer.gradient_checkpointing = False
    policy._set_inference_cuda_graph_enabled(False)


def _run_molmo(device) -> dict:
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    sys.path.insert(0, str(FN / "scripts" / "stage_nfe"))
    from layer_skip import skip_vlm_layers
    from load_policy import CALIB_PATH, load_policy
    from materialize import materialize_molmo

    def prep(policy):
        _disable_molmo_overhead(policy)
        policy.eval()
        blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
        sample = blob["samples"][0]
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample["batch"].items()}
        return batch

    def generator():
        gen = torch.Generator(device=device)
        gen.manual_seed(1000)
        return gen

    policy = load_policy(str(device))
    batch = prep(policy)

    def act(steps, drop):
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with torch.no_grad(), ctx, policy._autocast_context():
            policy.reset()
            return policy.predict_action_chunk(batch, num_steps=steps, generator=generator())

    from contextlib import nullcontext

    out = {"stack": "molmoact2", "drop": MOLMO_DROP, "tf32": False}
    out["full_load"] = _sync_alloc()
    full_ref = act(1, []).cpu()
    out["eager_full"] = {
        "nfe10_ms": _time_ms(lambda: act(10, [])),
        "nfe1_ms": _time_ms(lambda: act(1, [])),
        "peak": _peak_of(lambda: act(1, [])),
    }
    _dump(OUT / "deployment_molmoact2.json", out)
    out["compile_full"] = _try_compile_time(lambda: act(1, []), full_ref.to(device))
    policy.config.enable_inference_cuda_graph = True
    policy._set_inference_cuda_graph_enabled(True)
    try:
        out["expert_graph_full"] = {
            "nfe1_ms": _time_ms(lambda: act(1, [])),
            "nfe10_ms": _time_ms(lambda: act(10, [])),
            "peak": _peak_of(lambda: act(1, [])),
            "action_max_abs": _max_abs(act(1, []), full_ref.to(device)),
        }
    except Exception as exc:
        out["expert_graph_full"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
    _dump(OUT / "deployment_molmoact2.json", out)
    _release(policy)

    policy = load_policy(str(device))
    batch = prep(policy)

    def act_pruned(steps):
        with torch.no_grad(), policy._autocast_context():
            policy.reset()
            return policy.predict_action_chunk(batch, num_steps=steps, generator=generator())

    with torch.no_grad(), skip_vlm_layers(MOLMO_DROP), policy._autocast_context():
        policy.reset()
        hooked = policy.predict_action_chunk(batch, num_steps=1, generator=generator())
    info = materialize_molmo(policy, MOLMO_DROP)
    mat = act_pruned(1)
    diff = _max_abs(mat, hooked)
    out["materialize"] = {**info, "freed_mib": round(info["freed"] * 2 / (1024 ** 2), 1)}
    out["action_vs_kvonly"] = {"nfe1": diff}
    out["pruned_load"] = _sync_alloc()
    if diff > 1e-4:
        out["stopped"] = "materialized actions differ from K/V-only eager"
        _dump(OUT / "deployment_molmoact2.json", out)
        raise SystemExit(out["stopped"])
    out["eager_pruned"] = {
        "nfe10_ms": _time_ms(lambda: act_pruned(10)),
        "nfe1_ms": _time_ms(lambda: act_pruned(1)),
        "peak": _peak_of(lambda: act_pruned(1)),
    }
    _dump(OUT / "deployment_molmoact2.json", out)
    out["compile_pruned"] = _try_compile_time(lambda: act_pruned(1), mat)
    policy.config.enable_inference_cuda_graph = True
    policy._set_inference_cuda_graph_enabled(True)
    try:
        out["expert_graph_pruned"] = {
            "nfe1_ms": _time_ms(lambda: act_pruned(1)),
            "nfe10_ms": _time_ms(lambda: act_pruned(10)),
            "peak": _peak_of(lambda: act_pruned(1)),
            "action_max_abs": _max_abs(act_pruned(1), mat),
        }
    except Exception as exc:
        out["expert_graph_pruned"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
    _dump(OUT / "deployment_molmoact2.json", out)
    return out


def _run_smol(device) -> dict:
    sys.path.insert(0, str(FN / "scripts" / "smolvla_libero"))
    sys.path.insert(0, str(FN / "scripts" / "stage_nfe"))
    from layer_skip import skip_vlm_layers
    from load_policy import CALIB_PATH, load_policy
    from materialize import materialize_smol
    from score import _ensure_policy_batch

    def make_policy():
        policy = load_policy(str(device))
        policy.config.compile_model = False
        policy.config.n_action_steps = 10
        policy.eval()
        blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
        raw = _ensure_policy_batch(blob["samples"][0]["batch"])
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
        gen = torch.Generator(device="cpu")
        gen.manual_seed(1000)
        noise = torch.randn(
            batch["observation.state"].shape[0],
            int(policy.config.chunk_size),
            int(policy.config.max_action_dim),
            generator=gen,
        ).to(device)
        return policy, batch, noise

    from contextlib import nullcontext

    policy, batch, noise = make_policy()

    def act(steps, drop):
        policy.config.num_steps = steps
        ctx = skip_vlm_layers(drop) if drop else nullcontext()
        with torch.no_grad(), ctx:
            policy.reset()
            return policy.predict_action_chunk(batch, noise=noise.clone())

    out = {"stack": "smolvla", "drop": SMOL_DROP, "tf32": False}
    out["full_load"] = _sync_alloc()
    full_ref = act(2, []).cpu()
    out["eager_full"] = {
        "nfe10_ms": _time_ms(lambda: act(10, [])),
        "nfe2_ms": _time_ms(lambda: act(2, [])),
        "peak": _peak_of(lambda: act(2, [])),
    }
    _dump(OUT / "deployment_smolvla.json", out)
    out["compile_full"] = _try_compile_time(lambda: act(2, []), full_ref.to(device))
    _dump(OUT / "deployment_smolvla.json", out)
    _release(policy)

    policy, batch, noise = make_policy()

    def act_pruned(steps):
        policy.config.num_steps = steps
        with torch.no_grad():
            policy.reset()
            return policy.predict_action_chunk(batch, noise=noise.clone())

    with torch.no_grad(), skip_vlm_layers(SMOL_DROP):
        policy.config.num_steps = 2
        policy.reset()
        hooked = policy.predict_action_chunk(batch, noise=noise.clone())
    info = materialize_smol(policy, SMOL_DROP)
    mat = act_pruned(2)
    diff = _max_abs(mat, hooked)
    out["materialize"] = {**info, "freed_mib": round(info["freed"] * 2 / (1024 ** 2), 1)}
    out["action_vs_kvonly"] = {"nfe2": diff}
    out["pruned_load"] = _sync_alloc()
    if diff > 1e-4:
        out["stopped"] = "materialized actions differ from K/V-only eager"
        _dump(OUT / "deployment_smolvla.json", out)
        raise SystemExit(out["stopped"])
    out["eager_pruned"] = {
        "nfe10_ms": _time_ms(lambda: act_pruned(10)),
        "nfe2_ms": _time_ms(lambda: act_pruned(2)),
        "peak": _peak_of(lambda: act_pruned(2)),
    }
    _dump(OUT / "deployment_smolvla.json", out)
    out["compile_pruned"] = _try_compile_time(lambda: act_pruned(2), mat)
    _dump(OUT / "deployment_smolvla.json", out)
    return out


def _run_molmo_pruned(device) -> dict:
    """Second process: the full-model compile leaves a CUDA graph that blocks a reload."""
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    sys.path.insert(0, str(FN / "scripts" / "stage_nfe"))
    from layer_skip import skip_vlm_layers
    from load_policy import CALIB_PATH, load_policy
    from materialize import materialize_molmo

    path = OUT / "deployment_molmoact2.json"
    out = json.loads(path.read_text())
    policy = load_policy(str(device))
    _disable_molmo_overhead(policy)
    policy.eval()
    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    batch = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in blob["samples"][0]["batch"].items()
    }

    def generator():
        gen = torch.Generator(device=device)
        gen.manual_seed(1000)
        return gen

    def act_pruned(steps):
        with torch.no_grad(), policy._autocast_context():
            policy.reset()
            return policy.predict_action_chunk(batch, num_steps=steps, generator=generator())

    with torch.no_grad(), skip_vlm_layers(MOLMO_DROP), policy._autocast_context():
        policy.reset()
        hooked = policy.predict_action_chunk(batch, num_steps=1, generator=generator())
    info = materialize_molmo(policy, MOLMO_DROP)
    mat = act_pruned(1)
    diff = _max_abs(mat, hooked)
    out["materialize"] = {**info, "freed_mib": round(info["freed"] * 2 / (1024 ** 2), 1)}
    out["action_vs_kvonly"] = {"nfe1": diff}
    out["pruned_load"] = _sync_alloc()
    _dump(path, out)
    if diff > 1e-4:
        out["stopped"] = "materialized actions differ from K/V-only eager"
        _dump(path, out)
        raise SystemExit(out["stopped"])
    out["eager_pruned"] = {
        "nfe10_ms": _time_ms(lambda: act_pruned(10)),
        "nfe1_ms": _time_ms(lambda: act_pruned(1)),
        "peak": _peak_of(lambda: act_pruned(1)),
    }
    _dump(path, out)
    out["compile_pruned"] = _try_compile_time(lambda: act_pruned(1), mat)
    policy.config.enable_inference_cuda_graph = True
    policy._set_inference_cuda_graph_enabled(True)
    try:
        out["expert_graph_pruned"] = {
            "nfe1_ms": _time_ms(lambda: act_pruned(1)),
            "nfe10_ms": _time_ms(lambda: act_pruned(10)),
            "peak": _peak_of(lambda: act_pruned(1)),
            "action_max_abs": _max_abs(act_pruned(1), mat),
        }
    except Exception as exc:
        out["expert_graph_pruned"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
    _dump(path, out)
    return out


def main() -> None:
    stack = sys.argv[1]
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    runners = {
        "pi05": _run_pi05,
        "molmoact2": _run_molmo,
        "molmoact2-pruned": _run_molmo_pruned,
        "smolvla": _run_smol,
    }
    if stack not in runners:
        raise SystemExit("usage: measure_deployment.py pi05|molmoact2|molmoact2-pruned|smolvla")
    runners[stack](device)


if __name__ == "__main__":
    main()

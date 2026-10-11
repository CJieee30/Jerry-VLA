#!/usr/bin/env python3
"""Last VLM layer keep-KV skip vs full-model action chunk on 32 calib frames.

Paper operator (keep-KV identity residual). NFE=1 and NFE=10. Not a closed-loop
eval. Action expert is never dropped.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import torch

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
OUT_DIR = FN / "results" / "stage1_recovery_free"
OUT_JSON = OUT_DIR / "last_layer_identity.json"
OUT_MD = OUT_DIR / "last_layer_identity.md"
N_FRAMES = 32
NFES = (1, 10)
ATOL = 1e-4


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm().item())


def _stats(rows: list[dict]) -> dict:
    max_abs = [r["max_abs"] for r in rows]
    l2 = [r["l2"] for r in rows]
    return {
        "n": len(rows),
        "mean_max_abs": sum(max_abs) / len(max_abs),
        "max_max_abs": max(max_abs),
        "mean_l2": sum(l2) / len(l2),
        "max_l2": max(l2),
        "identity_atol": ATOL,
        "all_identity": all(v <= ATOL for v in max_abs),
    }


def _summarize(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Last-layer open-loop identity",
        "",
        "Keep-KV skip of the last VLM layer vs the full-model action chunk.",
        f"{N_FRAMES} calibration frames, NFE=1 and NFE=10, seed offset 1000.",
        "Not a LIBERO eval. Action expert is not dropped.",
        "",
        "| stack | last layer | NFE | mean max-abs | max max-abs | identity (≤1e-4) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, block in payload.get("stacks", {}).items():
        last = block.get("last_layer")
        for nfe, stats in block.get("nfe", {}).items():
            ident = "yes" if stats.get("all_identity") else "no"
            lines.append(
                f"| {name} | {last} | {nfe} | {stats['mean_max_abs']:.3e} | "
                f"{stats['max_max_abs']:.3e} | {ident} |"
            )
    lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n")


def _pi05_frames(n: int):
    sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
    from common import DEFAULT_CALIB_FILE, load_calib_data, slice_batch

    calib = load_calib_data(DEFAULT_CALIB_FILE)
    frames = []
    for batch in calib:
        if len(frames) >= n:
            break
        frames.append(slice_batch(batch, 0, 1))
    if len(frames) < n:
        raise RuntimeError(f"pi05 calib only yielded {len(frames)} frames")
    return frames[:n]


def probe_pi05(device: torch.device, payload: dict) -> None:
    sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
    from common import DEFAULT_CHECKPOINT, load_policy, move_to_device
    from layer_skip import skip_llm_layers

    frames = _pi05_frames(N_FRAMES)
    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.compile_model = False
    last = len(policy.model._paligemma_language_model_core().layers) - 1
    chunk = policy.config.chunk_size
    adim = policy.config.max_action_dim
    k_calls = {"full": 0, "drop": 0}

    def hook(_mod, _inp, _out, bucket=k_calls, key="full"):
        bucket[key] += 1

    k_proj = policy.model._paligemma_language_model_core().layers[last].self_attn.k_proj
    nfe_out = {}
    for nfe in NFES:
        policy.config.num_inference_steps = int(nfe)
        rows = []
        for i, raw in enumerate(frames):
            batch = move_to_device(raw, device)
            gen = torch.Generator(device=device)
            gen.manual_seed(1000 + i * 100 + nfe)
            noise = torch.randn(1, chunk, adim, device=device, dtype=torch.float32, generator=gen)
            policy.reset()
            h_full = k_proj.register_forward_hook(lambda m, inp, out: hook(m, inp, out, k_calls, "full"))
            with torch.no_grad():
                full = policy.predict_action_chunk(batch, noise=noise.clone(), num_steps=int(nfe))
            h_full.remove()
            policy.reset()
            h_drop = k_proj.register_forward_hook(lambda m, inp, out: hook(m, inp, out, k_calls, "drop"))
            with torch.no_grad(), skip_llm_layers(policy, [last], keep_kv=True):
                dropped = policy.predict_action_chunk(batch, noise=noise.clone(), num_steps=int(nfe))
            h_drop.remove()
            rows.append({"i": i, "max_abs": _max_abs(full, dropped), "l2": _l2(full, dropped)})
            print(
                f"pi05 L{last} nfe={nfe} frame={i} max_abs={rows[-1]['max_abs']:.3e} l2={rows[-1]['l2']:.3e}",
                flush=True,
            )
        nfe_out[str(nfe)] = _stats(rows)
        nfe_out[str(nfe)]["frames"] = rows
    payload["stacks"]["pi05"] = {
        "last_layer": last,
        "operator": "dtr_appendix_b_prefix_kv",
        "k_proj_calls": k_calls,
        "nfe": nfe_out,
    }
    del policy
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _summarize(payload)


def probe_molmo(device: torch.device, payload: dict) -> None:
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    from layer_skip import skip_vlm_layers
    from load_policy import CALIB_PATH, load_policy

    policy = load_policy(str(device))
    policy.eval()
    policy.config.compile_model = False
    policy.config.enable_inference_cuda_graph = False
    backbone = policy._backbone()
    if getattr(backbone, "action_cuda_graph_manager", None) is not None:
        backbone.action_cuda_graph_manager = None
    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    samples = blob["samples"][:N_FRAMES]
    last = len(backbone.transformer.blocks) - 1
    nfe_out = {}
    for nfe in NFES:
        rows = []
        for i, sample in enumerate(samples):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in sample["batch"].items()
            }
            seed = 1000 + i * 100 + nfe

            def run(drop: bool):
                gen = torch.Generator(device=device)
                gen.manual_seed(seed)
                policy.reset()
                with torch.no_grad(), policy._autocast_context():
                    if drop:
                        with skip_vlm_layers([last]):
                            return policy.predict_action_chunk(
                                batch, num_steps=int(nfe), generator=gen
                            )
                    return policy.predict_action_chunk(batch, num_steps=int(nfe), generator=gen)

            full = run(False)
            dropped = run(True)
            rows.append({"i": i, "max_abs": _max_abs(full, dropped), "l2": _l2(full, dropped)})
            print(
                f"molmo L{last} nfe={nfe} frame={i} max_abs={rows[-1]['max_abs']:.3e} l2={rows[-1]['l2']:.3e}",
                flush=True,
            )
        nfe_out[str(nfe)] = _stats(rows)
        nfe_out[str(nfe)]["frames"] = rows
    payload["stacks"]["molmoact2"] = {
        "last_layer": last,
        "operator": "keep_kv_identity_residual",
        "nfe": nfe_out,
    }
    del policy
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _summarize(payload)


def probe_smol(device: torch.device, payload: dict) -> None:
    sys.path.insert(0, str(FN / "scripts" / "smolvla_libero"))
    from layer_skip import skip_vlm_layers
    from load_policy import CALIB_PATH, load_policy
    from score import _ensure_policy_batch

    policy = load_policy(str(device))
    policy.eval()
    policy.config.compile_model = False
    policy.config.n_action_steps = 10
    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    samples = blob["samples"][:N_FRAMES]
    last = int(policy.model.vlm_with_expert.num_vlm_layers) - 1
    chunk = int(policy.config.chunk_size)
    adim = int(policy.config.max_action_dim)
    nfe_out = {}
    for nfe in NFES:
        policy.config.num_steps = int(nfe)
        rows = []
        for i, sample in enumerate(samples):
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in _ensure_policy_batch(sample["batch"]).items()
            }
            gen = torch.Generator(device="cpu")
            gen.manual_seed(1000 + i * 100 + nfe)
            noise = torch.randn(batch["observation.state"].shape[0], chunk, adim, generator=gen).to(
                device
            )
            policy.reset()
            with torch.no_grad():
                full = policy.predict_action_chunk(batch, noise=noise.clone())
            policy.reset()
            with torch.no_grad(), skip_vlm_layers([last]):
                dropped = policy.predict_action_chunk(batch, noise=noise.clone())
            rows.append({"i": i, "max_abs": _max_abs(full, dropped), "l2": _l2(full, dropped)})
            print(
                f"smolvla L{last} nfe={nfe} frame={i} max_abs={rows[-1]['max_abs']:.3e} l2={rows[-1]['l2']:.3e}",
                flush=True,
            )
        nfe_out[str(nfe)] = _stats(rows)
        nfe_out[str(nfe)]["frames"] = rows
    payload["stacks"]["smolvla"] = {
        "last_layer": last,
        "operator": "keep_kv_identity_residual",
        "nfe": nfe_out,
    }
    del policy
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _summarize(payload)


def _purge_stack_imports() -> None:
    for name in list(sys.modules):
        if name in {"layer_skip", "load_policy", "score", "common", "adapt_lerobot_official"}:
            sys.modules.pop(name, None)


def main() -> None:
    stacks = sys.argv[1:] or ["pi05", "molmoact2", "smolvla"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = {
        "protocol": "last_vlm_layer_keep_kv_vs_full_action_chunk",
        "n_frames": N_FRAMES,
        "nfes": list(NFES),
        "atol": ATOL,
        "stacks": {},
    }
    if OUT_JSON.is_file():
        try:
            payload = json.loads(OUT_JSON.read_text())
            payload.setdefault("stacks", {})
        except json.JSONDecodeError:
            pass
    runners = {"pi05": probe_pi05, "molmoact2": probe_molmo, "smolvla": probe_smol}
    for name in stacks:
        if name in payload.get("stacks", {}) and payload["stacks"][name].get("nfe"):
            print(f"[skip] {name} already in {OUT_JSON}", flush=True)
            continue
        _purge_stack_imports()
        runners[name](device, payload)
        _purge_stack_imports()
    _summarize(payload)
    for name, block in payload["stacks"].items():
        for nfe, stats in block["nfe"].items():
            print(
                f"{name} L{block['last_layer']} nfe={nfe} "
                f"mean_max_abs={stats['mean_max_abs']:.3e} "
                f"max={stats['max_max_abs']:.3e} identity={stats['all_identity']}",
                flush=True,
            )


if __name__ == "__main__":
    main()

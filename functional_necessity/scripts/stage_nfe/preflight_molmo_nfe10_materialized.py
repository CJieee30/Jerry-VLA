#!/usr/bin/env python3
"""Molmo Taylor Drop-16: materialize before first forward; match K/V-only at NFE=10."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

FN = Path(__file__).resolve().parents[2]
MOLMO_DROP = [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
ATOL = 1e-4
NFE = 10


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    sys.path.insert(0, str(FN / "scripts" / "stage_nfe"))
    from contextlib import nullcontext

    import layer_skip
    from layer_skip import skip_vlm_layers
    from load_policy import CALIB_PATH, load_policy
    from materialize import materialize_molmo

    def prep(policy):
        policy.config.compile_model = False
        policy.config.enable_inference_cuda_graph = False
        policy._set_inference_cuda_graph_enabled(False)
        backbone = policy._backbone()
        if getattr(backbone, "action_cuda_graph_manager", None) is not None:
            backbone.action_cuda_graph_manager = None
        if getattr(backbone.transformer, "gradient_checkpointing", False):
            backbone.transformer.gradient_checkpointing = False
        policy.eval()
        blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in blob["samples"][0]["batch"].items()
        }
        return batch

    def generator():
        gen = torch.Generator(device=device)
        gen.manual_seed(1000)
        return gen

    layer_skip.install()
    layer_skip._DROP = set(MOLMO_DROP)

    policy = load_policy(str(device))
    batch = prep(policy)
    params_before = sum(p.numel() for p in policy.parameters())

    with torch.no_grad(), skip_vlm_layers(MOLMO_DROP), policy._autocast_context():
        policy.reset()
        ref = policy.predict_action_chunk(batch, num_steps=NFE, generator=generator()).cpu()

    info = materialize_molmo(policy, MOLMO_DROP)
    params_after = sum(p.numel() for p in policy.parameters())

    with torch.no_grad(), policy._autocast_context():
        policy.reset()
        mat = policy.predict_action_chunk(batch, num_steps=NFE, generator=generator()).cpu()

    diff = _max_abs(mat, ref)
    freed = params_before - params_after
    out = {
        "status": "pass" if diff <= ATOL and freed > 0 else "fail",
        "nfe": NFE,
        "drop": MOLMO_DROP,
        "action_max_abs_vs_kvonly": diff,
        "params_before": params_before,
        "params_after": params_after,
        "params_freed": freed,
        "materialize": info,
        "code_sha256": {
            "layer_skip.py": _sha(FN / "scripts" / "molmoact2_libero" / "layer_skip.py"),
            "materialize.py": _sha(FN / "scripts" / "stage_nfe" / "materialize.py"),
        },
        "device": str(device),
        "compile_model": False,
        "enable_inference_cuda_graph": False,
    }
    tag = os.environ.get("MOLMO_PREFLIGHT_OUT", "").strip()
    if tag:
        path = FN / "results" / "stage_nfe" / tag / "preflight_materialized.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {path}")
    print(json.dumps(out, indent=2))
    if out["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

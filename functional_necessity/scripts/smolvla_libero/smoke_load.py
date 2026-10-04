#!/usr/bin/env python3
"""Load HuggingFaceVLA/smolvla_libero and check VLM depth / n_action_steps / MuJoCo."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import mujoco
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CKPT = "HuggingFaceVLA/smolvla_libero"
OUT = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity/results/smolvla_libero")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {CKPT} on {device}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(CKPT).to(device).eval()
    vlm = policy.model.vlm_with_expert
    n_vlm = int(vlm.num_vlm_layers)
    n_expert = int(vlm.num_expert_layers)
    n_text = len(vlm.get_vlm_model().text_model.layers)
    cfg = policy.config
    info = {
        "checkpoint": CKPT,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "mujoco": mujoco.__version__,
        "n_action_steps": int(cfg.n_action_steps),
        "chunk_size": int(cfg.chunk_size),
        "num_steps": int(cfg.num_steps),
        "num_vlm_layers_config": int(cfg.num_vlm_layers),
        "num_vlm_layers_runtime": n_vlm,
        "text_model_n_layers": n_text,
        "num_expert_layers": n_expert,
        "expert_width_multiplier": float(cfg.expert_width_multiplier),
        "attention_mode": cfg.attention_mode,
        "vlm_model_name": cfg.vlm_model_name,
        "freeze_vision_encoder": bool(cfg.freeze_vision_encoder),
        "train_expert_only": bool(cfg.train_expert_only),
        "input_features": {k: list(v.shape) if hasattr(v, "shape") else str(v) for k, v in cfg.input_features.items()},
        "ok_32_layers": n_vlm == 32 and n_text == 32,
        "ok_n_action_steps_1": int(cfg.n_action_steps) == 1,
    }
    (OUT / "smoke_load.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)
    if not info["ok_32_layers"]:
        raise SystemExit(f"expected 32 VLM layers, got runtime={n_vlm} text={n_text}")
    if not info["ok_n_action_steps_1"]:
        raise SystemExit(f"expected n_action_steps=1, got {cfg.n_action_steps}")
    print("SMOKE_LOAD_OK", flush=True)


if __name__ == "__main__":
    main()

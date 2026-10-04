#!/usr/bin/env python3
"""CPU inspect of lerobot/smolvla_libero cameras, state stats, and architecture."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_STATE

from adapt_lerobot_official import (
    CKPT,
    LIBERO_STATE_DIM,
    RENAME_MAP,
    patch_policy_state_dim,
    patch_processor_state_dim,
)

OUT = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity/results/smolvla_libero")
OUT.mkdir(parents=True, exist_ok=True)


def _stats_shapes(path: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            if k.startswith(f"{OBS_STATE}.") or k.startswith("action."):
                out[k] = list(f.get_tensor(k).shape)
    return out


def main() -> None:
    snap = Path(snapshot_download(CKPT))
    snapshot_download(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        ignore_patterns=["onnx/**", "*.onnx", "*.onnx_data"],
    )
    cfg = SmolVLAConfig.from_pretrained(CKPT)
    cfg.device = "cpu"
    pre_json = json.loads((snap / "policy_preprocessor.json").read_text())
    rename_saved = pre_json["steps"][0]["config"]["rename_map"]
    pre_feat_state = pre_json["steps"][-1]["config"]["features"][OBS_STATE]["shape"]
    stats = _stats_shapes(snap / "policy_preprocessor_step_5_normalizer_processor.safetensors")

    preprocessor, _ = make_pre_post_processors(
        cfg,
        pretrained_path=CKPT,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "rename_observations_processor": {"rename_map": dict(RENAME_MAP)},
        },
    )
    patched_steps = patch_processor_state_dim(preprocessor)
    pre_state_after = None
    for step in preprocessor.steps:
        feats = getattr(step, "features", None)
        if feats and OBS_STATE in feats:
            pre_state_after = list(feats[OBS_STATE].shape)

    print("loading policy on CPU (architecture check)", flush=True)
    policy = SmolVLAPolicy.from_pretrained(CKPT).to("cpu").eval()
    patched_policy = patch_policy_state_dim(policy)
    vlm = policy.model.vlm_with_expert
    n_text = len(vlm.get_vlm_model().text_model.layers)
    info = {
        "checkpoint": CKPT,
        "snapshot": str(snap),
        "cameras": {
            "config_keys": [k for k in cfg.input_features if "images" in k],
            "saved_preprocessor_rename_map": rename_saved,
            "eval_rename_map": dict(RENAME_MAP),
            "empty_cameras": int(cfg.empty_cameras),
        },
        "state": {
            "config_shape": list(cfg.input_features[OBS_STATE].shape),
            "preprocessor_json_shape": pre_feat_state,
            "normalizer_stats_shapes": stats,
            "libero_processor_dim": LIBERO_STATE_DIM,
            "patched_processor_steps": patched_steps,
            "preprocessor_features_shape_after_patch": pre_state_after,
            "policy_config_shape_after_patch": list(policy.config.input_features[OBS_STATE].shape),
            "patched_policy": patched_policy,
            "max_state_dim": int(cfg.max_state_dim),
        },
        "arch": {
            "n_action_steps_ckpt": int(cfg.n_action_steps),
            "chunk_size": int(cfg.chunk_size),
            "num_steps": int(cfg.num_steps),
            "num_vlm_layers_config": int(cfg.num_vlm_layers),
            "num_vlm_layers_runtime": int(vlm.num_vlm_layers),
            "text_model_n_layers": n_text,
            "num_expert_layers": int(vlm.num_expert_layers),
            "expert_width_multiplier": float(cfg.expert_width_multiplier),
            "vlm_model_name": cfg.vlm_model_name,
            "state_proj": [int(policy.model.state_proj.in_features), int(policy.model.state_proj.out_features)],
        },
        "ok_state_stats_8": stats.get(f"{OBS_STATE}.mean") == [8],
        "ok_rename_map": rename_saved == RENAME_MAP,
        "ok_16_vlm": int(vlm.num_vlm_layers) == 16 and n_text == 16,
    }
    (OUT / "lerobot_official_adapt.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)
    if not info["ok_state_stats_8"]:
        raise SystemExit("expected 8-dim observation.state normalizer stats")
    if not info["ok_rename_map"]:
        raise SystemExit("saved preprocessor rename_map != training camera map")
    print("ADAPT_INSPECT_OK", flush=True)


if __name__ == "__main__":
    main()

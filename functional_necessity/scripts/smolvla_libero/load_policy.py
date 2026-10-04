"""Load lerobot/smolvla_libero with the LIBERO camera rename and 8-dim state patch."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from adapt_lerobot_official import (
    CKPT,
    RENAME_MAP,
    patch_policy_state_dim,
    patch_processor_state_dim,
)

FN_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = FN_ROOT / "results" / "smolvla_libero"
CALIB_PATH = OUT_DIR / "calib" / "calib_32_seed42.pt"
DATASET_REPO = "lerobot/libero"

METHODS = (
    "magnitude",
    "cossim",
    "cka",
    "ppl",
    "taylor",
    "fisher",
    "igia",
    "gateprobe",
    "hessian",
)


def load_policy(device: str = "cuda:0"):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(CKPT)
    policy.config.compile_model = False
    policy.config.device = device
    patch_policy_state_dim(policy)
    policy.to(device)
    policy.eval()
    return policy


def vlm_blocks(policy):
    text = policy.model.vlm_with_expert.get_vlm_model().text_model
    n = int(policy.model.vlm_with_expert.num_vlm_layers)
    return list(text.layers)[:n], text.norm


def load_preprocessor(device: str = "cpu"):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    cfg = SmolVLAConfig.from_pretrained(CKPT)
    preprocessor, _ = make_pre_post_processors(
        cfg,
        pretrained_path=CKPT,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": dict(RENAME_MAP)},
        },
    )
    patch_processor_state_dim(preprocessor)
    return preprocessor

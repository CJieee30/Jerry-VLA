"""Load the LeRobot MolmoAct2 LIBERO checkpoint without the 20GB base shards.

The released LeRobot file already contains the finetuned weights. The HF
``checkpoint_path`` is used only to build the module tree.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FN_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_CKPT = FN_ROOT / "checkpoints" / "MolmoAct2-LIBERO-LeRobot"
HF_CONFIG = FN_ROOT / "checkpoints" / "MolmoAct2-LIBERO"
OUT_DIR = FN_ROOT / "results" / "molmoact2_libero"
CALIB_PATH = OUT_DIR / "calib" / "calib_32_seed42.pt"

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


def install_loader_patches() -> None:
    """Build the module tree from the local HF config. Weights come from the LeRobot file."""
    import lerobot.policies.molmoact2.modeling_molmoact2 as pol

    def _from_config(cls, pretrained_model_name_or_path=None, config=None, **kwargs):
        del kwargs
        if config is None:
            config = pol.HFMolmoAct2Config.from_pretrained(pretrained_model_name_or_path)
        return cls(config)

    pol.MolmoAct2ForConditionalGeneration.from_pretrained = classmethod(_from_config)
    pol._strict_load_safetensors_weights = lambda *args, **kwargs: None


def load_policy(device: str = "cuda:0"):
    import json

    import lerobot.policies.molmoact2.modeling_molmoact2 as pol

    install_loader_patches()

    raw = json.loads((LEROBOT_CKPT / "config.json").read_text())
    if "dtype" not in raw and "model_dtype" in raw:
        raw["dtype"] = raw.pop("model_dtype")
    else:
        raw.pop("model_dtype", None)
    for key in ("enable_lora_vlm", "enable_lora_action_expert", "train_action_expert_only"):
        raw.pop(key, None)
    raw["train_mode_vlm"] = "fft"
    clean_dir = OUT_DIR / "policy_config"
    clean_dir.mkdir(parents=True, exist_ok=True)
    (clean_dir / "config.json").write_text(json.dumps(raw) + "\n")
    cfg = pol.MolmoAct2Config.from_pretrained(clean_dir)
    cfg.device = device
    cfg.checkpoint_path = str(HF_CONFIG)
    cfg.compile_model = False
    cfg.enable_inference_cuda_graph = False
    cfg.gradient_checkpointing = True
    cfg.llm_residual_dropout = 0.0
    cfg.action_mode = "continuous"
    cfg.inference_action_mode = "continuous"
    cfg.num_flow_timesteps = 1
    cfg.train_mode_vlm = "fft"
    policy = pol.MolmoAct2Policy.from_pretrained(LEROBOT_CKPT, config=cfg, strict=True)
    policy.eval()
    return policy


def vlm_blocks(policy):
    transformer = policy._backbone().transformer
    return transformer.blocks, transformer.ln_f

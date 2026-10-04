"""Load the joint OFT checkpoint, L1 action head, and proprio projector."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

from paths import CKPT, OFT_REPO, bootstrap


def cap_cuda_memory() -> None:
    """Cap this process so the job already on the card keeps a reserve."""
    import torch

    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    override = os.environ.get("OPENVLA_OFT_MEM_FRACTION")
    if override:
        frac = float(override)
    else:
        reserve = 384 * 1024 * 1024
        frac = max(0.50, min(0.92, (free - reserve) / total))
    torch.cuda.set_per_process_memory_fraction(frac)
    print(
        f"cuda memory fraction {frac:.3f} (free {free / 1024**2:.0f}MiB / {total / 1024**2:.0f}MiB)",
        flush=True,
    )


def _ensure_repo_path() -> None:
    bootstrap()
    # Local checkpoint sync walks ./prismatic from the repo root.
    os.chdir(OFT_REPO)


def make_cfg():
    _ensure_repo_path()
    return SimpleNamespace(
        pretrained_checkpoint=str(CKPT),
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        unnorm_key="",
        lora_rank=32,
        num_diffusion_steps_train=50,
        num_diffusion_steps_inference=50,
    )


def load_policy(device: str = "cuda:0"):
    """Return (vla, action_head, proprio_projector) on `device`."""
    cfg = make_cfg()
    from experiments.robot.openvla_utils import get_action_head, get_proprio_projector, get_vla

    vla = get_vla(cfg)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    return vla, action_head, proprio


def load_processor():
    cfg = make_cfg()
    from experiments.robot.openvla_utils import get_processor

    return get_processor(cfg)


def llama_parts(vla):
    core = vla.language_model.model
    return core.layers, core.norm


def prepare_for_grad(vla, action_head, proprio) -> None:
    """Checkpoint the Llama stack. Vision, projector, and heads stay frozen."""
    vla.train()
    action_head.eval()
    proprio.eval()
    vla.vision_backbone.eval()
    vla.projector.eval()
    frozen = 0
    for module in (vla.vision_backbone, vla.projector, action_head, proprio):
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad_(False)
                frozen += param.numel()
    vla.language_model.gradient_checkpointing_enable()
    vla.language_model.config.use_cache = False
    if hasattr(vla, "config"):
        vla.config.use_cache = False
    print(f"grad scoring: froze {frozen:,} non-Llama params; Llama checkpointing on", flush=True)


def prepare_for_forward(vla, action_head, proprio) -> None:
    vla.eval()
    action_head.eval()
    proprio.eval()
    vla.language_model.gradient_checkpointing_disable()
    vla.language_model.config.use_cache = False
    if hasattr(vla, "config"):
        vla.config.use_cache = False

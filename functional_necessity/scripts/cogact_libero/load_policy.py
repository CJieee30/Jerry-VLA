#!/usr/bin/env python3
"""Load Dexmal/libero-db-cogact the same way as score.py."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, "/home/guorj/dexbotic")

import torch
from transformers import AutoTokenizer, CLIPImageProcessor

from dexbotic.model.cogact.cogact_arch import CogACTForCausalLM

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
CKPT = FN / "checkpoints" / "libero-db-cogact"
OUT = FN / "results" / "cogact_libero"


def load_model(device: torch.device):
    model = CogACTForCausalLM.from_pretrained(
        CKPT,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    tower = model.model.mm_vision_tower
    if not hasattr(tower, "image_processor"):
        tower.image_processor = CLIPImageProcessor.from_pretrained(tower.vision_tower_name)
        tower.is_loaded = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    stats = json.loads((CKPT / "norm_stats.json").read_text())["default"]
    return model, tokenizer, stats

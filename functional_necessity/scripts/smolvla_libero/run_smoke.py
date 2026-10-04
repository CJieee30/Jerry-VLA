#!/usr/bin/env python3
"""One-episode LIBERO-Spatial t0 smoke for HuggingFaceVLA/smolvla_libero."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OUT = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity/results/smolvla_libero/smoke_eval")
OUT.mkdir(parents=True, exist_ok=True)

sys.argv = [
    "lerobot-eval",
    "--policy.path=HuggingFaceVLA/smolvla_libero",
    "--policy.device=cuda",
    "--policy.n_action_steps=1",
    "--policy.num_steps=10",
    "--env.type=libero",
    "--env.task=libero_spatial",
    "--env.task_ids=[0]",
    "--env.control_mode=relative",
    "--env.max_parallel_tasks=1",
    "--eval.batch_size=1",
    "--eval.n_episodes=1",
    "--seed=1000",
    f"--output_dir={OUT}",
]

from lerobot.scripts.lerobot_eval import main

if __name__ == "__main__":
    main()

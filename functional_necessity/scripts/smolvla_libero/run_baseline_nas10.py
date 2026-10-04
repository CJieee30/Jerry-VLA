#!/usr/bin/env python3
"""400-episode LIBERO contrast for HuggingFaceVLA/smolvla_libero at n_action_steps=10.

Same checkpoint / seed / suites / episodes as run_baseline.py (n_action_steps=1).
Only the executed chunk horizon changes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OUT = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity/results/smolvla_libero/baseline_eval_nas10")
OUT.mkdir(parents=True, exist_ok=True)

META = {
    "checkpoint": "HuggingFaceVLA/smolvla_libero",
    "n_action_steps": 10,
    "num_steps": 10,
    "chunk_size": 50,
    "seed": 1000,
    "n_episodes": 10,
    "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    "control_mode": "relative",
    "contrast_to": "results/smolvla_libero/baseline_eval (n_action_steps=1)",
}
(OUT / "protocol.json").write_text(json.dumps(META, indent=2) + "\n")

sys.argv = [
    "lerobot-eval",
    "--policy.path=HuggingFaceVLA/smolvla_libero",
    "--policy.device=cuda",
    "--policy.n_action_steps=10",
    "--policy.num_steps=10",
    "--env.type=libero",
    "--env.task=libero_spatial,libero_object,libero_goal,libero_10",
    "--env.control_mode=relative",
    "--env.max_parallel_tasks=1",
    "--eval.batch_size=1",
    "--eval.n_episodes=10",
    "--seed=1000",
    f"--output_dir={OUT}",
]

from lerobot.scripts.lerobot_eval import main

if __name__ == "__main__":
    main()

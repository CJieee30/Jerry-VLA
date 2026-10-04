#!/usr/bin/env python3
"""1-episode LIBERO-Spatial t0 smoke for adapted lerobot/smolvla_libero (nas=10)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from adapt_lerobot_official import CKPT, N_ACTION_STEPS, NUM_STEPS, RENAME_MAP, eval_argv, install_eval_patches

OUT = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity/results/smolvla_libero/lerobot_official_smoke")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "protocol.json").write_text(
    json.dumps(
        {
            "checkpoint": CKPT,
            "n_action_steps": N_ACTION_STEPS,
            "num_steps": NUM_STEPS,
            "rename_map": RENAME_MAP,
            "state": "LIBERO 8-dim (eef_pos, axis-angle, gripper); patch config 6 -> 8",
            "task": "libero_spatial",
            "task_ids": [0],
            "n_episodes": 1,
        },
        indent=2,
    )
    + "\n"
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
install_eval_patches()
sys.argv = eval_argv(str(OUT), n_episodes=1, task="libero_spatial", task_ids="[0]")

from lerobot.scripts.lerobot_eval import main

if __name__ == "__main__":
    main()

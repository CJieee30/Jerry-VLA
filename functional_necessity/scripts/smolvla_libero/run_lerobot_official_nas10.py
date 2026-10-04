#!/usr/bin/env python3
"""400-episode LIBERO eval for adapted lerobot/smolvla_libero at n_action_steps=10.

Protocol: 4 suites × 10 tasks × 10 episodes, seed=1000, relative control.
Cameras: train/eval rename_map image->camera1, image2->camera2.
State: keep LIBERO 8-dim; overwrite stale SO100 config shape 6.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from adapt_lerobot_official import (
    CKPT,
    LIBERO_STATE_DIM,
    N_ACTION_STEPS,
    NUM_STEPS,
    RENAME_MAP,
    SEED,
    SUITES,
    eval_argv,
    install_eval_patches,
)

OUT = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity/results/smolvla_libero/lerobot_official_nas10")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "protocol.json").write_text(
    json.dumps(
        {
            "checkpoint": CKPT,
            "n_action_steps": N_ACTION_STEPS,
            "num_steps": NUM_STEPS,
            "chunk_size": 50,
            "seed": SEED,
            "n_episodes": 10,
            "suites": SUITES.split(","),
            "control_mode": "relative",
            "rename_map": RENAME_MAP,
            "state_dim": LIBERO_STATE_DIM,
            "empty_cameras": 0,
        },
        indent=2,
    )
    + "\n"
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
install_eval_patches()
sys.argv = eval_argv(str(OUT), n_episodes=10, task=SUITES)

from lerobot.scripts.lerobot_eval import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SmolVLA keep-KV skip of VLM layer 0 only. One suite per process.

Protocol matches the nas=10 baseline: seed=1000, 10 episodes/task, relative
control, camera rename_map, state dim 8, compile_model=false.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layer_skip
from adapt_lerobot_official import (
    CKPT,
    LIBERO_STATE_DIM,
    N_ACTION_STEPS,
    N_EPISODES,
    NUM_STEPS,
    RENAME_MAP,
    SEED,
    eval_argv,
    install_eval_patches,
)

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
TAG = os.environ.get("SMOLVLA_DROP_TAG", "l0_only").strip() or "l0_only"
SUITE = os.environ.get("SMOLVLA_SUITE", "").strip()
if not SUITE:
    raise SystemExit("SMOLVLA_SUITE is required")
DROP = [0]
OUT = FN / "results" / "smolvla_libero" / TAG / SUITE
if (OUT / "eval_info.json").is_file():
    raise SystemExit(f"ALREADY_DONE {TAG}/{SUITE}")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "drop.json").write_text(
    json.dumps(
        {
            "checkpoint": CKPT,
            "method": "l0_only",
            "methods": ["l0_only"],
            "k": 1,
            "drop": DROP,
            "operator": "keep_kv_identity_residual",
            "exclude_layer0": False,
            "n_action_steps": N_ACTION_STEPS,
            "num_steps": NUM_STEPS,
            "seed": SEED,
            "n_episodes": N_EPISODES,
            "suites": [SUITE],
            "rename_map": RENAME_MAP,
            "state_dim": LIBERO_STATE_DIM,
            "compile_model": False,
        },
        indent=2,
    )
    + "\n"
)
layer_skip.install()
layer_skip._DROP = set(DROP)
install_eval_patches()
sys.argv = eval_argv(str(OUT), n_episodes=N_EPISODES, task=SUITE) + [
    "--policy.compile_model=false",
]
from lerobot.scripts.lerobot_eval import main as eval_main

if __name__ == "__main__":
    eval_main()

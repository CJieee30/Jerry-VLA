#!/usr/bin/env python3
"""SmolVLA LIBERO eval at NFE=1, 10 episodes per task (400 total).

Writes under results/stage_nfe/. Does not reuse historical NFE=10 directories.
Protocol matches run_lerobot_official_nas10.py / run_drop4.py except num_steps=1.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
sys.path.insert(0, str(FN / "scripts" / "smolvla_libero"))

import layer_skip
from adapt_lerobot_official import (
    CKPT,
    LIBERO_STATE_DIM,
    N_ACTION_STEPS,
    N_EPISODES,
    RENAME_MAP,
    SEED,
    SUITES,
    eval_argv,
    install_eval_patches,
)

TAG = os.environ.get("SMOLVLA_DROP_TAG", "").strip()
RAW = os.environ.get("SMOLVLA_DROP", "").strip()
if not TAG:
    raise SystemExit("SMOLVLA_DROP_TAG is required")
DROP = [int(part.strip()) for part in RAW.split(",") if part.strip()] if RAW else []
K = int(os.environ.get("SMOLVLA_DROP_K", str(len(DROP))))
METHOD = os.environ.get("SMOLVLA_DROP_METHOD", "nfe1").strip() or "nfe1"
NFE = int(os.environ.get("SMOLVLA_NFE", "1"))
N_EP = int(os.environ.get("SMOLVLA_N_EPISODES", str(N_EPISODES)))

OUT = FN / "results" / "stage_nfe" / TAG
if (OUT / "eval_info.json").is_file():
    raise SystemExit(f"ALREADY_DONE {TAG}")

layer_skip.install()
layer_skip._DROP = set(DROP)
install_eval_patches()

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "drop.json").write_text(
    json.dumps(
        {
            "checkpoint": CKPT,
            "method": METHOD,
            "k": K,
            "drop": DROP,
            "operator": "keep_kv_identity_residual",
            "exclude_layer0": False,
            "n_action_steps": N_ACTION_STEPS,
            "num_steps": NFE,
            "chunk_size": 50,
            "seed": SEED,
            "n_episodes": N_EP,
            "suites": SUITES.split(","),
            "rename_map": RENAME_MAP,
            "state_dim": LIBERO_STATE_DIM,
            "empty_cameras": 0,
            "tag": TAG,
        },
        indent=2,
    )
    + "\n"
)

sys.argv = eval_argv(str(OUT), n_episodes=N_EP, task=SUITES)
sys.argv = [arg if not arg.startswith("--policy.num_steps=") else f"--policy.num_steps={NFE}" for arg in sys.argv]

from lerobot.scripts.lerobot_eval import main

if __name__ == "__main__":
    main()

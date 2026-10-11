#!/usr/bin/env python3
"""MolmoAct2 keep-KV skip of VLM layer 0 only. One suite per process.

400-episode protocol if all four suites run: 10 tasks × 10 episodes.
Does not overwrite historical 2000-ep dirs. compile_model=false.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layer_skip
from load_policy import HF_CONFIG, LEROBOT_CKPT, OUT_DIR, install_loader_patches

TAG = os.environ.get("MOLMO_DROP_TAG", "l0_only").strip() or "l0_only"
SUITE = os.environ.get("MOLMO_SUITE", "").strip()
if not SUITE:
    raise SystemExit("MOLMO_SUITE is required (one of libero_spatial,libero_object,libero_goal,libero_10)")
N_EPISODES = int(os.environ.get("MOLMO_N_EPISODES", "10"))
DROP = [0]
OUT = OUT_DIR / TAG / SUITE
if (OUT / "eval_info.json").is_file():
    raise SystemExit(f"ALREADY_DONE {TAG}/{SUITE}")

install_loader_patches()
layer_skip.install()
layer_skip._DROP = set(DROP)

EVAL_CKPT = OUT_DIR / "eval_ckpt"
EVAL_CKPT.mkdir(parents=True, exist_ok=True)
if not (EVAL_CKPT / "config.json").is_file():
    raw = json.loads((LEROBOT_CKPT / "config.json").read_text())
    raw.pop("model_dtype", None)
    for key in ("enable_lora_vlm", "enable_lora_action_expert", "train_action_expert_only"):
        raw.pop(key, None)
    raw["dtype"] = raw.get("dtype", "bfloat16")
    raw["train_mode_vlm"] = "fft"
    (EVAL_CKPT / "config.json").write_text(json.dumps(raw) + "\n")
for name in os.listdir(LEROBOT_CKPT):
    if name == "config.json":
        continue
    dest = EVAL_CKPT / name
    if dest.exists():
        continue
    dest.symlink_to(LEROBOT_CKPT / name)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "drop.json").write_text(
    json.dumps(
        {
            "methods": ["l0_only"],
            "k": 1,
            "drop": DROP,
            "operator": "keep_kv_identity_residual",
            "exclude_layer0": False,
            "shared": False,
            "tag": TAG,
            "suite": SUITE,
            "n_episodes": N_EPISODES,
            "seed": 1000,
            "compile_model": False,
        },
        indent=2,
    )
    + "\n"
)

sys.argv = [
    "lerobot-eval",
    f"--policy.path={EVAL_CKPT}",
    f"--policy.checkpoint_path={HF_CONFIG}",
    "--policy.inference_action_mode=continuous",
    "--policy.dtype=bfloat16",
    "--policy.enable_inference_cuda_graph=false",
    "--policy.device=cuda",
    "--policy.per_episode_seed=true",
    "--policy.eval_seed=1000",
    "--policy.compile_model=false",
    "--env.type=libero",
    f"--env.task={SUITE}",
    "--env.num_steps_wait=50",
    '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}',
    "--eval.batch_size=1",
    f"--eval.n_episodes={N_EPISODES}",
    "--seed=1000",
    f"--output_dir={OUT}",
]

from lerobot.scripts.lerobot_eval import main

if __name__ == "__main__":
    main()

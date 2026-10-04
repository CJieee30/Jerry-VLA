#!/usr/bin/env python3
"""Official MolmoAct2 LIBERO eval, loading weights from the local LeRobot file."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_policy import HF_CONFIG, LEROBOT_CKPT, OUT_DIR, install_loader_patches

install_loader_patches()

# The released config has fields this LeRobot checkout does not accept.
EVAL_CKPT = OUT_DIR / "eval_ckpt"
EVAL_CKPT.mkdir(parents=True, exist_ok=True)
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
    if not dest.exists():
        dest.symlink_to(LEROBOT_CKPT / name)

OUT = OUT_DIR / "baseline_eval"
OUT.mkdir(parents=True, exist_ok=True)
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
    "--env.task=libero_spatial,libero_object,libero_goal,libero_10",
    "--env.num_steps_wait=50",
    '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}',
    "--eval.batch_size=1",
    "--eval.n_episodes=50",
    "--seed=1000",
    f"--output_dir={OUT}",
]

from lerobot.scripts.lerobot_eval import main

if __name__ == "__main__":
    main()

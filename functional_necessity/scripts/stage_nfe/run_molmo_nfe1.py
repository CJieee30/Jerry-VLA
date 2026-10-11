#!/usr/bin/env python3
"""MolmoAct2 LIBERO eval at NFE=1, 10 episodes per task (400 total).

Writes under results/stage_nfe/. Does not reuse historical 2000-ep directories.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))

import layer_skip
from load_policy import HF_CONFIG, LEROBOT_CKPT, OUT_DIR, install_loader_patches

TAG = os.environ.get("MOLMO_DROP_TAG", "").strip()
RAW = os.environ.get("MOLMO_DROP", "").strip()
if not TAG:
    raise SystemExit("MOLMO_DROP_TAG is required")
DROP = [int(part.strip()) for part in RAW.split(",") if part.strip()] if RAW else []
K = int(os.environ.get("MOLMO_DROP_K", str(len(DROP))))
METHOD = os.environ.get("MOLMO_DROP_METHOD", "nfe1").strip() or "nfe1"
N_EP = int(os.environ.get("MOLMO_N_EPISODES", "10"))
NFE = int(os.environ.get("MOLMO_NFE", "1"))

OUT = FN / "results" / "stage_nfe" / TAG
if (OUT / "eval_info.json").is_file():
    raise SystemExit(f"ALREADY_DONE {TAG}")

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
            "methods": [METHOD],
            "k": K,
            "drop": DROP,
            "operator": "keep_kv_identity_residual",
            "num_inference_steps": NFE,
            "n_episodes_per_task": N_EP,
            "tag": TAG,
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
    f"--policy.num_inference_steps={NFE}",
    "--env.type=libero",
    "--env.task=libero_spatial,libero_object,libero_goal,libero_10",
    "--env.num_steps_wait=50",
    '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}',
    "--eval.batch_size=1",
    f"--eval.n_episodes={N_EP}",
    "--seed=1000",
    f"--output_dir={OUT}",
]

from lerobot.scripts.lerobot_eval import main

if __name__ == "__main__":
    main()

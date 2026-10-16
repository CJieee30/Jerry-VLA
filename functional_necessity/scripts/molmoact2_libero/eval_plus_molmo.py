#!/usr/bin/env python3
"""LIBERO-plus closed-loop eval for MolmoAct2 (official LeRobot load path).

Full model only in this entry; no layer skip / materialize. Uses libero_plus env
with the same task-id selection as pi05 ``run_eval_plus_lang.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

FN = Path(__file__).resolve().parents[2]
MOLMO_SCRIPTS = FN / "scripts" / "molmoact2_libero"
sys.path.insert(0, str(MOLMO_SCRIPTS))

from load_policy import HF_CONFIG, LEROBOT_CKPT, install_loader_patches  # noqa: E402

import layer_skip  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_eval_ckpt(base: Path) -> Path:
    ckpt = base / "eval_ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)
    if not (ckpt / "config.json").is_file():
        raw = json.loads((LEROBOT_CKPT / "config.json").read_text())
        raw.pop("model_dtype", None)
        for key in ("enable_lora_vlm", "enable_lora_action_expert", "train_action_expert_only"):
            raw.pop(key, None)
        raw["dtype"] = raw.get("dtype", "bfloat16")
        raw["train_mode_vlm"] = "fft"
        (ckpt / "config.json").write_text(json.dumps(raw) + "\n")
    for name in os.listdir(LEROBOT_CKPT):
        if name == "config.json":
            continue
        dest = ckpt / name
        if not dest.exists():
            dest.symlink_to(LEROBOT_CKPT / name)
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--job_name", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task_ids", required=True, help="JSON list of 0-based task ids")
    parser.add_argument("--category", default="Language Instructions")
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--classification_json", type=Path, default=None)
    args = parser.parse_args()

    out: Path = args.output_dir
    if (out / "eval_info.json").is_file():
        print(f"ALREADY_DONE {out / 'eval_info.json'}")
        return

    task_ids = json.loads(args.task_ids)
    if not task_ids:
        raise SystemExit("empty task_ids")

    install_loader_patches()
    layer_skip._DROP = set()

    shared_ckpt_root = FN / "results" / "eval_libero_plus" / "_molmo_shared"
    eval_ckpt = _ensure_eval_ckpt(shared_ckpt_root)

    nfe = int(args.num_inference_steps)
    run_cfg = {
        "model": "MolmoAct2-LIBERO-LeRobot",
        "checkpoint_path": str(HF_CONFIG),
        "lerobot_weights": str(LEROBOT_CKPT),
        "variant": "full",
        "inference_action_mode": "continuous",
        "dtype": "bfloat16",
        "num_inference_steps": nfe,
        "n_action_steps": 10,
        "num_steps_wait": 50,
        "seed": 1000,
        "eval_seed": 1000,
        "per_episode_seed": True,
        "compile_model": False,
        "enable_inference_cuda_graph": False,
        "n_episodes_per_task": 1,
        "env_type": "libero_plus",
        "env_task": args.suite,
        "env_control_mode": "relative",
        "env_max_parallel_tasks": 1,
        "eval_batch_size": 1,
        "category": args.category,
        "task_ids_n": len(task_ids),
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_sha256": {
            "eval_plus_molmo.py": _sha(Path(__file__)),
            "load_policy.py": _sha(MOLMO_SCRIPTS / "load_policy.py"),
            "layer_skip.py": _sha(MOLMO_SCRIPTS / "layer_skip.py"),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(run_cfg, indent=2) + "\n")
    (out / "task_manifest.json").write_text(
        json.dumps(
            {
                "suite": args.suite,
                "category": args.category,
                "task_ids": task_ids,
                "n_variants": len(task_ids),
                "classification_json": str(
                    args.classification_json
                    or "/home/guorj/LIBERO-plus/libero/libero/benchmark/task_classification.json"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    (out / "plus_meta.json").write_text(
        json.dumps(
            {
                "checkpoint": "allenai/MolmoAct2-LIBERO-LeRobot",
                "method": "full",
                "k": 0,
                "drop_layers": [],
                "operator": "full",
                "materialized": False,
                "job_name": args.job_name,
                "seed": 1000,
                "eval_seed": 1000,
                "num_inference_steps": nfe,
                "n_episodes_per_task": 1,
                "n_action_steps": 10,
                "compile_model": False,
                "enable_inference_cuda_graph": False,
            },
            indent=2,
        )
        + "\n"
    )

    sys.argv = [
        "lerobot-eval",
        f"--policy.path={eval_ckpt}",
        f"--policy.checkpoint_path={HF_CONFIG}",
        "--policy.inference_action_mode=continuous",
        "--policy.dtype=bfloat16",
        "--policy.enable_inference_cuda_graph=false",
        "--policy.device=cuda",
        "--policy.per_episode_seed=true",
        "--policy.eval_seed=1000",
        "--policy.compile_model=false",
        f"--policy.num_inference_steps={nfe}",
        "--policy.n_action_steps=10",
        "--env.type=libero_plus",
        f"--env.task={args.suite}",
        f"--env.task_ids={json.dumps(task_ids)}",
        "--env.num_steps_wait=50",
        "--env.control_mode=relative",
        "--env.max_parallel_tasks=1",
        '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}',
        "--eval.batch_size=1",
        "--eval.n_episodes=1",
        "--seed=1000",
        f"--output_dir={out}",
    ]

    from lerobot.scripts.lerobot_eval import main as lerobot_main

    lerobot_main()


if __name__ == "__main__":
    main()

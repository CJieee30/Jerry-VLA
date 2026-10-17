#!/usr/bin/env python3
"""LIBERO-plus closed-loop eval for MolmoAct2 (official LeRobot load path).

Variants: ``full`` or ``drop16_materialized`` (Taylor Drop-16, keep-KV + materialize).
Uses libero_plus env with the same task-id selection as pi05 ``run_eval_plus_lang.sh``.
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

DROP16 = [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]


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
    parser.add_argument(
        "--variant",
        choices=("full", "drop16_materialized"),
        default="full",
        help="full or Taylor Drop-16 materialized (keep-KV)",
    )
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
    nfe = int(args.num_inference_steps)
    variant = str(args.variant)

    def _install_materialize_hook() -> None:
        import lerobot.policies.molmoact2.modeling_molmoact2 as pol
        from layer_skip import skip_vlm_layers

        sys.path.insert(0, str(FN / "scripts" / "stage_nfe"))
        from materialize import materialize_molmo

        if getattr(pol.MolmoAct2Policy, "_plus_drop16_hook", False):
            return

        _orig = pol.MolmoAct2Policy.from_pretrained.__func__

        @classmethod
        def _from_pretrained(cls, *args, **kwargs):
            policy = _orig(cls, *args, **kwargs)
            policy.config.compile_model = False
            policy.config.enable_inference_cuda_graph = False
            policy._set_inference_cuda_graph_enabled(False)
            backbone = policy._backbone()
            if getattr(backbone, "action_cuda_graph_manager", None) is not None:
                backbone.action_cuda_graph_manager = None
            if getattr(backbone.transformer, "gradient_checkpointing", False):
                backbone.transformer.gradient_checkpointing = False
            import torch

            device = next(policy.parameters()).device
            from load_policy import CALIB_PATH

            blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in blob["samples"][0]["batch"].items()}
            layer_skip.install()
            layer_skip._DROP = set(DROP16)
            gen = torch.Generator(device=device)
            gen.manual_seed(1000)
            with torch.no_grad(), skip_vlm_layers(DROP16), policy._autocast_context():
                policy.reset()
                hooked = policy.predict_action_chunk(batch, num_steps=nfe, generator=gen)
            info = materialize_molmo(policy, DROP16)
            layer_skip._DROP = set()
            with torch.no_grad(), policy._autocast_context():
                policy.reset()
                gen.manual_seed(1000)
                mat = policy.predict_action_chunk(batch, num_steps=nfe, generator=gen)
            diff = float((mat - hooked).abs().max().item())
            if diff > 1e-4:
                raise RuntimeError(f"materialized NFE={nfe} action drift {diff} vs K/V-only hook")
            policy._materialize_info = {**info, "action_max_abs_vs_kvonly": diff}
            return policy

        pol.MolmoAct2Policy.from_pretrained = _from_pretrained
        pol.MolmoAct2Policy._plus_drop16_hook = True

    if variant == "drop16_materialized":
        _install_materialize_hook()

    shared_ckpt_root = FN / "results" / "eval_libero_plus" / "_molmo_shared"
    eval_ckpt = _ensure_eval_ckpt(shared_ckpt_root)

    run_cfg = {
        "model": "MolmoAct2-LIBERO-LeRobot",
        "checkpoint_path": str(HF_CONFIG),
        "lerobot_weights": str(LEROBOT_CKPT),
        "variant": variant,
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
                "method": "taylor" if variant == "drop16_materialized" else "full",
                "k": 16 if variant == "drop16_materialized" else 0,
                "drop_layers": DROP16 if variant == "drop16_materialized" else [],
                "operator": (
                    "keep_kv_identity_residual_then_materialize"
                    if variant == "drop16_materialized"
                    else "full"
                ),
                "materialized": variant == "drop16_materialized",
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

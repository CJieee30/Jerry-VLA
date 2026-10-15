#!/usr/bin/env python3
"""MolmoAct2 LIBERO closed loop: Full vs materialized Taylor Drop-16, NFE=10, 400 episodes.

Variant via MOLMO_EVAL_VARIANT=full|drop16_materialized. Does not overwrite 2000-ep dirs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

FN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
sys.path.insert(0, str(FN / "scripts" / "stage_nfe"))

DROP = [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
N_EP = int(os.environ.get("MOLMO_N_EPISODES", "10"))
NFE = int(os.environ.get("MOLMO_NFE", "10"))
VARIANT = os.environ.get("MOLMO_EVAL_VARIANT", "").strip()
if VARIANT not in ("full", "drop16_materialized"):
    raise SystemExit("MOLMO_EVAL_VARIANT must be full or drop16_materialized")

TAG = os.environ.get(
    "MOLMO_DROP_TAG",
    f"molmoact2_{'full' if VARIANT == 'full' else 'drop16_materialized'}_nfe{NFE}_400",
)
OUT = FN / "results" / "stage_nfe" / TAG
if (OUT / "eval_info.json").is_file():
    raise SystemExit(f"ALREADY_DONE {TAG}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_policy_hook() -> None:
    import lerobot.policies.molmoact2.modeling_molmoact2 as pol
    from layer_skip import skip_vlm_layers
    from materialize import materialize_molmo

    if getattr(pol.MolmoAct2Policy, "_nfe10_closed_loop_hook", False):
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
        if VARIANT == "drop16_materialized":
            device = next(policy.parameters()).device
            from load_policy import CALIB_PATH

            blob = __import__("torch").load(CALIB_PATH, map_location="cpu", weights_only=False)
            batch = {
                k: v.to(device) if __import__("torch").is_tensor(v) else v
                for k, v in blob["samples"][0]["batch"].items()
            }
            import layer_skip

            layer_skip.install()
            layer_skip._DROP = set(DROP)
            gen = __import__("torch").Generator(device=device)
            gen.manual_seed(1000)
            with __import__("torch").no_grad(), skip_vlm_layers(DROP), policy._autocast_context():
                policy.reset()
                hooked = policy.predict_action_chunk(batch, num_steps=NFE, generator=gen)
            info = materialize_molmo(policy, DROP)
            layer_skip._DROP = set()
            with __import__("torch").no_grad(), policy._autocast_context():
                policy.reset()
                gen.manual_seed(1000)
                mat = policy.predict_action_chunk(batch, num_steps=NFE, generator=gen)
            diff = float((mat - hooked).abs().max().item())
            if diff > 1e-4:
                raise RuntimeError(f"materialized NFE={NFE} action drift {diff} vs K/V-only hook")
            policy._materialize_info = {**info, "action_max_abs_vs_kvonly": diff}
        else:
            policy._materialize_info = None
        return policy

    pol.MolmoAct2Policy.from_pretrained = _from_pretrained
    pol.MolmoAct2Policy._nfe10_closed_loop_hook = True


import layer_skip
from load_policy import HF_CONFIG, LEROBOT_CKPT, OUT_DIR, install_loader_patches

install_loader_patches()
if VARIANT == "full":
    layer_skip._DROP = set()
else:
    layer_skip.install()
    layer_skip._DROP = set()

_install_policy_hook()

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
run_cfg = {
    "variant": VARIANT,
    "checkpoint": "MolmoAct2-LIBERO-LeRobot",
    "inference_action_mode": "continuous",
    "dtype": "bfloat16",
    "num_inference_steps": NFE,
    "n_action_steps": 10,
    "num_steps_wait": 50,
    "seed": 1000,
    "per_episode_seed": True,
    "compile_model": False,
    "enable_inference_cuda_graph": False,
    "n_episodes_per_task": N_EP,
    "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    "drop": DROP if VARIANT == "drop16_materialized" else [],
    "pruning": "materialized_kv_only" if VARIANT == "drop16_materialized" else "none",
    "operator": "keep_kv_identity_residual_then_materialize" if VARIANT == "drop16_materialized" else "full",
    "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "code_sha256": {
        "run_molmo_nfe10_closed_loop.py": _sha(Path(__file__)),
        "layer_skip.py": _sha(FN / "scripts" / "molmoact2_libero" / "layer_skip.py"),
        "materialize.py": _sha(FN / "scripts" / "stage_nfe" / "materialize.py"),
    },
}
(OUT / "run_config.json").write_text(json.dumps(run_cfg, indent=2) + "\n")
(OUT / "drop.json").write_text(
    json.dumps(
        {
            "methods": ["taylor"] if VARIANT == "drop16_materialized" else ["full"],
            "k": 16 if VARIANT == "drop16_materialized" else 0,
            "drop": DROP if VARIANT == "drop16_materialized" else [],
            "operator": run_cfg["operator"],
            "materialized": VARIANT == "drop16_materialized",
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

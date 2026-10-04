"""Adapt lerobot/smolvla_libero to the LeRobot LIBERO env (cameras + state).

Checkpoint leftover from SO100:
  cameras: observation.images.camera{1,2,3}
  config.input_features[observation.state].shape = 6
  empty_cameras = 0

LIBERO env (LiberoProcessorStep):
  cameras: observation.images.image, observation.images.image2
  state: 8-dim  [eef_pos(3), axis-angle(3), gripper_qpos(2)]

Training (train_config.json) already used the camera rename_map below on
lerobot/libero. Saved normalizer stats are 8-dim despite config saying 6.
Eval MUST pass the same rename_map on CLI: lerobot-eval always overrides
the saved preprocessor rename_map with cfg.rename_map (empty by default).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from lerobot.utils.constants import OBS_STATE

CKPT = "lerobot/smolvla_libero"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
LIBERO_STATE_DIM = 8
N_ACTION_STEPS = 10
NUM_STEPS = 10
SEED = 1000
N_EPISODES = 10
SUITES = "libero_spatial,libero_object,libero_goal,libero_10"


def rename_map_cli() -> str:
    return (
        '{"observation.images.image": "observation.images.camera1", '
        '"observation.images.image2": "observation.images.camera2"}'
    )


def _set_state_dim(features: dict[str, Any] | None, dim: int = LIBERO_STATE_DIM) -> bool:
    if not features or OBS_STATE not in features:
        return False
    feat = features[OBS_STATE]
    shape = tuple(getattr(feat, "shape", ()))
    if shape == (dim,):
        return False
    features[OBS_STATE] = replace(feat, shape=(dim,))
    return True


def patch_policy_state_dim(policy) -> bool:
    return _set_state_dim(getattr(policy.config, "input_features", None))


def patch_processor_state_dim(preprocessor) -> list[str]:
    changed: list[str] = []
    steps = getattr(preprocessor, "steps", None) or []
    for i, step in enumerate(steps):
        feats = getattr(step, "features", None)
        if _set_state_dim(feats):
            changed.append(f"step{i}:{type(step).__name__}")
    return changed


def install_eval_patches() -> None:
    """Patch lerobot-eval so state config matches the 8-dim LIBERO stats."""
    import lerobot.scripts.lerobot_eval as ev
    from lerobot.policies.factory import make_policy as orig_make_policy
    from lerobot.policies.factory import make_pre_post_processors as orig_pp

    def make_policy(*args, **kwargs):
        policy = orig_make_policy(*args, **kwargs)
        patch_policy_state_dim(policy)
        return policy

    def make_pre_post_processors(*args, **kwargs):
        preprocessor, postprocessor = orig_pp(*args, **kwargs)
        patch_processor_state_dim(preprocessor)
        return preprocessor, postprocessor

    ev.make_policy = make_policy
    ev.make_pre_post_processors = make_pre_post_processors


def eval_argv(output_dir: str, *, n_episodes: int, task: str, task_ids: str | None = None) -> list[str]:
    args = [
        "lerobot-eval",
        f"--policy.path={CKPT}",
        "--policy.device=cuda",
        f"--policy.n_action_steps={N_ACTION_STEPS}",
        f"--policy.num_steps={NUM_STEPS}",
        "--env.type=libero",
        f"--env.task={task}",
        "--env.control_mode=relative",
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=1",
        f"--eval.n_episodes={n_episodes}",
        f"--seed={SEED}",
        f"--rename_map={rename_map_cli()}",
        f"--output_dir={output_dir}",
    ]
    if task_ids is not None:
        args.append(f"--env.task_ids={task_ids}")
    return args

#!/usr/bin/env python3
"""Pilot: short-horizon closed-loop object-state drift.

Pre-registered rule, fixed before any of these rollouts are read:
  d(k) is the mean over episodes of the L2 distance between movable-object
  positions of the dropped policy and the full policy, at step H.
  sigma_state is the same distance between two full-policy rollouts that
  differ only in inference noise.
  k is inside the tolerance iff d(k) <= sigma_state. Coefficient is 1.

The pilot only checks the known cliffs:
  pi05 Fisher CARE k=7 (pass) vs k=8 (fail)
  MolmoAct2 Taylor CARE k=16 (pass) vs k=17 (fail)

H = 100. Episodes = 4 suites x task ids {0,2,4,6,8} x init state 0.
Terminal object poses are held if the episode ends early, so a reset cannot
replace them. Success is recorded and is not part of the rule.
"""

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
OUT = FN / "results" / "stage3_ledge" / "pilot_state"
HORIZON = 100
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASK_IDS = (0, 2, 4, 6, 8)
CHECKPOINTS = (25, 50, 75, 100)
NOISE_A = 1000
NOISE_B = 2000

PI05_K7 = [1, 2, 13, 14, 15, 16, 17]
PI05_K8 = [1, 2, 3, 13, 14, 15, 16, 17]
MOLMO_K16 = [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
MOLMO_K17 = [1, 2, 3, 4, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]


def _conditions(stack: str) -> list[tuple[str, list[int], int]]:
    if stack == "pi05":
        drops = {"full_a": [], "full_b": [], "k7": PI05_K7, "k8": PI05_K8}
        seeds = {"full_a": NOISE_A, "full_b": NOISE_B, "k7": NOISE_A, "k8": NOISE_A}
        safe, fail = "k7", "k8"
    elif stack == "molmoact2":
        drops = {"full_a": [], "full_b": [], "k16": MOLMO_K16, "k17": MOLMO_K17}
        seeds = {"full_a": NOISE_A, "full_b": NOISE_B, "k16": NOISE_A, "k17": NOISE_A}
        safe, fail = "k16", "k17"
    else:
        raise ValueError(stack)
    order = ("full_a", "full_b", safe, fail)
    return [(name, drops[name], seeds[name]) for name in order], safe, fail


def _prepare_molmo() -> tuple[str, str]:
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    import layer_skip
    from load_policy import HF_CONFIG, LEROBOT_CKPT, OUT_DIR, install_loader_patches

    install_loader_patches()
    layer_skip.install()
    eval_ckpt = OUT_DIR / "eval_ckpt"
    eval_ckpt.mkdir(parents=True, exist_ok=True)
    if not (eval_ckpt / "config.json").is_file():
        raw = json.loads((LEROBOT_CKPT / "config.json").read_text())
        raw.pop("model_dtype", None)
        for key in ("enable_lora_vlm", "enable_lora_action_expert", "train_action_expert_only"):
            raw.pop(key, None)
        raw["dtype"] = raw.get("dtype", "bfloat16")
        raw["train_mode_vlm"] = "fft"
        (eval_ckpt / "config.json").write_text(json.dumps(raw) + "\n")
    for name in os.listdir(LEROBOT_CKPT):
        if name == "config.json":
            continue
        dest = eval_ckpt / name
        if not dest.exists():
            dest.symlink_to(LEROBOT_CKPT / name)
    return str(eval_ckpt), str(HF_CONFIG)


def _cfg(stack: str, device: str):
    if stack == "molmoact2":
        eval_ckpt, hf_config = _prepare_molmo()
        argv = [
            "pilot_state",
            f"--policy.path={eval_ckpt}",
            f"--policy.checkpoint_path={hf_config}",
            "--policy.inference_action_mode=continuous",
            "--policy.dtype=bfloat16",
            "--policy.enable_inference_cuda_graph=false",
            "--policy.per_episode_seed=true",
            "--policy.eval_seed=1000",
            "--policy.compile_model=false",
            f"--policy.device={device}",
            "--env.type=libero",
            "--env.task=libero_spatial",
            "--env.num_steps_wait=50",
            '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}',
            "--env.control_mode=relative",
            "--eval.batch_size=1",
            "--eval.n_episodes=1",
            "--seed=1000",
            "--output_dir=/tmp/pilot_state_unused",
        ]
    else:
        sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))
        argv = [
            "pilot_state",
            "--policy.path=lerobot/pi05_libero_finetuned_v044",
            "--policy.n_action_steps=10",
            "--policy.compile_model=false",
            "--policy.gradient_checkpointing=false",
            f"--policy.device={device}",
            "--env.type=libero",
            "--env.task=libero_spatial",
            "--env.control_mode=relative",
            "--eval.batch_size=1",
            "--eval.n_episodes=1",
            "--seed=1000",
            "--output_dir=/tmp/pilot_state_unused",
        ]
    sys.argv = argv
    from lerobot.configs import parser
    from lerobot.configs.eval import EvalPipelineConfig

    holder: dict = {}

    @parser.wrap()
    def _go(cfg: EvalPipelineConfig):
        holder["cfg"] = cfg

    _go()
    return holder["cfg"]


def _object_positions(vec) -> np.ndarray:
    envs = getattr(vec, "envs", None)
    if envs is None:
        envs = vec.unwrapped.envs
    off = envs[0]._env
    sim = off.sim
    names = sorted(off.env.objects_dict)
    pts = []
    for name in names:
        body = off.env.objects_dict[name].root_body
        bid = sim.model.body_name2id(body)
        pts.append(np.asarray(sim.data.body_xpos[bid], dtype=np.float64))
    if not pts:
        raise RuntimeError("no movable objects")
    return np.concatenate(pts)


def _eef(vec) -> np.ndarray:
    envs = getattr(vec, "envs", None) or vec.unwrapped.envs
    off = envs[0]._env
    sim = off.sim
    for name in ("gripper0_eef", "robot0_right_hand", "robot0_eef"):
        try:
            bid = sim.model.body_name2id(name)
        except Exception:
            continue
        return np.asarray(sim.data.body_xpos[bid], dtype=np.float64).copy()
    return np.zeros(3, dtype=np.float64)


def _bind_pi05_noise(policy, seed: int):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    def sample_noise(shape, device):
        noise = torch.normal(mean=0.0, std=1.0, size=shape, generator=gen, dtype=torch.float32)
        return noise.to(device)

    policy.model.sample_noise = sample_noise
    return gen


def _success_from_info(info, n: int) -> bool:
    if "final_info" in info:
        final = info["final_info"]
        if isinstance(final, dict):
            flag = final.get("is_success", False)
            return bool(np.asarray(flag).reshape(-1)[0])
        item = final[0]
        if isinstance(item, dict) and "is_success" in item:
            return bool(item["is_success"])
    if "is_success" in info:
        return bool(np.asarray(info["is_success"]).reshape(-1)[0])
    return False


def _rollout(env, policy, processors, *, horizon: int, noise_seed: int, stack: str, env_seed: int) -> dict:
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION, preprocess_observation
    from lerobot.utils.constants import ACTION

    env_pre, env_post, pre, post = processors
    envs = getattr(env, "envs", None) or env.unwrapped.envs
    envs[0].init_state_id = 0
    policy.reset()
    generator = None
    if stack == "pi05":
        _bind_pi05_noise(policy, noise_seed)
    else:
        generator = torch.Generator(device="cuda")
        generator.manual_seed(int(noise_seed))

    observation, _info = env.reset(seed=[env_seed], options={NEW_ROLLOUT_OPTION: True})
    origin = _object_positions(env)
    objects = []
    eefs = []
    done = False
    success = False
    for _step in range(horizon):
        if done:
            objects.append(objects[-1])
            eefs.append(eefs[-1])
            continue
        batch = preprocess_observation(observation)
        batch["task"] = list(env.call("task_description"))
        batch = env_pre(batch)
        batch = pre(batch)
        with torch.inference_mode():
            if generator is None:
                action = policy.select_action(batch)
            else:
                action = policy.select_action(batch, generator=generator)
        action = post(action)
        action = env_post({ACTION: action})[ACTION]
        action_np = action.detach().to("cpu").numpy()
        if action_np.ndim == 1:
            action_np = action_np[None, :]
        observation, _reward, terminated, truncated, info = env.step(action_np)
        objects.append(_object_positions(env))
        eefs.append(_eef(env))
        success = success or _success_from_info(info, 1)
        done = bool(np.asarray(terminated).reshape(-1)[0] or np.asarray(truncated).reshape(-1)[0])
    return {
        "origin": origin,
        "objects": np.stack(objects),
        "eef": np.stack(eefs),
        "success": success,
        "stopped": int(done),
    }


def _make_env(cfg, suite: str, task_id: int, horizon: int):
    import gymnasium as gym
    from lerobot.envs.libero import create_libero_envs

    gym_kwargs = dict(cfg.env.gym_kwargs)
    gym_kwargs.pop("task_ids", None)
    gym_kwargs["task_ids"] = [task_id]
    created = create_libero_envs(
        task=suite,
        n_envs=1,
        camera_name=cfg.env.camera_name,
        init_states=True,
        gym_kwargs=gym_kwargs,
        env_cls=gym.vector.SyncVectorEnv,
        control_mode=cfg.env.control_mode,
        episode_length=horizon,
        camera_name_mapping=cfg.env.camera_name_mapping,
        is_libero_plus=False,
    )
    return created[suite][task_id]


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _summarize(episodes: list[dict], safe: str, fail: str, horizon: int) -> dict:
    out = {}
    steps = tuple(s for s in CHECKPOINTS if s <= horizon)
    if horizon not in steps:
        steps = steps + (horizon,)
    for step in steps:
        idx = step - 1
        sigma, d_safe, d_fail, moved = [], [], [], []
        for ep in episodes:
            fa = ep["full_a"]["objects"][idx]
            fb = ep["full_b"]["objects"][idx]
            sigma.append(_dist(fa, fb))
            d_safe.append(_dist(ep[safe]["objects"][idx], fa))
            d_fail.append(_dist(ep[fail]["objects"][idx], fa))
            moved.append(_dist(fa, ep["full_a"]["origin"]))
        sigma_m = float(np.mean(sigma))
        safe_m = float(np.mean(d_safe))
        fail_m = float(np.mean(d_fail))
        out[str(step)] = {
            "sigma_state": sigma_m,
            "d_safe": safe_m,
            "d_fail": fail_m,
            "safe_over_sigma": safe_m / sigma_m if sigma_m > 0 else None,
            "fail_over_sigma": fail_m / sigma_m if sigma_m > 0 else None,
            "safe_within": bool(safe_m <= sigma_m),
            "fail_exceeds": bool(fail_m > sigma_m),
            "object_motion_full": float(np.mean(moved)),
            "per_episode_sigma": sigma,
            "per_episode_d_safe": d_safe,
            "per_episode_d_fail": d_fail,
        }
    decision = dict(out[str(horizon)])
    decision["mechanism_ok"] = bool(decision["safe_within"] and decision["fail_exceeds"])
    decision["horizon"] = horizon
    return {"by_step": out, "decision": decision}


def _dump_episode(ep: dict) -> dict:
    packed = {"suite": ep["suite"], "task_id": ep["task_id"]}
    for name, run in ep.items():
        if name in ("suite", "task_id"):
            continue
        packed[name] = {
            "success": run["success"],
            "stopped_early": run["stopped"],
            "origin": run["origin"].tolist(),
            "objects_at": {
                str(s): run["objects"][s - 1].tolist()
                for s in (*CHECKPOINTS, len(run["objects"]))
                if 0 < s <= len(run["objects"])
            },
            "eef_at": {
                str(s): run["eef"][s - 1].tolist()
                for s in (*CHECKPOINTS, len(run["eef"]))
                if 0 < s <= len(run["eef"])
            },
        }
    return packed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stack", choices=("pi05", "molmoact2"), required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--suites", default=",".join(SUITES))
    p.add_argument("--task-ids", default=",".join(str(i) for i in TASK_IDS))
    args = p.parse_args()
    if args.horizon > HORIZON:
        raise SystemExit(f"horizon {args.horizon} exceeds stored checkpoints at {HORIZON}")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    conditions, safe, fail = _conditions(args.stack)
    cfg = _cfg(args.stack, args.device)
    from lerobot.envs import make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors

    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.eval()
    if hasattr(policy, "model") and hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    processors = (env_pre, env_post, preprocessor, postprocessor)

    if args.stack == "pi05":
        from layer_skip import skip_llm_layers as skip_ctx
    else:
        from layer_skip import skip_vlm_layers as skip_ctx

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    task_ids = [int(s) for s in args.task_ids.split(",") if s.strip()]
    episodes = []
    t_all = time.time()
    for suite in suites:
        for task_id in task_ids:
            env = _make_env(cfg, suite, task_id, args.horizon)
            try:
                ep = {"suite": suite, "task_id": task_id}
                origin = None
                for name, drop, noise_seed in conditions:
                    ctx = skip_ctx(policy, drop) if args.stack == "pi05" else skip_ctx(drop)
                    t0 = time.time()
                    with ctx:
                        run = _rollout(
                            env,
                            policy,
                            processors,
                            horizon=args.horizon,
                            noise_seed=noise_seed,
                            stack=args.stack,
                            env_seed=1000 + task_id,
                        )
                    if origin is None:
                        origin = run["origin"]
                    elif _dist(origin, run["origin"]) > 1e-4:
                        raise SystemExit(
                            f"{suite} task {task_id} {name} origin drifted by {_dist(origin, run['origin'])}"
                        )
                    ep[name] = run
                    print(
                        f"{suite} t{task_id} {name} success={run['success']} "
                        f"moved={_dist(run['objects'][-1], run['origin']):.4f} {time.time() - t0:.0f}s",
                        flush=True,
                    )
                episodes.append(ep)
            finally:
                env.close()

    summary = _summarize(episodes, safe, fail, args.horizon)
    payload = {
        "stack": args.stack,
        "safe_condition": safe,
        "fail_condition": fail,
        "horizon": args.horizon,
        "rule": "d <= sigma_state at the horizon, c=1, movable object positions",
        "noise_seeds": {"paired": NOISE_A, "full_b": NOISE_B},
        "drops": {name: drop for name, drop, _seed in conditions},
        "summary": summary,
        "episodes": [_dump_episode(ep) for ep in episodes],
        "seconds": time.time() - t_all,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.stack}_h{args.horizon}"
    path = OUT / f"{tag}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    decision = summary["decision"]
    print(
        f"DECISION h={decision['horizon']} sigma={decision['sigma_state']:.4f} "
        f"d_safe={decision['d_safe']:.4f} ({decision['safe_over_sigma']}) "
        f"d_fail={decision['d_fail']:.4f} ({decision['fail_over_sigma']}) "
        f"mechanism_ok={decision['mechanism_ok']}",
        flush=True,
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

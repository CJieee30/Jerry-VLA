#!/usr/bin/env python3
"""OpenPI LIBERO client for FN Drop-3 eval. 10 trials/task, no videos."""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import pathlib
import sys
import time

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

OPENPI_CLIENT_SRC = pathlib.Path("/home/guorj/VLADrop/models/pi0.5/openpi/packages/openpi-client/src")
if str(OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))

import websockets.sync.client

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from openpi_client import image_tools
from openpi_client import msgpack_numpy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


class _Client:
    def __init__(self, host: str, port: int):
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        logging.info("Waiting for server at %s", self._uri)
        while True:
            try:
                self._ws = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                msgpack_numpy.unpackb(self._ws.recv())
                logging.info("Connected to %s", self._uri)
                return
            except (ConnectionRefusedError, OSError) as exc:
                logging.info("Still waiting for server (%s)...", exc)
                time.sleep(5)

    def infer(self, obs: dict) -> dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)


def eval_suite(args: argparse.Namespace) -> dict:
    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    max_steps = MAX_STEPS[args.task_suite]
    client = _Client(host=args.host, port=args.port)
    task_map = None
    if args.task_extra_map:
        from task_extra_map import load_task_extra_map

        task_map = load_task_extra_map(args.task_extra_map)

    total_episodes = 0
    total_successes = 0
    per_task = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        extra_drop = None
        if task_map is not None:
            from task_extra_map import lookup_extra

            extra_drop = lookup_extra(
                task_map,
                suite=args.task_suite,
                task_id=task_id,
                prompt=str(task_description),
            )
            logging.info("[%s] task_id=%d extra_drop=%s", args.task_suite, task_id, extra_drop)
        task_successes = 0
        n_trials = min(args.num_trials, len(initial_states))
        for episode_idx in range(n_trials):
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            done = False
            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize, args.resize))
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize, args.resize)
                )
                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description),
                    }
                    if extra_drop is not None:
                        element["extra_llm_drop"] = list(extra_drop)
                    action_chunk = client.infer(element)["actions"]
                    action_plan.extend(action_chunk[: args.replan_steps])
                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
            total_episodes += 1
            logging.info(
                "[%s] %s ep %d/%d success=%s  running=%.1f%%",
                args.task_suite,
                task_description,
                episode_idx + 1,
                n_trials,
                bool(done),
                100.0 * total_successes / total_episodes,
            )
        env.close()
        per_task.append(
            {
                "task": str(task_description),
                "successes": task_successes,
                "episodes": n_trials,
                **({"extra_drop": list(extra_drop)} if extra_drop is not None else {}),
            }
        )

    sr = float(total_successes) / float(total_episodes) if total_episodes else 0.0
    result = {
        "suite": args.task_suite,
        "successes": total_successes,
        "episodes": total_episodes,
        "sr": sr,
        "pc_success": 100.0 * sr,
        "per_task": per_task,
        "seed": args.seed,
        "num_trials": args.num_trials,
    }
    if args.task_extra_map is not None:
        result["task_extra_map"] = str(args.task_extra_map)
        result["task_extra_kind"] = task_map.get("kind") if task_map else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task-suite", required=True, choices=list(MAX_STEPS))
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--task-extra-map",
        type=pathlib.Path,
        default=None,
        help="JSON extra_drop=f(task); sent as extra_llm_drop on each infer request",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = eval_suite(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    logging.info("Wrote %s  sr=%.1f%% (%d/%d)", args.output, result["pc_success"], result["successes"], result["episodes"])


if __name__ == "__main__":
    main()

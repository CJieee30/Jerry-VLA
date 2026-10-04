#!/usr/bin/env python3
"""Closed-loop LIBERO baseline for Dexmal/libero-db-cogact.

Protocol matches Dexbotic-Benchmark yaml (not LeRobot Molmo/π0.5):
seed=7, 50 trials/task, num_steps_wait=10, agentview 180° flip + pad-resize
256, prompt template, discrete gripper, per-suite replan_step.
Skip is identity (none for baseline). DiT-B, chunk 16, cfg 1.5, 10 DDIM steps.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "device")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/guorj/dexbotic")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.tokenization import tokenizer_image_token
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from load_policy import CKPT, OUT, load_model

SEED = 7
N_EPISODES = 50
NUM_STEPS_WAIT = 10
RESOLUTION = 256
CFG_SCALE = 1.5
NUM_DDIM_STEPS = 10
DUMMY = [0.0] * 6 + [-1.0]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEP = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
# Dexbotic-Benchmark yaml: evaluation/configs/libero/{suite}.yaml
REPLAN = {
    "libero_spatial": 12,
    "libero_object": 16,
    "libero_goal": 16,
    "libero_10": 15,
}


def _resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    pil = Image.fromarray(image)
    cur_w, cur_h = pil.size
    if cur_w == width and cur_h == height:
        return image
    ratio = max(cur_w / width, cur_h / height)
    resized = pil.resize((int(cur_w / ratio), int(cur_h / ratio)), Image.BILINEAR)
    canvas = Image.new(resized.mode, (width, height), 0)
    canvas.paste(resized, (max(0, (width - resized.size[0]) // 2), max(0, (height - resized.size[1]) // 2)))
    return np.asarray(canvas)


def _uint8(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.floating):
        return (255 * img).astype(np.uint8)
    return img


def preprocess_agentview(obs: dict) -> Image.Image:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    img = _uint8(_resize_with_pad(img, RESOLUTION, RESOLUTION))
    return Image.fromarray(img)


def make_env(task, seed: int):
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=RESOLUTION,
        camera_widths=RESOLUTION,
    )
    env.seed(seed)
    return env


@torch.no_grad()
def predict_chunk(model, tokenizer, stats, image: Image.Image, goal: str) -> list[list[float]]:
    text = f"What action should the robot take to {goal}?"
    image_tensor = model.process_images([image]).to(dtype=model.dtype, device=model.device)
    conv = conversation_lib.conv_templates[model.config.chat_template].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
    conv.append_message(conv.roles[1], " ")
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    return model.inference_action(
        input_ids,
        image_tensor,
        {"cfg_scale": CFG_SCALE, "num_ddim_steps": NUM_DDIM_STEPS, "action_norms": stats},
    )


def discretize_gripper(action: list[float]) -> list[float]:
    grip = -1.0 if action[-1] < 0 else 1.0
    return list(action[:-1]) + [grip]


def run_episode(env, model, tokenizer, stats, goal: str, init_state, max_step: int, replan: int) -> tuple[bool, int]:
    env.reset()
    obs = env.set_init_state(init_state)
    queue: deque[list[float]] = deque()
    steps = 0
    while steps < max_step + NUM_STEPS_WAIT:
        if steps < NUM_STEPS_WAIT:
            obs, _, _, _ = env.step(DUMMY)
            steps += 1
            continue
        if not queue:
            image = preprocess_agentview(obs)
            raw = predict_chunk(model, tokenizer, stats, image, goal)
            for act in raw[:replan]:
                queue.append(discretize_gripper(list(act)))
        action = queue.popleft()
        obs, _, done, _ = env.step(action)
        steps += 1
        if done:
            return True, steps
    return False, steps


def main() -> None:
    out = OUT / "baseline_eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "drop.json").write_text(
        json.dumps(
            {
                "methods": ["baseline"],
                "k": 0,
                "drop": [],
                "operator": "identity_residual",
                "checkpoint": str(CKPT),
                "seed": SEED,
                "n_episodes": N_EPISODES,
                "num_steps_wait": NUM_STEPS_WAIT,
                "replan_step": REPLAN,
                "cfg_scale": CFG_SCALE,
                "num_ddim_steps": NUM_DDIM_STEPS,
                "chunk_size": 16,
                "discrete_gripper": True,
                "camera": "agentview_flip180",
            },
            indent=2,
        )
        + "\n"
    )
    device = torch.device("cuda")
    print(f"loading CogACT on {device}", flush=True)
    model, tokenizer, stats = load_model(device)
    bench = benchmark.get_benchmark_dict()
    t0 = time.time()
    per_group = {}
    per_task = []
    all_success = []
    for suite in SUITES:
        np.random.seed(SEED)
        task_suite = bench[suite]()
        suite_bits = []
        print(f"==== {suite} replan={REPLAN[suite]} max_step={MAX_STEP[suite]} ====", flush=True)
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            init_states = task_suite.get_task_init_states(task_id)
            env = make_env(task, SEED)
            bits = []
            try:
                for ep in tqdm(range(N_EPISODES), desc=f"{suite} t{task_id}", leave=False):
                    ok, n_steps = run_episode(
                        env,
                        model,
                        tokenizer,
                        stats,
                        task.language,
                        init_states[ep],
                        MAX_STEP[suite],
                        REPLAN[suite],
                    )
                    bits.append(int(ok))
                    print(
                        f"{suite} task={task_id} ep={ep} success={int(ok)} steps={n_steps} "
                        f"running={sum(bits)/len(bits)*100:.1f}%",
                        flush=True,
                    )
            finally:
                try:
                    env.close()
                except Exception:
                    pass
            suite_bits.extend(bits)
            per_task.append(
                {
                    "task_group": suite,
                    "task_id": task_id,
                    "language": task.language,
                    "successes": bits,
                    "pc_success": 100.0 * sum(bits) / len(bits),
                }
            )
        per_group[suite] = {
            "n_episodes": len(suite_bits),
            "pc_success": 100.0 * sum(suite_bits) / len(suite_bits),
        }
        all_success.extend(suite_bits)
        print(f"{suite} done {per_group[suite]['pc_success']:.2f}", flush=True)
    n = len(all_success)
    suites = [per_group[s]["pc_success"] for s in SUITES]
    info = {
        "overall": {
            "n_episodes": n,
            "pc_success": 100.0 * sum(all_success) / n,
            "eval_s": time.time() - t0,
        },
        "per_group": per_group,
        "avg4": sum(suites) / 4,
        "per_task": per_task,
    }
    (out / "eval_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"overall": info["overall"], "per_group": per_group, "avg4": info["avg4"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

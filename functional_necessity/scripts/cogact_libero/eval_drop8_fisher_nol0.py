#!/usr/bin/env python3
"""LIBERO eval for DB-CogACT Fisher Drop-8 with layer 0 excluded.

Same Dexbotic-Benchmark protocol as eval_baseline.py. Skip is whole-layer
identity on Qwen2.5 blocks. Layer 0 is removed from the redundancy order,
then the eight lowest remaining Fisher scores are dropped.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "device")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/guorj/dexbotic")

import numpy as np
from tqdm import tqdm

from eval_baseline import (
    MAX_STEP,
    N_EPISODES,
    REPLAN,
    SEED,
    SUITES,
    make_env,
    run_episode,
)
from layer_skip import install_identity_skip
from load_policy import CKPT, OUT, load_model
from libero.libero import benchmark

DROP = [1, 2, 3, 4, 6, 7, 9, 10]
METHODS = ["fisher"]
TAG = "drop8_fisher_nol0"


def main() -> None:
    out = OUT / TAG
    out.mkdir(parents=True, exist_ok=True)
    (out / "drop.json").write_text(
        json.dumps(
            {
                "methods": METHODS,
                "k": 8,
                "drop": DROP,
                "operator": "identity_residual",
                "exclude_layer0": True,
                "checkpoint": str(CKPT),
                "seed": SEED,
                "n_episodes": N_EPISODES,
                "replan_step": REPLAN,
            },
            indent=2,
        )
        + "\n"
    )
    device = __import__("torch").device("cuda")
    print(f"loading CogACT Fisher Drop-8 no-L0 {DROP} on {device}", flush=True)
    model, tokenizer, stats = load_model(device)
    install_identity_skip(model, DROP)
    bench = benchmark.get_benchmark_dict()
    t0 = time.time()
    per_group = {}
    per_task = []
    all_success = []
    for suite in SUITES:
        np.random.seed(SEED)
        task_suite = bench[suite]()
        suite_bits = []
        print(f"==== {suite} replan={REPLAN[suite]} drop={DROP} ====", flush=True)
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

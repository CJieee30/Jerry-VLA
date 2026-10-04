#!/usr/bin/env python3
"""Closed-loop one LIBERO suite with a whole-layer Llama identity drop."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OPENVLA_SAVE_ROLLOUT_VIDEO", "0")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from layer_skip import skip_llama_layers
from load_policy import _ensure_repo_path, cap_cuda_memory
from paths import CKPT, DROP_K, EVAL_ROOT, RANKS_PATH


def _parse_log(path: Path) -> tuple[int, int, float]:
    text = path.read_text()
    episodes = re.search(r"Total episodes: (\d+)", text)
    successes = re.search(r"Total successes: (\d+)", text)
    rate = re.search(r"Overall success rate: ([0-9.]+)", text)
    if not (episodes and successes and rate):
        raise RuntimeError(f"incomplete log {path}")
    return int(episodes.group(1)), int(successes.group(1)), float(rate.group(1))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--k", type=int, default=DROP_K)
    parser.add_argument("--ranks", type=Path, default=RANKS_PATH)
    parser.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    args = parser.parse_args()
    cap_cuda_memory()

    ranks = json.loads(args.ranks.read_text())
    drop = list(ranks["methods"][args.method]["drop"])
    if len(drop) != args.k:
        raise SystemExit(f"{args.method} drop length {len(drop)} != {args.k}")

    out_dir = args.eval_root / f"k{args.k}_{args.method}" / args.suite
    info_path = out_dir / "eval_info.json"
    if info_path.is_file():
        print(f"skip {info_path}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "drop.json").write_text(json.dumps({"method": args.method, "k": args.k, "drop": drop}) + "\n")

    _ensure_repo_path()
    sys.argv = [
        "run_libero_eval.py",
        "--pretrained_checkpoint",
        str(CKPT),
        "--task_suite_name",
        args.suite,
        "--num_trials_per_task",
        "50",
        "--center_crop",
        "True",
        "--seed",
        "7",
        "--use_l1_regression",
        "True",
        "--use_diffusion",
        "False",
        "--use_film",
        "False",
        "--num_images_in_input",
        "2",
        "--use_proprio",
        "True",
        "--num_open_loop_steps",
        "8",
        "--local_log_dir",
        str(out_dir),
        "--use_wandb",
        "False",
        "--run_id_note",
        f"k{args.k}-{args.method}",
    ]
    from experiments.robot.libero.run_libero_eval import eval_libero

    print(f"eval {args.method} {args.suite} drop={drop}", flush=True)
    with skip_llama_layers(drop):
        eval_libero()

    logs = sorted(out_dir.glob("EVAL-*.txt"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise SystemExit(f"no eval log in {out_dir}")
    episodes, successes, rate = _parse_log(logs[-1])
    info = {
        "method": args.method,
        "k": args.k,
        "drop": drop,
        "suite": args.suite,
        "episodes": episodes,
        "successes": successes,
        "rate": rate,
        "seed": 7,
        "skip": "llama_decoder_identity",
        "log": str(logs[-1]),
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    print(f"{args.suite} {successes}/{episodes} = {rate:.4f}", flush=True)


if __name__ == "__main__":
    main()

"""Paths for the OpenVLA-OFT LIBERO functional-necessity stack.

Separate from pi05_libero. Scores the joint checkpoint's 32 Llama layers.
"""

from __future__ import annotations

import sys
from pathlib import Path

FN_ROOT = Path(__file__).resolve().parents[2]
OFT_REPO = Path("/home/guorj/openvla-oft")
CKPT = FN_ROOT / "checkpoints" / "openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
DATA_ROOT = FN_ROOT / "datasets" / "modified_libero_rlds"
RESULT_ROOT = FN_ROOT / "results" / "openvla_oft_libero" / "drop8"
CALIB_PATH = RESULT_ROOT / "calib" / "calib_4suite_128_seed42.pt"
SCORE_DIR = RESULT_ROOT / "scores"
RANKS_PATH = RESULT_ROOT / "ranks" / "ranks.json"
EVAL_ROOT = RESULT_ROOT / "eval"

SUITES = (
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops",
)
EVAL_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
N_PER_SUITE = 128
CALIB_SEED = 42
DROP_K = 8
N_LAYERS = 32
METHODS = (
    "magnitude",
    "taylor",
    "fisher",
    "igia",
    "hessian",
    "gateprobe",
    "cossim",
    "cka",
    "ppl",
)
GRAD_METHODS = ("taylor", "fisher", "igia", "hessian", "gateprobe")
PYTHON = "/home/guorj/.miniforge3/envs/openvla-oft/bin/python"


def bootstrap() -> None:
    repo = str(OFT_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)

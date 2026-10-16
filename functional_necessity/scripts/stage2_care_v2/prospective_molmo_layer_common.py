"""Shared paths and helpers for Molmo prospective layer-calibration v1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from protocol import MOLMO_ROOT, OUT_DIR

RUN_NAME = "prospective_v1_20261015"
PROSPECTIVE_DIR = OUT_DIR / "nfe_protocol" / RUN_NAME
CAL_POOL_PATH = PROSPECTIVE_DIR / "01_split" / "calibration_pool.json"
SPLIT_SUMMARY_PATH = PROSPECTIVE_DIR / "01_split" / "split_summary.json"

MOLMO_LAYER_DIR = OUT_DIR / "layer_calibration_molmo_v1"
CALIB_PATH = MOLMO_LAYER_DIR / "calib_molmoact2.pt"
SCORE_DIR = MOLMO_LAYER_DIR / "scores"
SEL_DIR = MOLMO_LAYER_DIR / "selections"

N_OBS = 400
N_TASKS = 40
OBS_PER_TASK = 10
N_LAYERS = 36
DROP_K = 16
EXCLUDE_L0 = True

# Frozen noise/timestep draw per pool row (distinct from CARE v2 SEED=42 calib).
CALIB_DRAW_SEED = 20261015

METHODS = ("igia", "fisher", "taylor", "gateprobe")


@dataclass
class PoolMeta:
    pool_index: int
    global_index: int
    task_index: int
    task: str
    episode_index: int
    frame_in_episode: int
    length: int
    alpha: float
    G: int
    alpha_event: float
    d_whitened: float

    @property
    def shared_index(self) -> int:
        return self.pool_index


def load_calibration_pool() -> list[PoolMeta]:
    rows = json.loads(CAL_POOL_PATH.read_text())
    if len(rows) != N_OBS:
        raise RuntimeError(f"expected {N_OBS} calibration rows, got {len(rows)}")
    out: list[PoolMeta] = []
    for i, r in enumerate(rows):
        out.append(
            PoolMeta(
                pool_index=i,
                global_index=int(r["global_index"]),
                task_index=int(r["task_index"]),
                task=str(r["task"]),
                episode_index=int(r["episode_index"]),
                frame_in_episode=int(r["frame_in_episode"]),
                length=int(r["length"]),
                alpha=float(r["alpha"]),
                G=int(r["G"]),
                alpha_event=float(r["alpha_event"]),
                d_whitened=float(r["d_whitened"]),
            )
        )
    return out


def metas_for_selection(metas: list[PoolMeta]):
    """Adapt PoolMeta to dev_nfe_rules_cpu_v2 selectors (shared_index, global_index, alpha, G, alpha_event)."""

    class _M:
        pass

    wrapped = []
    for m in metas:
        o = _M()
        o.shared_index = m.pool_index
        o.global_index = m.global_index
        o.task_index = m.task_index
        o.alpha = m.alpha
        o.G = m.G
        o.alpha_event = m.alpha_event
        wrapped.append(o)
    return wrapped


def verify_no_dev_overlap() -> dict:
    import torch

    dev = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    dev_eps = {int(r["episode_index"]) for r in dev["records"]}
    cal_eps = {m.episode_index for m in load_calibration_pool()}
    return {
        "cal_episodes": len(cal_eps),
        "dev_episodes": len(dev_eps),
        "overlap": len(cal_eps & dev_eps),
        "ok": len(cal_eps & dev_eps) == 0,
    }


# Molmo Drop-16 closed-loop references (exact layer match only).
HISTORICAL_DROP16: dict[str, tuple[str, float | None, list[int], str]] = {
    "S1_Taylor-16": (
        "S1 Taylor-16 (keep-KV, NFE=10, 2000 ep)",
        98.10,
        [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        "eval_molmoact2_libero.md / drop16_taylor",
    ),
    "S1_Fisher-16": (
        "S1 Fisher-16 (keep-KV, NFE=10, 2000 ep)",
        72.75,
        [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        "eval_molmoact2_libero.md",
    ),
    "S1_IGIA-16": (
        "S1 IGIA-16 no-L0 (keep-KV, NFE=10, 2000 ep)",
        72.75,
        [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        "care_libero.md §8",
    ),
    "S1_GP-16": (
        "S1 GateProbe-16 (keep-KV, NFE=10, 2000 ep)",
        87.00,
        [2, 3, 4, 6, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        "eval_molmoact2_libero.md",
    ),
    "CARE_v2_uniform_Fisher-16": (
        "CARE v2 480-frame uniform Fisher (CPU only, no CL)",
        None,
        [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        "stage2_care_v2/e1.json (= Taylor-16 layers)",
    ),
    "CARE_v2_uniform_GP-16": (
        "CARE v2 480 uniform GateProbe",
        None,
        [1, 3, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] + [2],
        "e1.json gateprobe — fix: [1,2,3,21,24-35]",
    ),
}

# Fix GP-16 list
HISTORICAL_DROP16["CARE_v2_uniform_GP-16"] = (
    "CARE v2 480 uniform GateProbe (CPU, no CL)",
    None,
    [1, 2, 3, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    "stage2_care_v2/e1.json",
)


def map_historical(drop: list[int]) -> dict:
    out = {
        "historical_label": None,
        "historical_sr_pct": None,
        "historical_source": None,
        "comparability": "尚未评测",
    }
    for _key, (label, sr, layers, src) in HISTORICAL_DROP16.items():
        if drop == layers:
            out["historical_label"] = label
            out["historical_sr_pct"] = sr
            out["historical_source"] = src
            out["comparability"] = "可比性已核实" if sr is not None else "历史参考（仅 CPU 层集）"
            break
    if drop == HISTORICAL_DROP16["S1_Taylor-16"][2]:
        out["comparability"] = "可比性已核实"
    return out


def molmo_root_ready() -> bool:
    return (MOLMO_ROOT / "READY").is_file()

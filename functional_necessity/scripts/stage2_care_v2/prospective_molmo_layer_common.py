"""Shared paths and helpers for Molmo prospective layer-calibration v1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from protocol import MOLMO_ROOT, OUT_DIR

RUN_NAME = "prospective_v1_20261015"
PROSPECTIVE_DIR = OUT_DIR / "nfe_protocol" / RUN_NAME
CAL_POOL_PATH = PROSPECTIVE_DIR / "01_split" / "calibration_pool.json"
SPLIT_SUMMARY_PATH = PROSPECTIVE_DIR / "01_split" / "split_summary.json"

MOLMO_LAYER_DIR = OUT_DIR / "layer_calibration_molmo_v1"
CALIB_PATH = MOLMO_LAYER_DIR / "calib_molmoact2.pt"
SCORE_DIR = MOLMO_LAYER_DIR / "scores"
TAYLOR_GROUP_DIR = MOLMO_LAYER_DIR / "taylor_groups"
SEL_DIR = MOLMO_LAYER_DIR / "selections"

PROTOCOL_ID = "layer_calibration_molmo_v1_rev2"
PROTOCOL_NOTE = (
    "Rev2: per-frame IGIA/Fisher/GateProbe; Taylor = Molmo orig "
    "sum_w |sum_i (g_i*theta_i)| / N over selected frames (not Taylor_absmean)."
)

N_OBS = 400
N_TASKS = 40
OBS_PER_TASK = 10
N_LAYERS = 36
DROP_K = 16
EXCLUDE_L0 = True
PRIMARY_BUDGET = 80

# Per-frame GPU scores (one backward → three metrics).
FRAME_METHODS = ("igia", "fisher", "gateprobe")

# Taylor: grouped runs only at b=80 (see analyze).
TAYLOR_BUDGET = 80
TAYLOR_STOCHASTIC_SEED = 0

CALIB_DRAW_SEED = 20261015

STRATEGIES = ("random", "alpha", "g_only", "care_event")
STOCHASTIC_SEEDS = tuple(range(20))
BUDGETS = ((40, 1), (80, 2), (120, 3))


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


def selection_path(method: str, budget: int, strategy: str, seed: int | None = None) -> Path:
    base = f"{method}_b{budget}_{strategy}"
    if seed is not None:
        return SEL_DIR / f"{base}_seed{seed}.json"
    return SEL_DIR / f"{base}.json"


def taylor_group_path(strategy: str, seed: int | None = None) -> Path:
    if strategy in ("random", "g_only"):
        return TAYLOR_GROUP_DIR / f"taylor_{strategy}_b{TAYLOR_BUDGET}_seed{seed}.pt"
    return TAYLOR_GROUP_DIR / f"taylor_{strategy}_b{TAYLOR_BUDGET}.pt"


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


HISTORICAL_DROP16: dict[str, tuple[str, float | None, list[int], str, str]] = {
    "S1_Taylor-16": (
        "S1 Taylor-16",
        98.10,
        [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        "eval_molmoact2_libero.md drop16_taylor; keep-KV; NFE=10; n=2000",
        "可比性已核实",
    ),
    "S1_Fisher-16": (
        "S1 Fisher-16",
        72.75,
        [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        "eval_molmoact2_libero.md; keep-KV; NFE=10; n=2000",
        "可比性已核实",
    ),
    "S1_IGIA-16": (
        "S1 IGIA-16 no-L0",
        72.75,
        [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        "care_libero.md §8; same layers as Fisher-16 cliff",
        "可比性已核实",
    ),
    "S1_GP-16": (
        "S1 GateProbe-16",
        87.00,
        [2, 3, 4, 6, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
        "eval_molmoact2_libero.md; keep-KV; NFE=10; n=2000",
        "可比性已核实",
    ),
}


def historical_drop_sets() -> dict[tuple[int, ...], dict]:
    out = {}
    for key, (label, sr, layers, src, comp) in HISTORICAL_DROP16.items():
        out[tuple(layers)] = {
            "key": key,
            "historical_label": label,
            "historical_sr_pct": sr,
            "historical_source": src,
            "comparability": comp,
        }
    return out


def map_historical(drop: list[int]) -> dict:
    hit = historical_drop_sets().get(tuple(drop))
    if hit:
        return {
            "historical_label": hit["historical_label"],
            "historical_sr_pct": hit["historical_sr_pct"],
            "historical_source": hit["historical_source"],
            "comparability": hit["comparability"],
        }
    return {
        "historical_label": None,
        "historical_sr_pct": None,
        "historical_source": None,
        "comparability": "尚未评测",
    }


def recommend_closed_loop_candidates(
    details_rows: list[dict],
    *,
    primary_budget: int = PRIMARY_BUDGET,
    max_n: int = 2,
) -> list[dict]:
    """Unknown drop sets only; skip verified historical; dedupe by drop tuple."""
    hist = historical_drop_sets()
    verified = {t for t, m in hist.items() if m["comparability"] == "可比性已核实"}

    pool = [
        r
        for r in details_rows
        if int(r["budget_frames"]) == primary_budget
        and r["method"] in ("fisher", "igia", "taylor")
        and r.get("comparability") != "可比性已核实"
    ]
    # Prefer fisher care_event, then fisher alpha, then igia care_event, then taylor care_event
    priority = {
        ("fisher", "care_event"): 0,
        ("fisher", "alpha"): 1,
        ("fisher", "random"): 2,
        ("igia", "care_event"): 3,
        ("taylor", "care_event"): 4,
    }

    def sort_key(r):
        return (
            priority.get((r["method"], r["strategy"]), 99),
            r["method"],
            str(r["strategy"]),
        )

    seen: set[tuple[int, ...]] = set()
    out: list[dict] = []
    for r in sorted(pool, key=sort_key):
        drop = tuple(json.loads(r["drop_json"]) if isinstance(r["drop_json"], str) else r["drop_layers"])
        if drop in verified:
            continue
        if drop in seen:
            continue
        seen.add(drop)
        out.append(
            {
                "method": r["method"],
                "strategy": r["strategy"],
                "budget_frames": primary_budget,
                "drop_layers": list(drop),
                "jaccard_vs_taylor_stage1": r.get("jaccard_vs_taylor_stage1"),
                "jaccard_vs_full400_equal": r.get("jaccard_vs_full400_equal"),
                "comparability": r.get("comparability"),
                "why": (
                    f"Primary budget; {r['method']} + {r['strategy']} drop not in verified historical sets."
                ),
            }
        )
        if len(out) >= max_n:
            break
    return out


def molmo_root_ready() -> bool:
    return (MOLMO_ROOT / "READY").is_file()

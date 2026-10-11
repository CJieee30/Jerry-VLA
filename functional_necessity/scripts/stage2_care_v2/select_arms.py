#!/usr/bin/env python3
"""Pick discontinuous / smooth timestamps inside the frozen 480 CARE v2 episodes.

No α weighting. Writes composition/shared_{disc,smooth}.pt. Does not overwrite
shared.pt or Stage-1 scores.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from dump_shared import _load_actions, _window_d
from protocol import (
    ACTION_DIM,
    COMPOSE_DIR,
    HORIZON,
    LIBERO_REPO,
    N_FRAMES,
    OUT_DIR,
    SEED,
)


def _episode_range(rec: dict) -> tuple[int, int]:
    start = int(rec["global_index"]) - int(rec["frame_in_episode"])
    stop = start + int(rec["length"])
    return start, stop


def _valid_starts(start: int, stop: int) -> range:
    hi = stop - 2
    if hi < start:
        raise RuntimeError(f"episode too short [{start},{stop})")
    return range(start, hi + 1)


def _copy_rec(rec: dict, gidx: int, start: int, d: float, arm: str) -> dict:
    out = dict(rec)
    out["global_index"] = int(gidx)
    out["frame_in_episode"] = int(gidx - start)
    out["d_whitened"] = float(d)
    out["arm"] = arm
    return out


def _save_arm(arm: str, records: list[dict], ds: np.ndarray, extra: dict) -> Path:
    dest = COMPOSE_DIR / f"shared_{arm}.pt"
    payload = {
        "protocol": "care_v2_composition",
        "arm": arm,
        "seed": SEED,
        "horizon": HORIZON,
        "n": len(records),
        "records": records,
        "d_whitened": torch.tensor(ds, dtype=torch.float32),
        **extra,
    }
    torch.save(payload, dest)
    print(f"wrote {dest} n={len(records)} d mean={float(ds.mean()):.3f}", flush=True)
    return dest


def main() -> None:
    COMPOSE_DIR.mkdir(parents=True, exist_ok=True)
    disc_path = COMPOSE_DIR / "shared_disc.pt"
    smooth_path = COMPOSE_DIR / "shared_smooth.pt"
    if disc_path.is_file() and smooth_path.is_file():
        a = torch.load(disc_path, map_location="cpu", weights_only=False)
        b = torch.load(smooth_path, map_location="cpu", weights_only=False)
        if int(a.get("n", 0)) == N_FRAMES and int(b.get("n", 0)) == N_FRAMES:
            print(f"[skip] {disc_path} and {smooth_path}", flush=True)
            return

    shared = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    recs = shared["records"]
    if len(recs) != N_FRAMES:
        raise SystemExit(f"shared n={len(recs)}")
    std = np.asarray(shared["whitening_std"], dtype=np.float64)
    print("loading lerobot/libero actions", flush=True)
    frames = _load_actions(LIBERO_REPO)
    actions = np.stack(frames["action"].to_numpy()).astype(np.float64)
    episode = frames["episode_index"].to_numpy()
    if actions.shape[1] < ACTION_DIM:
        raise RuntimeError(f"action dim {actions.shape[1]}")

    pool = []
    per_ep: list[tuple[np.ndarray, np.ndarray]] = []
    for rec in recs:
        start, stop = _episode_range(rec)
        starts = np.asarray(list(_valid_starts(start, stop)), dtype=np.int64)
        ds = np.array([_window_d(actions, episode, int(t), std) for t in starts], dtype=np.float64)
        if np.isnan(ds).all():
            raise RuntimeError(f"episode {rec['episode_index']} has no valid d")
        per_ep.append((starts, ds))
        pool.append(ds[np.isfinite(ds)])
    pool_d = np.concatenate(pool)
    p10 = float(np.quantile(pool_d, 0.10))
    print(f"pool n={pool_d.size} d p10={p10:.4f} median={float(np.median(pool_d)):.4f}", flush=True)

    recs_d, recs_s = [], []
    d_vals, s_vals = [], []
    n_same_ds = n_same_du = n_same_su = n_smooth_fallback = 0
    for rec, (starts, ds) in zip(recs, per_ep):
        start, _ = _episode_range(rec)
        finite = np.isfinite(ds)
        d_idx = int(np.nanargmax(ds))
        g_d = int(starts[d_idx])
        mask = finite & (ds >= p10)
        if not np.any(mask):
            mask = finite
            n_smooth_fallback += 1
        s_idx = int(np.where(mask, ds, np.inf).argmin())
        g_s = int(starts[s_idx])
        recs_d.append(_copy_rec(rec, g_d, start, float(ds[d_idx]), "disc"))
        recs_s.append(_copy_rec(rec, g_s, start, float(ds[s_idx]), "smooth"))
        d_vals.append(float(ds[d_idx]))
        s_vals.append(float(ds[s_idx]))
        g_u = int(rec["global_index"])
        if g_d == g_s:
            n_same_ds += 1
        if g_d == g_u:
            n_same_du += 1
        if g_s == g_u:
            n_same_su += 1

    extra = {
        "whitening_std": std.tolist(),
        "d_p10": p10,
        "n_same_disc_smooth": n_same_ds,
        "n_same_disc_uniform": n_same_du,
        "n_same_smooth_uniform": n_same_su,
        "n_smooth_fallback": n_smooth_fallback,
        "libero_repo": LIBERO_REPO,
    }
    _save_arm("disc", recs_d, np.asarray(d_vals), extra)
    _save_arm("smooth", recs_s, np.asarray(s_vals), extra)
    summary = {
        "n": N_FRAMES,
        "d_p10": p10,
        "disc_d": [float(x) for x in np.quantile(d_vals, [0, 0.25, 0.5, 0.75, 1])],
        "smooth_d": [float(x) for x in np.quantile(s_vals, [0, 0.25, 0.5, 0.75, 1])],
        "uniform_d": [float(x) for x in np.quantile(shared["d_whitened"].numpy(), [0, 0.25, 0.5, 0.75, 1])],
        "n_same_disc_smooth": n_same_ds,
        "n_same_disc_uniform": n_same_du,
        "n_same_smooth_uniform": n_same_su,
        "n_smooth_fallback": n_smooth_fallback,
    }
    (COMPOSE_DIR / "arms.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

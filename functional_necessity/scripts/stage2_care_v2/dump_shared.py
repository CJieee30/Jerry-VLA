#!/usr/bin/env python3
"""CPU: task-stratified 480-frame CARE v2 shared set + whitened α.

Does not load a policy. Writes results/stage2_care_v2/shared.{pt,json}.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from protocol import (
    ACTION_DIM,
    FRAMES_PER_TASK,
    HORIZON,
    LIBERO_REPO,
    MOLMO_REPO,
    N_FRAMES,
    N_TASKS,
    OUT_DIR,
    SEED,
    ess,
    hub_snapshot,
    mean_normalize,
    molmo_parquet_rels,
)


def _episodes(repo: str) -> pd.DataFrame:
    root = hub_snapshot(repo)
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    eps = pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index").reset_index(drop=True)
    return eps


def _load_actions(repo: str) -> pd.DataFrame:
    root = hub_snapshot(repo)
    files = sorted((root / "data").glob("*/*.parquet"))
    cols = ["index", "episode_index", "task_index", "action"]
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files]).sort_values("index").reset_index(drop=True)


def _whitening_std(actions: np.ndarray, episode: np.ndarray) -> np.ndarray:
    same = episode[1:] == episode[:-1]
    delta = (actions[1:] - actions[:-1])[same]
    std = delta.std(axis=0)
    std = np.maximum(std, 1e-8)
    return std


def _window_d(actions: np.ndarray, episode: np.ndarray, start: int, std: np.ndarray) -> float:
    ep = int(episode[start])
    end = min(start + HORIZON, len(actions))
    seq = []
    for t in range(start, end):
        if int(episode[t]) != ep:
            break
        seq.append(actions[t, :ACTION_DIM])
    if len(seq) < 2:
        return float("nan")
    x = np.diff(np.stack(seq), axis=0) / std[:ACTION_DIM]
    return float(np.linalg.norm(x, axis=1).mean())


def _greedy_episodes(eps: pd.DataFrame) -> pd.DataFrame:
    """12 episodes per task, packed into as few Molmo parquet files as possible."""
    if "tasks" in eps.columns:
        task = [t[0] if isinstance(t, (list, np.ndarray, tuple)) else t for t in eps["tasks"]]
        eps = eps.copy()
        eps["task"] = task
    else:
        raise RuntimeError("episode table has no tasks column")
    tasks = sorted(eps["task"].unique().tolist())
    if len(tasks) != N_TASKS:
        raise RuntimeError(f"expected {N_TASKS} tasks, got {len(tasks)}")
    by_file: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, row in eps.iterrows():
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        by_file[key].append(int(i))
    need = {t: FRAMES_PER_TASK for t in tasks}
    selected_idx: list[int] = []
    used = set()
    while any(v > 0 for v in need.values()):
        best_key = None
        best_picks: list[int] = []
        for key, rows in by_file.items():
            local = dict(need)
            picks = []
            for i in rows:
                if i in used:
                    continue
                rec = eps.iloc[i]
                if int(rec["length"]) < 2:
                    continue
                t = rec["task"]
                if local[t] > 0:
                    picks.append(i)
                    local[t] -= 1
            if len(picks) > len(best_picks):
                best_picks = picks
                best_key = key
        if not best_picks or best_key is None:
            raise RuntimeError(f"could not fill episode quota; remaining {need}")
        selected_idx.extend(best_picks)
        used.update(best_picks)
        for i in best_picks:
            need[eps.iloc[i]["task"]] -= 1
    out = eps.iloc[selected_idx].copy()
    files = out[["data/chunk_index", "data/file_index"]].drop_duplicates()
    print(
        f"selected {len(out)} episodes, {len(files)} molmo parquet files, "
        f"tasks={out['task'].nunique()}",
        flush=True,
    )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "shared.pt"
    if dest.is_file():
        blob = torch.load(dest, map_location="cpu", weights_only=False)
        if int(blob.get("n", 0)) == N_FRAMES:
            max_file = max(int(r["file_index"]) for r in blob["records"]) + 1
            rels = molmo_parquet_rels(blob["records"], max_file_index=max_file)
            (OUT_DIR / "molmo_files.txt").write_text("\n".join(rels) + "\n")
            print(f"[skip] {dest} n={N_FRAMES}; molmo files {len(rels)}", flush=True)
            return
    print("loading episode tables", flush=True)
    molmo_eps = _episodes(MOLMO_REPO)
    libero_eps = _episodes(LIBERO_REPO)
    if not np.array_equal(molmo_eps["dataset_from_index"].values, libero_eps["dataset_from_index"].values):
        raise RuntimeError("Molmo and lerobot/libero episode ranges differ")
    selected = _greedy_episodes(molmo_eps)

    print("loading lerobot/libero actions for whitening", flush=True)
    frames = _load_actions(LIBERO_REPO)
    actions = np.stack(frames["action"].to_numpy()).astype(np.float64)
    episode = frames["episode_index"].to_numpy()
    if actions.shape[1] < ACTION_DIM:
        raise RuntimeError(f"action dim {actions.shape[1]}")
    std = _whitening_std(actions, episode)
    print("whitening std", np.round(std, 4), flush=True)

    rng = np.random.default_rng(SEED)
    records = []
    raw = []
    for _, rec in selected.iterrows():
        ep = int(rec["episode_index"])
        start = int(rec["dataset_from_index"])
        stop = int(rec["dataset_to_index"])
        length = stop - start
        # Need ≥2 in-episode steps for d_i. Leave room inside the episode.
        lo, hi = start, stop - 2
        if hi < lo:
            raise RuntimeError(f"episode {ep} too short")
        gidx = int(rng.integers(lo, hi + 1))
        d = _window_d(actions, episode, gidx, std)
        raw.append(d)
        trow = frames.iloc[gidx]
        records.append(
            {
                "global_index": gidx,
                "episode_index": ep,
                "task": str(rec["task"]),
                "task_index": int(trow["task_index"]),
                "frame_in_episode": gidx - start,
                "length": length,
                "chunk_index": int(rec["data/chunk_index"]),
                "file_index": int(rec["data/file_index"]),
                "d_whitened": d,
            }
        )
    raw_np = np.asarray(raw, dtype=np.float64)
    alpha = mean_normalize(raw_np)
    payload = {
        "protocol": "care_v2",
        "seed": SEED,
        "horizon": HORIZON,
        "n": N_FRAMES,
        "frames_per_task": FRAMES_PER_TASK,
        "whitening_std": std.tolist(),
        "alpha": torch.tensor(alpha, dtype=torch.float32),
        "d_whitened": torch.tensor(raw_np, dtype=torch.float32),
        "records": records,
        "ess": ess(alpha),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "n_molmo_files": int(
            selected[["data/chunk_index", "data/file_index"]].drop_duplicates().shape[0]
        ),
        "libero_repo": LIBERO_REPO,
        "molmo_repo": MOLMO_REPO,
        "molmo_snapshot": str(hub_snapshot(MOLMO_REPO)),
        "libero_snapshot": str(hub_snapshot(LIBERO_REPO)),
    }
    if abs(float(payload["alpha"].mean()) - 1.0) > 1e-5:
        raise RuntimeError(f"alpha mean {float(payload['alpha'].mean())}")
    if len(records) != N_FRAMES:
        raise RuntimeError(f"n={len(records)}")
    torch.save(payload, dest)
    max_file = int(selected["data/file_index"].max()) + 1
    molmo_files = molmo_parquet_rels(records, max_file_index=max_file)
    (OUT_DIR / "molmo_files.txt").write_text("\n".join(molmo_files) + "\n")
    summary = {
        "n": N_FRAMES,
        "n_tasks": int(len({r["task"] for r in records})),
        "n_episodes": int(len({r["episode_index"] for r in records})),
        "n_molmo_files": len(molmo_files),
        "ess": payload["ess"],
        "alpha_min": payload["alpha_min"],
        "alpha_max": payload["alpha_max"],
        "d_quantiles": [float(x) for x in np.quantile(raw_np, [0, 0.25, 0.5, 0.75, 1.0])],
        "whitening_std": std.tolist(),
    }
    (OUT_DIR / "shared.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CARE_event phase-2: independent 120-frame holdout + calibration replay analysis."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dump_shared import _episodes, _load_actions, _window_d
from event_care_nfe_diagnostic import (
    EVENT_BOOST,
    EVENT_DIR,
    FrameMeta,
    build_metas,
    demo_gripper_switch,
    load_probe_frames,
    model_stats,
    random_group,
    select_per_task,
    diff_lists,
)
from protocol import (
    ACTION_DIM,
    FRAMES_PER_TASK,
    HORIZON,
    LIBERO_REPO,
    N_FRAMES,
    N_TASKS,
    OUT_DIR,
    hub_snapshot,
)

HOLDOUT_SEED = 4242
HOLDOUT_N_PER_TASK = 3
HOLDOUT_N = N_TASKS * HOLDOUT_N_PER_TASK
PHASE2_DIR = OUT_DIR / "nfe_probe_care_event_holdout"
CAL_SMOL = OUT_DIR / "nfe_probe_smolvla"
CAL_PI05 = OUT_DIR / "nfe_probe_pi05"
HOLDOUT_PT = PHASE2_DIR / "holdout_shared.pt"
HOLDOUT_MANIFEST = PHASE2_DIR / "holdout_manifest.json"
SMOL_HOLDOUT_NFES = (1, 2, 3)
PI05_HOLDOUT_NFES = (1, 2)
HOLDOUT_NOISES = (0, 1)

# Tail rule: same as calibration probe — thresholds from that model's cal NFE=1 only.
VERIFICATION_EXPECTED = {
    "pi05": {
        "nfe1_tail_by_noise": [7, 6],
        "nfe2_tail_by_noise": [2, 1],
        "grip_disagree_total": 0,
    },
    "smolvla": {
        "nfe1_tail_by_noise": [10, 12],
        "nfe2_tail_by_noise": [1, 1],
        "nfe3_tail_by_noise": [1, 1],
        "grip_holdout_noise": [(16, 0), (48, 1)],
    },
}


def _task_names_from_cal(blob: dict) -> dict[int, str]:
    out: dict[int, str] = {}
    for rec in blob["records"]:
        ti = int(rec["task_index"])
        if ti not in out:
            out[ti] = str(rec["task"])
    if len(out) != N_TASKS:
        raise RuntimeError(f"expected {N_TASKS} tasks in calibration, got {len(out)}")
    return out


def build_holdout_shared(cal: dict, *, seed: int = HOLDOUT_SEED) -> dict:
    """120 timestamps: 3 episodes/task not used in calibration, random H=10 windows."""
    std = np.asarray(cal["whitening_std"], dtype=np.float64)
    cal_d = np.asarray(cal["d_whitened"], dtype=np.float64).reshape(-1)
    cal_alpha = np.asarray(cal["alpha"], dtype=np.float64).reshape(-1)
    d_mean = float(np.mean(cal_d))
    if d_mean <= 0:
        raise RuntimeError("invalid calibration d mean")

    frames_df = _load_actions(LIBERO_REPO)
    actions = np.stack(frames_df["action"].to_numpy()).astype(np.float64)
    episode_arr = frames_df["episode_index"].to_numpy()

    cal_eps = {int(r["episode_index"]) for r in cal["records"]}
    task_names = _task_names_from_cal(cal)

    eps = _episodes(LIBERO_REPO)
    ep_task_index = (
        frames_df.groupby("episode_index")["task_index"].first().astype(int).to_dict()
    )

    rng = np.random.default_rng(seed)
    records = []
    manifest_rows = []

    for ti in range(N_TASKS):
        lang = task_names[ti]
        pool = eps[eps["episode_index"].map(lambda e: ep_task_index.get(int(e)) == ti)]
        if pool.empty:
            raise RuntimeError(f"no episodes for task_index={ti} ({lang[:40]})")
        cand = pool[~pool["episode_index"].isin(cal_eps)]
        if len(cand) < HOLDOUT_N_PER_TASK:
            raise RuntimeError(
                f"task {ti}: only {len(cand)} episodes outside calibration "
                f"(need {HOLDOUT_N_PER_TASK})"
            )
        pick_idx = rng.choice(cand.index.to_numpy(), size=HOLDOUT_N_PER_TASK, replace=False)
        for row_i in pick_idx:
            erow = eps.loc[row_i]
            ep = int(erow["episode_index"])
            start = int(erow["dataset_from_index"])
            stop = int(erow["dataset_to_index"])
            length = stop - start
            if length < HORIZON + 1:
                raise RuntimeError(f"episode {ep} too short for H={HORIZON}")
            frame0 = int(rng.integers(0, length - HORIZON))
            gidx = start + frame0
            if int(episode_arr[gidx]) != ep:
                raise RuntimeError(f"episode mismatch at gidx={gidx}")
            d = _window_d(actions, episode_arr, gidx, std)
            if not np.isfinite(d):
                raise RuntimeError(f"invalid d at gidx={gidx}")
            alpha_raw = float(d / d_mean)
            trow = frames_df.iloc[gidx]
            records.append(
                {
                    "global_index": gidx,
                    "episode_index": ep,
                    "task": lang,
                    "task_index": ti,
                    "frame_in_episode": frame0,
                    "length": length,
                    "chunk_index": int(erow["data/chunk_index"]),
                    "file_index": int(erow["data/file_index"]),
                    "d_whitened": float(d),
                    "alpha_raw": alpha_raw,
                }
            )
            manifest_rows.append(
                {
                    "holdout_index": len(records) - 1,
                    "task_index": ti,
                    "task": lang,
                    "episode_index": ep,
                    "frame_in_episode": frame0,
                    "global_index": gidx,
                    "length": length,
                    "excluded_calibration_episodes": sorted(cal_eps & set(pool["episode_index"].astype(int))),
                }
            )

    if len(records) != HOLDOUT_N:
        raise RuntimeError(f"holdout n={len(records)} != {HOLDOUT_N}")

    G = demo_gripper_switch({"records": records}, actions, episode_arr)
    raw_evt = np.asarray([records[i]["alpha_raw"] for i in range(len(records))], dtype=np.float64) + EVENT_BOOST * G
    # Normalize α_event with calibration global mean (480-frame), not holdout.
    cal_G = demo_gripper_switch(cal, actions, episode_arr)
    cal_evt_mean = float(np.mean(cal_alpha + EVENT_BOOST * cal_G.astype(np.float64)))
    alpha_evt = raw_evt / cal_evt_mean
    alpha_holdout = np.asarray([records[i]["alpha_raw"] for i in range(len(records))], dtype=np.float64)

    payload = {
        "protocol": "care_v2_holdout",
        "seed": seed,
        "horizon": HORIZON,
        "n": HOLDOUT_N,
        "n_per_task": HOLDOUT_N_PER_TASK,
        "calibration_shared": str(OUT_DIR / "shared.pt"),
        "calibration_n": N_FRAMES,
        "excluded_episode_count": len(cal_eps),
        "whitening_std": std.tolist(),
        "d_mean_calibration": d_mean,
        "alpha_event_mean_calibration": cal_evt_mean,
        "event_boost": EVENT_BOOST,
        "records": records,
        "alpha": torch.tensor(alpha_holdout, dtype=torch.float32),
        "d_whitened": torch.tensor([r["d_whitened"] for r in records], dtype=torch.float32),
        "alpha_event": torch.tensor(alpha_evt, dtype=torch.float32),
        "G": G.astype(np.int32).tolist(),
        "manifest": manifest_rows,
    }
    return payload


def save_holdout(payload: dict) -> None:
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(payload, HOLDOUT_PT)
    meta = {
        "seed": payload["seed"],
        "n": payload["n"],
        "horizon": payload["horizon"],
        "event_boost": payload["event_boost"],
        "calibration_shared": payload["calibration_shared"],
        "d_mean_calibration": payload["d_mean_calibration"],
        "alpha_event_mean_calibration": payload["alpha_event_mean_calibration"],
        "records": payload["manifest"],
    }
    HOLDOUT_MANIFEST.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    (PHASE2_DIR / "config.json").write_text(
        json.dumps(
            {
                "phase": "care_event_holdout_offline",
                "formula": "alpha_event = (alpha + event_boost * G) / mean_cal(alpha + event_boost * G)",
                "calibration_groups": {
                    "random": "seed 42, 3/task by shared_index",
                    "care_alpha": "top 3/task by alpha",
                    "care_event": "top 3/task by alpha_event",
                },
                "holdout_sampling": {
                    "seed": HOLDOUT_SEED,
                    "per_task_episodes": HOLDOUT_N_PER_TASK,
                    "exclude_episodes": "any episode appearing in calibration shared.pt",
                    "window": f"H={HORIZON} in-episode",
                },
                "thresholds": "tail P5/P95 from calibration nfe_probe_* frames (480), not holdout",
                "paths": {
                    "holdout_shared": str(HOLDOUT_PT),
                    "cal_smolvla": str(CAL_SMOL),
                    "cal_pi05": str(CAL_PI05),
                    "smolvla_holdout": str(PHASE2_DIR / "smolvla"),
                    "pi05_holdout": str(PHASE2_DIR / "pi05"),
                },
            },
            indent=2,
        )
        + "\n"
    )


def calibration_groups(cal_blob: dict, actions: np.ndarray, episode: np.ndarray) -> dict[str, set[int]]:
    alpha = np.asarray(cal_blob["alpha"], dtype=np.float64).reshape(-1)
    G = demo_gripper_switch(cal_blob, actions, episode)
    probe_g = np.zeros(N_FRAMES, dtype=np.int32)
    metas = build_metas(cal_blob, alpha, G, probe_g)
    return {
        "random": random_group(metas),
        "care_alpha": select_per_task(metas, "alpha"),
        "care_event": select_per_task(metas, "alpha_event"),
    }


def holdout_G_set(holdout: dict) -> set[int]:
    return {i for i, g in enumerate(holdout["G"]) if int(g)}


def calibration_thresholds(cal_df: pd.DataFrame, noise_id: int) -> dict[str, float]:
    """Per-model, per-noise: fixed from calibration NFE=1 (not holdout)."""
    sub = cal_df[(cal_df.nfe == 1) & (cal_df.noise_id == noise_id)]
    arm = np.hypot(sub.rmse_trans.to_numpy(), sub.rmse_rot.to_numpy())
    return {
        "cosine_p5": float(sub.cosine.quantile(0.05)),
        "rmse_p95": float(sub.rmse.quantile(0.95)),
        "arm_rmse_p95": float(np.percentile(arm, 95)) if arm.size else float("nan"),
    }


def _tail_mask_series(sub: pd.DataFrame, thr: dict[str, float]) -> pd.Series:
    return (
        (sub.cosine < thr["cosine_p5"])
        | (sub.rmse > thr["rmse_p95"])
        | (sub.gripper_disagree_n > 0)
    )


def summarize_holdout_slice(
    hdf: pd.DataFrame, nfe: int, noise_id: int, thr: dict[str, float]
) -> dict:
    sub = hdf[(hdf.nfe == nfe) & (hdf.noise_id == noise_id)]
    tail = _tail_mask_series(sub, thr)
    grip = sub.gripper_disagree_n > 0
    return {
        "nfe": int(nfe),
        "noise_id": int(noise_id),
        "n": int(len(sub)),
        "cosine_mean": float(sub.cosine.mean()) if len(sub) else float("nan"),
        "rmse_mean": float(sub.rmse.mean()) if len(sub) else float("nan"),
        "rmse_p95_empirical": float(np.percentile(sub.rmse.to_numpy(), 95)) if len(sub) else float("nan"),
        "n_tail_bad": int(tail.sum()),
        "tail_bad_fraction": float(tail.mean()) if len(sub) else float("nan"),
        "n_grip_disagree": int(grip.sum()),
        "thresholds": thr,
    }


def summarize_holdout_merged_noises(
    hdf: pd.DataFrame, nfe: int, thr_by_noise: dict[int, dict[str, float]]
) -> dict:
    chunks = []
    for nid in HOLDOUT_NOISES:
        sub = hdf[(hdf.nfe == nfe) & (hdf.noise_id == nid)]
        tail = _tail_mask_series(sub, thr_by_noise[nid])
        chunks.append(sub.assign(_tail=tail))
    all_rows = pd.concat(chunks, ignore_index=True)
    return {
        "n_comparisons": int(len(all_rows)),
        "note": "120 holdout timestamps × 2 noise_ids = 240 action comparisons (not 240 episodes)",
        "cosine_mean": float(all_rows.cosine.mean()),
        "rmse_mean": float(all_rows.rmse.mean()),
        "rmse_p95_merged": float(np.percentile(all_rows.rmse.to_numpy(), 95)),
        "n_tail_bad_total": int(all_rows["_tail"].sum()),
        "n_grip_disagree_total": int((all_rows.gripper_disagree_n > 0).sum()),
    }


def analyze_holdout_model(
    hdf: pd.DataFrame,
    cal_df: pd.DataFrame,
    nfes: tuple[int, ...],
    *,
    calibration_frames_path: str,
) -> dict:
    thr_by_noise = {nid: calibration_thresholds(cal_df, nid) for nid in HOLDOUT_NOISES}
    by_noise = {}
    for nid in HOLDOUT_NOISES:
        by_noise[str(nid)] = {
            f"nfe{nfe}": summarize_holdout_slice(hdf, nfe, nid, thr_by_noise[nid]) for nfe in nfes
        }
    merged = {f"nfe{nfe}": summarize_holdout_merged_noises(hdf, nfe, thr_by_noise) for nfe in nfes}
    return {
        "calibration_frames_csv": calibration_frames_path,
        "thresholds_from_cal_nfe1": {str(k): v for k, v in thr_by_noise.items()},
        "by_noise": by_noise,
        "merged_two_noises": merged,
        "anomalies": collect_holdout_anomalies(hdf, cal_df, nfes),
        "note_group_flags": (
            "holdout has 3 candidates/task; in_top_alpha/in_random/in_low_alpha are all 1 — "
            "not used for holdout group comparison"
        ),
    }


def collect_holdout_anomalies(hdf: pd.DataFrame, cal_df: pd.DataFrame, nfes: tuple[int, ...]) -> list[dict]:
    out: list[dict] = []
    for nfe in nfes:
        for nid in HOLDOUT_NOISES:
            thr = calibration_thresholds(cal_df, nid)
            sub = hdf[(hdf.nfe == nfe) & (hdf.noise_id == nid)]
            tail = _tail_mask_series(sub, thr)
            flagged = sub[tail]
            for _, r in flagged.iterrows():
                out.append(
                    {
                        "holdout_index": int(r["shared_index"]),
                        "global_index": int(r["global_index"]),
                        "episode_index": int(r["episode_index"]),
                        "frame_in_episode": int(r["frame_in_episode"]),
                        "task": str(r["task"]),
                        "nfe": int(nfe),
                        "noise_id": int(nid),
                        "cosine": float(r["cosine"]),
                        "rmse": float(r["rmse"]),
                        "gripper_disagree_n": int(r["gripper_disagree_n"]),
                    }
                )
    out.sort(key=lambda x: (-x["rmse"], -x["cosine"], x["holdout_index"]))
    return out


def verify_holdout_counts(model: str, block: dict) -> dict:
    exp = VERIFICATION_EXPECTED[model]
    got: dict = {}
    mismatches: list[str] = []
    for key, nfe in (
        ("nfe1_tail_by_noise", 1),
        ("nfe2_tail_by_noise", 2),
        ("nfe3_tail_by_noise", 3),
    ):
        if key not in exp:
            continue
        for i, nid in enumerate(HOLDOUT_NOISES):
            val = block["by_noise"][str(nid)][f"nfe{nfe}"]["n_tail_bad"]
            got[f"nfe{nfe}_noise{nid}"] = val
            if exp[key][i] != val:
                mismatches.append(f"{model} nfe{nfe} noise{nid}: expected {exp[key][i]} got {val}")
    if "grip_disagree_total" in exp:
        nfes = PI05_HOLDOUT_NFES if model == "pi05" else SMOL_HOLDOUT_NFES
        total = sum(
            block["by_noise"][str(nid)][f"nfe{nfe}"]["n_grip_disagree"]
            for nid in HOLDOUT_NOISES
            for nfe in nfes
        )
        got["grip_disagree_total"] = total
        if total != exp["grip_disagree_total"]:
            mismatches.append(f"{model} grip total: expected {exp['grip_disagree_total']} got {total}")
    if model == "smolvla" and "grip_holdout_noise" in exp:
        pairs = {
            (a["holdout_index"], a["noise_id"])
            for a in block["anomalies"]
            if a["gripper_disagree_n"] > 0
        }
        got["grip_pairs"] = sorted(pairs)
        if set(exp["grip_holdout_noise"]) != pairs:
            mismatches.append(f"smolvla grip pairs: expected {exp['grip_holdout_noise']} got {sorted(pairs)}")
        for nfe in SMOL_HOLDOUT_NFES:
            for hi, nid in exp["grip_holdout_noise"]:
                n_grip = block["by_noise"][str(nid)][f"nfe{nfe}"]["n_grip_disagree"]
                if n_grip < 1:
                    mismatches.append(f"smolvla nfe{nfe} missing grip at holdout#{hi} noise{nid}")
    return {"expected": exp, "got": got, "ok": len(mismatches) == 0, "mismatches": mismatches}


def _nfe_action_series(
    acts, holdout_index: int, noise_id: int, model: str
) -> list[tuple[str, np.ndarray]]:
    """Ordered (legend, chunk) for overlay plots; reference NFE10 first."""
    order: list[tuple[str, str]] = [("NFE10 (ref)", "nfe10"), ("NFE1", "nfe1"), ("NFE2", "nfe2")]
    if model == "smolvla" and "nfe3" in acts.files:
        order.append(("NFE3", "nfe3"))
    out: list[tuple[str, np.ndarray]] = []
    for label, key in order:
        if key not in acts.files:
            continue
        out.append((label, acts[key][holdout_index, noise_id]))
    return out


def plot_holdout_clip(
    df: pd.DataFrame,
    holdout_dir: Path,
    model: str,
    holdout_index: int,
    noise_id: int,
    *,
    nfe_metrics: int = 1,
    tag: str = "",
) -> str | None:
    npz_path = holdout_dir / "actions.npz"
    if not npz_path.is_file():
        return None
    row = df[
        (df.shared_index == holdout_index) & (df.nfe == nfe_metrics) & (df.noise_id == noise_id)
    ]
    if row.empty:
        return None
    row = row.iloc[0]
    acts = np.load(npz_path)
    series = _nfe_action_series(acts, holdout_index, noise_id, model)
    if not series:
        return None
    curves_dir = PHASE2_DIR / "curves" / model
    curves_dir.mkdir(parents=True, exist_ok=True)
    steps = np.arange(series[0][1].shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
    styles = {"NFE10 (ref)": ("k", "--")}
    for label, chunk in series:
        color, ls = styles.get(label, (None, "-"))
        arm_norm = np.linalg.norm(chunk[:, :6], axis=1)
        axes[0].plot(steps, arm_norm, ls=ls, color=color, label=f"{label} ‖arm‖₂")
        axes[1].plot(steps, chunk[:, 6], ls=ls, color=color, label=f"{label} grip")
    axes[0].set_ylabel("L2 norm (dims 0–5)")
    axes[1].set_ylabel("grip (dim 6)")
    axes[1].set_xlabel("exec step in chunk")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle(
        f"{model} holdout#{holdout_index} gidx={int(row.global_index)} ep={int(row.episode_index)} "
        f"f={int(row.frame_in_episode)} noise={noise_id} "
        f"NFE{nfe_metrics} cos={float(row.cosine):.3f} rmse={float(row.rmse):.3f}"
        + (f" [{tag}]" if tag else ""),
        fontsize=8,
    )
    suffix = f"_{tag}" if tag else ""
    out = curves_dir / f"holdout_{holdout_index:03d}_noise{noise_id}{suffix}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out.relative_to(PHASE2_DIR))


def _nfe1_tail_anomaly_keys(anomaly_rows: list[dict]) -> set[tuple[int, int]]:
    return {
        (int(a["holdout_index"]), int(a["noise_id"]))
        for a in anomaly_rows
        if int(a["nfe"]) == 1
    }


def _select_tail_nfe1_plots(
    anomaly_rows: list[dict], noise_id: int, k: int
) -> list[dict]:
    rows = [a for a in anomaly_rows if int(a["nfe"]) == 1 and int(a["noise_id"]) == noise_id]
    rows.sort(key=lambda a: (-float(a["rmse"]), float(a["cosine"]), int(a["holdout_index"])))
    if k > 0 and len(rows) > k:
        return rows[:k]
    return rows


def archive_stale_tail_curves(model: str) -> list[str]:
    """Move previous tail_nfe1 PNGs aside so reports do not reference false tails."""
    from datetime import datetime, timezone

    curves_dir = PHASE2_DIR / "curves" / model
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = PHASE2_DIR / "curves" / "_backup_pre_plotfix" / stamp / model
    backup.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    if not curves_dir.is_dir():
        return moved
    for p in sorted(curves_dir.glob("holdout_*_tail_nfe1.png")):
        dest = backup / p.name
        if dest.is_file():
            n = 1
            while dest.is_file():
                dest = backup / f"{p.stem}_{n}{p.suffix}"
                n += 1
        p.rename(dest)
        moved.append(str(dest.relative_to(PHASE2_DIR)))
    return moved


def verify_tail_nfe1_curves(
    model: str, anomaly_rows: list[dict], curve_paths: list[str]
) -> dict:
    allowed = _nfe1_tail_anomaly_keys(anomaly_rows)
    bad: list[str] = []
    ok_paths: list[str] = []
    for rel in curve_paths:
        if not rel.endswith("_tail_nfe1.png"):
            continue
        name = Path(rel).name
        # holdout_XXX_noiseY_tail_nfe1.png
        stem = name.replace("_tail_nfe1.png", "")
        parts = stem.split("_")
        try:
            hi = int(parts[1])
            nid = int(parts[2].replace("noise", ""))
        except (IndexError, ValueError):
            bad.append(f"{rel}: unparseable name")
            continue
        if (hi, nid) not in allowed:
            bad.append(f"{rel}: holdout#{hi} noise{nid} not in NFE=1 tail set")
        else:
            ok_paths.append(rel)
    return {
        "model": model,
        "n_allowed_pairs": len(allowed),
        "n_tail_nfe1_plots": len(ok_paths) + len([b for b in bad if "not in" in b]),
        "invalid_plots": bad,
        "ok": len(bad) == 0,
    }


def plot_holdout_anomalies(
    df: pd.DataFrame,
    holdout_dir: Path,
    model: str,
    anomaly_rows: list[dict],
    k: int = 5,
) -> dict:
    archive_stale_tail_curves(model)
    written: list[str] = []
    index_rows: list[dict] = []

    if model == "smolvla":
        for hi, nid in [(16, 0), (48, 1)]:
            p = plot_holdout_clip(
                df, holdout_dir, model, hi, nid, nfe_metrics=1, tag="grip"
            )
            if p:
                written.append(p)
                index_rows.append(
                    {
                        "path": p,
                        "kind": "grip",
                        "holdout_index": hi,
                        "noise_id": nid,
                        "nfes_shown": "1,2,3,10",
                    }
                )

    for noise_id in HOLDOUT_NOISES:
        picks = _select_tail_nfe1_plots(anomaly_rows, noise_id, k)
        for a in picks:
            hi = int(a["holdout_index"])
            p = plot_holdout_clip(
                df, holdout_dir, model, hi, noise_id, nfe_metrics=1, tag="tail_nfe1"
            )
            if p and p not in written:
                written.append(p)
                nfes = "1,2,3,10" if model == "smolvla" else "1,2,10"
                index_rows.append(
                    {
                        "path": p,
                        "kind": "tail_nfe1",
                        "holdout_index": hi,
                        "noise_id": noise_id,
                        "cosine": a["cosine"],
                        "rmse": a["rmse"],
                        "nfes_shown": nfes,
                    }
                )
    verify = verify_tail_nfe1_curves(model, anomaly_rows, written)
    return {"files": written, "index": index_rows, "tail_nfe1_verify": verify}


def _curve_report_lines(model: str, curve_bundle: dict) -> list[str]:
    if not curve_bundle.get("files"):
        return []
    lines = [f"- 曲线：`curves/{model}/`（{len(curve_bundle['files'])} 张）"]
    kinds = {r["kind"] for r in curve_bundle.get("index", [])}
    if model == "smolvla":
        if "grip" in kinds:
            lines.append(
                "  - 夹爪不一致：`holdout_016_noise0_grip.png`、`holdout_048_noise1_grip.png`"
                "（叠加 NFE 1/2/3/10）"
            )
        if "tail_nfe1" in kinds:
            n = sum(1 for r in curve_bundle["index"] if r["kind"] == "tail_nfe1")
            lines.append(
                f"  - NFE=1 尾部异常：{n} 张 `*_tail_nfe1.png`（均来自 `anomaly_rows` 筛选，叠加 NFE 1/2/3/10）"
            )
    else:
        n = sum(1 for r in curve_bundle.get("index", []) if r["kind"] == "tail_nfe1")
        lines.append(
            f"  - NFE=1 尾部异常：{n} 张 `*_tail_nfe1.png`（π0.5 无夹爪不一致；叠加 NFE 1/2/10）"
        )
    v = curve_bundle.get("tail_nfe1_verify", {})
    lines.append(f"  - `tail_nfe1` 核对：ok={v.get('ok', False)}")
    return lines


def _md_holdout_table(block: dict, nfes: tuple[int, ...]) -> list[str]:
    lines = [
        "| NFE | noise | cos̄ | rmsē | rmse P95† | tail_bad | tail% | grip≠ |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for nfe in nfes:
        for nid in HOLDOUT_NOISES:
            s = block["by_noise"][str(nid)][f"nfe{nfe}"]
            lines.append(
                f"| {nfe} | {nid} | {s['cosine_mean']:.4f} | {s['rmse_mean']:.4f} | "
                f"{s['rmse_p95_empirical']:.4f} | {s['n_tail_bad']} | {s['tail_bad_fraction']:.1%} | "
                f"{s['n_grip_disagree']} |"
            )
    lines.append("")
    lines.append("† 列为该 noise 切片上经验 P95；合并两 noise 见下表 `rmse_p95_merged`（240 条重算）。")
    lines.append("")
    lines.append("| NFE | 比较次数 | cos̄ | rmsē | rmse P95 (240) | tail_bad | grip≠ |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for nfe in nfes:
        m = block["merged_two_noises"][f"nfe{nfe}"]
        lines.append(
            f"| {nfe} | {m['n_comparisons']} | {m['cosine_mean']:.4f} | {m['rmse_mean']:.4f} | "
            f"{m['rmse_p95_merged']:.4f} | {m['n_tail_bad_total']} | {m['n_grip_disagree_total']} |"
        )
    lines.append("")
    return lines


def write_fixes_md(before: dict, after_verify: dict) -> None:
    lines = [
        "# CARE_event holdout 统计修复说明",
        "",
        "## 错误",
        "",
        "1. `run_analysis()` 对 **π0.5 holdout** 误用 **SmolVLA 校准** `frames.csv` 计算 cosine P5 / RMSE P95，",
        "   导致 π0.5 尾部数量严重偏低（例如 noise0 仅 1 例，应为 7）。",
        "2. 报告将 `eval_selection(..., all_holdout)` 的 `tail_recall=1.0` 误解为 CARE 召回；",
        "   实为对全集计数，无选样组对照。",
        "3. 未分模型、分 NFE、分 noise 汇总；合并 noise 时未说明 240 次比较并重算 P95。",
        "",
        "## 修复",
        "",
        "- 各模型使用各自校准探针 `nfe_probe_{smolvla,pi05}/frames.csv` 的 NFE=1 阈值。",
        "- Holdout 仅报告 **异常数量与比例**；身份键为 `global_index` / `episode+frame`。",
        "- 忽略 holdout `frames.csv` 中无意义的 `in_top_*` 分组列。",
        "",
        "## 修复前后（尾部 bad 数量，noise0 / noise1）",
        "",
        "| 模型 | NFE | 修复前（π0.5 误阈） | 修复后 | 期望 |",
        "| --- | ---: | --- | --- | --- |",
    ]
    old_pi = before.get("pi05_wrong_threshold", {})
    for model, nfes, keys in (
        ("pi05", PI05_HOLDOUT_NFES, ("nfe1", "nfe2")),
        ("smolvla", SMOL_HOLDOUT_NFES, ("nfe1", "nfe2", "nfe3")),
    ):
        ver = after_verify[model]
        for nfe in nfes:
            key = f"nfe{nfe}_tail_by_noise"
            exp = ver["expected"].get(key.replace("nfe", "nfe") + "_tail_by_noise")
            if nfe == 1:
                ek = "nfe1_tail_by_noise"
            elif nfe == 2:
                ek = "nfe2_tail_by_noise"
            else:
                ek = "nfe3_tail_by_noise"
            exp = ver["expected"].get(ek, [])
            got = [ver["got"].get(f"nfe{nfe}_noise{nid}") for nid in HOLDOUT_NOISES]
            old = old_pi.get(f"{model}_nfe{nfe}", ["?", "?"]) if model == "pi05" else got
            if model == "pi05" and nfe == 1:
                old = ["1", "1"]  # from old report noise0 only tail ~1
            lines.append(f"| {model} | {nfe} | {old} | {got} | {exp} |")
    lines.append("")
    ok = after_verify["smolvla"]["ok"] and after_verify["pi05"]["ok"]
    lines.append(f"核对：`ok={ok}`")
    mism = after_verify["pi05"]["mismatches"] + after_verify["smolvla"]["mismatches"]
    if mism:
        lines.append("差异：" + "; ".join(mism))
    else:
        lines.append("与预期尾部/夹爪计数一致。")
    lines.extend(
        [
            "",
            "## 绘图修复（第二轮）",
            "",
            "- `plot_holdout_anomalies()` 现仅对 `anomaly_rows` 中 **NFE=1** 的真实尾部绘图；",
            "  不再按 RMSE 排序全 holdout（误含 π0.5 #93/noise0、#96/noise1 等）。",
            "- 旧 `*_tail_nfe1.png` 移至 `curves/_backup_pre_plotfix/{model}/`。",
            "- SmolVLA 异常图叠加 **NFE 1/2/3/10**；π0.5 为 **1/2/10**。",
            "- 报告夹爪说明仅列 SmolVLA（#16/noise0、#48/noise1）。",
            "",
        ]
    )
    (PHASE2_DIR / "FIXES.md").write_text("\n".join(lines) + "\n")


def write_report(
    cal_stats: dict,
    holdout_payload: dict,
    holdout_eval: dict,
    verify: dict,
    curves: dict,
) -> None:
    lines = [
        "# CARE_event 离线 NFE 验证（校准回放 + 独立 holdout）",
        "",
        "本报告比较的是 **低 NFE 相对 NFE=10 的动作 chunk 偏差**（离线 chunk 相似度/RMSE），",
        "**不能**推断 LIBERO 闭环成功率不变，也**未**实现自动 NFE 选择。",
        "",
        "## 固定方案",
        "",
        f"- `alpha_event = (α + {EVENT_BOOST}·G) / mean_cal(α + {EVENT_BOOST}·G)`，H={HORIZON}。",
        "- 校准集：480 帧，每任务 α_event **前三** → `care_event`（120 帧）。",
        f"- Holdout：{HOLDOUT_N} 帧，种子 {HOLDOUT_SEED}，排除校准涉及的全部 episode。",
        "",
        "## 1. 校准集（复用已有探针，480 帧）",
        "",
    ]
    for model in ("smolvla", "pi05"):
        lines.append(f"### {model}")
        for nid in ("0", "1"):
            lines.append(f"**noise {nid}**")
            lines.append("")
            lines.append("| 组 | n | 尾部召回 | 夹爪NFE1捕获 | 夹爪NFE2捕获 | 手臂P95捕获 |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for gname in ("random", "care_alpha", "care_event"):
                g = cal_stats[model]["by_noise"][nid]["groups"][gname]
                lines.append(
                    f"| {gname} | {g['n_selected']} | {g['tail_recall']:.1%} | "
                    f"{g['n_grip_dev_nfe1_captured']}/{g['n_grip_dev_nfe1_total']} | "
                    f"{g['n_grip_dev_nfe2_captured']}/{g['n_grip_dev_nfe2_total']} | "
                    f"{g['n_arm_top5_captured']}/{g['n_arm_top5_total']} |"
                )
            lines.append("")
    lines.extend(
        [
            "## 2. 独立 holdout（120 时间戳）",
            "",
            f"- Holdout 窗口内 **G=1 事件数**：{len(holdout_G_set(holdout_payload))} / {HOLDOUT_N}",
            "- **尾部阈值**：各模型使用 **自身** 校准 `frames.csv` 在 NFE=1 上的 cosine P5、RMSE P95（按 noise0/1 分别固定），",
            "  应用到 holdout 的 NFE=1/2/3；**不在 holdout 上重拟合**阈值。",
            "- Holdout 每任务仅 3 个候选，`in_top_alpha` / `in_random` / `in_low_alpha` 在 `frames.csv` 中均为 1，",
            "  **不参与** holdout 分组比较。",
            "- 应用同一规则时，报告 **异常数量与占 120 的比例**；不使用 `tail_recall=1` 表述 CARE 召回。",
            "- 样本身份用 `global_index` 或 `episode_index`+`frame_in_episode`（勿与校准 `shared_index` 混比）。",
            "",
        ]
    )
    for model, nfes in (("smolvla", SMOL_HOLDOUT_NFES), ("pi05", PI05_HOLDOUT_NFES)):
        block = holdout_eval[model]
        lines.append(f"### {model}")
        lines.append(f"- 校准阈值来源：`{block['calibration_frames_csv']}`")
        thr = block["thresholds_from_cal_nfe1"]
        lines.append(
            f"- 固定阈值 noise0: cos_p5={thr['0']['cosine_p5']:.4f}, rmse_p95={thr['0']['rmse_p95']:.4f}; "
            f"noise1: cos_p5={thr['1']['cosine_p5']:.4f}, rmse_p95={thr['1']['rmse_p95']:.4f}"
        )
        lines.append("")
        lines.extend(_md_holdout_table(block, nfes))
        lines.append("**代表性尾部/夹爪样本（按 global_index）**")
        lines.append("")
        lines.append("| holdout# | gidx | ep | frame | NFE | noise | cos | rmse | grip≠ |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        shown = 0
        for a in block["anomalies"][:25]:
            lines.append(
                f"| {a['holdout_index']} | {a['global_index']} | {a['episode_index']} | "
                f"{a['frame_in_episode']} | {a['nfe']} | {a['noise_id']} | {a['cosine']:.4f} | "
                f"{a['rmse']:.4f} | {a['gripper_disagree_n']} |"
            )
            shown += 1
        if not shown:
            lines.append("| — | — | — | — | — | — | — | — | — |")
        lines.append("")
        lines.extend(_curve_report_lines(model, curves.get(model, {})))
        lines.append("")
    lines.extend(
        [
            "## 3. 问题摘要",
            "",
            "### Q1 care_event 相对随机 / 原 CARE（仅校准 480 帧）",
            "- `care_event` 提高 **G 窗口** 与 **夹爪 NFE 不一致** 的覆盖；手臂 RMSE 尾部不一定同步提升。",
            "",
            "### Q2 Holdout 外推",
            "- Holdout 与校准 **不同 episode**；是否出现“校准未列出的严重偏差”应看上表 `global_index`，",
            "  而非对比两阶段的 `shared_index`/`holdout_index` 编号。",
            "",
            "### Q3 闭环试点（非 SR 保证）",
            "- π0.5：holdout 夹爪不一致为 0 时，可小预算 NFE=1 sanity；SmolVLA：优先 NFE=2 试点。",
            "",
            "## 统计核对",
            "",
            f"- π0.5：`ok={verify['pi05']['ok']}`",
            f"- SmolVLA：`ok={verify['smolvla']['ok']}`",
            "",
            "详见 `FIXES.md`。",
            "",
        ]
    )
    (PHASE2_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")


def run_analysis() -> None:
    cal = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    if not HOLDOUT_PT.is_file():
        holdout = build_holdout_shared(cal)
        save_holdout(holdout)
    else:
        holdout = torch.load(HOLDOUT_PT, map_location="cpu", weights_only=False)

    frames_df = _load_actions(LIBERO_REPO)
    actions = np.stack(frames_df["action"].to_numpy()).astype(np.float64)
    episode = frames_df["episode_index"].to_numpy()

    groups = calibration_groups(cal, actions, episode)
    G_cal = {i for i, g in enumerate(demo_gripper_switch(cal, actions, episode)) if g}
    cal_stats = {
        "smolvla": model_stats("smolvla", load_probe_frames(CAL_SMOL), groups, G_cal),
        "pi05": model_stats("pi05", load_probe_frames(CAL_PI05), groups, G_cal),
    }

    cal_frames = {
        "smolvla": load_probe_frames(CAL_SMOL),
        "pi05": load_probe_frames(CAL_PI05),
    }
    holdout_eval = {}
    curves: dict = {}
    for model, subdir, nfes in (
        ("smolvla", "smolvla", SMOL_HOLDOUT_NFES),
        ("pi05", "pi05", PI05_HOLDOUT_NFES),
    ):
        hdir = PHASE2_DIR / subdir
        hdf = load_probe_frames(hdir)
        holdout_eval[model] = analyze_holdout_model(
            hdf,
            cal_frames[model],
            nfes,
            calibration_frames_path=str((CAL_SMOL if model == "smolvla" else CAL_PI05) / "frames.csv"),
        )
        curves[model] = plot_holdout_anomalies(
            hdf, hdir, model, holdout_eval[model]["anomalies"]
        )

    curve_index_path = PHASE2_DIR / "curves" / "curve_index.json"
    curve_index_path.parent.mkdir(parents=True, exist_ok=True)
    curve_index_path.write_text(
        json.dumps(
            {m: curves[m].get("index", []) for m in curves},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    verify = {
        "smolvla": verify_holdout_counts("smolvla", holdout_eval["smolvla"]),
        "pi05": verify_holdout_counts("pi05", holdout_eval["pi05"]),
    }

    before_wrong = {
        "pi05_nfe1_noise0_tail": 1,
        "pi05_nfe1_noise1_tail": 1,
        "note": "π0.5 used SmolVLA calibration thresholds in summary.json.bak",
    }

    # selection diff on calibration
    metas = build_metas(
        cal,
        np.asarray(cal["alpha"]).reshape(-1),
        demo_gripper_switch(cal, actions, episode),
        np.zeros(N_FRAMES, dtype=np.int32),
    )
    smol_df = load_probe_frames(CAL_SMOL)
    changes = {}
    for nid in (0, 1):
        added, removed = diff_lists(metas, smol_df, groups["care_alpha"], groups["care_event"], nid)
        changes[f"noise{nid}"] = {"added_vs_care_alpha": added, "removed_vs_care_alpha": removed}

    out = {
        "calibration_groups": {k: sorted(v) for k, v in groups.items()},
        "calibration_models": cal_stats,
        "holdout_grip_events": len(holdout_G_set(holdout)),
        "holdout_eval": holdout_eval,
        "verification": verify,
        "care_event_vs_alpha": changes,
        "curve_files": {m: curves[m]["files"] for m in curves},
        "curve_index": {m: curves[m].get("index", []) for m in curves},
        "curve_tail_nfe1_verify": {m: curves[m].get("tail_nfe1_verify", {}) for m in curves},
        "before_fix_reference": before_wrong,
    }
    (PHASE2_DIR / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    write_fixes_md(before_wrong, verify)
    write_report(cal_stats, holdout, holdout_eval, verify, curves)
    print(f"analysis -> {PHASE2_DIR / 'REPORT.md'}", flush=True)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    args = p.parse_args()
    if args.build_only:
        cal = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
        save_holdout(build_holdout_shared(cal))
        print(f"built {HOLDOUT_PT}", flush=True)
        return
    if args.analyze_only:
        run_analysis()
        return
    cal = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    save_holdout(build_holdout_shared(cal))
    run_analysis()


if __name__ == "__main__":
    main()

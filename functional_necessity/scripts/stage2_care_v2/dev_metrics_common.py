"""Shared LIBERO-unified metrics for CARE dev screening (CPU, from actions.npz)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from protocol import ACTION_DIM, LIBERO_REPO, OUT_DIR, hub_snapshot

MODEL_DIRS = {
    "pi05": OUT_DIR / "nfe_probe_pi05",
    "smolvla": OUT_DIR / "nfe_probe_smolvla",
    "molmoact2": OUT_DIR / "nfe_probe_molmoact2",
}

SCREENING_DIR = OUT_DIR / "dev_three_model_screening"
POOL_CSV = SCREENING_DIR / "pool_observation_level.csv"
CONVERSION_MANIFEST_PATH = SCREENING_DIR / "conversion_manifest.json"
CONVERSION_CHECK_PATH = SCREENING_DIR / "conversion_check.json"


def libero_mean_std() -> tuple[np.ndarray, np.ndarray]:
    stats = json.load(open(hub_snapshot(LIBERO_REPO) / "meta" / "stats.json"))
    m = np.asarray(stats["action"]["mean"], dtype=np.float64)[:ACTION_DIM]
    s = np.asarray(stats["action"]["std"], dtype=np.float64)[:ACTION_DIM]
    return m, np.maximum(s, 1e-8)


def molmo_quantile_stats() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from safetensors.torch import load_file
    from load_policy import LEROBOT_CKPT

    d = load_file(next(LEROBOT_CKPT.glob("policy_postprocessor_step_1_*.safetensors")))
    q01 = d["action.q01"].float().numpy()[:ACTION_DIM]
    q99 = d["action.q99"].float().numpy()[:ACTION_DIM]
    mask = d["action.mask"].float().numpy()[:ACTION_DIM]
    return q01, q99, mask


def native_to_physical(model: str, x: np.ndarray) -> np.ndarray:
    """x: (10, 7) policy output before env postprocessor."""
    mean, std = libero_mean_std()
    if model in ("pi05", "smolvla"):
        return x.astype(np.float64) * std + mean
    q01, q99, mask = molmo_quantile_stats()
    span = np.maximum(q99 - q01, 1e-8)
    out = x.astype(np.float64).copy()
    m = mask > 0.5
    out[:, m] = (x[:, m] + 1.0) * 0.5 * span[m] + q01[m]
    out[:, ~m] = x[:, ~m]
    return out


def to_libero_normalized(phys: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (phys - mean) / std


def grip_closed_physical(phys: np.ndarray) -> np.ndarray:
    return phys[:, 6] > 0.0


def compare_chunk(model: str, pred: np.ndarray, ref: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict[str, Any]:
    lp = to_libero_normalized(native_to_physical(model, pred), mean, std)
    lr = to_libero_normalized(native_to_physical(model, ref), mean, std)
    err = lp - lr
    err_arm = err[:, :6]
    rmse7 = float(np.sqrt((err ** 2).mean()))
    rmse6 = float(np.sqrt((err_arm ** 2).mean()))
    cos = float(np.dot(lp.ravel(), lr.ravel()) / (np.linalg.norm(lp) * np.linalg.norm(lr) + 1e-12))
    gc = grip_closed_physical(native_to_physical(model, pred))
    gr = grip_closed_physical(native_to_physical(model, ref))
    disagree = gc != gr
    return {
        "cosine_libero": cos,
        "rmse7_libero": rmse7,
        "rmse6_arm_libero": rmse6,
        "grip_disagree_steps": int(disagree.sum()),
        "grip_any_disagree": int(disagree.any()),
    }


def obs_arm_rmse_merged_two_noises(
    model: str, z: np.lib.npyio.NpzFile, si: int, nfe: int, mean: np.ndarray, std: np.ndarray
) -> float:
    """Per observation: RMSE over 2 noises × 10 steps × 6 arm dims (LIBERO norm vs NFE10)."""
    errs = []
    for nid in (0, 1):
        pred = z[f"nfe{nfe}"][si, nid]
        ref = z["nfe10"][si, nid]
        lp = to_libero_normalized(native_to_physical(model, pred), mean, std)
        lr = to_libero_normalized(native_to_physical(model, ref), mean, std)
        errs.append((lp[:, :6] - lr[:, :6]).reshape(-1))
    cat = np.concatenate(errs)
    return float(np.sqrt((cat ** 2).mean()))


def native_metrics_row(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    err = pred - ref
    cos = float(np.dot(pred.ravel(), ref.ravel()) / (np.linalg.norm(pred) * np.linalg.norm(ref) + 1e-12))
    rmse = float(np.sqrt((err ** 2).mean()))
    arm = float(np.hypot(err[:, :3], err[:, :3]).mean()) if err.ndim == 2 else float("nan")
    return {"cosine_native": cos, "rmse_native": rmse, "arm_proxy_native": arm}


def export_pool_rows(model: str, z: np.lib.npyio.NpzFile, records: list[dict]) -> list[dict]:
    mean, std = libero_mean_std()
    nfes = sorted(int(k.replace("nfe", "")) for k in z.files if k.startswith("nfe") and k != "nfe10")
    rows: list[dict] = []
    for si, rec in enumerate(records):
        if si >= z["nfe10"].shape[0]:
            break
        base = {
            "model": model,
            "shared_index": si,
            "global_index": int(rec["global_index"]),
            "episode_index": int(rec["episode_index"]),
            "task_index": int(rec["task_index"]),
            "task": str(rec.get("task", "")),
            "frame_in_episode": int(rec["frame_in_episode"]),
        }
        for nid in (0, 1):
            for nfe in nfes:
                pred = z[f"nfe{nfe}"][si, nid]
                ref = z["nfe10"][si, nid]
                m = compare_chunk(model, pred, ref, mean, std)
                nat = native_metrics_row(pred, ref)
                rows.append(
                    {
                        **base,
                        "noise_id": nid,
                        "nfe": nfe,
                        **nat,
                        **m,
                    }
                )
    return rows


def grip_stats_for_selection(df: pd.DataFrame, selected: set[int], nfe: int) -> dict[str, int]:
    sub = df[(df.nfe == nfe) & (df.shared_index.isin(selected))]
    n_cmp = len(sub)
    n_cmp_bad = int((sub.grip_disagree_steps > 0).sum())
    n_obs = int(sub.shared_index.nunique())
    n_obs_bad = int(sub.loc[sub.grip_disagree_steps > 0, "shared_index"].nunique())
    return {
        "grip_comparisons_with_disagree": n_cmp_bad,
        "grip_comparisons_total": n_cmp,
        "grip_unique_obs_with_disagree": n_obs_bad,
        "grip_unique_obs_denominator": n_obs,
    }


def tail_stats_merged_noises(df_native: pd.DataFrame, selected: set[int]) -> dict[str, int]:
    from event_care_nfe_diagnostic import tail_mask_nfe1

    tail_union: set[int] = set()
    cmp_bad = 0
    cmp_total = 0
    for nid in (0, 1):
        tail_bad, _, _ = tail_mask_nfe1(df_native, nid)
        tail_union |= tail_bad
        sub = df_native[(df_native.nfe == 1) & (df_native.noise_id == nid) & (df_native.shared_index.isin(selected))]
        cmp_total += len(sub)
        for r in sub.itertuples():
            if int(r.shared_index) in tail_bad:
                cmp_bad += 1
    cap_unique = len(selected & tail_union)
    return {
        "tail_comparisons_captured": cmp_bad,
        "tail_comparisons_total": cmp_total,
        "tail_unique_obs_captured": cap_unique,
        "tail_unique_obs_denominator_pool": len(tail_union),
        "tail_unique_obs_denominator_selected": len(selected),
    }


def write_conversion_manifest() -> dict:
    mean, std = libero_mean_std()
    q01, q99, mask = molmo_quantile_stats()
    man = {
        "common_normalized_space": "lerobot/libero meta/stats.json action mean/std on physical 7-d",
        "libero_mean": mean.tolist(),
        "libero_std": std.tolist(),
        "pi05_smolvla_physical": "physical = policy_norm * checkpoint_mean_std (MEAN_STD)",
        "molmo_physical": "masked dims: (norm+1)/2*(q99-q01)+q01; grip dim mask=0: raw channel unchanged",
        "molmo_q01": q01.tolist(),
        "molmo_q99": q99.tolist(),
        "molmo_mask": mask.tolist(),
        "gripper_rule": "physical gripper > 0 => closed (LIBERO demo convention)",
        "rmse7_unit": "LIBERO-normalized 7-d RMSE vs NFE10 reference (same noise_id)",
        "rmse6_unit": "LIBERO-normalized 6-d arm RMSE vs NFE10 reference (same noise_id)",
        "obs_arm_p95": "per obs: sqrt(mean sq err) pooled over 2 noises × 10 steps × 6 arm; P95 over selected obs",
        "postprocessor_check": str(CONVERSION_CHECK_PATH.relative_to(OUT_DIR)),
    }
    SCREENING_DIR.mkdir(parents=True, exist_ok=True)
    CONVERSION_MANIFEST_PATH.write_text(json.dumps(man, indent=2) + "\n")
    return man


def build_conversion_check(models: tuple[str, ...] = ("pi05", "smolvla", "molmoact2")) -> dict:
    from event_care_nfe_diagnostic import load_probe_frames

    mean, std = libero_mean_std()
    q01, q99, mask = molmo_quantile_stats()
    out: dict[str, Any] = {"libero_meta_grip_mean_std": {"mean6": float(mean[6]), "std6": float(std[6])}}
    for model in models:
        mdir = MODEL_DIRS[model]
        if not (mdir / "actions.npz").is_file():
            out[model] = {"status": "missing"}
            continue
        z = np.load(mdir / "actions.npz")
        si = 0
        pred = z["nfe1"][si, 0]
        phys_manual = native_to_physical(model, pred)
        block: dict[str, Any] = {
            "manual_physical_grip_sample": float(phys_manual[0, 6]),
            "manual_vs_libero_norm_roundtrip_max_abs": float(
                np.max(np.abs(to_libero_normalized(phys_manual, mean, std) * std + mean - phys_manual))
            ),
        }
        if model == "molmoact2":
            block["molmo_grip_mask"] = float(mask[6])
            block["note"] = "grip uses mask=0 pass-through; closed rule uses physical>0 after quantile on arm only"
        if (mdir / "frames.csv").is_file():
            frames = load_probe_frames(mdir)
            r = frames[(frames.shared_index == si) & (frames.nfe == 1) & (frames.noise_id == 0)].iloc[0]
            block["frames_native_rmse_nfe1"] = float(r.rmse)
            nat_rmse = float(np.sqrt(((pred - z["nfe10"][si, 0]) ** 2).mean()))
            block["npz_native_rmse_recompute_delta"] = abs(nat_rmse - float(r.rmse))
        out[model] = block
    CONVERSION_CHECK_PATH.write_text(json.dumps(out, indent=2) + "\n")
    return out

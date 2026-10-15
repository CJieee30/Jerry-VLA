#!/usr/bin/env python3
"""CPU: four selection strategies × three models — native + LIBERO-unified metrics.

Reads one actions.npz per model (480 pool). Does not run inference or tune thresholds.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "molmoact2_libero"))

from dump_shared import _load_actions
from event_care_nfe_diagnostic import (
    EVENT_DIR,
    build_metas,
    demo_gripper_switch,
    eval_selection,
    g_only_group,
    grip_deviation_set,
    load_probe_frames,
    random_group,
    select_per_task,
    tail_mask_nfe1,
    arm_top5_set,
)
from protocol import ACTION_DIM, LIBERO_REPO, OUT_DIR, hub_snapshot

OUT = OUT_DIR / "dev_three_model_screening"
SELECTION_DIR = OUT_DIR / "nfe_probe_event_care" / "budget_compare" / "selections"
MODEL_DIRS = {
    "pi05": OUT_DIR / "nfe_probe_pi05",
    "smolvla": OUT_DIR / "nfe_probe_smolvla",
    "molmoact2": OUT_DIR / "nfe_probe_molmoact2",
}
STRATEGIES = {
    "random": SELECTION_DIR / "random_seed42.json",
    "alpha": SELECTION_DIR / "care_alpha.json",
    "g_only": SELECTION_DIR / "g_only_seed42.json",
    "alpha_event": SELECTION_DIR / "care_event.json",
}
CLOSED_LOOP = {
    "pi05": {"nfe1": 97.00, "nfe2": None, "nfe10": 97.25, "n": 400, "source": "stage_nfe full"},
    "smolvla": {"nfe1": 79.25, "nfe2": 82.00, "nfe3": 81.00, "nfe10": 83.00, "n": 400, "source": "stage_nfe full"},
    "molmoact2": {"nfe1_2000": 97.80, "nfe10_2000": 97.50, "n": 2000, "source": "molmoact2_full_nfe1_2000 / baseline"},
}


def libero_mean_std() -> tuple[np.ndarray, np.ndarray]:
    stats = json.load(open(hub_snapshot(LIBERO_REPO) / "meta" / "stats.json"))
    m = np.asarray(stats["action"]["mean"], dtype=np.float64)[:ACTION_DIM]
    s = np.asarray(stats["action"]["std"], dtype=np.float64)[:ACTION_DIM]
    return m, np.maximum(s, 1e-8)


def pi_smol_to_libero(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


def molmo_q01_q99():
    from safetensors.torch import load_file
    from load_policy import LEROBOT_CKPT

    d = load_file(next(LEROBOT_CKPT.glob("policy_postprocessor_step_1_*.safetensors")))
    q01 = d["action.q01"].float().numpy()[:ACTION_DIM]
    q99 = d["action.q99"].float().numpy()[:ACTION_DIM]
    mask = d["action.mask"].float().numpy()[:ACTION_DIM]
    return q01, q99, mask


def molmo_to_libero(x: np.ndarray, q01: np.ndarray, q99: np.ndarray, mask: np.ndarray) -> np.ndarray:
    span = np.maximum(q99 - q01, 1e-8)
    out = x.copy()
    m = mask > 0.5
    out[..., m] = (x[..., m] + 1.0) * 0.5 * span[m] + q01[m]
    out[..., ~m] = x[..., ~m]
    return out


def norm_libero(phys: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (phys - mean) / std


def metrics_pair(pred: np.ndarray, ref: np.ndarray, mean, std) -> dict:
    """pred/ref (10,7) in common LIBERO-normalized space."""
    err = pred - ref
    arm = err[:, :6]
    cos = float(np.dot(pred.ravel(), ref.ravel()) / (np.linalg.norm(pred) * np.linalg.norm(ref) + 1e-12))
    rmse = float(np.sqrt((err ** 2).mean()))
    rmse_arm = float(np.sqrt((arm ** 2).mean()))
    rmse_arm_p95 = float(np.sqrt((arm ** 2).mean(axis=1)).mean())  # per-step then mean; report obs-level p95 separately
    pu = pred[:, 6] * std[6] + mean[6]
    ru = ref[:, 6] * std[6] + mean[6]
    grip = int(np.any((pu > 0) != (ru > 0)))
    return {"cosine_libero": cos, "rmse_libero": rmse, "rmse_arm_libero": rmse_arm, "grip_disagree": grip}


def native_metrics_row(pred: np.ndarray, ref: np.ndarray, model: str) -> dict:
    err = pred - ref
    cos = float(np.dot(pred.ravel(), ref.ravel()) / (np.linalg.norm(pred) * np.linalg.norm(ref) + 1e-12))
    rmse = float(np.sqrt((err ** 2).mean()))
    arm = np.hypot(err[:, :3], err[:, 3:6]).mean() if err.ndim == 2 else float("nan")
    return {"cosine_native": cos, "rmse_native": rmse, "arm_proxy_native": float(arm)}


def load_selections() -> dict[str, set[int]]:
    out = {}
    for name, path in STRATEGIES.items():
        blob = json.loads(path.read_text())
        key = "shared_indices" if "shared_indices" in blob else "pool_index"
        out[name] = set(int(x) for x in blob[key])
    return out


def convert_native_to_libero_phys(model: str, arr: np.ndarray) -> np.ndarray:
    mean, std = libero_mean_std()
    if model in ("pi05", "smolvla"):
        return pi_smol_to_libero(arr, mean, std)
    q01, q99, mask = molmo_q01_q99()
    return molmo_to_libero(arr, q01, q99, mask)


def verify_conversion(model: str, z: dict, demo_actions: np.ndarray, n_check: int = 8) -> dict:
    """Native: recompute RMSE from npz vs frames.csv; unified: demo gripper dim stats."""
    frames = load_probe_frames(MODEL_DIRS[model])
    mean, std = libero_mean_std()
    rows = frames[(frames.nfe == 1) & (frames.noise_id == 0)].head(n_check)
    native_delta = []
    for r in rows.itertuples():
        si = int(r.shared_index)
        pred, ref = z["nfe1"][si, 0], z["nfe10"][si, 0]
        nat_rmse = float(np.sqrt(((pred - ref) ** 2).mean()))
        native_delta.append(abs(nat_rmse - float(r.rmse)))
    demo_g = demo_actions[:, 6]
    return {
        "model": model,
        "n": len(native_delta),
        "native_rmse_recompute_max_delta": float(max(native_delta)) if native_delta else None,
        "libero_grip_demo_frac_positive": float((demo_g > 0).mean()),
        "unified_norm": "libero meta mean/std on physical actions",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shared = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    frames_df = _load_actions(LIBERO_REPO)
    actions = np.stack(frames_df["action"].to_numpy()).astype(np.float64)
    episode = frames_df["episode_index"].to_numpy()
    G_flags = demo_gripper_switch(shared, actions, episode)
    metas = build_metas(shared, np.asarray(shared["alpha"]), G_flags, G_flags)
    G_all = {m.shared_index for m in metas if m.G == 1}

    selections = load_selections()
    mean, std = libero_mean_std()
    obs_rows = []
    summary = {"models": {}, "closed_loop_labels": CLOSED_LOOP, "selections": {k: len(v) for k, v in selections.items()}}

    for model, mdir in MODEL_DIRS.items():
        if not (mdir / "actions.npz").is_file():
            summary["models"][model] = {"status": "missing_probe"}
            continue
        z = np.load(mdir / "actions.npz")
        nfes = sorted(int(k.replace("nfe", "")) for k in z.files if k.startswith("nfe"))
        df = load_probe_frames(mdir) if (mdir / "frames.csv").is_file() else None
        conv = (
            verify_conversion(model, z, actions)
            if model != "molmoact2" or (mdir / "frames.csv").is_file()
            else {"note": "pending"}
        )
        model_sum = {"nfes": nfes, "conversion_check": conv, "strategies": {}}

        for strat, selected in selections.items():
            per = []
            for si in sorted(selected):
                for nid in (0, 1):
                    for nfe in nfes:
                        if nfe == 10:
                            continue
                        pred = z[f"nfe{nfe}"][si, nid]
                        ref = z["nfe10"][si, nid]
                        phys_p = convert_native_to_libero_phys(model, pred)
                        phys_r = convert_native_to_libero_phys(model, ref)
                        lib_p = norm_libero(phys_p, mean, std)
                        lib_r = norm_libero(phys_r, mean, std)
                        uni = metrics_pair(lib_p, lib_r, mean, std)
                        nat = native_metrics_row(pred, ref, model)
                        row = {
                            "model": model,
                            "strategy": strat,
                            "shared_index": si,
                            "noise_id": nid,
                            "nfe": nfe,
                            **nat,
                            **uni,
                            "rmse_arm_p95_libero": float(np.sqrt(((lib_p[:, :6] - lib_r[:, :6]) ** 2).mean(axis=1)).max()),
                        }
                        obs_rows.append(row)
                        per.append(row)
            sub = pd.DataFrame(per)
            model_sum["strategies"][strat] = {
                "n_obs": len(selected),
                "G1": len(selected & G_all),
                "cosine_libero_mean": float(sub["cosine_libero"].mean()),
                "rmse_libero_mean": float(sub["rmse_libero"].mean()),
                "rmse_arm_p95_libero_mean": float(sub.groupby(["shared_index", "nfe"])["rmse_arm_p95_libero"].max().mean()),
                "grip_rate_nfe1": float(sub[sub.nfe == 1]["grip_disagree"].mean()) if (sub.nfe == 1).any() else None,
            }
            if df is not None:
                tail0, _, _ = tail_mask_nfe1(df, 0)
                tail1, _, _ = tail_mask_nfe1(df, 1)
                tail_bad = tail0 | tail1
                g1 = grip_deviation_set(df, 0, 1) | grip_deviation_set(df, 1, 1)
                g2 = grip_deviation_set(df, 0, 2) | grip_deviation_set(df, 1, 2)
                arm5, _ = arm_top5_set(df, 0)
                ev = eval_selection(strat, selected, G_all, tail_bad, g1, g2, arm5, df, 0)
                model_sum["strategies"][strat]["risk_native"] = ev
        summary["models"][model] = model_sum

    pd.DataFrame(obs_rows).to_csv(OUT / "observation_level.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# 三模型开发筛查（统一 LIBERO 口径 + 原生指标）", ""]
    for model, ms in summary["models"].items():
        if ms.get("status") == "missing_probe":
            lines.append(f"- **{model}**: 探针缺失")
            continue
        lines.append(f"## {model}")
        lines.append(f"转换核对: `{json.dumps(ms.get('conversion_check'), ensure_ascii=False)}`")
        lines.append("| 策略 | G=1 | cos(mean) | rmse(mean) | arm P95(mean) | grip NFE1 | 尾部捕获 |")
        lines.append("|---|---|---|---|---|---|---|")
        for strat, st in ms["strategies"].items():
            risk = st.get("risk_native", {})
            tail = f"{risk.get('n_tail_captured','?')}/{risk.get('n_tail_bad_total','?')}" if risk else "—"
            lines.append(
                f"| {strat} | {st['G1']} | {st['cosine_libero_mean']:.4f} | {st['rmse_libero_mean']:.4f} | "
                f"{st['rmse_arm_p95_libero_mean']:.4f} | {st.get('grip_rate_nfe1')} | {tail} |"
            )
        lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

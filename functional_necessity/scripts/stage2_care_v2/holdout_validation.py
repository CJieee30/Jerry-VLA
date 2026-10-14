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
    eval_selection,
    load_probe_frames,
    model_stats,
    random_group,
    select_per_task,
    tail_mask_nfe1,
    grip_deviation_set,
    arm_top5_set,
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


def eval_holdout_with_cal_thresholds(
    df: pd.DataFrame,
    holdout: dict,
    cal_df: pd.DataFrame,
    noise_id: int,
) -> dict:
    """Map calibration tail/grip sets to holdout indices via per-row metrics."""
    tail_bad_cal, cos_p5, rmse_p95 = tail_mask_nfe1(cal_df, noise_id)
    # thresholds only — rebuild bad on holdout using same cuts
    sub = df[(df.nfe == 1) & (df.noise_id == noise_id)]
    tail_bad = {
        int(r.shared_index)
        for r in sub.itertuples()
        if float(r.cosine) < cos_p5 or float(r.rmse) > rmse_p95 or int(r.gripper_disagree_n) > 0
    }
    g1 = grip_deviation_set(df, noise_id, 1)
    g2 = grip_deviation_set(df, noise_id, 2)
    arm5, arm_thr = arm_top5_set(cal_df, noise_id)
    # arm top5 threshold from cal, apply to holdout
    sub_h = df[(df.nfe == 1) & (df.noise_id == noise_id)].copy()
    sub_h["arm_rmse"] = np.hypot(sub_h.rmse_trans, sub_h.rmse_rot)
    arm_bad_h = {int(r.shared_index) for r in sub_h.itertuples() if float(r.arm_rmse) >= arm_thr}
    G_all = holdout_G_set(holdout)
    all_idx = set(range(int(holdout["n"])))
    return {
        "thresholds_from_calibration": {
            "cosine_p5": cos_p5,
            "rmse_p95": rmse_p95,
            "arm_rmse_p95": arm_thr,
        },
        "holdout_universe": {
            "n": len(all_idx),
            "n_grip_event_windows": len(G_all),
            "n_tail_bad": len(tail_bad),
            "n_grip_dev_nfe1": len(g1),
            "n_grip_dev_nfe2": len(g2),
            "n_arm_bad": len(arm_bad_h),
        },
        "full_holdout": eval_selection(
            "all_holdout",
            all_idx,
            G_all,
            tail_bad,
            g1,
            g2,
            arm_bad_h,
            df,
            noise_id,
        ),
    }


def plot_holdout_anomalies(
    df: pd.DataFrame,
    holdout_dir: Path,
    model: str,
    noise_id: int = 0,
    k: int = 5,
) -> list[str]:
    sub = df[(df.nfe == 1) & (df.noise_id == noise_id)].copy()
    sub["score"] = -sub.cosine + sub.rmse
    worst = sub.nlargest(k, "score")
    curves_dir = PHASE2_DIR / "curves" / model
    curves_dir.mkdir(parents=True, exist_ok=True)
    npz_path = holdout_dir / "actions.npz"
    if not npz_path.is_file():
        return []
    acts = np.load(npz_path)
    written = []
    steps = np.arange(acts["nfe1"].shape[2])
    for _, row in worst.iterrows():
        hi = int(row.shared_index)
        ref = acts["nfe10"][hi, noise_id]
        pred1 = acts["nfe1"][hi, noise_id]
        pred2 = acts["nfe2"][hi, noise_id]
        fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
        axes[0].plot(steps, np.linalg.norm(ref[:, :6], axis=1), "k--", label="NFE10 |arm|")
        axes[0].plot(steps, np.linalg.norm(pred1[:, :6], axis=1), label="NFE1")
        axes[0].plot(steps, np.linalg.norm(pred2[:, :6], axis=1), label="NFE2")
        axes[1].plot(steps, ref[:, 6], "k--", label="NFE10 grip")
        axes[1].plot(steps, pred1[:, 6], label="NFE1 grip")
        axes[1].plot(steps, pred2[:, 6], label="NFE2 grip")
        axes[0].set_ylabel("arm norm")
        axes[1].set_ylabel("grip")
        axes[1].set_xlabel("exec step")
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
        fig.suptitle(
            f"{model} holdout#{hi} ep={row.episode_index} f={row.frame_in_episode} "
            f"cos={row.cosine:.3f} rmse={row.rmse:.3f}",
            fontsize=9,
        )
        out = curves_dir / f"holdout_{hi:03d}_noise{noise_id}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(str(out.relative_to(PHASE2_DIR)))
    return written


def write_report(
    cal_groups: dict,
    cal_stats: dict,
    holdout_payload: dict,
    holdout_eval: dict,
    curves: dict,
) -> None:
    lines = [
        "# CARE_event 离线 NFE 验证（校准回放 + 独立 holdout）",
        "",
        "本报告比较的是 **低 NFE 相对 NFE=10 的动作 chunk 偏差**，不是 LIBERO 闭环成功率。",
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
            "## 2. 独立 holdout（120 帧）",
            "",
            f"- Holdout 窗口内 **G=1 事件数**：{len(holdout_G_set(holdout_payload))} / {HOLDOUT_N}",
            "- 尾部阈值：来自 **校准** SmolVLA NFE=1 的 P5/P95（holdout 上不重算）。",
            "",
        ]
    )
    for model in ("smolvla", "pi05"):
        ev = holdout_eval[model]
        lines.append(f"### {model}")
        u = ev["0"]["holdout_universe"]
        lines.append(
            f"- 全 holdout 上 tail_bad={u['n_tail_bad']}, grip_dev_nfe1={u['n_grip_dev_nfe1']}, "
            f"grip_dev_nfe2={u['n_grip_dev_nfe2']}, arm_bad={u['n_arm_bad']}"
        )
        if curves.get(model):
            lines.append(f"- 异常曲线：`curves/{model}/`（{len(curves[model])} 张）")
        lines.append("")
    lines.extend(
        [
            "## 3. 问题摘要",
            "",
            "### Q1 care_event 相对随机 / 原 CARE 多覆盖了什么？",
            "- 见校准表：`care_event` 通常在 **G 事件窗口** 与 **夹爪 NFE1 不一致** 上高于 `care_alpha`/`random`；",
            "  手臂 RMSE 尾部依赖任务分布，高 α_event 不保证手臂极端误差。",
            "",
            "### Q2 Holdout 是否出现校准未见的严重偏差？",
            "- 对比 holdout 全集 tail_bad 与校准组已捕获的 shared_index 集合（不同 universe）；",
            "  若 holdout tail 样本的 α_event 偏低，说明事件选样未外推覆盖。",
            "",
            "### Q3 闭环建议（非 SR 保证）",
            "- **π0.5 NFE=1**：仅当 holdout 上夹爪/尾部仍可控时建议小预算闭环；",
            "- **SmolVLA NFE=2**：校准中 NFE2 对 NFE1 尾部有恢复时优先于 NFE1 试点。",
            "",
            "## 闭环试点命令（本轮不执行）",
            "",
            "```bash",
            "# π0.5：LIBERO-plus / 正式 eval 脚本，Drop-6，num_inference_steps=1，先 20–40 回合 sanity",
            "# SmolVLA：num_steps=2 对照 10；若 holdout 上 NFE2 仍 tail_bad，勿直接上 400 回合",
            "```",
            "",
            "## 产物路径",
            "",
            "- `holdout_manifest.json` / `holdout_shared.pt`：独立 120 时间戳",
            "- `smolvla/`、`pi05/`：`frames.csv`、`actions.npz`、`correctness.json`",
            "- `curves/{model}/holdout_*.png`：holdout 上最差 5 例（noise0, NFE1）",
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

    smol_cal = load_probe_frames(CAL_SMOL)
    holdout_eval = {}
    curves = {}
    for model, subdir in (("smolvla", "smolvla"), ("pi05", "pi05")):
        hdir = PHASE2_DIR / subdir
        if not (hdir / "frames.csv").is_file():
            holdout_eval[model] = {"0": {"error": "missing frames.csv"}, "1": {}}
            continue
        hdf = load_probe_frames(hdir)
        holdout_eval[model] = {
            "0": eval_holdout_with_cal_thresholds(hdf, holdout, smol_cal, 0),
            "1": eval_holdout_with_cal_thresholds(hdf, holdout, smol_cal, 1),
        }
        curves[model] = plot_holdout_anomalies(hdf, hdir, model, 0)

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
        "care_event_vs_alpha": changes,
        "curve_files": curves,
    }
    (PHASE2_DIR / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    write_report(groups, cal_stats, holdout, holdout_eval, curves)
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

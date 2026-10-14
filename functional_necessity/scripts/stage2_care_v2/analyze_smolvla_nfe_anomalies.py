#!/usr/bin/env python3
"""Deep-dive on selected SmolVLA NFE probe frames (offline actions only).

Reads nfe_probe_smolvla/{actions.npz,frames.csv,config.json,correctness.json} and
shared.pt metadata. Does not re-run inference.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FN_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "smolvla_libero"))

from protocol import OUT_DIR

PROBE_DIR = OUT_DIR / "nfe_probe_smolvla"
CASE_DIR = PROBE_DIR / "anomaly_supplement"
NFES_EXPORT = (1, 2, 3, 10)
N_NOISE = 2
EXEC_STEPS = 10
ACTION_DIM = 7

# NFE=2 仍异常（recovery.examples_remaining）；高 α 未覆盖的 NFE=1 异常代表
TARGETS = {
    "nfe2_remaining": [89, 381, 437, 451],
    "care_miss_examples": [124, 121],
}
ALL_SHARED = sorted(set(TARGETS["nfe2_remaining"]) | set(TARGETS["care_miss_examples"]))


def load_mean_std(correctness: dict) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(correctness["action_mean"], dtype=torch.float32)
    std = torch.tensor(correctness["action_std"], dtype=torch.float32)
    return mean, std


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    import torch.nn.functional as F

    fa = F.normalize(_flat(a).float()[None], dim=-1)
    fb = F.normalize(_flat(b).float()[None], dim=-1)
    return float((fa * fb).sum().item())


def rmse(err: torch.Tensor) -> float:
    e = err.float()
    return float(torch.sqrt((e * e).mean()).item())


def unnormalize(normed: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return normed.float() * std + mean


def grip_closed(unnorm_g: torch.Tensor) -> torch.Tensor:
    return unnorm_g.float() > 0.0


def metrics_row(pred: torch.Tensor, ref: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> dict:
    err = pred - ref
    pred_u = unnormalize(pred, mean, std)
    ref_u = unnormalize(ref, mean, std)
    disagree = grip_closed(pred_u[:, 6]) != grip_closed(ref_u[:, 6])
    return {
        "cosine": cosine(pred, ref),
        "rmse": rmse(err),
        "max_abs": float(err.abs().max().item()),
        "rmse_trans": rmse(err[:, :3]),
        "rmse_rot": rmse(err[:, 3:6]),
        "rmse_grip": rmse(err[:, 6:7]),
        "rmse_tail3": rmse(err[-3:]),
        "gripper_disagree": float(disagree.float().mean().item()),
        "gripper_disagree_n": int(disagree.sum().item()),
    }


def step_records(
    shared_index: int,
    noise_id: int,
    nfe: int,
    act: np.ndarray,
    ref: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    meta: dict,
) -> list[dict]:
    pred = torch.from_numpy(act)
    reference = torch.from_numpy(ref)
    pred_u = unnormalize(pred, mean, std)
    ref_u = unnormalize(reference, mean, std)
    rows = []
    for t in range(EXEC_STEPS):
        rows.append(
            {
                "shared_index": shared_index,
                "noise_id": noise_id,
                "nfe": nfe,
                "step": t,
                "task": meta["task"],
                "episode_index": meta["episode_index"],
                "alpha": meta["alpha"],
                "pred_grip_norm": float(pred[t, 6].item()),
                "ref_grip_norm": float(reference[t, 6].item()),
                "pred_grip_exec": float(pred_u[t, 6].item()),
                "ref_grip_exec": float(ref_u[t, 6].item()),
                "pred_grip_closed": int(grip_closed(pred_u[t, 6:7]).item()),
                "ref_grip_closed": int(grip_closed(ref_u[t, 6:7]).item()),
                **{f"pred_d{d}": float(pred[t, d].item()) for d in range(ACTION_DIM)},
                **{f"ref_d{d}": float(reference[t, d].item()) for d in range(ACTION_DIM)},
            }
        )
    return rows


def plot_case(shared_index: int, arrays: dict, mean: torch.Tensor, std: torch.Tensor, meta: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    fig.suptitle(
        f"shared={shared_index} ep={meta['episode_index']} α={meta['alpha']:.3f}\n{meta['task'][:70]}",
        fontsize=10,
    )
    colors = {1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c", 10: "#1f77b4"}
    steps = np.arange(EXEC_STEPS)
    for nid in range(N_NOISE):
        ax_g = axes[nid, 0]
        ax_a = axes[nid, 1]
        ref = arrays[(nid, 10)]
        ref_u = unnormalize(torch.from_numpy(ref), mean, std).numpy()
        ax_g.plot(steps, ref_u[:, 6], "k--", lw=1.2, label="ref NFE=10")
        ax_g.axhline(0.0, color="gray", lw=0.6, alpha=0.5)
        for nfe in (1, 2, 3, 10):
            pred_u = unnormalize(torch.from_numpy(arrays[(nid, nfe)]), mean, std).numpy()
            ax_g.plot(
                steps,
                pred_u[:, 6],
                color=colors[nfe],
                marker="o",
                ms=3,
                lw=1.0,
                label=f"NFE={nfe}",
            )
        ax_g.set_ylabel(f"noise {nid}\ngrip (exec)")
        ax_g.legend(fontsize=7, loc="best")
        ax_g.grid(True, alpha=0.25)
        arm_rmse = []
        for nfe in (1, 2, 3, 10):
            pred = arrays[(nid, nfe)]
            err = pred - ref
            arm_rmse.append(float(np.sqrt((err[:, :6] ** 2).mean())))
        ax_a.bar([1, 2, 3, 10], arm_rmse, color=[colors[k] for k in (1, 2, 3, 10)], alpha=0.85)
        ax_a.set_xticks([1, 2, 3, 10])
        ax_a.set_ylabel(f"noise {nid}\narm RMSE (norm)")
        ax_a.grid(True, axis="y", alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_xlabel("exec step")
    fig.tight_layout()
    out = CASE_DIR / f"shared_{shared_index:03d}_curves.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)


def classify_case(shared_index: int, by_noise: dict[int, dict]) -> str:
    """Short Chinese tag for report (动作偏差 / 偏差缓解 wording)."""
    notes = []
    for nid in range(N_NOISE):
        m1 = by_noise[nid][1]
        m2 = by_noise[nid][2]
        m3 = by_noise[nid][3]
        if m2["gripper_disagree_n"] > 0 and m1["gripper_disagree_n"] > 0:
            if m3["gripper_disagree_n"] > 0:
                notes.append(
                    f"noise{nid}: 开合执行态在 NFE=1–3 均与参照不一致（{m1['gripper_disagree_n']}→{m2['gripper_disagree_n']}→{m3['gripper_disagree_n']} 步），"
                    "更像切换时机偏移而非单步积分噪声。"
                )
            else:
                notes.append(
                    f"noise{nid}: NFE=2 仍保留开合偏差（{m2['gripper_disagree_n']} 步），NFE=3 已缓解；"
                    "低步数下夹爪阈值附近敏感。"
                )
        elif m1["gripper_disagree_n"] > 0 and m2["gripper_disagree_n"] == 0:
            notes.append(f"noise{nid}: 开合偏差在 NFE=2 已缓解（NFE=1 有 {m1['gripper_disagree_n']} 步不一致）。")
        if m1["rmse_trans"] > 0.12 or m1["rmse_rot"] > 0.12:
            notes.append(
                f"noise{nid}: NFE=1 手臂归一化 RMSE 偏大（trans={m1['rmse_trans']:.3f}, rot={m1['rmse_rot']:.3f}），"
                f"NFE=2 降至 trans={m2['rmse_trans']:.3f}, rot={m2['rmse_rot']:.3f}。"
            )
        if m1["cosine"] < 0.94 and m2["cosine"] < 0.98:
            notes.append(
                f"noise{nid}: 余弦在 NFE=2 仍偏低（{m1['cosine']:.3f}→{m2['cosine']:.3f}），"
                "以手臂方向偏差为主。"
            )
    if not notes:
        return "逐步积分后动作偏差整体已缓解；残余主要为小幅归一化尺度误差。"
    return " ".join(notes)


def main() -> None:
    npz_path = PROBE_DIR / "actions.npz"
    if not npz_path.is_file():
        (CASE_DIR / "MISSING_ACTIONS_NPZ.md").write_text(
            "# actions.npz 缺失\n\n未找到 `actions.npz`；跳过异常导出，请先完成 SmolVLA 探针推理。\n"
        )
        print("missing actions.npz", flush=True)
        return

    CASE_DIR.mkdir(parents=True, exist_ok=True)
    shared = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    correctness = json.loads((PROBE_DIR / "correctness.json").read_text())
    mean, std = load_mean_std(correctness)
    z = np.load(npz_path)
    alpha = z["alpha"]

    import pandas as pd

    frames = pd.read_csv(PROBE_DIR / "frames.csv")
    frame_lookup = {
        (int(r.shared_index), int(r.noise_id), int(r.nfe)): r
        for r in frames.itertuples()
    }

    verification = []
    case_summaries = []

    for si in ALL_SHARED:
        rec = shared["records"][si]
        meta = {
            "shared_index": si,
            "global_index": int(rec["global_index"]),
            "episode_index": int(rec["episode_index"]),
            "task": rec["task"],
            "task_index": int(rec["task_index"]),
            "frame_in_episode": int(rec["frame_in_episode"]),
            "alpha": float(alpha[si]),
            "in_top_alpha": int(
                frames[(frames.shared_index == si) & (frames.nfe == 1) & (frames.noise_id == 0)].iloc[0][
                    "in_top_alpha"
                ]
            ),
            "in_grip_switch": int(
                frames[(frames.shared_index == si) & (frames.nfe == 1) & (frames.noise_id == 0)].iloc[0][
                    "in_grip_switch"
                ]
            ),
        }
        arrays = {}
        csv_rows = []
        metrics_by_noise = {}
        for nid in range(N_NOISE):
            metrics_by_noise[nid] = {}
            ref = z[f"nfe10"][si, nid]
            for nfe in NFES_EXPORT:
                act = z[f"nfe{nfe}"][si, nid]
                arrays[(nid, nfe)] = act
                m = metrics_row(torch.from_numpy(act), torch.from_numpy(ref), mean, std)
                metrics_by_noise[nid][nfe] = m
                csv_rows.extend(step_records(si, nid, nfe, act, ref, mean, std, meta))
                fr = frame_lookup.get((si, nid, nfe))
                if fr is not None and nfe != 10:
                    verification.append(
                        {
                            "shared_index": si,
                            "noise_id": nid,
                            "nfe": nfe,
                            "csv_cosine": float(fr.cosine),
                            "recomputed_cosine": m["cosine"],
                            "cosine_abs_diff": abs(float(fr.cosine) - m["cosine"]),
                            "csv_rmse": float(fr.rmse),
                            "recomputed_rmse": m["rmse"],
                            "rmse_abs_diff": abs(float(fr.rmse) - m["rmse"]),
                            "csv_grip_n": int(fr.gripper_disagree_n),
                            "recomputed_grip_n": m["gripper_disagree_n"],
                        }
                    )

        plot_case(si, arrays, mean, std, meta)

        payload = {
            "meta": meta,
            "metrics_vs_nfe10": {
                str(nid): {str(nfe): metrics_by_noise[nid][nfe] for nfe in NFES_EXPORT if nfe != 10}
                for nid in range(N_NOISE)
            },
            "interpretation": classify_case(si, metrics_by_noise),
            "bucket": (
                "nfe2_remaining"
                if si in TARGETS["nfe2_remaining"]
                else "care_miss_examples"
            ),
        }
        (CASE_DIR / f"shared_{si:03d}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        with (CASE_DIR / f"shared_{si:03d}_steps.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        case_summaries.append(payload)

    vdf = pd.DataFrame(verification)
    max_cos = float(vdf["cosine_abs_diff"].max()) if len(vdf) else 0.0
    max_rmse = float(vdf["rmse_abs_diff"].max()) if len(vdf) else 0.0
    (CASE_DIR / "verification.json").write_text(
        json.dumps(
            {
                "n_checks": len(verification),
                "max_cosine_abs_diff": max_cos,
                "max_rmse_abs_diff": max_rmse,
                "all_grip_match": bool((vdf["csv_grip_n"] == vdf["recomputed_grip_n"]).all()) if len(vdf) else True,
                "rows": verification,
            },
            indent=2,
        )
        + "\n"
    )

    lines = [
        "# SmolVLA NFE 探针 — 异常样本补充（离线动作）",
        "",
        "用语：**动作偏差**、**偏差缓解**；不将单帧低 NFE 动作差异表述为闭环失败或失败修复。",
        "",
        f"数据源：`{PROBE_DIR.relative_to(FN_ROOT)}` 的 `actions.npz` / `frames.csv`。",
        f"独立复算与 `frames.csv` 核对：{len(verification)} 条，max |Δcosine|={max_cos:.2e}，max |Δrmse|={max_rmse:.2e}。",
        "",
        "## 样本列表",
        "",
        "| shared | 分组 | α | topα | gripH | 解读 |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for p in case_summaries:
        m = p["meta"]
        lines.append(
            f"| {m['shared_index']} | {p['bucket']} | {m['alpha']:.3f} | {m['in_top_alpha']} | "
            f"{m['in_grip_switch']} | {p['interpretation'][:120]}… |"
        )
    lines += [
        "",
        "## NFE=2 后仍偏大的帧（89/381 @ noise0；437/451 @ noise1）",
        "",
        "- **89 / 381（noise0）**：NFE=2 余弦已升高，但 **开合执行态仍整段不一致**（`gripper_disagree_n=1` 贯穿 NFE=1–3），"
        "连续夹爪值在阈值附近偏移，手臂 RMSE 已明显下降 → **切换时机 / 阈值敏感** 为主，不是单纯手臂大幅跑偏。",
        "- **437（noise1）**：NFE=1 手臂与夹爪偏差都大；NFE=2 **夹爪仍不一致** 且余弦 <0.98 → 低步数下 **夹爪+手臂** 复合偏差。",
        "- **451（noise1）**：NFE=1 余弦最低档（~0.84）且夹爪 1 步不一致；**NFE=2 夹爪已对齐**，余弦升至 ~0.98 → "
        "典型 **NFE=2 偏差缓解**；残余为手臂归一化误差。",
        "",
        "## 高 α 未覆盖的代表（124 / 121）",
        "",
        "- 二者 **不在 topα 组**（每任务 α 最高 3 帧），但 NFE=1 在 noise1 上余弦/RMSE 仍进入全体尾部（尤其 124）。",
        "- **NFE=2 均明显缓解**（夹爪不一致归零，余弦 >0.99）→ CARE α **漏检** 的是「中等 α 但仍对低 NFE 敏感」的帧，"
        "不能仅靠高 α 子集筛尽动作偏差。",
        "",
        "## 文件",
        "",
        "- `shared_XXX.json` / `shared_XXX_steps.csv` / `shared_XXX_curves.png`",
        "- `verification.json`",
        "",
    ]
    (CASE_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {CASE_DIR / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()

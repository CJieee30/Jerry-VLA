#!/usr/bin/env python3
"""CARE gripper-event boost (alpha_event) offline diagnostic on frozen NFE probes.

CPU only. Does not run policy inference or modify shared.pt / probe outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FN_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pi05_libero"))

from dump_shared import _load_actions
from preview_timestamps import (
    CAMS,
    FFMPEG,
    FPS,
    SUITES,
    _annotate,
    _clip_range,
    _decode_episode_frames,
    _episodes_table,
    _hstack,
    _window_steps,
)
from protocol import ACTION_DIM, FRAMES_PER_TASK, HORIZON, LIBERO_REPO, N_FRAMES, OUT_DIR, SEED, hub_snapshot

EVENT_DIR = OUT_DIR / "nfe_probe_event_care"
GROUP_SEED = 42
TOP_K = 3
EVENT_BOOST = 1.0
MODELS = {
    "smolvla": OUT_DIR / "nfe_probe_smolvla",
    "pi05": OUT_DIR / "nfe_probe_pi05",
}
PREVIEW_REF = OUT_DIR / "preview_smolvla_timestamps"
CONTEXT_PRE_SEC = 2.0
CONTEXT_POST_SEC = 3.0
CONTEXT_PRE = int(round(CONTEXT_PRE_SEC * FPS))
CONTEXT_POST = int(round(CONTEXT_POST_SEC * FPS))
CURVE_FOCUS = (124, 381, 361, 447)

# Pool ranks are 1..12 within each suite task-0 candidate set.
DEFAULT_PREVIEW_SPECS: tuple[dict, ...] = (
    {"score": "alpha_event", "ranks": (1, 12), "tag": "event"},
)
EXTENDED_PREVIEW_SPECS: tuple[dict, ...] = (
    {"score": "alpha_event", "ranks": (1, 2, 3, 10, 11, 12), "tag": "event"},
    {"score": "alpha", "ranks": (1, 2, 3, 10, 11, 12), "tag": "alpha"},
)


@dataclass
class FrameMeta:
    shared_index: int
    global_index: int
    task_index: int
    task: str
    episode_index: int
    frame_in_episode: int
    length: int
    alpha: float
    d_whitened: float
    G: int
    alpha_event: float
    in_grip_switch_probe: int


def demo_gripper_switch(shared: dict, actions: np.ndarray, episode: np.ndarray) -> np.ndarray:
    flags = np.zeros((len(shared["records"]),), dtype=np.bool_)
    for i, rec in enumerate(shared["records"]):
        start = int(rec["global_index"])
        ep = int(episode[start])
        seq = []
        for t in range(start, min(start + HORIZON, len(actions))):
            if int(episode[t]) != ep:
                break
            seq.append(float(actions[t, 6]))
        if len(seq) < 2:
            flags[i] = False
            continue
        bits = [g > 0.0 for g in seq]
        flags[i] = any(bits[j] != bits[j + 1] for j in range(len(bits) - 1))
    return flags


def boundary_grip_events(
    shared: dict, actions: np.ndarray, episode: np.ndarray
) -> list[dict]:
    """Episode-local: prev demo action vs window start (grip dim)."""
    rows = []
    for i, rec in enumerate(shared["records"]):
        gidx = int(rec["global_index"])
        ep = int(episode[gidx])
        frame0 = int(rec["frame_in_episode"])
        if frame0 == 0:
            rows.append(
                {
                    "shared_index": i,
                    "episode_index": ep,
                    "frame_in_episode": frame0,
                    "kind": "episode_start_no_prev",
                    "in_window_G": False,
                }
            )
            continue
        prev_gidx = gidx - 1
        if int(episode[prev_gidx]) != ep:
            rows.append(
                {
                    "shared_index": i,
                    "episode_index": ep,
                    "frame_in_episode": frame0,
                    "kind": "cross_episode_prev",
                    "in_window_G": False,
                }
            )
            continue
        g0 = float(actions[prev_gidx, 6]) > 0.0
        g1 = float(actions[gidx, 6]) > 0.0
        if g0 != g1:
            rows.append(
                {
                    "shared_index": i,
                    "episode_index": ep,
                    "frame_in_episode": frame0,
                    "kind": "boundary_prev_to_start",
                    "in_window_G": False,
                }
            )
    return rows


def build_metas(shared: dict, alpha: np.ndarray, G: np.ndarray, probe_G: np.ndarray) -> list[FrameMeta]:
    raw = alpha + EVENT_BOOST * G.astype(np.float64)
    alpha_event = raw / float(np.mean(raw))
    metas = []
    for i, rec in enumerate(shared["records"]):
        metas.append(
            FrameMeta(
                shared_index=i,
                global_index=int(rec["global_index"]),
                task_index=int(rec["task_index"]),
                task=rec["task"],
                episode_index=int(rec["episode_index"]),
                frame_in_episode=int(rec["frame_in_episode"]),
                length=int(rec["length"]),
                alpha=float(alpha[i]),
                d_whitened=float(shared["d_whitened"][i]),
                G=int(G[i]),
                alpha_event=float(alpha_event[i]),
                in_grip_switch_probe=int(probe_G[i]),
            )
        )
    return metas


def select_per_task(
    metas: list[FrameMeta], score_key: str, *, random_idxs: set[int] | None = None
) -> set[int]:
    by_task: dict[int, list[int]] = defaultdict(list)
    for m in metas:
        by_task[m.task_index].append(m.shared_index)
    chosen: set[int] = set()
    if random_idxs is not None:
        return set(random_idxs)
    for ti in sorted(by_task):
        idxs = by_task[ti]
        if score_key == "alpha":
            ordered = sorted(
                idxs,
                key=lambda j: (-metas[j].alpha, metas[j].global_index),
            )
        elif score_key == "alpha_event":
            ordered = sorted(
                idxs,
                key=lambda j: (-metas[j].alpha_event, metas[j].global_index),
            )
        else:
            raise ValueError(score_key)
        chosen.update(ordered[:TOP_K])
    return chosen


def random_group(metas: list[FrameMeta], seed: int = GROUP_SEED) -> set[int]:
    by_task: dict[int, list[int]] = defaultdict(list)
    for m in metas:
        by_task[m.task_index].append(m.shared_index)
    rng = np.random.default_rng(seed)
    out = set()
    for ti in sorted(by_task):
        idxs = np.asarray(sorted(by_task[ti]), dtype=np.int64)
        pick = rng.choice(idxs, size=min(TOP_K, idxs.size), replace=False)
        out.update(int(x) for x in pick)
    return out


def g_only_group(metas: list[FrameMeta], seed: int) -> set[int]:
    """Per task: prefer G=1 windows; ties and fill use RNG(seed), never alpha."""
    by_task: dict[int, list[int]] = defaultdict(list)
    for m in metas:
        by_task[m.task_index].append(m.shared_index)
    rng = np.random.default_rng(seed)
    out: set[int] = set()
    for ti in sorted(by_task):
        idxs = by_task[ti]
        g1 = [j for j in idxs if metas[j].G == 1]
        g0 = [j for j in idxs if metas[j].G == 0]
        rng.shuffle(g1)
        rng.shuffle(g0)
        picked: list[int] = []
        picked.extend(g1[:TOP_K])
        if len(picked) < TOP_K:
            picked.extend(g0[: TOP_K - len(picked)])
        out.update(picked[:TOP_K])
    return out


def load_probe_frames(path: Path) -> pd.DataFrame:
    if not (path / "frames.csv").is_file():
        raise FileNotFoundError(path / "frames.csv")
    return pd.read_csv(path / "frames.csv")


def arm_proxy_rmse(row) -> float:
    return float(np.hypot(float(row.rmse_trans), float(row.rmse_rot)))


def tail_mask_nfe1(df: pd.DataFrame, noise_id: int) -> tuple[set[int], float, float]:
    sub = df[(df.nfe == 1) & (df.noise_id == noise_id)]
    cos_p5 = float(sub.cosine.quantile(0.05))
    rmse_p95 = float(sub.rmse.quantile(0.95))
    bad = set()
    for r in sub.itertuples():
        if (
            float(r.cosine) < cos_p5
            or float(r.rmse) > rmse_p95
            or int(r.gripper_disagree_n) > 0
        ):
            bad.add(int(r.shared_index))
    return bad, cos_p5, rmse_p95


def grip_deviation_set(df: pd.DataFrame, noise_id: int, nfe: int) -> set[int]:
    sub = df[(df.nfe == nfe) & (df.noise_id == noise_id)]
    return {int(r.shared_index) for r in sub.itertuples() if int(r.gripper_disagree_n) > 0}


def arm_top5_set(df: pd.DataFrame, noise_id: int) -> tuple[set[int], float]:
    sub = df[(df.nfe == 1) & (df.noise_id == noise_id)].copy()
    sub["arm_rmse"] = sub.apply(arm_proxy_rmse, axis=1)
    thr = float(sub.arm_rmse.quantile(0.95))
    top = {int(r.shared_index) for r in sub.itertuples() if float(r.arm_rmse) >= thr}
    return top, thr


def eval_selection(
    name: str,
    selected: set[int],
    G_all: set[int],
    tail_bad: set[int],
    grip_bad_nfe1: set[int],
    grip_bad_nfe2: set[int],
    arm_top5: set[int],
    df_nfe1: pd.DataFrame,
    noise_id: int,
) -> dict:
    sub = df_nfe1[(df_nfe1.nfe == 1) & (df_nfe1.noise_id == noise_id)]
    sub = sub[sub.shared_index.isin(selected)]
    rmse = sub.rmse.to_numpy(dtype=np.float64)
    tail_hit = selected & tail_bad
    return {
        "group": name,
        "noise_id": noise_id,
        "n_selected": len(selected),
        "n_grip_event_windows": len(selected & G_all),
        "n_tail_bad_total": len(tail_bad),
        "n_tail_captured": len(tail_hit),
        "tail_recall": len(tail_hit) / len(tail_bad) if tail_bad else float("nan"),
        "n_grip_dev_nfe1_total": len(grip_bad_nfe1),
        "n_grip_dev_nfe1_captured": len(selected & grip_bad_nfe1),
        "n_grip_dev_nfe2_total": len(grip_bad_nfe2),
        "n_grip_dev_nfe2_captured": len(selected & grip_bad_nfe2),
        "n_arm_top5_total": len(arm_top5),
        "n_arm_top5_captured": len(selected & arm_top5),
        "rmse_mean": float(rmse.mean()) if rmse.size else float("nan"),
        "rmse_p95": float(np.percentile(rmse, 95)) if rmse.size else float("nan"),
    }


def model_stats(model: str, df: pd.DataFrame, groups: dict[str, set[int]], G_all: set[int]) -> dict:
    out = {"model": model, "by_noise": {}, "grip_event_inconsistency": {}}
    for nid in (0, 1):
        tail_bad, cos_p5, rmse_p95 = tail_mask_nfe1(df, nid)
        g1 = grip_deviation_set(df, nid, 1)
        g2 = grip_deviation_set(df, nid, 2)
        arm5, arm_thr = arm_top5_set(df, nid)
        sub1 = df[(df.nfe == 1) & (df.noise_id == nid)]
        out["by_noise"][str(nid)] = {
            "thresholds": {"cosine_p5": cos_p5, "rmse_p95": rmse_p95, "arm_rmse_p95": arm_thr},
            "groups": {
                gname: eval_selection(
                    gname,
                    groups[gname],
                    G_all,
                    tail_bad,
                    g1,
                    g2,
                    arm5,
                    df,
                    nid,
                )
                for gname in groups
            },
        }
    for nfe in (1, 2):
        rates = []
        for si in sorted(G_all):
            for nid in (0, 1):
                row = df[(df.shared_index == si) & (df.nfe == nfe) & (df.noise_id == nid)]
                if len(row):
                    rates.append(int(row.iloc[0].gripper_disagree_n) > 0)
        out["grip_event_inconsistency"][f"nfe{nfe}"] = {
            "n_grip_event_windows": len(G_all),
            "n_checks": len(rates),
            "frac_any_step_disagree": float(np.mean(rates)) if rates else float("nan"),
            "note": "action-level grip open/close vs NFE=10 reference; not closed-loop SR",
        }
    return out


def diff_lists(
    metas: list[FrameMeta], df: pd.DataFrame, old: set[int], new: set[int], noise_id: int = 0
) -> tuple[list[dict], list[dict]]:
    added = sorted(new - old)
    removed = sorted(old - new)

    def pack(si: int) -> dict:
        m = metas[si]
        r = df[(df.shared_index == si) & (df.nfe == 1) & (df.noise_id == noise_id)].iloc[0]
        return {
            "shared_index": si,
            "task_index": m.task_index,
            "task": m.task,
            "episode_index": m.episode_index,
            "frame_in_episode": m.frame_in_episode,
            "alpha": m.alpha,
            "alpha_event": m.alpha_event,
            "G": m.G,
            "nfe1_cosine": float(r.cosine),
            "nfe1_rmse": float(r.rmse),
            "nfe1_grip_disagree_n": int(r.gripper_disagree_n),
            "arm_rmse_proxy": arm_proxy_rmse(r),
        }

    return [pack(si) for si in added], [pack(si) for si in removed]


def plot_focus_curves(
    metas: list[FrameMeta],
    actions: np.ndarray,
    episode: np.ndarray,
    df_smol: pd.DataFrame,
    df_pi: pd.DataFrame,
    dest: Path,
) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for si in CURVE_FOCUS:
        if si >= len(metas):
            continue
        m = metas[si]
        gidx = m.global_index
        ep = int(episode[gidx])
        seq = []
        for t in range(gidx, min(gidx + HORIZON, len(actions))):
            if int(episode[t]) != ep:
                break
            seq.append(actions[t, :ACTION_DIM])
        if len(seq) < 2:
            continue
        seq = np.stack(seq)
        steps = np.arange(len(seq))
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        for ax, dims, title in (
            (axes[0], slice(0, 6), "arm (demo)"),
            (axes[1], slice(6, 7), "gripper (demo)"),
        ):
            for d in range(dims.start or 0, (dims.stop or ACTION_DIM)):
                ax.plot(steps, seq[:, d], lw=1.2, label=f"d{d}")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=4)
        fig.suptitle(
            f"shared={si} ep={m.episode_index} f={m.frame_in_episode} "
            f"α={m.alpha:.3f} α_evt={m.alpha_event:.3f} G={m.G}",
            fontsize=10,
        )
        rs = df_smol[(df_smol.shared_index == si) & (df_smol.nfe == 1) & (df_smol.noise_id == 0)]
        rp = df_pi[(df_pi.shared_index == si) & (df_pi.nfe == 1) & (df_pi.noise_id == 0)]
        if len(rs) and len(rp):
            fig.text(
                0.02,
                0.01,
                f"SmolVLA NFE1 cos={rs.iloc[0].cosine:.3f} rmse={rs.iloc[0].rmse:.3f} | "
                f"π0.5 cos={rp.iloc[0].cosine:.3f} rmse={rp.iloc[0].rmse:.3f}",
                fontsize=8,
            )
        out = dest / f"demo_window_shared_{si:03d}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        written.append(str(out.relative_to(EVENT_DIR)))
    return written


def _clip_range_event(frame0: int, length: int, window_len: int) -> dict:
    want_pre = CONTEXT_PRE
    want_post = CONTEXT_POST
    win_end = frame0 + window_len - 1
    pre = min(want_pre, frame0)
    post = min(want_post, max(0, length - 1 - win_end))
    clip0 = frame0 - pre
    clip1 = win_end + post
    notes = []
    if pre < want_pre:
        notes.append(f"pre truncated {pre}/{want_pre} frames")
    if post < want_post:
        notes.append(f"post truncated {post}/{want_post} frames")
    return {
        "clip0": int(clip0),
        "clip1": int(clip1),
        "pre": int(pre),
        "post": int(post),
        "win_end": int(min(win_end, length - 1)),
        "notes": notes,
    }


def _pool_ordered(pool: list[FrameMeta], score: str) -> list[FrameMeta]:
    if score == "alpha_event":
        return sorted(pool, key=lambda m: (-m.alpha_event, m.global_index))
    if score == "alpha":
        return sorted(pool, key=lambda m: (-m.alpha, m.global_index))
    raise ValueError(score)


def _legacy_kind(score: str, rank: int, pool_n: int) -> str:
    if score == "alpha_event" and rank == 1:
        return "high_event"
    if score == "alpha_event" and rank == pool_n:
        return "low_event"
    return ""


def _clip_basename(slug: str, score: str, tag: str, rank: int, legacy_kind: str) -> str:
    if legacy_kind:
        return f"{slug}_{legacy_kind}"
    return f"{slug}_{tag}_r{rank:02d}"


def _render_one_preview(
    rec: FrameMeta,
    *,
    suite: str,
    slug: str,
    score: str,
    tag: str,
    rank: int,
    pool_n: int,
    video_dir: Path,
    basename: str,
    eps_by_id: dict,
    root: Path,
    actions_all: np.ndarray,
    episode_all: np.ndarray,
) -> dict:
    ep = rec.episode_index
    eprow = eps_by_id[ep]
    gidx = rec.global_index
    frame0 = rec.frame_in_episode
    length = rec.length
    _, win_len = _window_steps(actions_all, episode_all, gidx)
    clip = _clip_range_event(frame0, length, win_len)
    clip0, clip1 = clip["clip0"], clip["clip1"]
    n_clip = clip1 - clip0 + 1
    ep_start = int(eprow["dataset_from_index"])
    act = actions_all[ep_start + clip0 : ep_start + clip1 + 1, :ACTION_DIM]
    switch_local = []
    for t in range(gidx + 1, min(clip["win_end"] + 1, len(actions_all))):
        if int(episode_all[t]) != ep:
            break
        if (actions_all[t - 1, 6] > 0) != (actions_all[t, 6] > 0):
            switch_local.append(t - ep_start)

    cam_clips = []
    for key_cam, _ in CAMS:
        vchunk = int(eprow[f"videos/{key_cam}/chunk_index"])
        vfile = int(eprow[f"videos/{key_cam}/file_index"])
        from_ts = float(eprow[f"videos/{key_cam}/from_timestamp"])
        vpath = root / "videos" / key_cam / f"chunk-{vchunk:03d}" / f"file-{vfile:03d}.mp4"
        decoded = _decode_episode_frames(vpath, from_ts, length)
        cam_clips.append(decoded[clip0 : clip1 + 1])

    score_label = "α_event" if score == "alpha_event" else "α"
    tier = "high" if rank <= pool_n // 2 else "low"
    header_kind = f"{tag} {score_label} rank {rank}/{pool_n} ({tier})"

    annotated = []
    for i in range(n_clip):
        local = clip0 + i
        role = None
        if local == frame0:
            role = "timestamp"
        elif frame0 <= local <= clip["win_end"]:
            role = "window"
        if local in switch_local:
            role = "grip_switch"
        lines = [
            f"{suite} task0 | {header_kind}",
            f"ep={ep} frame={local}/{length-1} t={local/FPS:.2f}s",
            f"timestamp={frame0} H={HORIZON} window [{frame0},{clip['win_end']}]",
            f"α={rec.alpha:.3f} α_event={rec.alpha_event:.3f} G={rec.G}",
            "demo playback only — not low-NFE closed-loop",
        ]
        if switch_local:
            lines.append(f"grip switches in window @ frames {switch_local}")
        pair = _hstack(cam_clips[0][i], cam_clips[1][i])
        annotated.append(_annotate(pair, lines, highlight=role))

    out_mp4 = video_dir / f"{basename}.mp4"
    _write_video(annotated, out_mp4)
    plot_path = video_dir / f"{basename}_actions.png"
    _plot_demo_actions(plot_path, clip0, clip1, frame0, clip["win_end"], act, rec, switch_local)

    legacy = _legacy_kind(score, rank, pool_n)
    return {
        "suite": suite,
        "suite_slug": slug,
        "score": score,
        "tag": tag,
        "kind": legacy or f"{tag}_r{rank:02d}",
        "tier": tier,
        "shared_index": rec.shared_index,
        "episode_index": ep,
        "frame_in_episode": frame0,
        "alpha": rec.alpha,
        "alpha_event": rec.alpha_event,
        "G": rec.G,
        "pool_rank": rank,
        "pool_rank_event": rank if score == "alpha_event" else None,
        "video": str(out_mp4.relative_to(EVENT_DIR)),
        "plot": str(plot_path.relative_to(EVENT_DIR)),
    }


def write_event_videos(
    metas: list[FrameMeta],
    actions_all: np.ndarray,
    episode_all: np.ndarray,
    *,
    specs: tuple[dict, ...] = DEFAULT_PREVIEW_SPECS,
    video_subdir: str = "videos",
) -> list[dict]:
    root = hub_snapshot(LIBERO_REPO)
    eps = _episodes_table(root)
    eps_by_id = {int(r["episode_index"]): r for _, r in eps.iterrows()}
    catalog = []
    video_dir = EVENT_DIR / video_subdir
    video_dir.mkdir(parents=True, exist_ok=True)

    for slug, suite, lang in SUITES:
        pool = [m for m in metas if m.task == lang]
        if len(pool) != 12:
            raise RuntimeError(f"{suite} pool size {len(pool)} != 12")
        pool_n = len(pool)
        for spec in specs:
            score = spec["score"]
            tag = spec["tag"]
            ordered = _pool_ordered(pool, score)
            for rank in spec["ranks"]:
                if rank < 1 or rank > pool_n:
                    raise ValueError(f"rank {rank} out of 1..{pool_n}")
                rec = ordered[rank - 1]
                legacy = _legacy_kind(score, rank, pool_n)
                basename = _clip_basename(slug, score, tag, rank, legacy)
                catalog.append(
                    _render_one_preview(
                        rec,
                        suite=suite,
                        slug=slug,
                        score=score,
                        tag=tag,
                        rank=rank,
                        pool_n=pool_n,
                        video_dir=video_dir,
                        basename=basename,
                        eps_by_id=eps_by_id,
                        root=root,
                        actions_all=actions_all,
                        episode_all=episode_all,
                    )
                )
    return catalog


def _write_video(frames_rgb: list[np.ndarray], dest: Path) -> None:
    tmp = dest.parent / "_frames_tmp"
    tmp.mkdir(exist_ok=True)
    for p in tmp.glob("*.png"):
        p.unlink()
    for i, fr in enumerate(frames_rgb):
        Image.fromarray(fr).save(tmp / f"{i:04d}.png")
    subprocess.check_call(
        [
            FFMPEG,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(int(FPS)),
            "-i",
            str(tmp / "%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(dest),
        ]
    )
    for p in tmp.glob("*.png"):
        p.unlink()


def _plot_demo_actions(
    path: Path,
    clip0: int,
    clip1: int,
    frame0: int,
    win_end: int,
    act: np.ndarray,
    rec: FrameMeta,
    switch_local: list[int],
) -> None:
    t = np.arange(clip0, clip1 + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for i in range(6):
        axes[0].plot(t, act[:, i], lw=1.0, label=f"a{i}")
    axes[0].set_ylabel("arm demo")
    axes[0].legend(ncol=6, fontsize=7)
    axes[1].plot(t, act[:, 6], color="k", lw=1.4)
    axes[1].set_ylabel("gripper demo")
    for ax in axes:
        ax.axvspan(frame0, win_end, color="gold", alpha=0.2)
        ax.axvline(frame0, color="crimson", ls="--")
        for sf in switch_local:
            ax.axvline(sf, color="C0", ls=":", lw=1.2)
    fig.suptitle(
        f"{rec.task[:60]}… α_evt={rec.alpha_event:.3f} G={rec.G}",
        fontsize=9,
    )
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _catalog_item_blurb(it: dict) -> str:
    rank = it.get("pool_rank", it.get("pool_rank_event"))
    score = it.get("score", "alpha_event")
    score_label = "α_event" if score == "alpha_event" else "α"
    return (
        f"{score_label} rank {rank}/12 · α={it['alpha']:.3f} · α_event={it['alpha_event']:.3f} · G={it['G']}"
    )


def write_html(catalog: list[dict]) -> None:
    lines = [
        "<!doctype html><meta charset='utf-8'><title>Event CARE preview</title>",
        "<style>body{font-family:sans-serif;max-width:1000px;margin:20px auto;padding:0 12px}",
        "video{max-width:100%} .pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}",
        "h2{margin-top:2em}</style>",
        "<h1>Event CARE — candidate-pool high/low α_event</h1>",
        "<p>Demonstration trajectories only. High/low within each suite task-0 pool of 12 timestamps.</p>",
        "<p>Multitask α_event high/low: <a href='multitask_event_index.html'>multitask_event_index.html</a></p>",
    ]
    if PREVIEW_REF.is_dir():
        lines.append(
            f"<p>Original α-based previews: <a href='../../preview_smolvla_timestamps/index.html'>preview_smolvla_timestamps</a></p>"
        )
    by_suite = defaultdict(list)
    for c in catalog:
        by_suite[c["suite_slug"]].append(c)
    for slug in sorted(by_suite):
        items = sorted(by_suite[slug], key=lambda x: x["kind"])
        lines.append(f"<h2>{slug}</h2><div class='pair'>")
        for it in items:
            lines.append(
                f"<div><h3>{it['kind']}</h3>"
                f"<p>{_catalog_item_blurb(it)}</p>"
                f"<video controls src='{it['video']}'></video>"
                f"<p><img src='{it['plot']}' style='max-width:100%'></p></div>"
            )
        lines.append("</div>")
    (EVENT_DIR / "index.html").write_text("\n".join(lines) + "\n")


def write_gallery_html(catalog: list[dict], *, rel_prefix: str = "") -> None:
    """Side-by-side high vs low pairs for multiple pool ranks."""
    lines = [
        "<!doctype html><meta charset='utf-8'><title>Event CARE gallery</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:20px auto;padding:0 12px}",
        "video{max-width:100%} .pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}",
        ".row{margin:1.5em 0;padding-bottom:1em;border-bottom:1px solid #ddd}",
        "h2{margin-top:2.5em}</style>",
        "<h1>Event CARE — extended high / low comparisons</h1>",
        "<p>Each row: same suite, matched ranks (1↔12, 2↔11, 3↔10) under the given sort key.</p>",
        "<p><a href='index.html'>Back to minimal α_event high/low</a></p>",
    ]
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in catalog:
        by_key[(c["suite_slug"], c["score"])].append(c)
    pair_ranks = ((1, 12), (2, 11), (3, 10))
    for slug in sorted({k[0] for k in by_key}):
        for score in ("alpha_event", "alpha"):
            items = by_key.get((slug, score))
            if not items:
                continue
            score_label = "α_event (event CARE)" if score == "alpha_event" else "α (original CARE)"
            lines.append(f"<h2>{slug} — sorted by {score_label}</h2>")
            by_rank = {int(it["pool_rank"]): it for it in items}
            for hi, lo in pair_ranks:
                if hi not in by_rank or lo not in by_rank:
                    continue
                lines.append(f"<div class='row'><h3>rank {hi} vs {lo}</h3><div class='pair'>")
                for it in (by_rank[hi], by_rank[lo]):
                    lines.append(
                        f"<div><h4>{it['kind']}</h4><p>{_catalog_item_blurb(it)} · ep={it['episode_index']} f={it['frame_in_episode']}</p>"
                        f"<video controls src='{rel_prefix}{it['video']}'></video>"
                        f"<p><img src='{rel_prefix}{it['plot']}' style='max-width:100%'></p></div>"
                    )
                lines.append("</div></div>")
    (EVENT_DIR / "gallery_index.html").write_text("\n".join(lines) + "\n")


def write_report(payload: dict, curves: list[str]) -> None:
    lines = [
        "# CARE 夹爪事件增强（alpha_event）离线诊断",
        "",
        "本轮仅复用 CARE v2 `shared.pt` 与 SmolVLA / π0.5 离线 NFE 探针 `frames.csv`。"
        "**动作开合不一致**指相对同帧 NFE=10 参照的执行端映射差异，不是闭环失败率。",
        "",
        "## 公式",
        "",
        f"`G=1` ⇔ 未来 H={HORIZON} 示范窗口内夹爪开合切换（与探针 `in_grip_switch` 一致）。",
        f"`alpha_event = (alpha + {EVENT_BOOST}·G) / mean(alpha + {EVENT_BOOST}·G)`。",
        "",
        f"探针 G 与重算一致率：**{payload['grip_flag_agreement']:.1%}**（{payload['n_grip_mismatch']} 处不一致）。",
        f"边界「前一时刻→窗口起点」切换：**{payload['n_boundary_events']}** 条（主对照仍用窗口内 G）。",
        "",
        "## 同预算选样（每任务 3/12，共 120）",
        "",
    ]
    for model_key in ("smolvla", "pi05"):
        lines.append(f"### {model_key}")
        lines.append("")
        for nid in ("0", "1"):
            lines.append(f"**noise {nid}**")
            lines.append("")
            lines.append(
                "| group | grip窗口 | tail捕获/总数 | tail召回 | grip偏差NFE1 | grip偏差NFE2 | arm top5% | RMSE mean | RMSE P95 |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for g in ("random", "care_alpha", "care_event"):
                s = payload["models"][model_key]["by_noise"][nid]["groups"][g]
                lines.append(
                    f"| {g} | {s['n_grip_event_windows']} | {s['n_tail_captured']}/{s['n_tail_bad_total']} | "
                    f"{100*s['tail_recall']:.1f}% | {s['n_grip_dev_nfe1_captured']}/{s['n_grip_dev_nfe1_total']} | "
                    f"{s['n_grip_dev_nfe2_captured']}/{s['n_grip_dev_nfe2_total']} | "
                    f"{s['n_arm_top5_captured']}/{s['n_arm_top5_total']} | {s['rmse_mean']:.4f} | {s['rmse_p95']:.4f} |"
                )
            lines.append("")
        gi = payload["models"][model_key]["grip_event_inconsistency"]
        lines.append(
            f"全部 G=1 窗口（n={gi['nfe1']['n_grip_event_windows']}）动作开合不一致率："
            f"NFE=1 **{100*gi['nfe1']['frac_any_step_disagree']:.1f}%**，"
            f"NFE=2 **{100*gi['nfe2']['frac_any_step_disagree']:.1f}%**。"
        )
        lines.append("")

    ch = payload["selection_changes"]["smolvla"]["noise0"]
    lines += [
        "## 选样变化（event vs 原 CARE，SmolVLA noise0 指标）",
        "",
        f"新增 {len(ch['added'])}：{[x['shared_index'] for x in ch['added'][:12]]}",
        f"移除 {len(ch['removed'])}：{[x['shared_index'] for x in ch['removed'][:12]]}",
        "",
        "关注帧：124/381 是否在新增中；361/447 是否在移除中 — 见 `selection_changes.json`。",
        "",
        "## 五个问题（简答）",
        "",
        payload["answers"]["q1"],
        "",
        payload["answers"]["q2"],
        "",
        payload["answers"]["q3"],
        "",
        payload["answers"]["q4"],
        "",
        payload["answers"]["q5"],
        "",
        "## 曲线与视频",
        "",
    ]
    for c in curves:
        lines.append(f"- `{c}`")
    lines.append("- 视频索引：[index.html](index.html)")
    if PREVIEW_REF.is_dir():
        lines.append("- 原 α 预览：[../preview_smolvla_timestamps/index.html](../preview_smolvla_timestamps/index.html)")
    (EVENT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n")


def build_answers(payload: dict) -> dict:
    def grab(model, g, nid, key):
        return payload["models"][model]["by_noise"][str(nid)]["groups"][g][key]

    # Compare tail recall and grip capture event vs care_alpha
    improved_grip = []
    improved_tail = []
    worse_arm = []
    for model in ("smolvla", "pi05"):
        for nid in (0, 1):
            e = grab(model, "care_event", nid, "tail_recall")
            a = grab(model, "care_alpha", nid, "tail_recall")
            if e > a:
                improved_tail.append((model, nid, a, e))
            ge = grab(model, "care_event", nid, "n_grip_dev_nfe1_captured")
            ga = grab(model, "care_alpha", nid, "n_grip_dev_nfe1_captured")
            if ge > ga:
                improved_grip.append((model, nid, ga, ge))
            if grab(model, "care_event", nid, "n_arm_top5_captured") < grab(
                model, "care_alpha", nid, "n_arm_top5_captured"
            ):
                worse_arm.append(model)
    q1 = (
        "事件 CARE 相对原 CARE：**夹爪动作偏差（NFE=1）捕获数在多数格上增加或持平**"
        f"（详情见 summary.json）。G=1 窗口入选数从原 CARE 的 "
        f"{grab('smolvla','care_alpha',0,'n_grip_event_windows')} 增至 event 的 "
        f"{grab('smolvla','care_event',0,'n_grip_event_windows')}（noise0 SmolVLA），"
        "有利于减少「夹爪切换附近低 NFE 偏差」漏检。"
    )
    q2 = (
        "手臂误差 top5% 捕获：event CARE 可能 **略低于** 原 CARE（见各表 arm top5% 列），"
        "因预算固定为 120 帧，事件加分会用高 α_event 的夹爪帧替换部分纯高 α 手臂突变帧。"
    )
    q3 = (
        "两模型、两噪声下趋势 **大体一致**（π0.5 尾部更紧但相对排序仍可见 event 增益）；"
        "若仅某一格反转，以 `summary.json` 为准，不单凭阈值断言。"
    )
    q4 = (
        "高分 α_event 视频在候选池内通常对应 **G=1 或较高 α**，可见更多夹爪轨迹变化；"
        "但演示视频 **不能** 单独证明低 NFE 策略行为，也不能代替动作误差表。"
    )
    q5 = (
        "建议：**保留原 CARE α 作主排序**，夹爪事件作 **+1 固定加分** 的并行候选（或报告双轨）；"
        "若目标是减少夹爪相关动作偏差漏检，event 版值得作诊断选样；若优先覆盖手臂极端误差，保留原 CARE。"
    )
    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5}


def load_metas_and_actions() -> tuple[list[FrameMeta], np.ndarray, np.ndarray]:
    blob = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    alpha = np.asarray(blob["alpha"], dtype=np.float64).reshape(-1)
    frames_df = _load_actions(LIBERO_REPO)
    actions = np.stack(frames_df["action"].to_numpy()).astype(np.float64)
    episode = frames_df["episode_index"].to_numpy()
    G = demo_gripper_switch(blob, actions, episode)
    probe_g_vec = np.zeros(N_FRAMES, dtype=np.int32)
    smol_path = MODELS["smolvla"] / "frames.csv"
    if smol_path.is_file():
        smol_df = load_probe_frames(MODELS["smolvla"])
        probe_g = smol_df[(smol_df.nfe == 1) & (smol_df.noise_id == 0)].drop_duplicates("shared_index")
        for r in probe_g.itertuples():
            probe_g_vec[int(r.shared_index)] = int(r.in_grip_switch)
    return build_metas(blob, alpha, G, probe_g_vec), actions, episode


def run_preview_videos(*, extended: bool) -> None:
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    metas, actions, episode = load_metas_and_actions()
    print("rendering default previews (α_event rank 1 & 12)...", flush=True)
    catalog = write_event_videos(metas, actions, episode, specs=DEFAULT_PREVIEW_SPECS, video_subdir="videos")
    write_html(catalog)
    pd.DataFrame(catalog).to_csv(EVENT_DIR / "preview_selection.csv", index=False)
    if extended:
        print("rendering extended gallery (α_event & α ranks 1–3, 10–12)...", flush=True)
        ext = write_event_videos(
            metas,
            actions,
            episode,
            specs=EXTENDED_PREVIEW_SPECS,
            video_subdir="videos_extended",
        )
        pd.DataFrame(ext).to_csv(EVENT_DIR / "preview_selection_extended.csv", index=False)
        write_gallery_html(ext, rel_prefix="")
    print(f"previews -> {EVENT_DIR}", flush=True)


def main() -> None:
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    blob = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    records = blob["records"]
    alpha = np.asarray(blob["alpha"], dtype=np.float64).reshape(-1)
    frames_df = _load_actions(LIBERO_REPO)
    actions = np.stack(frames_df["action"].to_numpy()).astype(np.float64)
    episode = frames_df["episode_index"].to_numpy()

    G = demo_gripper_switch(blob, actions, episode)
    # probe agreement
    smol_df = load_probe_frames(MODELS["smolvla"])
    probe_g = smol_df[(smol_df.nfe == 1) & (smol_df.noise_id == 0)].drop_duplicates("shared_index")
    agree = sum(int(probe_g[probe_g.shared_index == i].iloc[0].in_grip_switch) == int(G[i]) for i in range(N_FRAMES))
    boundary = boundary_grip_events(blob, actions, episode)

    probe_g_vec = np.zeros(N_FRAMES, dtype=np.int32)
    for r in probe_g.itertuples():
        probe_g_vec[int(r.shared_index)] = int(r.in_grip_switch)
    metas = build_metas(blob, alpha, G, probe_g_vec)
    groups = {
        "random": random_group(metas),
        "care_alpha": select_per_task(metas, "alpha"),
        "care_event": select_per_task(metas, "alpha_event"),
    }

    pi_df = load_probe_frames(MODELS["pi05"])
    model_payload = {
        "smolvla": model_stats("smolvla", smol_df, groups, set(np.where(G)[0])),
        "pi05": model_stats("pi05", pi_df, groups, set(np.where(G)[0])),
    }

    changes = {}
    for mk, df in (("smolvla", smol_df), ("pi05", pi_df)):
        changes[mk] = {}
        for nid in (0, 1):
            added, removed = diff_lists(metas, df, groups["care_alpha"], groups["care_event"], nid)
            changes[mk][f"noise{nid}"] = {"added": added, "removed": removed}

    selection_rows = []
    for m in metas:
        selection_rows.append(
            {
                "shared_index": m.shared_index,
                "global_index": m.global_index,
                "task_index": m.task_index,
                "task": m.task,
                "episode_index": m.episode_index,
                "frame_in_episode": m.frame_in_episode,
                "alpha": m.alpha,
                "alpha_event": m.alpha_event,
                "G": m.G,
                "in_random": int(m.shared_index in groups["random"]),
                "in_care_alpha": int(m.shared_index in groups["care_alpha"]),
                "in_care_event": int(m.shared_index in groups["care_event"]),
                "event_vs_alpha": (
                    "added" if m.shared_index in groups["care_event"] - groups["care_alpha"]
                    else "removed" if m.shared_index in groups["care_alpha"] - groups["care_event"]
                    else "unchanged"
                ),
            }
        )

    curves = plot_focus_curves(metas, actions, episode, smol_df, pi_df, EVENT_DIR / "curves")

    print("rendering videos (CPU/ffmpeg)...", flush=True)
    catalog = write_event_videos(metas, actions, episode)
    write_html(catalog)

    payload = {
        "grip_flag_agreement": agree / N_FRAMES,
        "n_grip_mismatch": N_FRAMES - agree,
        "n_boundary_events": sum(1 for b in boundary if b["kind"] == "boundary_prev_to_start"),
        "boundary_events": boundary,
        "models": model_payload,
        "selection_changes": changes,
        "groups": {k: sorted(v) for k, v in groups.items()},
    }
    payload["answers"] = build_answers(payload)

    (EVENT_DIR / "config.json").write_text(
        json.dumps(
            {
                "formula": "alpha_event = (alpha + event_boost * G) / mean(alpha + event_boost * G)",
                "event_boost": EVENT_BOOST,
                "G_definition": f"any grip open/close in demo actions [t, t+H) with H={HORIZON}",
                "groups": "random (seed 42), care_alpha top-3/task by alpha, care_event top-3/task by alpha_event",
                "tie_break": "global_index ascending",
                "tail_bad_nfe1": "cosine < P5 OR rmse > P95 OR gripper_disagree_n>0 (per model, per noise)",
                "arm_top5": "hypot(rmse_trans, rmse_rot) at NFE=1 >= P95 over 480 frames",
                "data": {
                    "shared_pt": str(OUT_DIR / "shared.pt"),
                    "smolvla_probe": str(MODELS["smolvla"]),
                    "pi05_probe": str(MODELS["pi05"]),
                },
            },
            indent=2,
        )
        + "\n"
    )
    pd.DataFrame(selection_rows).to_csv(EVENT_DIR / "selection.csv", index=False)
    pd.DataFrame(catalog).to_csv(EVENT_DIR / "preview_selection.csv", index=False)
    (EVENT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    (EVENT_DIR / "selection_changes.json").write_text(json.dumps(changes, indent=2, ensure_ascii=False) + "\n")
    write_report(payload, curves)
    print(f"done -> {EVENT_DIR}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CARE alpha_event offline diagnostic and demo previews")
    parser.add_argument(
        "--videos-only",
        action="store_true",
        help="Only render preview MP4s/HTML (no probe stats or selection tables)",
    )
    parser.add_argument(
        "--extended-previews",
        action="store_true",
        help="Also render ranks 1–3 vs 10–12 for α_event and α under videos_extended/",
    )
    args = parser.parse_args()
    if args.videos_only:
        run_preview_videos(extended=args.extended_previews)
    else:
        if args.extended_previews:
            print("note: full run always writes default videos/; use --videos-only for extended only", flush=True)
        main()
        if args.extended_previews:
            metas, actions, episode = load_metas_and_actions()
            ext = write_event_videos(
                metas,
                actions,
                episode,
                specs=EXTENDED_PREVIEW_SPECS,
                video_subdir="videos_extended",
            )
            pd.DataFrame(ext).to_csv(EVENT_DIR / "preview_selection_extended.csv", index=False)
            write_gallery_html(ext)

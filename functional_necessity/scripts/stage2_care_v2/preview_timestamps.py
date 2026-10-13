#!/usr/bin/env python3
"""Preview CARE v2 timestamps for four LIBERO suite task-0 pools.

Reads the frozen 480-frame CARE v2 set. Does not load a policy, does not
overwrite shared.pt / scores / historical evals.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dump_shared import _load_actions
from protocol import ACTION_DIM, HORIZON, LIBERO_REPO, OUT_DIR, hub_snapshot

FPS = 10.0
CONTEXT_SEC = 1.0
CONTEXT_FRAMES = int(round(CONTEXT_SEC * FPS))
SUITES = (
    (
        "spatial",
        "libero_spatial",
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
    ),
    (
        "object",
        "libero_object",
        "pick up the alphabet soup and place it in the basket",
    ),
    (
        "goal",
        "libero_goal",
        "open the middle drawer of the cabinet",
    ),
    (
        "long",
        "libero_10",
        "put both the alphabet soup and the tomato sauce in the basket",
    ),
)
ARM_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")
GRIP_NAME = "gripper"
CAMS = (
    ("observation.images.image", "agentview"),
    ("observation.images.image2", "wrist"),
)
PREVIEW_DIR = OUT_DIR / "preview_smolvla_timestamps"
FFMPEG = "/home/guorj/.miniforge3/envs/lerobot/bin/ffmpeg"


def _episodes_table(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    eps = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return eps.sort_values("episode_index").reset_index(drop=True)


def _window_steps(actions: np.ndarray, episode: np.ndarray, start: int) -> tuple[np.ndarray, int]:
    ep = int(episode[start])
    seq = []
    for t in range(start, min(start + HORIZON, len(actions))):
        if int(episode[t]) != ep:
            break
        seq.append(actions[t, :ACTION_DIM])
    if len(seq) < 2:
        raise RuntimeError(f"CARE window at {start} has <2 in-episode steps")
    return np.stack(seq), len(seq)


def _adjacent_whitened(seq: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(seq, axis=0) / std[:ACTION_DIM], axis=1)


def _pick_high_mid_low(pool: list[dict]) -> dict[str, dict]:
    ordered = sorted(pool, key=lambda r: -float(r["alpha"]))
    for i, rec in enumerate(ordered):
        rec["pool_rank"] = i + 1
        rec["pool_n"] = len(ordered)
    alphas = np.asarray([float(r["alpha"]) for r in ordered], dtype=np.float64)
    median_val = float(np.median(alphas))
    best_i = None
    best_key = None
    for i, rec in enumerate(ordered):
        dist = abs(float(rec["alpha"]) - median_val)
        rank_dist = abs((i + 1) - 6.5)
        key = (dist, rank_dist, i)
        if best_key is None or key < best_key:
            best_key = key
            best_i = i
    mid = ordered[int(best_i)]
    mid["median_target_alpha"] = median_val
    return {"high": ordered[0], "mid": mid, "low": ordered[-1]}


def _clip_range(frame0: int, length: int, window_len: int) -> dict:
    want_pre = CONTEXT_FRAMES
    want_post = CONTEXT_FRAMES
    win_end = frame0 + window_len - 1
    pre = min(want_pre, frame0)
    post = min(want_post, max(0, length - 1 - win_end))
    clip0 = frame0 - pre
    clip1 = win_end + post
    notes = []
    if pre < want_pre:
        notes.append(
            f"pre-context truncated: requested {want_pre} frames ({CONTEXT_SEC:.0f}s at {FPS:.0f} FPS), got {pre}"
        )
    if post < want_post:
        notes.append(
            f"post-context truncated: requested {want_post} frames ({CONTEXT_SEC:.0f}s at {FPS:.0f} FPS), got {post}"
        )
    if win_end > length - 1:
        notes.append(f"scoring window clipped by episode end (length={length})")
    return {
        "clip0": int(clip0),
        "clip1": int(clip1),
        "pre": int(pre),
        "post": int(post),
        "win_end": int(min(win_end, length - 1)),
        "notes": notes,
    }


def _decode_episode_frames(video_path: Path, from_ts: float, n_frames: int) -> list[np.ndarray]:
    import av

    out = [None] * n_frames
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        start = max(0.0, from_ts - 1.0 / FPS)
        container.seek(int(start * av.time_base), any_frame=False, backward=True)
        for frame in container.decode(stream):
            ts = float(frame.time) if frame.time is not None else float(frame.pts * stream.time_base)
            rel = ts - from_ts
            idx = int(round(rel * FPS))
            if idx < 0:
                continue
            if idx >= n_frames:
                if rel > (n_frames / FPS) + 0.15:
                    break
                continue
            if out[idx] is None:
                out[idx] = frame.to_ndarray(format="rgb24")
        if any(im is None for im in out):
            got = sum(im is not None for im in out)
            raise RuntimeError(f"decoded {got}/{n_frames} frames from {video_path} ts={from_ts:.3f}")
    return out


def _annotate(frame: np.ndarray, lines: list[str], *, highlight: str | None) -> np.ndarray:
    img = Image.fromarray(frame)
    header_h = 118
    canvas = Image.new("RGB", (img.width, img.height + header_h), (18, 18, 18))
    canvas.paste(img, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    y = 4
    for line in lines:
        draw.text((8, y), line, fill=(240, 240, 240), font=font)
        y += 14
    if highlight:
        color = {
            "timestamp": (220, 50, 47),
            "window": (255, 200, 40),
            "maxdiff": (31, 119, 180),
            "context": (160, 160, 160),
        }.get(highlight, (255, 255, 255))
        draw.rectangle([0, header_h, canvas.width - 1, canvas.height - 1], outline=color, width=4)
        draw.text((8, header_h - 16), highlight, fill=color, font=font)
    return np.asarray(canvas)


def _hstack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        b = np.asarray(Image.fromarray(b).resize((a.shape[1], a.shape[0])))
    return np.concatenate([a, b], axis=1)


def _role_of_local(local: int, frame0: int, win_end: int, maxdiff_frame: int) -> str:
    if local == maxdiff_frame:
        return "maxdiff"
    if local == frame0:
        return "timestamp"
    if frame0 <= local <= win_end:
        return "window"
    return "context"


def _plot_actions(
    path: Path,
    *,
    frames: np.ndarray,
    actions: np.ndarray,
    whitened: np.ndarray,
    raw_l2: np.ndarray,
    rec: dict,
    clip: dict,
    maxdiff_pair: int,
) -> None:
    frame0 = int(rec["frame_in_episode"])
    win_end = int(clip["win_end"])
    t_rel = frames
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True, constrained_layout=True)
    ax0, ax1, ax2 = axes
    for i, name in enumerate(ARM_NAMES):
        ax0.plot(t_rel, actions[:, i], lw=1.1, label=name)
    ax0.set_ylabel("arm action")
    ax0.legend(loc="upper right", ncol=3, fontsize=8)
    ax0.set_title(
        f"{rec['suite']} task0  ep={rec['episode_index']}  "
        f"pool rank {rec['pool_rank']}/{rec['pool_n']}  α={float(rec['alpha']):.3f}"
    )
    ax1.plot(t_rel, actions[:, 6], color="0.15", lw=1.4, label=GRIP_NAME)
    ax1.set_ylabel("gripper")
    ax1.legend(loc="upper right", fontsize=8)
    # adjacent diffs live on the later frame of each pair
    diff_x = t_rel[1:]
    ax2.plot(diff_x, raw_l2, color="0.55", lw=1.0, label="raw ||Δa||_2 (not CARE)")
    ax2.plot(diff_x, whitened, color="C0", lw=1.4, label="whitened ||Δa||_2 (CARE d_t)")
    ax2.set_ylabel("adjacent action diff")
    ax2.set_xlabel("frame in episode")
    ax2.legend(loc="upper right", fontsize=8)
    maxdiff_frame = int(t_rel[1 + maxdiff_pair]) if 1 + maxdiff_pair < len(t_rel) else int(t_rel[-1])
    for ax in axes:
        ax.axvspan(frame0, win_end, color="gold", alpha=0.18, zorder=0)
        ax.axvline(frame0, color="crimson", ls="--", lw=1.2)
        ax.axvline(maxdiff_frame, color="C0", ls=":", lw=1.4)
        ax.grid(True, alpha=0.25)
    ax2.annotate(
        "timestamp (window start)",
        xy=(frame0, ax2.get_ylim()[1]),
        xytext=(4, -12),
        textcoords="offset points",
        color="crimson",
        fontsize=8,
    )
    ax2.annotate(
        "max CARE Δ in window",
        xy=(maxdiff_frame, float(whitened[maxdiff_pair])),
        xytext=(6, 8),
        textcoords="offset points",
        color="C0",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="C0", lw=0.8),
    )
    fig.supxlabel(
        f"H={HORIZON} window=[{frame0},{win_end}]  FPS={FPS:.0f}  "
        f"clip=[{clip['clip0']},{clip['clip1']}]  "
        + (" | ".join(clip["notes"]) if clip["notes"] else "full ±1s context"),
        fontsize=8,
    )
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_video(frames_rgb: list[np.ndarray], dest: Path) -> None:
    tmp = dest.parent / "_frames"
    if tmp.exists():
        for p in tmp.glob("*.png"):
            p.unlink()
    else:
        tmp.mkdir()
    for i, fr in enumerate(frames_rgb):
        Image.fromarray(fr).save(tmp / f"{i:04d}.png")
    cmd = [
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
    subprocess.check_call(cmd)
    for p in tmp.glob("*.png"):
        p.unlink()
    tmp.rmdir()


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")


def main() -> None:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    blob = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    if str(blob.get("protocol")) != "care_v2":
        raise RuntimeError(f"expected care_v2 shared.pt, got {blob.get('protocol')}")
    if int(blob.get("horizon")) != HORIZON:
        raise RuntimeError(f"horizon {blob.get('horizon')} != {HORIZON}")
    recs = blob["records"]
    alpha = np.asarray(blob["alpha"], dtype=np.float64).reshape(-1)
    std = np.asarray(blob["whitening_std"], dtype=np.float64)
    if abs(float(np.mean(alpha)) - 1.0) > 1e-5:
        raise RuntimeError(f"alpha mean {float(np.mean(alpha))}")

    print("loading demonstration actions", flush=True)
    frames_df = _load_actions(LIBERO_REPO)
    actions_all = np.stack(frames_df["action"].to_numpy()).astype(np.float64)
    episode_all = frames_df["episode_index"].to_numpy()
    if "timestamp" in frames_df.columns:
        ts_all = frames_df["timestamp"].to_numpy()
    else:
        ts_all = None

    root = hub_snapshot(LIBERO_REPO)
    eps = _episodes_table(root)
    eps_by_id = {int(r["episode_index"]): r for _, r in eps.iterrows()}

    selected = []
    for slug, suite, lang in SUITES:
        pool = []
        for i, rec in enumerate(recs):
            if rec["task"] != lang:
                continue
            item = dict(rec)
            item["alpha"] = float(alpha[i])
            item["suite"] = suite
            item["suite_slug"] = slug
            item["shared_index"] = i
            pool.append(item)
        if len(pool) != 12:
            raise RuntimeError(f"{suite} task0 has {len(pool)} CARE v2 timestamps, expected 12")
        picks = _pick_high_mid_low(pool)
        for kind, rec in picks.items():
            rec["kind"] = kind
            selected.append(rec)
        print(
            f"{suite}: high α={picks['high']['alpha']:.3f} rank {picks['high']['pool_rank']}/12; "
            f"mid α={picks['mid']['alpha']:.3f} rank {picks['mid']['pool_rank']}/12 "
            f"(median={picks['mid']['median_target_alpha']:.3f}); "
            f"low α={picks['low']['alpha']:.3f} rank {picks['low']['pool_rank']}/12",
            flush=True,
        )

    catalog = []
    for rec in selected:
        ep = int(rec["episode_index"])
        eprow = eps_by_id[ep]
        gidx = int(rec["global_index"])
        frame0 = int(rec["frame_in_episode"])
        length = int(rec["length"])
        ep_start = int(eprow["dataset_from_index"])
        if gidx != ep_start + frame0:
            raise RuntimeError(f"index mismatch ep={ep} gidx={gidx} start={ep_start} frame={frame0}")
        seq, win_len = _window_steps(actions_all, episode_all, gidx)
        d_check = float(np.mean(_adjacent_whitened(seq, std)))
        if abs(d_check - float(rec["d_whitened"])) > 1e-4:
            raise RuntimeError(
                f"d_whitened mismatch ep={ep}: recomputed {d_check} vs stored {rec['d_whitened']}"
            )
        clip = _clip_range(frame0, length, win_len)
        clip0, clip1 = clip["clip0"], clip["clip1"]
        n_clip = clip1 - clip0 + 1
        act = actions_all[ep_start + clip0 : ep_start + clip1 + 1, :ACTION_DIM]
        if act.shape[0] != n_clip:
            raise RuntimeError("action slice length")
        whitened = _adjacent_whitened(act, std)
        raw_l2 = np.linalg.norm(np.diff(act, axis=0), axis=1)
        # max CARE delta inside scoring window, not the whole clip
        win_off0 = frame0 - clip0
        win_off1 = clip["win_end"] - clip0
        pair0 = win_off0
        pair1 = win_off1
        local_w = whitened[pair0:pair1]
        if local_w.size == 0:
            raise RuntimeError("empty scoring-window diffs")
        maxdiff_pair_local = int(np.argmax(local_w))
        maxdiff_pair = pair0 + maxdiff_pair_local
        maxdiff_frame = int(clip0 + 1 + maxdiff_pair)
        ts_sel = None if ts_all is None else float(ts_all[gidx])
        ts_max = None if ts_all is None else float(ts_all[ep_start + maxdiff_frame])

        tag = f"{rec['suite_slug']}_task0_{rec['kind']}"
        out_dir = PREVIEW_DIR / tag
        out_dir.mkdir(exist_ok=True)

        cam_clips = []
        for key, _role in CAMS:
            vchunk = int(eprow[f"videos/{key}/chunk_index"])
            vfile = int(eprow[f"videos/{key}/file_index"])
            from_ts = float(eprow[f"videos/{key}/from_timestamp"])
            vpath = root / "videos" / key / f"chunk-{vchunk:03d}" / f"file-{vfile:03d}.mp4"
            decoded = _decode_episode_frames(vpath, from_ts, length)
            cam_clips.append(decoded[clip0 : clip1 + 1])

        annotated = []
        stills = {}
        for i in range(n_clip):
            local = clip0 + i
            t_s = local / FPS
            role = _role_of_local(local, frame0, clip["win_end"], maxdiff_frame)
            lines = [
                f"{rec['suite']} task 0 | {rec['task'][:70]}",
                f"episode {ep}  frame {local}/{length - 1}  t={t_s:.2f}s  FPS={FPS:.0f}",
                f"timestamp frame={frame0} t={frame0 / FPS:.2f}s  H={HORIZON} window [{frame0},{clip['win_end']}]",
                f"CARE alpha={float(rec['alpha']):.3f}  d={float(rec['d_whitened']):.3f}  "
                f"pool rank {rec['pool_rank']}/12 ({rec['kind']})",
                f"max whitened |dA| in window: frame {maxdiff_frame} (pair {maxdiff_frame - 1}->{maxdiff_frame})  "
                f"value={float(local_w[maxdiff_pair_local]):.3f}",
            ]
            if clip["notes"]:
                lines.append(" | ".join(clip["notes"]))
            pair = _hstack(cam_clips[0][i], cam_clips[1][i])
            labeled = Image.fromarray(pair)
            draw = ImageDraw.Draw(labeled)
            draw.text((8, 6), "agentview", fill=(255, 255, 255))
            draw.text((pair.shape[1] // 2 + 8, 6), "wrist", fill=(255, 255, 255))
            fr = _annotate(np.asarray(labeled), lines, highlight=role)
            annotated.append(fr)
            if local == maxdiff_frame:
                stills["maxdiff"] = fr
            if local == frame0:
                stills["timestamp"] = fr
            if local == clip["win_end"]:
                stills["window_end"] = fr

        video_path = out_dir / "clip.mp4"
        _write_video(annotated, video_path)
        still_path = out_dir / "still_maxdiff.png"
        Image.fromarray(stills["maxdiff"] if "maxdiff" in stills else annotated[0]).save(still_path)
        still_ts = out_dir / "still_timestamp.png"
        Image.fromarray(stills["timestamp"] if "timestamp" in stills else annotated[0]).save(still_ts)
        plot_path = out_dir / "actions.png"
        _plot_actions(
            plot_path,
            frames=np.arange(clip0, clip1 + 1),
            actions=act,
            whitened=whitened,
            raw_l2=raw_l2,
            rec=rec,
            clip=clip,
            maxdiff_pair=maxdiff_pair,
        )
        meta = {
            "suite": rec["suite"],
            "suite_slug": rec["suite_slug"],
            "kind": rec["kind"],
            "task": rec["task"],
            "task_index": rec["task_index"],
            "episode_index": ep,
            "shared_index": rec["shared_index"],
            "global_index": gidx,
            "frame_in_episode": frame0,
            "timestamp_s": ts_sel,
            "length": length,
            "horizon": HORIZON,
            "fps": FPS,
            "alpha": float(rec["alpha"]),
            "d_whitened": float(rec["d_whitened"]),
            "pool_rank": rec["pool_rank"],
            "pool_n": rec["pool_n"],
            "median_target_alpha": rec.get("median_target_alpha"),
            "window": [frame0, clip["win_end"]],
            "window_len": win_len,
            "clip": [clip0, clip1],
            "context_notes": clip["notes"],
            "maxdiff_frame": maxdiff_frame,
            "maxdiff_pair": [maxdiff_frame - 1, maxdiff_frame],
            "maxdiff_whitened": float(local_w[maxdiff_pair_local]),
            "maxdiff_timestamp_s": ts_max,
            "whitening_std": std.tolist(),
            "video": str(video_path.relative_to(PREVIEW_DIR)),
            "plot": str(plot_path.relative_to(PREVIEW_DIR)),
            "still_maxdiff": str(still_path.relative_to(PREVIEW_DIR)),
            "still_timestamp": str(still_ts.relative_to(PREVIEW_DIR)),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        catalog.append(meta)
        print(f"wrote {tag} rank {rec['pool_rank']}/12 maxdiff_frame={maxdiff_frame}", flush=True)

    (PREVIEW_DIR / "selection.json").write_text(
        json.dumps(
            {
                "protocol": "care_v2",
                "stack_note": "timestamps and alpha are the shared CARE v2 480-frame set used by SmolVLA scoring",
                "horizon": HORIZON,
                "fps": FPS,
                "n_pool_per_task": 12,
                "selection": "within each suite's task-0 pool of 12: highest alpha, alpha closest to median, lowest alpha",
                "not_global_rank": True,
                "clips": catalog,
            },
            indent=2,
        )
        + "\n"
    )
    _write_index(catalog)
    print(f"wrote index {PREVIEW_DIR / 'INDEX.md'}", flush=True)


def _write_index(catalog: list[dict]) -> None:
    by = {}
    for item in catalog:
        by.setdefault(item["suite_slug"], []).append(item)
    lines = [
        "# SmolVLA CARE v2 timestamp preview",
        "",
        "只读 CARE v2 已冻结的 480 帧校准集（`results/stage2_care_v2/shared.pt`）。",
        "H=10，7 维示范动作按维白化后的相邻 L2 均值再除以全体均值，得到 α（均值精确为 1）。",
        "四个套件各取 **task 0**，在该任务已有的 **12 个候选 timestamp** 内按 α 取最高、最接近中位、最低，共 12 段。",
        "**这是候选池内排序，不是整条轨迹、也不是 480 帧全局排序。**",
        "",
        "α 大表示该 timestamp 之后 H 步里动作变化更急；这里只客观展示，不把高 α 直接叫成「关键动作」。",
        "数据集 FPS=**10**，窗口前后各约 1 秒（10 帧）；轨迹边界不够时在片段上标注截断。",
        "红色虚线 = 所选 timestamp（窗口起点）。蓝点线 = 窗口内 whitened |Δa| 最大的那一步的后一帧。不要把窗口起点当成突变发生点。",
        "",
        "未跑模型。未覆盖 `shared.pt`、分数文件和历史闭环。",
        "",
    ]
    html = [
        "<!doctype html><meta charset='utf-8'><title>SmolVLA CARE v2 timestamp preview</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;line-height:1.4}",
        "video,img{max-width:100%} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:4px 8px}",
        "h2{margin-top:2.2em}</style>",
        "<h1>SmolVLA CARE v2 timestamp preview</h1>",
        "<p>Candidate-pool ranks on CARE v2 α (H=10, mean-normalized). Not a global trajectory ranking. High α is not labeled as a key action.</p>",
    ]
    kind_title = {"high": "highest α in pool", "mid": "closest to pool median α", "low": "lowest α in pool"}
    for slug, suite, lang in SUITES:
        items = by[slug]
        pool_note = items[0]
        lines += [
            f"## {suite} · task 0",
            "",
            f"Instruction: `{_md_escape(lang)}`",
            "",
            f"Dataset `task_index`={pool_note['task_index']}. 12 CARE v2 timestamps in this task's pool.",
            "",
            "| pick | pool rank | α | d_whitened | episode | timestamp frame | window | max-Δ frame | clip | notes |",
            "|---|---:|---:|---:|---:|---:|---|---:|---|---|",
        ]
        html.append(f"<h2>{suite} · task 0</h2><p>{lang}</p>")
        html.append(
            "<table><tr><th>pick</th><th>rank</th><th>α</th><th>d</th><th>ep</th>"
            "<th>t frame</th><th>window</th><th>max-Δ</th><th>notes</th></tr>"
        )
        order = {"high": 0, "mid": 1, "low": 2}
        items = sorted(items, key=lambda x: order[x["kind"]])
        for it in items:
            notes = "; ".join(it["context_notes"]) if it["context_notes"] else "full ±1s"
            win = f"[{it['window'][0]}, {it['window'][1]}]"
            lines.append(
                f"| {it['kind']} | {it['pool_rank']}/12 | {it['alpha']:.3f} | {it['d_whitened']:.3f} | "
                f"{it['episode_index']} | {it['frame_in_episode']} | {win} | {it['maxdiff_frame']} | "
                f"`{it['video']}` | {notes} |"
            )
            html.append(
                f"<tr><td>{it['kind']}</td><td>{it['pool_rank']}/12</td><td>{it['alpha']:.3f}</td>"
                f"<td>{it['d_whitened']:.3f}</td><td>{it['episode_index']}</td>"
                f"<td>{it['frame_in_episode']}</td><td>{win}</td><td>{it['maxdiff_frame']}</td>"
                f"<td>{notes}</td></tr>"
            )
        html.append("</table>")
        lines.append("")
        for it in items:
            rel_v = it["video"]
            rel_p = it["plot"]
            rel_s = it["still_maxdiff"]
            lines += [
                f"### {it['kind']} — {kind_title[it['kind']]} (rank {it['pool_rank']}/12)",
                "",
                f"- episode `{it['episode_index']}`, timestamp frame `{it['frame_in_episode']}` "
                f"(t={it['frame_in_episode'] / FPS:.2f}s), length `{it['length']}`",
                f"- CARE α=`{it['alpha']:.3f}`, d_whitened=`{it['d_whitened']:.3f}`",
                f"- scoring window H=10: frames `{it['window'][0]}`–`{it['window'][1]}`",
                f"- max whitened adjacent diff **inside the window**: frame `{it['maxdiff_frame']}` "
                f"(pair `{it['maxdiff_pair'][0]}`→`{it['maxdiff_pair'][1]}`), value `{it['maxdiff_whitened']:.3f}`",
                "",
                f"[video]({rel_v}) · still at max-Δ: ![]({rel_s})",
                "",
                f"![]({rel_p})",
                "",
            ]
            html += [
                f"<h3>{it['kind']} — {kind_title[it['kind']]} (rank {it['pool_rank']}/12)</h3>",
                f"<p>ep {it['episode_index']}, timestamp frame {it['frame_in_episode']}, "
                f"α={it['alpha']:.3f}, window {it['window']}, max-Δ frame {it['maxdiff_frame']}.</p>",
                f"<video controls src='{rel_v}'></video>",
                f"<p><img src='{rel_p}' alt='action curves'></p>",
            ]
        lines.append("")
    (PREVIEW_DIR / "INDEX.md").write_text("\n".join(lines) + "\n")
    (PREVIEW_DIR / "index.html").write_text("\n".join(html) + "\n")


if __name__ == "__main__":
    main()

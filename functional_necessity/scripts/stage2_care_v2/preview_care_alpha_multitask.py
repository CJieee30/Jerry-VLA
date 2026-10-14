#!/usr/bin/env python3
"""Preview event CARE (α_event) high/low within selected task_index pools (12 frames each).

CPU + ffmpeg only. Does not modify shared.pt or probe outputs.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dump_shared import _load_actions
from event_care_nfe_diagnostic import (
    EVENT_BOOST,
    EVENT_DIR,
    FrameMeta,
    _render_one_preview,
    demo_gripper_switch,
)
from preview_timestamps import _episodes_table, hub_snapshot
from protocol import FRAMES_PER_TASK, LIBERO_REPO, OUT_DIR

VIDEO_SUBDIR = "videos_multitask_event"
# Two tasks per LIBERO suite (task_index: long 0–9, goal 10–19, object 20–29, spatial 30–39).
DEFAULT_TASK_INDICES = (3, 5, 10, 16, 24, 28, 34, 37)


def suite_slug(task_index: int) -> str:
    if task_index < 10:
        return "long"
    if task_index < 20:
        return "goal"
    if task_index < 30:
        return "object"
    return "spatial"


def metas_for_indices(blob: dict, alpha: np.ndarray, G: np.ndarray, task_indices: tuple[int, ...]) -> list[FrameMeta]:
    raw = alpha + EVENT_BOOST * G.astype(np.float64)
    alpha_event = raw / float(np.mean(raw))
    want = set(task_indices)
    out: list[FrameMeta] = []
    for i, rec in enumerate(blob["records"]):
        ti = int(rec["task_index"])
        if ti not in want:
            continue
        out.append(
            FrameMeta(
                shared_index=i,
                global_index=int(rec["global_index"]),
                task_index=ti,
                task=rec["task"],
                episode_index=int(rec["episode_index"]),
                frame_in_episode=int(rec["frame_in_episode"]),
                length=int(rec["length"]),
                alpha=float(alpha[i]),
                d_whitened=float(blob["d_whitened"][i]),
                G=int(G[i]),
                alpha_event=float(alpha_event[i]),
                in_grip_switch_probe=0,
            )
        )
    return out


def _basename_for_rank(slug: str, ti: int, rank: int, pool_n: int) -> tuple[str, str]:
    if rank == 1:
        return f"{slug}_t{ti:02d}_event_high", "high"
    if rank == pool_n:
        return f"{slug}_t{ti:02d}_event_low", "low"
    return f"{slug}_t{ti:02d}_event_r{rank:02d}", f"rank{rank}"


def write_multitask_event_videos(
    blob: dict,
    alpha: np.ndarray,
    actions: np.ndarray,
    episode: np.ndarray,
    task_indices: tuple[int, ...],
    *,
    pool_ranks: tuple[int, ...] = (1, FRAMES_PER_TASK),
    video_subdir: str = VIDEO_SUBDIR,
) -> list[dict]:
    G = demo_gripper_switch(blob, actions, episode)
    metas = metas_for_indices(blob, alpha, G, task_indices)
    by_task: dict[int, list[FrameMeta]] = defaultdict(list)
    for m in metas:
        by_task[m.task_index].append(m)

    root = hub_snapshot(LIBERO_REPO)
    eps = _episodes_table(root)
    eps_by_id = {int(r["episode_index"]): r for _, r in eps.iterrows()}
    video_dir = EVENT_DIR / video_subdir
    video_dir.mkdir(parents=True, exist_ok=True)
    catalog: list[dict] = []

    for ti in sorted(task_indices):
        pool = by_task.get(ti, [])
        if len(pool) != FRAMES_PER_TASK:
            raise RuntimeError(f"task_index {ti}: pool size {len(pool)}, expected {FRAMES_PER_TASK}")
        ordered = sorted(pool, key=lambda m: (-m.alpha_event, m.global_index))
        slug = suite_slug(ti)
        for rank in pool_ranks:
            if rank < 1 or rank > FRAMES_PER_TASK:
                raise ValueError(f"pool rank {rank} out of 1..{FRAMES_PER_TASK}")
            rec = ordered[rank - 1]
            basename, tier = _basename_for_rank(slug, ti, rank, FRAMES_PER_TASK)
            catalog.append(
                _render_one_preview(
                    rec,
                    suite=f"libero ({slug})",
                    slug=slug,
                    score="alpha_event",
                    tag=f"t{ti:02d}",
                    rank=rank,
                    pool_n=FRAMES_PER_TASK,
                    video_dir=video_dir,
                    basename=basename,
                    eps_by_id=eps_by_id,
                    root=root,
                    actions_all=actions,
                    episode_all=episode,
                )
            )
            catalog[-1]["task_index"] = ti
            catalog[-1]["task_short"] = rec.task[:72]
    return catalog


def write_tasks_html(
    catalog: list[dict],
    *,
    html_name: str = "multitask_event_index.html",
    title: str = "Event CARE α_event — high vs low per task (12-frame pool)",
    blurb: str | None = None,
) -> None:
    if blurb is None:
        blurb = (
            f"Sorted by <b>α_event = (α + {EVENT_BOOST}·G) / mean(α+G)</b> within each "
            "<code>task_index</code> pool (global mean over 480 frames)."
        )
    lines = [
        "<!doctype html><meta charset='utf-8'><title>Event CARE multitask previews</title>",
        "<style>body{font-family:sans-serif;max-width:1100px;margin:20px auto;padding:0 12px}",
        "video{max-width:100%} .pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}",
        ".grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}",
        ".row{margin:1.5em 0;padding-bottom:1em;border-bottom:1px solid #eee}",
        "h2{margin-top:2em}</style>",
        f"<h1>{title}</h1>",
        f"<p>{blurb}</p>",
        "<p><a href='index.html'>Suite task-0 α_event</a> · "
        "<a href='multitask_event_index.html'>multitask rank 1 &amp; 12</a></p>",
    ]
    by_task: dict[int, list[dict]] = defaultdict(list)
    for c in catalog:
        by_task[int(c["task_index"])].append(c)
    for ti in sorted(by_task):
        items = sorted(by_task[ti], key=lambda x: -x["pool_rank"])
        slug = suite_slug(ti)
        task_name = items[0].get("task_short", "")
        ncol = 2 if len(items) <= 2 else 2
        grid = "pair" if ncol == 2 else "grid"
        lines.append(f"<div class='row'><h2>{slug} · task_index={ti}</h2><p>{task_name}</p><div class='{grid}'>")
        for it in items:
            lines.append(
                f"<div><h3>α_event rank {it['pool_rank']}/12 ({it['tier']})</h3>"
                f"<p>α={it['alpha']:.3f} α_event={it['alpha_event']:.3f} G={it['G']} · "
                f"ep={it['episode_index']} frame={it['frame_in_episode']}</p>"
                f"<video controls src='{it['video']}'></video>"
                f"<p><img src='{it['plot']}' style='max-width:100%'></p></div>"
            )
        lines.append("</div></div>")
    (EVENT_DIR / html_name).write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-indices",
        type=int,
        nargs="+",
        default=list(DEFAULT_TASK_INDICES),
        help=f"task_index values to preview (default: {DEFAULT_TASK_INDICES})",
    )
    parser.add_argument(
        "--pool-ranks",
        type=int,
        nargs="+",
        default=[1, FRAMES_PER_TASK],
        help="α_event pool ranks to render (1=highest, 12=lowest). Example: 2 3 for sub-top high.",
    )
    parser.add_argument(
        "--video-subdir",
        type=str,
        default=VIDEO_SUBDIR,
        help=f"output folder under nfe_probe_event_care/ (default: {VIDEO_SUBDIR})",
    )
    parser.add_argument(
        "--html-index",
        type=str,
        default="multitask_event_index.html",
        help="HTML catalog filename under nfe_probe_event_care/",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="preview_multitask_event.csv",
        help="CSV catalog filename",
    )
    args = parser.parse_args()
    task_indices = tuple(sorted(set(args.task_indices)))
    pool_ranks = tuple(sorted(set(args.pool_ranks)))
    EVENT_DIR.mkdir(parents=True, exist_ok=True)

    blob = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)
    alpha = np.asarray(blob["alpha"], dtype=np.float64).reshape(-1)
    frames_df = _load_actions(LIBERO_REPO)
    actions = np.stack(frames_df["action"].to_numpy()).astype(np.float64)
    episode = frames_df["episode_index"].to_numpy()

    print(
        f"rendering α_event ranks {pool_ranks} for task_index {task_indices} -> {args.video_subdir}/",
        flush=True,
    )
    catalog = write_multitask_event_videos(
        blob,
        alpha,
        actions,
        episode,
        task_indices,
        pool_ranks=pool_ranks,
        video_subdir=args.video_subdir,
    )
    pd.DataFrame(catalog).to_csv(EVENT_DIR / args.csv_name, index=False)
    rank_desc = ", ".join(str(r) for r in pool_ranks)
    write_tasks_html(
        catalog,
        html_name=args.html_index,
        title=f"Event CARE α_event — pool ranks {rank_desc}",
        blurb=(
            f"Per-task 12-frame pool sorted by α_event (event boost G={EVENT_BOOST}). "
            f"This page: ranks <b>{rank_desc}</b> (not necessarily rank 1)."
        ),
    )
    print(f"done: {EVENT_DIR / args.html_index}", flush=True)


if __name__ == "__main__":
    main()

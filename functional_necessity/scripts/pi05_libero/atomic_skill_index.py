#!/usr/bin/env python3
"""Map lerobot/libero frames to AtomicVLA verbs and apply the P0 interior filter.

Looks up ``primary_action_verb`` with AtomicVLA's first-matching segment rule
(``start_frame <= frame_index <= end_frame``). Drops frames within
``SKILL_BOUNDARY_MARGIN`` of each segment boundary so a 10-step action chunk
does not cross skills. Writes per-verb dataset indices plus a coverage report.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from common import DEFAULT_DATASET, init_logging, resolve_dataset_root
from drop6_extra3_protocol import (
    ATOMIC_SKILLS,
    DEFAULT_ATOMIC_JSON,
    DEFAULT_SKILL_INDEX_DIR,
    SKILL_BATCH_SIZE,
    SKILL_BOUNDARY_MARGIN,
    SKILL_NUM_BATCHES,
    SUITE_LABEL,
    SUITE_SKILL_CELLS,
    dump_json,
    load_task_catalog,
    suite_skill_stem,
)


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def load_atomic_json(path: Path) -> dict[int, dict]:
    raw = json.loads(Path(path).read_text())
    out = {}
    for key, rec in raw.items():
        if str(key).isdigit():
            out[int(key)] = rec
    if not out:
        raise RuntimeError(f"{path} has no integer episode keys")
    return out


def first_match_verb(segments: list[dict], frame_index: int) -> str | None:
    for seg in segments:
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
        if end == -1:
            end = 10**9
        if start <= int(frame_index) <= end:
            verb = str(seg.get("primary_action_verb") or "").strip().lower()
            return verb or None
    return None


def load_task_language(dataset_root: Path) -> dict[int, str]:
    tasks = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    lang: dict[int, str] = {}
    for idx, row in tasks.iterrows():
        lang[int(row["task_index"])] = str(idx)
    return lang


def load_frame_table(dataset_root: Path):
    table = ds.dataset(str(dataset_root / "data"), format="parquet").to_table(
        columns=["index", "episode_index", "frame_index", "task_index"]
    )
    return (
        [int(x) for x in table.column("index").to_pylist()],
        [int(x) for x in table.column("episode_index").to_pylist()],
        [int(x) for x in table.column("frame_index").to_pylist()],
        [int(x) for x in table.column("task_index").to_pylist()],
    )


def build_index(
    *,
    atomic_json: Path,
    dataset_root: Path,
    margin: int,
) -> dict:
    ann = load_atomic_json(atomic_json)
    lang = load_task_language(dataset_root)
    catalog = load_task_catalog()
    prompt_to_short = {_norm(row["task"]): row["short"] for row in catalog}
    short_to_suite = {row["short"]: SUITE_LABEL[row["suite"]] for row in catalog}
    indices, episodes, frames, task_indices = load_frame_table(dataset_root)
    n_rows = len(indices)
    if n_rows != len(set(indices)):
        raise RuntimeError("parquet index column is not unique")
    if indices != list(range(n_rows)):
        logging.warning("parquet index is not 0..N-1; dump will use the index column as dataset idx")

    ep_rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    ep_to_ti: dict[int, int] = {}
    for index, ep, fi, ti in zip(indices, episodes, frames, task_indices):
        ep_rows[int(ep)].append((int(fi), int(index)))
        ep_to_ti[int(ep)] = int(ti)

    ds_eps = set(ep_rows)
    json_eps = set(ann)
    if ds_eps != json_eps:
        raise RuntimeError(
            f"episode set mismatch: json-only={sorted(json_eps - ds_eps)[:8]} "
            f"dataset-only={sorted(ds_eps - json_eps)[:8]}"
        )

    n_name_eq = 0
    mismatches = []
    for ep in sorted(json_eps):
        json_name = _norm(ann[ep]["segments"][0]["action_name"])
        ds_name = _norm(lang.get(ep_to_ti[ep], ""))
        if json_name == ds_name:
            n_name_eq += 1
        elif len(mismatches) < 8:
            mismatches.append({"episode": ep, "json": json_name, "dataset": ds_name})
    if n_name_eq != len(json_eps):
        raise RuntimeError(
            f"task-name alignment failed: {n_name_eq}/{len(json_eps)} matched; sample={mismatches}"
        )

    raw_verb = Counter()
    keep_verb = Counter()
    unknown_raw = 0
    unlabeled = 0
    short_segments = Counter()
    other_verbs = Counter()
    keep_ids: dict[str, list[int]] = {v: [] for v in ATOMIC_SKILLS}
    keep_ids_suite: dict[str, dict[str, list[int]]] = {
        lab: {v: [] for v in ATOMIC_SKILLS} for lab in SUITE_LABEL.values()
    }
    n_goal5_pick_excluded = 0
    per_task_raw: dict[str, Counter] = defaultdict(Counter)
    per_task_keep: dict[str, Counter] = defaultdict(Counter)
    boundary_touch = 0

    for ep, rec in ann.items():
        segments = list(rec.get("segments") or [])
        rows = sorted(ep_rows[ep])
        max_frame = rows[-1][0]
        frame_to_idx = {fi: idx for fi, idx in rows}
        task_name = _norm(segments[0]["action_name"]) if segments else ""
        short = prompt_to_short.get(task_name, task_name[:40])

        for a, b in zip(segments, segments[1:]):
            if int(b["start_frame"]) == int(a["end_frame"]):
                boundary_touch += 1

        labeled = {}
        for fi, idx in rows:
            verb = first_match_verb(segments, fi)
            labeled[fi] = verb
            if verb is None:
                unlabeled += 1
                continue
            if verb not in ATOMIC_SKILLS:
                other_verbs[verb] += 1
                unknown_raw += 1
                continue
            raw_verb[verb] += 1
            per_task_raw[short][verb] += 1

        for seg in segments:
            verb = str(seg.get("primary_action_verb") or "").strip().lower()
            if verb not in ATOMIC_SKILLS:
                continue
            start = max(int(seg["start_frame"]), 0)
            end = int(seg["end_frame"])
            if end == -1:
                end = max_frame
            end = min(end, max_frame)
            lo = start + margin
            hi = end - margin
            if lo > hi:
                short_segments[verb] += 1
                continue
            for fi in range(lo, hi + 1):
                if labeled.get(fi) != verb:
                    continue
                idx = frame_to_idx.get(fi)
                if idx is None:
                    continue
                keep_verb[verb] += 1
                per_task_keep[short][verb] += 1
                keep_ids[verb].append(idx)
                suite_label = short_to_suite.get(short)
                if suite_label is None:
                    continue
                if short == "Goal5" and verb == "pick":
                    n_goal5_pick_excluded += 1
                    continue
                keep_ids_suite[suite_label][verb].append(idx)

    for verb in ATOMIC_SKILLS:
        keep_ids[verb] = sorted(set(keep_ids[verb]))
        keep_verb[verb] = len(keep_ids[verb])
    suite_keep: dict[str, dict[str, int]] = {}
    for lab, verbs in keep_ids_suite.items():
        suite_keep[lab] = {}
        for verb in ATOMIC_SKILLS:
            verbs[verb] = sorted(set(verbs[verb]))
            suite_keep[lab][verb] = len(verbs[verb])

    return {
        "atomic_json": str(atomic_json),
        "dataset_root": str(dataset_root),
        "margin": margin,
        "n_episodes": len(ann),
        "n_frames": n_rows,
        "n_task_name_match": n_name_eq,
        "boundary_touch_segments": boundary_touch,
        "unlabeled_frames": unlabeled,
        "other_verbs": dict(other_verbs),
        "unknown_raw_frames": unknown_raw,
        "short_segments_dropped": dict(short_segments),
        "frames_raw": {v: int(raw_verb[v]) for v in ATOMIC_SKILLS},
        "frames_interior": {v: int(keep_verb[v]) for v in ATOMIC_SKILLS},
        "keep_ids": keep_ids,
        "keep_ids_suite": keep_ids_suite,
        "suite_frames_interior": suite_keep,
        "n_goal5_pick_excluded_from_goal_pick": n_goal5_pick_excluded,
        "per_task_raw": {k: dict(v) for k, v in sorted(per_task_raw.items())},
        "per_task_keep": {k: dict(v) for k, v in sorted(per_task_keep.items())},
        "warnings": _warnings(per_task_raw, keep_verb, suite_keep),
    }


def _warnings(
    per_task_raw: dict[str, Counter],
    keep_verb: Counter,
    suite_keep: dict[str, dict[str, int]] | None = None,
) -> list[str]:
    warnings = []
    push = "Goal5"
    if push in per_task_raw and set(per_task_raw[push]) == {"pick"}:
        warnings.append(
            "Goal5 'push the plate to the front of the stove' is labeled pick in AtomicVLA; "
            "excluded from Goal-pick suite×skill pool."
        )
    need = SKILL_NUM_BATCHES * SKILL_BATCH_SIZE
    for verb in ATOMIC_SKILLS:
        n = int(keep_verb[verb])
        if n < need:
            warnings.append(
                f"{verb}: only {n} interior frames after margin filter; "
                f"dump will oversample to {need}"
            )
    if suite_keep:
        for suite_label, verb in SUITE_SKILL_CELLS:
            n = int(suite_keep.get(suite_label, {}).get(verb, 0))
            if n < need:
                warnings.append(
                    f"{suite_label}/{verb}: only {n} interior frames; dump will oversample to {need}"
                )
    return warnings


def write_report(payload: dict, path: Path) -> None:
    raw = payload["frames_raw"]
    keep = payload["frames_interior"]
    lines = [
        "# AtomicVLA skill-frame index (P0)",
        "",
        f"- JSON: `{payload['atomic_json']}`",
        f"- Dataset: `{payload['dataset_root']}`",
        f"- Episodes: **{payload['n_episodes']}** (task-name match {payload['n_task_name_match']}/{payload['n_episodes']})",
        f"- Frames: **{payload['n_frames']}**",
        f"- Boundary margin: **{payload['margin']}** (drop this many frames at each segment end)",
        f"- Inclusive-touch segment pairs (`end == next.start`): {payload['boundary_touch_segments']}",
        f"- Unlabeled frames: {payload['unlabeled_frames']}",
        f"- Other verbs: `{payload['other_verbs']}`",
        "",
        "| Verb | Frames before filter | Frames after interior filter | Kept % |",
        "| --- | ---: | ---: | ---: |",
    ]
    for verb in ATOMIC_SKILLS:
        before = raw[verb]
        after = keep[verb]
        pct = (100.0 * after / before) if before else 0.0
        lines.append(f"| `{verb}` | {before} | {after} | {pct:.1f} |")
    lines += [
        "",
        f"Short segments dropped (length < {2 * payload['margin'] + 1}): `{payload['short_segments_dropped']}`",
        "",
        "## Warnings",
        "",
    ]
    if payload["warnings"]:
        for item in payload["warnings"]:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Eval-task interior frames (selected)",
        "",
        "| Short | Raw verbs | Interior verbs |",
        "| --- | --- | --- |",
    ]
    highlight = {"Goal0", "Goal5", "Goal7", "Goal3", "Long2", "Long3", "Long9", "Obj0", "Obj1"}
    for short in sorted(payload["per_task_keep"]):
        if short not in highlight:
            continue
        lines.append(
            f"| `{short}` | `{payload['per_task_raw'].get(short, {})}` | "
            f"`{payload['per_task_keep'].get(short, {})}` |"
        )
    suite_keep = payload.get("suite_frames_interior") or {}
    if suite_keep:
        lines += [
            "",
            "## Suite × skill interior frames (Goal-pick excludes Goal5)",
            "",
            "| Suite | pick | place | open | close | turn |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for lab in ("Spatial", "Object", "Goal", "Long"):
            row = suite_keep.get(lab, {})
            lines.append(
                f"| {lab} | {row.get('pick', 0)} | {row.get('place', 0)} | "
                f"{row.get('open', 0)} | {row.get('close', 0)} | {row.get('turn', 0)} |"
            )
        n_ex = payload.get("n_goal5_pick_excluded_from_goal_pick", 0)
        lines += ["", f"- Goal5 pick frames excluded from Goal-pick: **{n_ex}**"]
    lines += ["", f"JSON: `{path.with_suffix('.json')}`", ""]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atomic-json", type=Path, default=DEFAULT_ATOMIC_JSON)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SKILL_INDEX_DIR)
    parser.add_argument("--margin", type=int, default=SKILL_BOUNDARY_MARGIN)
    args = parser.parse_args()

    init_logging()
    dataset_root = resolve_dataset_root(args.dataset_repo_id, args.dataset_root)
    if dataset_root is None:
        raise FileNotFoundError(f"cannot resolve local dataset for {args.dataset_repo_id}")
    logging.info("Building skill index from %s + %s (margin=%d)", args.atomic_json, dataset_root, args.margin)
    built = build_index(atomic_json=args.atomic_json, dataset_root=dataset_root, margin=args.margin)
    keep_ids = built.pop("keep_ids")
    keep_ids_suite = built.pop("keep_ids_suite")
    out_dir = Path(args.out_dir)
    idx_dir = out_dir / "indices"
    idx_dir.mkdir(parents=True, exist_ok=True)
    index_paths = {}
    for verb in ATOMIC_SKILLS:
        path = idx_dir / f"{verb}.npy"
        np.save(path, np.asarray(keep_ids[verb], dtype=np.int64))
        index_paths[verb] = str(path)
        logging.info("%s: raw=%d interior=%d -> %s", verb, built["frames_raw"][verb], len(keep_ids[verb]), path)
    built["index_paths"] = index_paths
    suite_dir = idx_dir / "suite_skill"
    suite_dir.mkdir(parents=True, exist_ok=True)
    suite_index_paths: dict[str, str] = {}
    for suite_label, verb in SUITE_SKILL_CELLS:
        stem = suite_skill_stem(suite_label, verb)
        path = suite_dir / f"{stem}.npy"
        ids = keep_ids_suite[suite_label][verb]
        np.save(path, np.asarray(ids, dtype=np.int64))
        suite_index_paths[f"{suite_label}/{verb}"] = str(path)
        logging.info("%s/%s: interior=%d -> %s", suite_label, verb, len(ids), path)
    built["suite_skill_index_paths"] = suite_index_paths
    dump_json(built, out_dir / "coverage.json")
    write_report(built, out_dir / "coverage.md")
    print((out_dir / "coverage.md").read_text())
    if built["n_task_name_match"] != built["n_episodes"]:
        raise SystemExit("alignment failed")


if __name__ == "__main__":
    main()

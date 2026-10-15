#!/usr/bin/env python3
"""Staged CARE_event NFE recommendation protocol (v1).

v1 vs v0 draft: ledger stages are not silently re-appended; calibration frame draw uses
full H=10 window; optional closed-loop JSON import; `verify-ledger` re-checks file hashes.

Stages (each writes under <run_dir>/ and appends a hash entry to ledger.jsonl):

  freeze             config + code sha256; later stages refuse if either changed
  split              (prospective) calibration pool / validation set identities only
  select             Random / α / G-only / α_event from demo actions + frozen constants
  probe-plan         probe request for the union of selected observations
  probe-import       seal probe outputs of the union (prospective) ...
  seal-dev           ... or of existing dev probe arrays (dev rehearsal)
  recommend          reads only the sealed union, restricted per strategy
  validation-plan    validation probe request; requires a recommendation entry
  validation-import  seal validation probe outputs (prospective)
  validate           quality conditions on the independent set for each recommendation
  run-dev            freeze → select → seal-dev → recommend → validate on dev data

No stage runs a model. Probe runs are issued by the commands in probe-plan /
validation-plan and imported afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi05_libero"))

from dump_shared import _episodes, _load_actions, _window_d, _whitening_std  # noqa: E402
from event_care_nfe_diagnostic import (  # noqa: E402
    FrameMeta,
    demo_gripper_switch,
    g_only_group,
    random_group,
    select_per_task,
)
from protocol import ACTION_DIM, FN_ROOT, HORIZON, LIBERO_REPO, MOLMO_REPO, N_TASKS, OUT_DIR, hub_snapshot  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_FILES = (
    "nfe_protocol_v1.py",
    "event_care_nfe_diagnostic.py",
    "dump_shared.py",
    "protocol.py",
    "probe_pi05_nfe.py",
    "probe_smolvla_nfe.py",
    "probe_molmoact2_nfe.py",
)
NON_REPEATABLE_STAGES = frozenset(
    {
        "freeze",
        "split",
        "select",
        "probe-plan",
        "seal-dev",
        "recommend",
        "validation-plan",
        "validate",
    }
)
STRATEGIES = ("random", "alpha", "g_only", "alpha_event")
PROTOCOL_ROOT = OUT_DIR / "nfe_protocol"


# ----------------------------------------------------------------------------- io


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else FN_ROOT / p


def write_json(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    return sha256_file(path)


class Run:
    def __init__(self, config_path: Path, *, draft: bool = False):
        self.config_path = config_path.resolve()
        self.cfg = json.loads(self.config_path.read_text())
        self.dir = PROTOCOL_ROOT / self.cfg["run_name"]
        self.ledger = self.dir / "ledger.jsonl"
        self.mode = self.cfg["mode"]
        self.draft = draft
        if self.mode not in ("dev", "prospective"):
            raise SystemExit(f"mode must be dev|prospective, got {self.mode}")
        if draft and "DRAFT" not in self.cfg["run_name"]:
            raise SystemExit("--draft requires a run_name containing DRAFT")

    def entries(self) -> list[dict]:
        if not self.ledger.is_file():
            return []
        return [json.loads(x) for x in self.ledger.read_text().splitlines() if x.strip()]

    def last(self, stage: str) -> dict | None:
        hits = [e for e in self.entries() if e["stage"] == stage]
        return hits[-1] if hits else None

    def append(self, stage: str, files: dict[str, str], *, force: bool = False, **extra) -> None:
        if not force and stage in NON_REPEATABLE_STAGES and self.last(stage) is not None:
            raise SystemExit(
                f"stage '{stage}' already recorded at {self.last(stage)['utc']}; "
                f"use a new run_name or pass --force-restage"
            )
        self.dir.mkdir(parents=True, exist_ok=True)
        entry = {"stage": stage, "utc": utc_now(), "files": files, **extra}
        if self.draft:
            entry["draft"] = True
            entry["config_sha256"] = sha256_file(self.config_path)
        with open(self.ledger, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def require(self, stage: str) -> dict:
        e = self.last(stage)
        if e is None:
            raise SystemExit(f"stage '{stage}' has no ledger entry in {self.ledger}")
        for rel, digest in e["files"].items():
            path = self.dir / rel
            if not path.is_file() or sha256_file(path) != digest:
                raise SystemExit(f"{rel} changed since stage '{stage}' (ledger sha mismatch)")
        return e

    def check_frozen(self) -> None:
        if self.draft:
            return
        e = self.require("freeze")
        if sha256_file(self.config_path) != e["config_sha256"]:
            raise SystemExit("config changed after freeze")
        for name, digest in e["code_sha256"].items():
            if sha256_file(SCRIPT_DIR / name) != digest:
                raise SystemExit(f"code {name} changed after freeze")


# ----------------------------------------------------------------------- freeze


def stage_freeze(run: Run) -> None:
    if run.draft or (run.mode == "prospective" and "DRAFT" in run.cfg["run_name"]):
        raise SystemExit("a DRAFT run cannot be frozen; copy the reviewed config to a new run_name")
    if run.last("freeze") is not None:
        prev = run.last("freeze")
        if prev["config_sha256"] != sha256_file(run.config_path):
            raise SystemExit("run already frozen with a different config; use a new run_name")
        print(f"[skip] already frozen at {prev['utc']}")
        return
    run.dir.mkdir(parents=True, exist_ok=True)
    copy = run.dir / "config.frozen.json"
    copy.write_text(run.config_path.read_text())
    code = {name: sha256_file(SCRIPT_DIR / name) for name in CODE_FILES}
    run.append(
        "freeze",
        {"config.frozen.json": sha256_file(copy)},
        config_path=str(run.config_path),
        config_sha256=sha256_file(run.config_path),
        code_sha256=code,
        mode=run.mode,
    )
    print(f"frozen {run.dir}")


# ------------------------------------------------------------------------ demo


class Demo:
    """lerobot/libero demonstration actions (no model outputs)."""

    def __init__(self) -> None:
        self.frames = _load_actions(LIBERO_REPO)
        self.actions = np.stack(self.frames["action"].to_numpy()).astype(np.float64)
        self.episode = self.frames["episode_index"].to_numpy()
        self.eps = _episodes(LIBERO_REPO)
        ep_task = self.frames.groupby("episode_index")["task_index"].first().astype(int).to_dict()
        self.eps["task_index"] = self.eps["episode_index"].map(lambda e: ep_task[int(e)])


def score_records(records: list[dict], demo: Demo, const: dict) -> list[FrameMeta]:
    std = np.asarray(const["whitening_std"], dtype=np.float64)
    d = np.asarray([_window_d(demo.actions, demo.episode, int(r["global_index"]), std) for r in records])
    if not np.all(np.isfinite(d)):
        raise RuntimeError("non-finite d in pool")
    alpha = d / float(const["d_mean_cal"])
    G = demo_gripper_switch({"records": records}, demo.actions, demo.episode).astype(np.int64)
    alpha_event = (alpha + float(const["lambda"]) * G) / float(const["event_mean_cal"])
    metas = []
    for i, r in enumerate(records):
        metas.append(
            FrameMeta(
                shared_index=i,
                global_index=int(r["global_index"]),
                task_index=int(r["task_index"]),
                task=str(r["task"]),
                episode_index=int(r["episode_index"]),
                frame_in_episode=int(r["frame_in_episode"]),
                length=int(r["length"]),
                alpha=float(alpha[i]),
                d_whitened=float(d[i]),
                G=int(G[i]),
                alpha_event=float(alpha_event[i]),
                in_grip_switch_probe=0,
            )
        )
    return metas


# ------------------------------------------------------------------------ split


def dev_excluded_episodes(cfg: dict) -> set[int]:
    out: set[int] = set()
    for p in cfg["split"]["exclude_shared"]:
        blob = torch.load(resolve(p), map_location="cpu", weights_only=False)
        out |= {int(r["episode_index"]) for r in blob["records"]}
    return out


def stage_split(run: Run) -> None:
    run.check_frozen()
    if run.mode != "prospective":
        raise SystemExit("split is prospective-only; dev uses existing pool and holdout")
    sc = run.cfg["split"]
    const = run.cfg["constants"]
    demo = Demo()
    std = _whitening_std(demo.actions, demo.episode)[:ACTION_DIM]
    if not np.allclose(std, np.asarray(const["whitening_std"]), rtol=0, atol=1e-9):
        raise SystemExit(f"whitening std drifted: {std.tolist()}")
    excluded = dev_excluded_episodes(run.cfg)
    names = dev_task_names(run.cfg)
    rng = np.random.default_rng(int(sc["seed"]))
    n_cal, n_val, f_val = int(sc["calibration_episodes_per_task"]), int(sc["validation_episodes_per_task"]), int(
        sc["validation_frames_per_episode"]
    )
    cal_records, val_records, reserve, per_task = [], [], [], []
    for ti in range(N_TASKS):
        pool = demo.eps[demo.eps["task_index"] == ti].sort_values("episode_index")
        free = pool[~pool["episode_index"].isin(excluded)]
        per_task.append({"task_index": ti, "total": int(len(pool)), "excluded_dev": int(len(pool) - len(free)), "free": int(len(free))})
        if len(free) < n_cal + n_val:
            raise SystemExit(f"task {ti}: {len(free)} free episodes < {n_cal}+{n_val}")
        order = rng.permutation(free.index.to_numpy())
        val_rows, cal_rows, res_rows = order[:n_val], order[n_val : n_val + n_cal], order[n_val + n_cal :]
        for row_i in cal_rows:
            e = demo.eps.loc[row_i]
            start, stop = int(e["dataset_from_index"]), int(e["dataset_to_index"])
            gidx = int(rng.integers(start, stop - HORIZON + 1))
            cal_records.append(_record(demo, e, gidx, ti, names))
        for row_i in val_rows:
            e = demo.eps.loc[row_i]
            start, stop = int(e["dataset_from_index"]), int(e["dataset_to_index"])
            length = stop - start
            starts: list[int] = []
            cand = np.arange(0, length - HORIZON)
            while len(starts) < f_val:
                f0 = int(rng.choice(cand))
                if all(abs(f0 - s) >= HORIZON for s in starts):
                    starts.append(f0)
            for f0 in sorted(starts):
                val_records.append(_record(demo, e, start + f0, ti, names))
        reserve.extend(int(demo.eps.loc[r]["episode_index"]) for r in res_rows)

    metas = score_records(cal_records, demo, const)
    for r, m in zip(cal_records, metas):
        r.update({"d_whitened": m.d_whitened, "alpha": m.alpha, "G": m.G, "alpha_event": m.alpha_event})
    vmetas = score_records(val_records, demo, const)
    for r, m in zip(val_records, vmetas):
        r.update({"d_whitened": m.d_whitened, "alpha": m.alpha, "G": m.G, "alpha_event": m.alpha_event})

    cal_eps = {r["episode_index"] for r in cal_records}
    val_eps = {r["episode_index"] for r in val_records}
    if cal_eps & val_eps or cal_eps & excluded or val_eps & excluded:
        raise RuntimeError("episode overlap between roles")

    d = run.dir / "01_split"
    files = {}
    files["01_split/calibration_pool.json"] = write_json(d / "calibration_pool.json", cal_records)
    files["01_split/validation_set.json"] = write_json(d / "validation_set.json", val_records)
    shards = molmo_shards(demo, cal_eps, val_eps)
    videos = libero_video_coverage(demo, cal_eps | val_eps)
    summary = {
        "seed": int(sc["seed"]),
        "excluded_dev_episodes": len(excluded),
        "per_task_free_min": min(t["free"] for t in per_task),
        "per_task_free_max": max(t["free"] for t in per_task),
        "free_total": sum(t["free"] for t in per_task),
        "calibration": {"episodes": len(cal_eps), "observations": len(cal_records), "G1": int(sum(r["G"] for r in cal_records))},
        "validation": {"episodes": len(val_eps), "observations": len(val_records), "G1": int(sum(r["G"] for r in val_records))},
        "reserve_episodes": len(reserve),
        "molmo_shards": shards,
        "libero_videos": videos,
        "per_task": per_task,
    }
    files["01_split/split_summary.json"] = write_json(d / "split_summary.json", summary)
    files["01_split/reserve_episodes.json"] = write_json(d / "reserve_episodes.json", sorted(reserve))
    rows = [{"role": "calibration", **r} for r in cal_records] + [{"role": "validation", **r} for r in val_records]
    pd.DataFrame(rows).to_csv(d / "split_manifest.csv", index=False)
    files["01_split/split_manifest.csv"] = sha256_file(d / "split_manifest.csv")
    run.append("split", files)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=2))


def _record(demo: Demo, e: pd.Series, gidx: int, ti: int, task_names: dict[int, str]) -> dict:
    start = int(e["dataset_from_index"])
    if int(demo.episode[gidx]) != int(e["episode_index"]):
        raise RuntimeError(f"episode mismatch at {gidx}")
    return {
        "global_index": int(gidx),
        "episode_index": int(e["episode_index"]),
        "task_index": int(ti),
        "task": task_names[ti],
        "frame_in_episode": int(gidx - start),
        "length": int(e["dataset_to_index"] - start),
        "chunk_index": int(e["data/chunk_index"]),
        "file_index": int(e["data/file_index"]),
    }


def dev_task_names(cfg: dict) -> dict[int, str]:
    blob = torch.load(resolve(cfg["split"]["exclude_shared"][0]), map_location="cpu", weights_only=False)
    names = {int(r["task_index"]): str(r["task"]) for r in blob["records"]}
    if len(names) != N_TASKS:
        raise RuntimeError(f"task names: {len(names)}")
    return names


def molmo_shards(demo: Demo, cal_eps: set[int], val_eps: set[int]) -> dict:
    meps = _episodes(MOLMO_REPO)
    if not np.array_equal(meps["dataset_from_index"].values, demo.eps["dataset_from_index"].values):
        raise RuntimeError("Molmo and lerobot/libero episode ranges differ")
    key = {int(r["episode_index"]): (int(r["data/chunk_index"]), int(r["data/file_index"])) for _, r in meps.iterrows()}
    snap = hub_snapshot(MOLMO_REPO)
    local = {(int(p.parent.name.split("-")[1]), int(p.stem.split("-")[1])) for p in (snap / "data").glob("*/*.parquet")}
    local_sizes = [p.stat().st_size for p in (snap / "data").glob("*/*.parquet")]
    mean_mb = float(np.mean(local_sizes) / 1e6) if local_sizes else float("nan")

    def need(eps: set[int]) -> dict:
        files = {key[e] for e in eps}
        missing = sorted(files - local)
        return {"files": len(files), "missing_local": len(missing), "est_missing_gb": round(len(missing) * mean_mb / 1e3, 2)}

    return {"calibration": need(cal_eps), "validation": need(val_eps), "union": need(cal_eps | val_eps), "local_files": len(local)}


def libero_video_coverage(demo: Demo, eps: set[int]) -> dict:
    snap = hub_snapshot(LIBERO_REPO)
    out = {}
    for cam in ("observation.images.image", "observation.images.image2"):
        sub = demo.eps[demo.eps["episode_index"].isin(eps)]
        need = {(int(a), int(b)) for a, b in zip(sub[f"videos/{cam}/chunk_index"], sub[f"videos/{cam}/file_index"])}
        missing = [k for k in need if not (snap / "videos" / cam / f"chunk-{k[0]:03d}" / f"file-{k[1]:03d}.mp4").is_file()]
        out[cam] = {"files": len(need), "missing_local": len(missing)}
    return out


# ----------------------------------------------------------------------- select


def load_pool(run: Run) -> list[dict]:
    if run.mode == "dev":
        blob = torch.load(resolve(run.cfg["dev"]["pool_shared"]), map_location="cpu", weights_only=False)
        return [dict(r) for r in blob["records"]]
    run.require("split")
    return json.loads((run.dir / "01_split" / "calibration_pool.json").read_text())


def stage_select(run: Run) -> None:
    run.check_frozen()
    sel = run.cfg["selection"]
    records = load_pool(run)
    demo = Demo()
    metas = score_records(records, demo, run.cfg["constants"])
    if run.mode == "dev":
        blob = torch.load(resolve(run.cfg["dev"]["pool_shared"]), map_location="cpu", weights_only=False)
        ref = np.asarray(blob["alpha"], dtype=np.float64).reshape(-1)
        got = np.asarray([m.alpha for m in metas])
        if np.max(np.abs(ref - got)) > 1e-5:
            raise SystemExit(f"dev α mismatch vs shared.pt: {np.max(np.abs(ref - got))}")
    if sel["top_k_per_task"] != 3:
        raise SystemExit("top_k_per_task is fixed at 3 by event_care_nfe_diagnostic.TOP_K")
    groups = {
        "random": random_group(metas, seed=int(sel["random_seed"])),
        "alpha": select_per_task(metas, "alpha"),
        "g_only": g_only_group(metas, seed=int(sel["g_only_seed"])),
        "alpha_event": select_per_task(metas, "alpha_event"),
    }
    union = sorted(set().union(*groups.values()))
    out = {
        "pool_size": len(records),
        "pool_G1": int(sum(m.G for m in metas)),
        "strategies": {
            k: {
                "pool_index": sorted(v),
                "n": len(v),
                "G1": int(sum(metas[i].G for i in v)),
                "alpha_mean": float(np.mean([metas[i].alpha for i in v])),
            }
            for k, v in groups.items()
        },
        "union_pool_index": union,
        "pool_scores": [
            {"pool_index": m.shared_index, "global_index": m.global_index, "episode_index": m.episode_index,
             "task_index": m.task_index, "alpha": m.alpha, "G": m.G, "alpha_event": m.alpha_event}
            for m in metas
        ],
        "budget": budget(run.cfg, {k: len(v) for k, v in groups.items()}, len(union), len(records)),
    }
    digest = write_json(run.dir / "02_select" / "selections.json", out)
    run.append("select", {"02_select/selections.json": digest})
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "pool_index"} for k, v in out["strategies"].items()}, indent=2))
    print(json.dumps(out["budget"], indent=2))


def budget(cfg: dict, sizes: dict[str, int], n_union: int, n_pool: int) -> dict:
    rule = cfg["rule"]
    n_noise = len(rule["noise_ids"])
    out = {}
    for model, mc in cfg["models"].items():
        if mc.get("adapter") == "unavailable":
            continue
        cands = [n for n in rule["candidates"] if n in mc["probe_nfes"]]
        nfes = cands + [rule["reference_nfe"]]
        per_obs_chunks = n_noise * len(nfes)
        per_obs_vel = n_noise * sum(nfes)
        logical = {k: {"observations": s, "chunk_predictions": s * per_obs_chunks, "velocity_evals": s * per_obs_vel,
                       "nfe10_reference_velocity_evals": s * n_noise * rule["reference_nfe"]} for k, s in sizes.items()}
        sec = mc.get("historical_seconds_per_observation")
        out[model] = {
            "nfes_probed": nfes,
            "logical_per_strategy": logical,
            "logical_sum_over_strategies": {"observations": sum(sizes.values()), "velocity_evals": sum(sizes.values()) * per_obs_vel},
            "actual_union": {"observations": n_union, "chunk_predictions": n_union * per_obs_chunks, "velocity_evals": n_union * per_obs_vel,
                             "nfe10_reference_velocity_evals": n_union * n_noise * rule["reference_nfe"],
                             "est_gpu_seconds": None if sec is None else round(sec * n_union, 1)},
            "dev_historical_full_pool": {"observations": n_pool, "note": "dev probes already covered the whole pool; sunk cost"},
        }
    return out


# ------------------------------------------------------------------------ probe


def _probe_request(records: list[dict], index: list[int], scores: dict[int, dict]) -> dict:
    sub = []
    for i in index:
        r = dict(records[i])
        r["pool_index"] = int(i)
        sub.append(r)
    alpha = [scores[i]["alpha"] for i in index] if scores else [float(r.get("alpha", 1.0)) for r in sub]
    d = [float(r.get("d_whitened", np.nan)) for r in sub]
    return {"protocol": "care_event_nfe_protocol", "n": len(sub), "records": sub,
            "alpha": torch.tensor(alpha, dtype=torch.float32), "d_whitened": torch.tensor(d, dtype=torch.float32)}


def stage_probe_plan(run: Run) -> None:
    run.check_frozen()
    run.require("select")
    sel = json.loads((run.dir / "02_select" / "selections.json").read_text())
    records = load_pool(run)
    scores = {s["pool_index"]: s for s in sel["pool_scores"]}
    req = _probe_request(records, sel["union_pool_index"], scores)
    d = run.dir / "03_probe"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(req, d / "probe_request.pt")
    cmds = probe_commands(run, d / "probe_request.pt", d)
    (d / "COMMANDS.txt").write_text("\n".join(cmds) + "\n")
    run.append("probe-plan", {"03_probe/probe_request.pt": sha256_file(d / "probe_request.pt"),
                              "03_probe/COMMANDS.txt": sha256_file(d / "COMMANDS.txt")}, n=req["n"])
    print("\n".join(cmds))


def probe_commands(run: Run, shared: Path, out_root: Path) -> list[str]:
    cmds = []
    for model, mc in run.cfg["models"].items():
        if mc.get("adapter") == "unavailable":
            cmds.append(f"# {model}: adapter unavailable ({mc.get('reason', '')})")
            continue
        nfes = ",".join(str(n) for n in sorted(set(run.cfg["rule"]["candidates"]) | {run.cfg["rule"]["reference_nfe"]}))
        script = SCRIPT_DIR / mc["probe_script"]
        extra = " ".join(mc.get("probe_extra_args", []))
        cmds.append(
            f"{mc['python']} {script} --gpu <GPU> --shared-path {shared} --out-dir {out_root / model / 'raw'} "
            f"--skip-analyze --nfes {nfes} {extra}".strip()
        )
    return cmds


def _seal(model: str, actions: dict[str, np.ndarray], gidx: np.ndarray, pool_index: np.ndarray, episode: np.ndarray,
          mc: dict, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        pool_index=pool_index.astype(np.int64),
        global_index=gidx.astype(np.int64),
        episode_index=episode.astype(np.int64),
        grip_mean=np.float64(mc["gripper_unnormalize"]["mean"]),
        grip_std=np.float64(mc["gripper_unnormalize"]["std"]),
        **actions,
    )
    return sha256_file(dest)


def _probe_gidx(probe_dir: Path, n: int) -> np.ndarray:
    df = pd.read_csv(probe_dir / "frames.csv", usecols=["shared_index", "global_index"])
    g = df.drop_duplicates("shared_index").sort_values("shared_index")["global_index"].to_numpy()
    if len(g) != n:
        raise SystemExit(f"{probe_dir}: {len(g)} observations, expected {n}")
    return g


def stage_seal_dev(run: Run) -> None:
    """Dev only: the existing pool probes play the role of the union probe run."""
    run.check_frozen()
    if run.mode != "dev":
        raise SystemExit("seal-dev is dev-only")
    run.require("select")
    sel = json.loads((run.dir / "02_select" / "selections.json").read_text())
    records = load_pool(run)
    U = np.asarray(sel["union_pool_index"], dtype=np.int64)
    files = {}
    for model, mc in run.cfg["models"].items():
        if mc.get("adapter") == "unavailable":
            continue
        src = resolve(mc["dev_pool_probe"])
        z = np.load(src / "actions.npz")
        if not np.array_equal(z["shared_index"], np.arange(len(records))):
            raise SystemExit(f"{src}: shared_index is not the pool order")
        g = _probe_gidx(src, len(records))
        if not np.array_equal(g, [r["global_index"] for r in records]):
            raise SystemExit(f"{src}: global_index differs from pool")
        keep = [n for n in sorted(set(run.cfg["rule"]["candidates"]) | {run.cfg["rule"]["reference_nfe"]}) if f"nfe{n}" in z.files]
        acts = {f"nfe{n}": z[f"nfe{n}"][U] for n in keep}
        ep = np.asarray([records[i]["episode_index"] for i in U])
        rel = f"03_probe/{model}/sealed_union.npz"
        files[rel] = _seal(model, acts, g[U], U, ep, mc, run.dir / rel)
    run.append("seal-dev", files, note="dev rehearsal: sealed rows cut from probes that covered the full pool")
    print(json.dumps(files, indent=2))


def stage_probe_import(run: Run, model: str, probe_dir: Path) -> None:
    run.check_frozen()
    run.require("probe-plan")
    req = torch.load(run.dir / "03_probe" / "probe_request.pt", map_location="cpu", weights_only=False)
    _import(run, model, probe_dir, req, f"03_probe/{model}/sealed_union.npz", "probe-import")


def _import(run: Run, model: str, probe_dir: Path, req: dict, rel: str, stage: str) -> None:
    mc = run.cfg["models"][model]
    z = np.load(probe_dir / "actions.npz")
    n = int(req["n"])
    if not np.array_equal(z["shared_index"], np.arange(n)):
        raise SystemExit("probe output order differs from request")
    g = _probe_gidx(probe_dir, n)
    want = np.asarray([r["global_index"] for r in req["records"]])
    if not np.array_equal(g, want):
        raise SystemExit("probe output global_index differs from request")
    nfes = sorted(set(run.cfg["rule"]["candidates"]) | {run.cfg["rule"]["reference_nfe"]})
    missing = [n_ for n_ in nfes if f"nfe{n_}" not in z.files]
    if missing:
        raise SystemExit(f"probe output lacks NFE {missing}")
    acts = {f"nfe{n_}": z[f"nfe{n_}"] for n_ in nfes}
    pool_index = np.asarray([r.get("pool_index", r.get("validation_index", -1)) for r in req["records"]])
    ep = np.asarray([r["episode_index"] for r in req["records"]])
    digest = _seal(model, acts, g, pool_index, ep, mc, run.dir / rel)
    run.append(stage, {rel: digest}, model=model, probe_dir=str(probe_dir),
               probe_done_sha256=sha256_file(probe_dir / "DONE") if (probe_dir / "DONE").is_file() else None)
    print(f"sealed {rel}")


# ---------------------------------------------------------------------- metrics


def per_obs(z: dict, nfe: int, rule: dict) -> dict[str, np.ndarray]:
    """Observation-level quantities; both noises of one observation stay together."""
    a, ref = z[f"nfe{nfe}"].astype(np.float64), z[f"nfe{rule['reference_nfe']}"].astype(np.float64)
    arm = rule["arm_dims"]
    gd = rule["grip_dim"]
    mean, std = float(z["grip_mean"]), float(z["grip_std"])
    sev = int(rule["grip_severe_steps"])

    def closed(x):
        return (x[..., gd] * std + mean) > 0.0

    e = np.sqrt(((a - ref)[..., arm] ** 2).mean(axis=(2, 3))).mean(axis=1)
    dis = (closed(a) != closed(ref)).sum(axis=2)
    return {"e": e, "k": (dis > 0).mean(axis=1), "s": (dis >= sev).mean(axis=1), "dis_steps": dis}


def floor_obs(z: dict, rule: dict) -> dict[str, np.ndarray]:
    ref = z[f"nfe{rule['reference_nfe']}"].astype(np.float64)
    arm, gd = rule["arm_dims"], rule["grip_dim"]
    mean, std = float(z["grip_mean"]), float(z["grip_std"])
    sev = int(rule["grip_severe_steps"])
    f = np.sqrt(((ref[:, 0] - ref[:, 1])[..., arm] ** 2).mean(axis=(1, 2)))
    c = (ref[..., gd] * std + mean) > 0.0
    dis = (c[:, 0] != c[:, 1]).sum(axis=1)
    return {"f": f, "h": (dis > 0).astype(np.float64), "hs": (dis >= sev).astype(np.float64), "dis_steps": dis}


def stats(o: dict, fl: dict, idx: np.ndarray, rule: dict) -> dict[str, float]:
    q = float(rule["arm_quantile"])
    med_e, med_f = float(np.median(o["e"][idx])), float(np.median(fl["f"][idx]))
    q_e, q_f = float(np.quantile(o["e"][idx], q)), float(np.quantile(fl["f"][idx], q))
    return {
        "arm_median": med_e,
        "floor_median": med_f,
        "arm_q": q_e,
        "floor_q": q_f,
        "ratio_median": med_e / med_f if med_f > 0 else float("inf"),
        "ratio_q": q_e / q_f if q_f > 0 else float("inf"),
        "grip_rate": float(o["k"][idx].mean()),
        "floor_grip_rate": float(fl["h"][idx].mean()),
        "grip_diff": float(o["k"][idx].mean() - fl["h"][idx].mean()),
        "severe_rate": float(o["s"][idx].mean()),
        "floor_severe_rate": float(fl["hs"][idx].mean()),
        "severe_diff": float(o["s"][idx].mean() - fl["hs"][idx].mean()),
        "grip_comparisons_with_disagreement": int((o["dis_steps"][idx] > 0).sum()),
        "grip_comparisons": int(o["dis_steps"][idx].size),
        "floor_obs_with_disagreement": int((fl["dis_steps"][idx] > 0).sum()),
    }


def checks(s: dict, rule: dict) -> dict[str, bool]:
    return {
        "arm_median": s["ratio_median"] <= float(rule["arm_ratio_median"]),
        "arm_tail": s["ratio_q"] <= float(rule["arm_ratio_tail"]),
        "grip_rate": s["grip_diff"] <= float(rule["grip_rate_margin"]),
        "grip_severe": s["severe_diff"] <= float(rule["grip_severe_margin"]),
    }


def cluster_ci(o: dict, fl: dict, idx: np.ndarray, clusters: np.ndarray, rule: dict, B: int, seed: int) -> dict:
    keys = ("ratio_median", "ratio_q", "grip_diff", "severe_diff")
    uniq, inv = np.unique(clusters[idx], return_inverse=True)
    groups = [idx[inv == g] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in keys}
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        bi = np.concatenate([groups[p] for p in pick])
        s = stats(o, fl, bi, rule)
        for k in keys:
            vals[k].append(s[k])
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in vals.items()} | {"clusters": len(groups)}


def load_sealed(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


# -------------------------------------------------------------------- recommend


REASON = {
    "arm_median": "arm median > floor median",
    "arm_tail": "arm P95 > floor P95",
    "grip_rate": "gripper deviation rate > floor rate",
    "grip_severe": "severe gripper deviation rate > floor rate",
}


def recommend_one(z: dict, idx: np.ndarray, rule: dict, avail: list[int], boot: dict) -> dict:
    fl = floor_obs(z, rule)
    floor_med = float(np.median(fl["f"][idx]))
    rows = []
    for n in rule["candidates"]:
        if n not in avail:
            rows.append({"nfe": n, "status": "not_evaluable", "reasons": ["no probe output for this NFE"]})
            continue
        o = per_obs(z, n, rule)
        s = stats(o, fl, idx, rule)
        c = checks(s, rule)
        ci = cluster_ci(o, fl, idx, z["episode_index"], rule, int(boot["B"]), int(boot["seed"]))
        rows.append({"nfe": n, "status": "pass" if all(c.values()) else "reject", "checks": c, "stats": s, "ci95": ci,
                     "reasons": [REASON[k] for k, ok in c.items() if not ok]})
    degenerate = floor_med < float(rule["floor_degenerate_median"])
    passing = [r["nfe"] for r in rows if r["status"] == "pass"]
    unevaluable_below = []
    rec, why = rule["fallback_nfe"], "no candidate passed"
    if degenerate:
        why = f"floor median {floor_med:.2e} < {rule['floor_degenerate_median']}; stochastic floor unusable"
    else:
        for r in rows:
            if r["status"] == "not_evaluable":
                unevaluable_below.append(r["nfe"])
                continue
            if r["status"] == "pass":
                rec, why = r["nfe"], "lowest passing candidate"
                break
    larger_fail = [r["nfe"] for r in rows if r["status"] == "reject" and r["nfe"] > rec] if rec != rule["fallback_nfe"] else []
    return {
        "recommended_nfe": rec,
        "why": why,
        "fallback": rec == rule["fallback_nfe"],
        "floor_degenerate": degenerate,
        "floor_median": floor_med,
        "n_observations": int(idx.size),
        "skipped_unevaluable_below": unevaluable_below,
        "non_monotone_larger_rejected": larger_fail,
        "candidates": rows,
    }


def stage_recommend(run: Run) -> None:
    run.check_frozen()
    sel_entry = run.require("select")
    seal_stage = "seal-dev" if run.mode == "dev" else "probe-import"
    sel = json.loads((run.dir / "02_select" / "selections.json").read_text())
    rule = run.cfg["rule"]
    out = {"mode": run.mode, "label": "DEV REHEARSAL" if run.mode == "dev" else "PROSPECTIVE",
           "select_sha256": sel_entry["files"]["02_select/selections.json"], "models": {}}
    sealed_hashes = {}
    for model, mc in run.cfg["models"].items():
        if mc.get("adapter") == "unavailable":
            out["models"][model] = {"status": "not_evaluated", "reason": mc.get("reason", "")}
            continue
        rel = f"03_probe/{model}/sealed_union.npz"
        entry = next((e for e in reversed(run.entries()) if e["stage"] == seal_stage and rel in e["files"]), None)
        if entry is None or sha256_file(run.dir / rel) != entry["files"][rel]:
            raise SystemExit(f"{rel} missing or not matching its {seal_stage} ledger entry")
        sealed_hashes[rel] = entry["files"][rel]
        z = load_sealed(run.dir / rel)
        if not np.array_equal(z["pool_index"], sel["union_pool_index"]):
            raise SystemExit(f"{rel}: rows differ from the selected union")
        avail = [n for n in rule["candidates"] if f"nfe{n}" in z]
        pos = {int(p): i for i, p in enumerate(z["pool_index"])}
        res = {}
        for strat in STRATEGIES:
            pick = sel["strategies"][strat]["pool_index"]
            idx = np.asarray([pos[p] for p in pick], dtype=np.int64)
            res[strat] = recommend_one(z, idx, rule, avail, run.cfg["bootstrap"])
            res[strat]["G1"] = sel["strategies"][strat]["G1"]
        out["models"][model] = {"status": "evaluated", "available_candidates": avail, "strategies": res}
    d = run.dir / "04_recommend"
    digest = write_json(d / "recommendations.json", out)
    (d / "RECOMMEND.md").write_text(render_recommend(out, rule))
    run.append("recommend", {"04_recommend/recommendations.json": digest, "04_recommend/RECOMMEND.md": sha256_file(d / "RECOMMEND.md")},
               sealed=sealed_hashes)
    print((d / "RECOMMEND.md").read_text())


def _fmt_ci(ci: list[float]) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def render_recommend(out: dict, rule: dict) -> str:
    lines = [f"# NFE recommendation — {out['label']}", "",
             "Screening metrics on selected observations. Event-enriched rates are stress-test rates, not deployment rates.", "",
             "| model | strategy | G=1 | rec. NFE | NFE | status | arm med ratio | arm P95 ratio | grip dev / floor (comparisons) | severe dev / floor | reasons |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for model, m in out["models"].items():
        if m["status"] != "evaluated":
            lines.append(f"| {model} | — | — | — | — | not evaluated | | | | | {m['reason']} |")
            continue
        for strat, r in m["strategies"].items():
            for c in r["candidates"]:
                if c["status"] == "not_evaluable":
                    lines.append(f"| {model} | {strat} | {r['G1']} | {r['recommended_nfe']} | {c['nfe']} | not evaluable | | | | | {'; '.join(c['reasons'])} |")
                    continue
                s = c["stats"]
                lines.append(
                    f"| {model} | {strat} | {r['G1']} | {r['recommended_nfe']} | {c['nfe']} | {c['status']} | "
                    f"{s['ratio_median']:.2f} {_fmt_ci(c['ci95']['ratio_median'])} | {s['ratio_q']:.2f} {_fmt_ci(c['ci95']['ratio_q'])} | "
                    f"{s['grip_comparisons_with_disagreement']}/{s['grip_comparisons']} vs {s['floor_obs_with_disagreement']}/{r['n_observations']} | "
                    f"{s['severe_rate']:.3f} vs {s['floor_severe_rate']:.3f} | {'; '.join(c['reasons']) or '—'} |"
                )
    lines += ["", f"Rule: candidates {rule['candidates']}, reference NFE {rule['reference_nfe']}, fallback {rule['fallback_nfe']}; "
              f"arm ratio ≤ {rule['arm_ratio_median']} (median) and ≤ {rule['arm_ratio_tail']} (P{int(rule['arm_quantile']*100)}); "
              f"gripper rate diff ≤ {rule['grip_rate_margin']}; severe (≥{rule['grip_severe_steps']} steps) diff ≤ {rule['grip_severe_margin']}.", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------- validation


def stage_validation_plan(run: Run) -> None:
    run.check_frozen()
    if run.mode != "prospective":
        raise SystemExit("validation-plan is prospective-only")
    run.require("recommend")
    run.require("split")
    val = json.loads((run.dir / "01_split" / "validation_set.json").read_text())
    for i, r in enumerate(val):
        r["validation_index"] = i
    req = {"protocol": "care_event_nfe_protocol_validation", "n": len(val), "records": val,
           "alpha": torch.tensor([r["alpha"] for r in val], dtype=torch.float32),
           "d_whitened": torch.tensor([r["d_whitened"] for r in val], dtype=torch.float32)}
    d = run.dir / "05_validation"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(req, d / "probe_request.pt")
    cmds = probe_commands(run, d / "probe_request.pt", d)
    (d / "COMMANDS.txt").write_text("\n".join(cmds) + "\n")
    run.append("validation-plan", {"05_validation/probe_request.pt": sha256_file(d / "probe_request.pt"),
                                   "05_validation/COMMANDS.txt": sha256_file(d / "COMMANDS.txt")},
               recommend_sha256=run.last("recommend")["files"]["04_recommend/recommendations.json"])
    print("\n".join(cmds))


def stage_validation_import(run: Run, model: str, probe_dir: Path) -> None:
    run.check_frozen()
    run.require("validation-plan")
    req = torch.load(run.dir / "05_validation" / "probe_request.pt", map_location="cpu", weights_only=False)
    _import(run, model, probe_dir, req, f"05_validation/{model}/sealed_validation.npz", "validation-import")


def dev_validation_sealed(run: Run, model: str, mc: dict) -> dict:
    src = resolve(mc["dev_validation_probe"])
    blob = torch.load(resolve(run.cfg["dev"]["validation_shared"]), map_location="cpu", weights_only=False)
    z = np.load(src / "actions.npz")
    n = len(blob["records"])
    g = _probe_gidx(src, n)
    if not np.array_equal(g, [r["global_index"] for r in blob["records"]]):
        raise SystemExit(f"{src}: global_index differs from validation records")
    out = {k: z[k] for k in z.files if k.startswith("nfe")}
    out.update(pool_index=np.arange(n), global_index=g, episode_index=np.asarray([r["episode_index"] for r in blob["records"]]),
               grip_mean=np.float64(mc["gripper_unnormalize"]["mean"]), grip_std=np.float64(mc["gripper_unnormalize"]["std"]))
    return out


def stage_validate(run: Run) -> None:
    run.check_frozen()
    rec_entry = run.require("recommend")
    rec = json.loads((run.dir / "04_recommend" / "recommendations.json").read_text())
    rule = run.cfg["rule"]
    vq = run.cfg["validation"]
    out = {"mode": run.mode, "label": rec["label"], "recommend_sha256": rec_entry["files"]["04_recommend/recommendations.json"],
           "recommend_utc": rec_entry["utc"], "models": {}}
    for model, mc in run.cfg["models"].items():
        if rec["models"].get(model, {}).get("status") != "evaluated":
            out["models"][model] = {"status": "not_evaluated"}
            continue
        if run.mode == "dev":
            z = dev_validation_sealed(run, model, mc)
            source = str(resolve(mc["dev_validation_probe"]))
        else:
            rel = f"05_validation/{model}/sealed_validation.npz"
            e = next((x for x in reversed(run.entries()) if x["stage"] == "validation-import" and rel in x["files"]), None)
            if e is None or e["utc"] < rec_entry["utc"]:
                raise SystemExit(f"{rel}: no validation import after recommendation")
            z = load_sealed(run.dir / rel)
            source = rel
        idx = np.arange(len(z["global_index"]))
        fl = floor_obs(z, rule)
        per_nfe = {}
        for n in rule["candidates"]:
            if f"nfe{n}" not in z:
                per_nfe[n] = {"status": "not_evaluable"}
                continue
            o = per_obs(z, n, rule)
            s = stats(o, fl, idx, rule)
            c = checks(s, rule)
            ci = cluster_ci(o, fl, idx, z["episode_index"], rule, int(run.cfg["bootstrap"]["B"]), int(run.cfg["bootstrap"]["seed"]))
            per_nfe[n] = {"status": "pass" if all(c.values()) else "fail", "checks": c, "stats": s, "ci95": ci}
        verdict = {}
        for strat, r in rec["models"][model]["strategies"].items():
            n_star = r["recommended_nfe"]
            if n_star == rule["fallback_nfe"]:
                offline = {"status": "reference", "note": "fallback NFE is the reference itself"}
            else:
                offline = {"status": per_nfe[n_star]["status"], "failed": [k for k, ok in per_nfe[n_star]["checks"].items() if not ok]}
            rejected_but_pass = [c["nfe"] for c in r["candidates"] if c["status"] == "reject" and per_nfe.get(c["nfe"], {}).get("status") == "pass"]
            verdict[strat] = {"recommended_nfe": n_star, "offline_quality": offline,
                              "closed_loop_quality": closed_loop_check(vq, rule, model, n_star),
                              "rejected_in_screening_but_pass_validation": rejected_but_pass}
        out["models"][model] = {"status": "evaluated", "source": source, "n_observations": int(idx.size),
                                "n_episodes": int(np.unique(z["episode_index"]).size), "per_nfe": {str(k): v for k, v in per_nfe.items()},
                                "strategies": verdict}
    d = run.dir / "06_validate"
    digest = write_json(d / "validation.json", out)
    (d / "VALIDATE.md").write_text(render_validate(out))
    run.append("validate", {"06_validate/validation.json": digest, "06_validate/VALIDATE.md": sha256_file(d / "VALIDATE.md")})
    print((d / "VALIDATE.md").read_text())


def stage_verify_ledger(run: Run) -> None:
    """Re-hash every ledger file entry; does not mutate the run."""
    entries = run.entries()
    if not entries:
        raise SystemExit(f"empty ledger {run.ledger}")
    ok = True
    for e in entries:
        for rel, digest in e.get("files", {}).items():
            path = run.dir / rel
            if not path.is_file():
                print(f"MISSING {rel} ({e['stage']})")
                ok = False
                continue
            got = sha256_file(path)
            if got != digest:
                print(f"MISMATCH {rel} stage={e['stage']}")
                ok = False
    fr = run.last("freeze")
    if fr:
        if sha256_file(run.config_path) != fr["config_sha256"]:
            print("MISMATCH config vs freeze")
            ok = False
        for name, digest in fr.get("code_sha256", {}).items():
            p = SCRIPT_DIR / name
            if p.is_file() and sha256_file(p) != digest:
                print(f"MISMATCH code {name}")
                ok = False
    raise SystemExit(0 if ok else 1)


def stage_import_closed_loop(run: Run, source: Path) -> None:
    """Import closed-loop success rates from an external JSON (not from probe errors)."""
    run.check_frozen()
    src = source if source.is_absolute() else resolve(source)
    if not src.is_file():
        raise SystemExit(f"missing closed-loop source {src}")
    payload = json.loads(src.read_text())
    frozen = run.dir / "config.frozen.json"
    cfg = json.loads(frozen.read_text()) if frozen.is_file() else run.cfg
    cfg.setdefault("validation", {}).setdefault("closed_loop", {})["results"] = payload.get(
        "results", payload
    )
    cfg["validation"]["closed_loop"]["import_source"] = str(src)
    cfg["validation"]["closed_loop"]["import_utc"] = utc_now()
    digest = write_json(run.dir / "closed_loop_import.json", cfg["validation"]["closed_loop"])
    run.append(
        "closed-loop-import",
        {"closed_loop_import.json": digest},
        source=str(src),
        force=True,
    )
    print(f"imported closed-loop results from {src}")


def closed_loop_check(vq: dict, rule: dict, model: str, n_star: int) -> dict:
    cl = vq["closed_loop"]
    if n_star == rule["fallback_nfe"]:
        return {"status": "reference", "note": "fallback NFE needs no closed-loop comparison"}
    res = cl.get("results", {}).get(model, {})
    ref, cand = res.get(str(rule["reference_nfe"])), res.get(str(n_star))
    if ref is None or cand is None:
        return {"status": "pending", "rule": f"Avg ≥ ref − {cl['avg_margin_pp']} pp and every suite ≥ ref − {cl['suite_drop_pp']} pp, same protocol and n"}
    avg_d = float(np.mean(list(cand.values())) - np.mean(list(ref.values())))
    suite_d = {k: float(cand[k] - ref[k]) for k in ref}
    ok = avg_d >= -float(cl["avg_margin_pp"]) and min(suite_d.values()) >= -float(cl["suite_drop_pp"])
    return {"status": "pass" if ok else "fail", "avg_delta_pp": avg_d, "suite_delta_pp": suite_d}


def render_validate(out: dict) -> str:
    lines = [f"# Independent-set quality conditions — {out['label']}", "",
             "Uniform (non-enriched) observations; observation = episode cluster for CIs.", "",
             "| model | NFE | status | arm med ratio | arm P95 ratio | grip rate vs floor | severe vs floor |", "|---|---|---|---|---|---|---|"]
    for model, m in out["models"].items():
        if m["status"] != "evaluated":
            continue
        for n, r in m["per_nfe"].items():
            if r["status"] == "not_evaluable":
                lines.append(f"| {model} | {n} | not evaluable | | | | |")
                continue
            s, ci = r["stats"], r["ci95"]
            lines.append(f"| {model} | {n} | {r['status']} | {s['ratio_median']:.2f} {_fmt_ci(ci['ratio_median'])} | "
                         f"{s['ratio_q']:.2f} {_fmt_ci(ci['ratio_q'])} | {s['grip_rate']:.3f} vs {s['floor_grip_rate']:.3f} | "
                         f"{s['severe_rate']:.3f} vs {s['floor_severe_rate']:.3f} |")
    lines += ["", "| model | strategy | rec. NFE | offline quality | closed loop | rejected but pass on validation |", "|---|---|---|---|---|---|"]
    for model, m in out["models"].items():
        if m["status"] != "evaluated":
            continue
        for strat, v in m["strategies"].items():
            off = v["offline_quality"]
            lines.append(f"| {model} | {strat} | {v['recommended_nfe']} | {off['status']}{' ' + ','.join(off.get('failed', [])) if off.get('failed') else ''} | "
                         f"{v['closed_loop_quality']['status']} | {v['rejected_in_screening_but_pass_validation'] or '—'} |")
    return "\n".join(lines) + "\n"


# -------------------------------------------------------------------------- cli


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "stage",
        choices=[
            "freeze",
            "split",
            "select",
            "probe-plan",
            "probe-import",
            "seal-dev",
            "recommend",
            "validation-plan",
            "validation-import",
            "validate",
            "verify-ledger",
            "closed-loop-import",
            "run-dev",
        ],
    )
    p.add_argument("--model", default=None)
    p.add_argument("--probe-dir", type=Path, default=None)
    p.add_argument("--closed-loop-json", type=Path, default=None, help="for closed-loop-import")
    p.add_argument("--force-restage", action="store_true", help="allow re-appending a non-repeatable stage")
    p.add_argument("--draft", action="store_true", help="split/select/probe-plan preview without freeze (identities only)")
    a = p.parse_args()
    if a.draft and a.stage not in ("split", "select", "probe-plan"):
        raise SystemExit("--draft only allowed for split, select, probe-plan")
    run = Run(a.config, draft=a.draft)
    if a.force_restage:
        _orig_append = run.append

        def _append(stage, files, **extra):
            return _orig_append(stage, files, force=True, **extra)

        run.append = _append  # type: ignore[method-assign]
    if a.stage == "verify-ledger":
        stage_verify_ledger(run)
        return
    if a.stage == "closed-loop-import":
        if not a.closed_loop_json:
            raise SystemExit("--closed-loop-json required")
        stage_import_closed_loop(run, a.closed_loop_json)
        return
    if a.stage == "run-dev":
        if run.mode != "dev":
            raise SystemExit("run-dev requires mode=dev")
        for fn in (stage_freeze, stage_select, stage_seal_dev, stage_recommend, stage_validate):
            fn(run)
        return
    if a.stage in ("probe-import", "validation-import"):
        if not a.model or not a.probe_dir:
            raise SystemExit("--model and --probe-dir required")
        (stage_probe_import if a.stage == "probe-import" else stage_validation_import)(run, a.model, a.probe_dir)
        return
    {"freeze": stage_freeze, "split": stage_split, "select": stage_select, "probe-plan": stage_probe_plan,
     "seal-dev": stage_seal_dev, "recommend": stage_recommend, "validation-plan": stage_validation_plan,
     "validate": stage_validate}[a.stage](run)


if __name__ == "__main__":
    main()

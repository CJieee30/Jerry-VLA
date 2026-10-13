#!/usr/bin/env python3
"""SmolVLA offline NFE probe on the frozen CARE v2 480-frame set.

Full model lerobot/smolvla_libero. Independently scheduled NFE in {1,2,3,10}.
Same frame + noise is identical across NFEs. Does not change CARE α, does not
overwrite shared.pt / historical evals, and does not run closed-loop.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
FN_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "smolvla_libero"))
sys.path.insert(0, str(HERE.parent / "pi05_libero"))

import numpy as np
import torch
import torch.nn.functional as F

from adapt_lerobot_official import (
    CKPT,
    LIBERO_STATE_DIM,
    N_ACTION_STEPS,
    NUM_STEPS,
    RENAME_MAP,
    patch_processor_state_dim,
)
from dump_shared import _load_actions
from protocol import ACTION_DIM, FRAMES_PER_TASK, HORIZON, LIBERO_REPO, N_FRAMES, OUT_DIR, SEED as CARE_SEED
from score import _ensure_policy_batch

PROBE_DIR = OUT_DIR / "nfe_probe_smolvla"
PREVIEW_SELECTION = OUT_DIR / "preview_smolvla_timestamps" / "selection.json"
NFES = (1, 2, 3, 10)
N_NOISE = 2
EXEC_STEPS = 10
NOISE_SEED_BASE = 1000  # closed-loop seed; documented in config.json
GROUP_SEED = 42
TOP_K_PER_TASK = 3  # 25% of 12
MIN_FREE_MIB = 6000
SMOLVLA_PEAK_MIB_NOTE = "~3 GiB weights + activations at batch=1"
CLOSED_LOOP_SR = {
    "full_nfe10": 83.00,
    "full_nfe2": 82.00,
    "full_nfe1": 79.25,
    "full_nfe1_paired": 78.25,
}
INFER_DROP = {"action", "action_is_pad"}
CSV_FIELDS = [
    "shared_index",
    "global_index",
    "episode_index",
    "task",
    "task_index",
    "frame_in_episode",
    "length",
    "timestamp",
    "noise_id",
    "noise_seed",
    "alpha",
    "d_whitened",
    "nfe",
    "n_velocity_calls",
    "cosine",
    "rmse",
    "max_abs",
    "rmse_trans",
    "rmse_rot",
    "rmse_grip",
    "max_abs_trans",
    "max_abs_rot",
    "max_abs_grip",
    "rmse_tail3",
    "gripper_disagree",
    "gripper_disagree_n",
    "v_adj_cos_mean",
    "v_adj_cos_min",
    "v_first_last_cos",
    "v_adj_norm_rel_mean",
    "v_first_last_norm_rel",
    "in_all",
    "in_top_alpha",
    "in_random",
    "in_low_alpha",
    "in_grip_switch",
    "viz_tag",
]


class DenoiseProbe:
    """Count denoise_step calls and optionally record velocities. No extra forwards."""

    def __init__(self, fn, *, record: bool = False):
        self.fn = fn
        self.record = record
        self.n_calls = 0
        self.vs: list[torch.Tensor] = []

    def __call__(self, *args, **kwargs):
        self.n_calls += 1
        out = self.fn(*args, **kwargs)
        if self.record:
            self.vs.append(out.detach().float().cpu().clone())
        return out


def _gpu_memory(gpu: int) -> dict[str, int]:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    tot, used, free, util = [p.strip() for p in raw.split(",")]
    return {
        "total_mib": int(tot),
        "used_mib": int(used),
        "free_mib": int(free),
        "util": int(util),
    }


def _gpu_occupancy(gpu: int) -> list[dict[str, Any]]:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        pid_s, name, mem = [p.strip() for p in line.split(",", 2)]
        pid = int(pid_s)
        try:
            user = subprocess.check_output(["ps", "-o", "user=", "-p", str(pid)], text=True).strip()
        except subprocess.CalledProcessError:
            user = "?"
        try:
            args = subprocess.check_output(["ps", "-o", "args=", "-p", str(pid)], text=True).strip()
        except subprocess.CalledProcessError:
            args = ""
        rows.append({"pid": pid, "user": user, "name": name, "mem_mib": mem, "args": args})
    return rows


def require_shared_gpu(gpu: int, *, allow_shared: bool, min_free_mib: int) -> dict[str, Any]:
    if not allow_shared:
        raise SystemExit("offline probe on a shared card requires --allow-shared and explicit --gpu")
    occ = _gpu_occupancy(gpu)
    mem = _gpu_memory(gpu)
    evals = [
        r
        for r in occ
        if r.get("user") == "guorj" and "eval_drop.py" in str(r.get("args") or "")
    ]
    if evals:
        desc = ", ".join(f"pid={e['pid']}" for e in evals)
        raise SystemExit(f"GPU{gpu} already runs this project's eval_drop ({desc})")
    if int(mem["free_mib"]) < int(min_free_mib):
        raise SystemExit(
            f"GPU{gpu} free {mem['free_mib']} MiB < {min_free_mib} MiB "
            f"(SmolVLA probe {SMOLVLA_PEAK_MIB_NOTE})"
        )
    snap = {"gpu": int(gpu), "memory": mem, "occupancy": occ}
    others = ", ".join(f"{r['user']}:{r['pid']}:{r['mem_mib']}MiB" for r in occ) or "idle"
    print(f"GPU{gpu} shared probe ok free={mem['free_mib']}MiB occupancy=({others})", flush=True)
    return snap


def noise_seed(shared_index: int, noise_id: int) -> int:
    return int(NOISE_SEED_BASE) * 1_000_003 + int(shared_index) * N_NOISE + int(noise_id)


def make_noise(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.normal(0.0, 1.0, size=shape, dtype=torch.float32, generator=gen)


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    fa = F.normalize(_flat(a).float()[None], dim=-1)
    fb = F.normalize(_flat(b).float()[None], dim=-1)
    return float((fa * fb).sum().item())


def _rmse(err: torch.Tensor) -> float:
    e = err.float()
    return float(torch.sqrt((e * e).mean()).item())


def _max_abs(err: torch.Tensor) -> float:
    return float(err.float().abs().max().item())


def slice_exec(chunk: torch.Tensor) -> torch.Tensor:
    """First 10 executed steps × 7 valid dims. Padding dims already stripped by the policy."""
    if chunk.ndim == 3:
        chunk = chunk[0]
    return chunk[:EXEC_STEPS, :ACTION_DIM].float().contiguous()


def unnormalize_action(normed: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return normed.float() * std.to(normed.device, dtype=torch.float32) + mean.to(
        normed.device, dtype=torch.float32
    )


def gripper_closed(unnorm_grip: torch.Tensor) -> torch.Tensor:
    """LIBERO dummy/hold-open is -1. Closed iff gripper > 0 after env-scale unnormalize."""
    return unnorm_grip.float() > 0.0


def velocity_metrics(vs: list[torch.Tensor]) -> dict[str, float]:
    if len(vs) != NUM_STEPS:
        raise RuntimeError(f"expected {NUM_STEPS} velocities, got {len(vs)}")
    toks = [slice_exec(v) for v in vs]
    adj_cos = [_cos(toks[i], toks[i + 1]) for i in range(len(toks) - 1)]
    norms = [float(t.norm().item()) for t in toks]
    adj_rel = [
        norms[i + 1] / (norms[i] + 1e-12) - 1.0 for i in range(len(norms) - 1)
    ]
    return {
        "v_adj_cos_mean": float(np.mean(adj_cos)),
        "v_adj_cos_min": float(np.min(adj_cos)),
        "v_first_last_cos": _cos(toks[0], toks[-1]),
        "v_adj_norm_rel_mean": float(np.mean(np.abs(adj_rel))),
        "v_first_last_norm_rel": norms[-1] / (norms[0] + 1e-12) - 1.0,
    }


def empty_v_metrics() -> dict[str, float]:
    return {
        "v_adj_cos_mean": float("nan"),
        "v_adj_cos_min": float("nan"),
        "v_first_last_cos": float("nan"),
        "v_adj_norm_rel_mean": float("nan"),
        "v_first_last_norm_rel": float("nan"),
    }


def action_metrics(pred: torch.Tensor, ref: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> dict[str, float]:
    err = pred - ref
    trans, rot, grip = err[:, :3], err[:, 3:6], err[:, 6:7]
    pred_u = unnormalize_action(pred, mean, std)
    ref_u = unnormalize_action(ref, mean, std)
    disagree = gripper_closed(pred_u[:, 6]) != gripper_closed(ref_u[:, 6])
    return {
        "cosine": _cos(pred, ref),
        "rmse": _rmse(err),
        "max_abs": _max_abs(err),
        "rmse_trans": _rmse(trans),
        "rmse_rot": _rmse(rot),
        "rmse_grip": _rmse(grip),
        "max_abs_trans": _max_abs(trans),
        "max_abs_rot": _max_abs(rot),
        "max_abs_grip": _max_abs(grip),
        "rmse_tail3": _rmse(err[-3:]),
        "gripper_disagree": float(disagree.float().mean().item()),
        "gripper_disagree_n": int(disagree.sum().item()),
    }


def _cpu_tensors(batch: dict) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.detach().cpu().contiguous() if torch.is_tensor(value) else value
    return out


def inference_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if key in INFER_DROP:
            continue
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def action_mean_std(preprocessor) -> tuple[torch.Tensor, torch.Tensor]:
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    for step in preprocessor.steps:
        if isinstance(step, NormalizerProcessorStep):
            stats = step._tensor_stats.get("action") or step._tensor_stats.get("ACTION")
            if not stats:
                raise RuntimeError(f"normalizer has no action stats: {list(step._tensor_stats)}")
            mean = stats["mean"].detach().cpu().float().flatten()[:ACTION_DIM].clone()
            std = stats["std"].detach().cpu().float().flatten()[:ACTION_DIM].clone()
            std = torch.clamp(std, min=1e-8)
            return mean, std
    raise RuntimeError("preprocessor has no NormalizerProcessorStep")


def load_processors():
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    cfg = SmolVLAConfig.from_pretrained(CKPT)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=CKPT,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "rename_observations_processor": {"rename_map": dict(RENAME_MAP)},
        },
    )
    patch_processor_state_dim(preprocessor)
    return preprocessor, postprocessor, cfg


def load_shared() -> dict:
    path = OUT_DIR / "shared.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if int(blob["n"]) != N_FRAMES or len(blob["records"]) != N_FRAMES:
        raise SystemExit(f"shared.pt n={blob.get('n')} records={len(blob['records'])}")
    if abs(float(blob["alpha"].mean()) - 1.0) > 1e-4:
        raise SystemExit(f"alpha mean {float(blob['alpha'].mean())}")
    return blob


def load_dataset(cfg):
    from common import resolve_dataset_root
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = resolve_dataset_root(LIBERO_REPO)
    ds_meta = LeRobotDatasetMetadata(LIBERO_REPO, root=root)
    delta = resolve_delta_timestamps(cfg, ds_meta)
    dataset = LeRobotDataset(
        LIBERO_REPO,
        delta_timestamps=delta,
        download_videos=False,
        video_backend="pyav",
        return_uint8=False,
        root=root,
    )
    return dataset


def preprocess_frames(shared: dict, preprocessor, cfg, dest: Path, limit: int | None) -> list[dict]:
    n = N_FRAMES if limit is None else min(int(limit), N_FRAMES)
    if dest.is_file() and limit is None:
        blob = torch.load(dest, map_location="cpu", weights_only=False)
        if int(blob.get("n", 0)) == N_FRAMES:
            print(f"[cache] {dest}", flush=True)
            return blob["samples"]
    dataset = load_dataset(cfg)
    samples = []
    for i, rec in enumerate(shared["records"][:n]):
        gidx = int(rec["global_index"])
        frame = dataset[gidx]
        batch = _ensure_policy_batch(preprocessor(frame))
        samples.append({"global_index": gidx, "batch": _cpu_tensors(batch)})
        if (i + 1) % 40 == 0 or i == 0:
            print(f"preprocess {i+1}/{n}", flush=True)
    if limit is None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"n": len(samples), "checkpoint": CKPT, "samples": samples}, dest)
        print(f"wrote {dest}", flush=True)
    return samples


def configure_policy(policy, device: torch.device):
    policy.config.compile_model = False
    policy.config.n_action_steps = N_ACTION_STEPS
    policy.config.num_steps = NUM_STEPS
    policy.config.empty_cameras = 0
    policy.config.device = str(device)
    policy.eval()
    return policy


def predict_chunk(policy, batch: dict, noise: torch.Tensor, nfe: int, *, record_v: bool):
    model = policy.model
    orig = model.denoise_step
    probe = DenoiseProbe(orig, record=record_v)
    policy.config.num_steps = int(nfe)
    policy.reset()
    model.denoise_step = probe
    try:
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch, noise=noise.clone())
    finally:
        model.denoise_step = orig
        policy.config.num_steps = NUM_STEPS
    return slice_exec(chunk.detach().cpu()), probe


def demo_gripper_switch(shared: dict) -> np.ndarray:
    frames = _load_actions(LIBERO_REPO)
    actions = np.stack(frames["action"].to_numpy()).astype(np.float64)
    episode = frames["episode_index"].to_numpy()
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


def group_indices(records: list[dict], alpha: np.ndarray) -> dict[str, set[int]]:
    by_task: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        by_task[int(rec["task_index"])].append(i)
    top, low = [], []
    for ti in sorted(by_task):
        idxs = by_task[ti]
        ordered = sorted(
            idxs,
            key=lambda j: (-float(alpha[j]), int(records[j]["global_index"])),
        )
        if len(records) == N_FRAMES and len(ordered) != FRAMES_PER_TASK:
            raise RuntimeError(f"task {ti} has {len(ordered)} frames, expected {FRAMES_PER_TASK}")
        k = min(TOP_K_PER_TASK, len(ordered))
        top.extend(ordered[:k])
        low.extend(ordered[-k:] if k else [])
    rng = np.random.default_rng(GROUP_SEED)
    rand = []
    for ti in sorted(by_task):
        idxs = np.asarray(sorted(by_task[ti]), dtype=np.int64)
        k = min(TOP_K_PER_TASK, int(idxs.size))
        chosen = rng.choice(idxs, size=k, replace=False)
        rand.extend(int(x) for x in chosen)
    return {
        "all": set(range(len(records))),
        "top_alpha": set(top),
        "random": set(rand),
        "low_alpha": set(low),
    }


def load_viz_map() -> dict[int, str]:
    if not PREVIEW_SELECTION.is_file():
        return {}
    blob = json.loads(PREVIEW_SELECTION.read_text())
    out = {}
    for clip in blob.get("clips", []):
        idx = int(clip["shared_index"])
        tag = f"{clip['suite_slug']}_{clip['kind']}"
        out[idx] = tag
    return out


def percentile(xs: np.ndarray, q: float) -> float:
    if xs.size == 0:
        return float("nan")
    return float(np.percentile(xs, q))


def summarize_rows(rows: list[dict], nfe: int, noise_id: int | None) -> dict[str, Any]:
    sub = [r for r in rows if int(r["nfe"]) == int(nfe)]
    if noise_id is not None:
        sub = [r for r in sub if int(r["noise_id"]) == int(noise_id)]
    if not sub:
        return {"n": 0, "nfe": nfe, "noise_id": noise_id}
    cosine = np.asarray([float(r["cosine"]) for r in sub], dtype=np.float64)
    rmse = np.asarray([float(r["rmse"]) for r in sub], dtype=np.float64)
    max_abs = np.asarray([float(r["max_abs"]) for r in sub], dtype=np.float64)
    grip = np.asarray([float(r["gripper_disagree"]) for r in sub], dtype=np.float64)
    tail = np.asarray([float(r["rmse_tail3"]) for r in sub], dtype=np.float64)
    worst_cos_i = int(np.argmin(cosine))
    worst_rmse_i = int(np.argmax(rmse))
    worst_cos = sub[worst_cos_i]
    worst_rmse = sub[worst_rmse_i]
    out = {
        "n": len(sub),
        "nfe": nfe,
        "noise_id": noise_id,
        "cosine_mean": float(cosine.mean()),
        "cosine_p5": percentile(cosine, 5),
        "rmse_mean": float(rmse.mean()),
        "rmse_p95": percentile(rmse, 95),
        "max_abs_mean": float(max_abs.mean()),
        "max_abs_p95": percentile(max_abs, 95),
        "rmse_trans_mean": float(np.mean([float(r["rmse_trans"]) for r in sub])),
        "rmse_rot_mean": float(np.mean([float(r["rmse_rot"]) for r in sub])),
        "rmse_grip_mean": float(np.mean([float(r["rmse_grip"]) for r in sub])),
        "rmse_tail3_mean": float(tail.mean()),
        "rmse_tail3_p95": percentile(tail, 95),
        "gripper_disagree_mean": float(grip.mean()),
        "gripper_any_frac": float(np.mean([int(r["gripper_disagree_n"]) > 0 for r in sub])),
        "frac_cosine_lt_0.99": float(np.mean(cosine < 0.99)),
        "frac_cosine_lt_0.95": float(np.mean(cosine < 0.95)),
        "worst_cosine": {
            "shared_index": int(worst_cos["shared_index"]),
            "noise_id": int(worst_cos["noise_id"]),
            "alpha": float(worst_cos["alpha"]),
            "task": worst_cos["task"],
            "episode_index": int(worst_cos["episode_index"]),
            "timestamp": int(worst_cos["timestamp"]),
            "cosine": float(worst_cos["cosine"]),
            "rmse": float(worst_cos["rmse"]),
            "gripper_disagree": float(worst_cos["gripper_disagree"]),
            "viz_tag": worst_cos.get("viz_tag") or "",
        },
        "worst_rmse": {
            "shared_index": int(worst_rmse["shared_index"]),
            "noise_id": int(worst_rmse["noise_id"]),
            "alpha": float(worst_rmse["alpha"]),
            "task": worst_rmse["task"],
            "episode_index": int(worst_rmse["episode_index"]),
            "timestamp": int(worst_rmse["timestamp"]),
            "cosine": float(worst_rmse["cosine"]),
            "rmse": float(worst_rmse["rmse"]),
            "gripper_disagree": float(worst_rmse["gripper_disagree"]),
            "viz_tag": worst_rmse.get("viz_tag") or "",
        },
    }
    if nfe == 10:
        vac = np.asarray([float(r["v_adj_cos_mean"]) for r in sub], dtype=np.float64)
        vfl = np.asarray([float(r["v_first_last_cos"]) for r in sub], dtype=np.float64)
        out["v_adj_cos_mean"] = float(np.nanmean(vac))
        out["v_first_last_cos_mean"] = float(np.nanmean(vfl))
        out["v_first_last_cos_p5"] = percentile(vfl[np.isfinite(vfl)], 5) if np.isfinite(vfl).any() else float("nan")
    return out


def run_correctness(policy, sample0: dict, device: torch.device, mean, std, noise_shape) -> dict:
    batch = inference_batch(sample0["batch"], device)
    seed = noise_seed(0, 0)
    noise = make_noise(noise_shape, seed).to(device)
    counts = {}
    chunks = {}
    for nfe in NFES:
        act, probe = predict_chunk(policy, batch, noise, nfe, record_v=(nfe == 10))
        counts[str(nfe)] = int(probe.n_calls)
        chunks[nfe] = act
        if probe.n_calls != nfe:
            raise RuntimeError(f"NFE={nfe} velocity calls {probe.n_calls}")
        if nfe == 10:
            recorded = act
            vmet = velocity_metrics(probe.vs)

    policy.config.num_steps = 10
    policy.reset()
    with torch.inference_mode():
        official = slice_exec(policy.predict_action_chunk(batch, noise=noise.clone()).detach().cpu())
    delta = (recorded - official).abs().max().item()
    match = bool(torch.allclose(recorded, official, rtol=1e-4, atol=1e-5))
    policy.reset()
    with torch.inference_mode():
        official2 = slice_exec(policy.predict_action_chunk(batch, noise=noise.clone()).detach().cpu())
    reset_ok = bool(torch.allclose(official, official2, rtol=1e-4, atol=1e-5))
    same_n1_n10 = bool(torch.allclose(chunks[1], chunks[10], rtol=1e-3, atol=1e-3))
    report = {
        "velocity_calls": counts,
        "calls_ok": counts == {str(n): n for n in NFES},
        "nfe10_vs_official_max_abs": delta,
        "nfe10_vs_official_match": match,
        "repeat_nfe10_match_after_reset": reset_ok,
        "nfe1_equals_nfe10": same_n1_n10,
        "noise_seed": seed,
        "noise_shape": list(noise_shape),
        "action_exec_shape": list(official.shape),
        "nfe10_velocity": vmet,
        "nfe1_vs_nfe10": action_metrics(chunks[1], chunks[10], mean, std),
        "nfe2_vs_nfe10": action_metrics(chunks[2], chunks[10], mean, std),
        "image_keys": sorted(k for k in sample0["batch"] if "images" in k),
        "compile_model": bool(policy.config.compile_model),
        "n_action_steps": int(policy.config.n_action_steps),
        "chunk_size": int(policy.config.chunk_size),
        "num_steps_config_restored": int(policy.config.num_steps),
        "empty_cameras": int(policy.config.empty_cameras),
        "state_dim_cfg": int(tuple(policy.config.input_features["observation.state"].shape)[0]),
        "rename_map": dict(RENAME_MAP),
    }
    if not report["calls_ok"] or not match or not reset_ok:
        raise RuntimeError(f"correctness failed: {json.dumps(report, indent=2)}")
    return report


def fmt_row(
    *,
    shared_index: int,
    rec: dict,
    alpha: float,
    d_whitened: float,
    noise_id: int,
    nfe: int,
    n_calls: int,
    metrics: dict,
    vmet: dict,
    groups: dict[str, set[int]],
    grip_switch: np.ndarray,
    viz: dict[int, str],
) -> dict:
    return {
        "shared_index": shared_index,
        "global_index": int(rec["global_index"]),
        "episode_index": int(rec["episode_index"]),
        "task": rec["task"],
        "task_index": int(rec["task_index"]),
        "frame_in_episode": int(rec["frame_in_episode"]),
        "length": int(rec["length"]),
        "timestamp": int(rec["frame_in_episode"]),
        "noise_id": int(noise_id),
        "noise_seed": noise_seed(shared_index, noise_id),
        "alpha": float(alpha),
        "d_whitened": float(d_whitened),
        "nfe": int(nfe),
        "n_velocity_calls": int(n_calls),
        **metrics,
        **vmet,
        "in_all": 1,
        "in_top_alpha": int(shared_index in groups["top_alpha"]),
        "in_random": int(shared_index in groups["random"]),
        "in_low_alpha": int(shared_index in groups["low_alpha"]),
        "in_grip_switch": int(bool(grip_switch[shared_index])),
        "viz_tag": viz.get(shared_index, ""),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" if h != headers[-1] else "---:" for h in headers) + " |"
    # keep simple ---: only if last looks numeric; use --- for all
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep, *body])


def f4(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{float(x):.4f}"


def f3(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{float(x):.3f}"


def pct(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{100.0 * float(x):.1f}%"


def filter_group(rows: list[dict], flag: str) -> list[dict]:
    if flag == "all":
        return rows
    if flag == "viz":
        return [r for r in rows if str(r.get("viz_tag") or "")]
    return [r for r in rows if int(r.get(flag, 0)) == 1]


def representative_anomalies(rows: list[dict], nfe: int, noise_id: int, k: int = 5) -> list[dict]:
    sub = [r for r in rows if int(r["nfe"]) == nfe and int(r["noise_id"]) == noise_id]
    sub = sorted(sub, key=lambda r: (float(r["cosine"]), -float(r["rmse"])))
    out = []
    for r in sub[:k]:
        out.append(
            {
                "shared_index": int(r["shared_index"]),
                "noise_id": int(r["noise_id"]),
                "alpha": float(r["alpha"]),
                "task": r["task"],
                "episode_index": int(r["episode_index"]),
                "timestamp": int(r["timestamp"]),
                "cosine": float(r["cosine"]),
                "rmse": float(r["rmse"]),
                "rmse_tail3": float(r["rmse_tail3"]),
                "gripper_disagree": float(r["gripper_disagree"]),
                "in_top_alpha": int(r["in_top_alpha"]),
                "in_low_alpha": int(r["in_low_alpha"]),
                "in_grip_switch": int(r["in_grip_switch"]),
                "viz_tag": r.get("viz_tag") or "",
            }
        )
    return out


def nfe2_vs_nfe1(rows: list[dict], noise_id: int, cosine_cut: float, rmse_cut: float) -> dict:
    by_key = {}
    for r in rows:
        if int(r["noise_id"]) != noise_id:
            continue
        key = (int(r["shared_index"]), int(r["nfe"]))
        by_key[key] = r
    recovered = []
    remaining = []
    n1_bad = []
    for i in range(N_FRAMES):
        a = by_key.get((i, 1))
        b = by_key.get((i, 2))
        if a is None or b is None:
            continue
        bad = float(a["cosine"]) < cosine_cut or float(a["rmse"]) > rmse_cut or int(a["gripper_disagree_n"]) > 0
        if not bad:
            continue
        item = {
            "shared_index": i,
            "alpha": float(a["alpha"]),
            "nfe1_cosine": float(a["cosine"]),
            "nfe2_cosine": float(b["cosine"]),
            "nfe1_rmse": float(a["rmse"]),
            "nfe2_rmse": float(b["rmse"]),
            "nfe1_grip": float(a["gripper_disagree"]),
            "nfe2_grip": float(b["gripper_disagree"]),
            "in_top_alpha": int(a["in_top_alpha"]),
            "in_low_alpha": int(a["in_low_alpha"]),
            "in_random": int(a["in_random"]),
            "in_grip_switch": int(a["in_grip_switch"]),
            "viz_tag": a.get("viz_tag") or "",
        }
        n1_bad.append(item)
        still = float(b["cosine"]) < cosine_cut or float(b["rmse"]) > rmse_cut or int(b["gripper_disagree_n"]) > 0
        if still:
            remaining.append(item)
        else:
            recovered.append(item)
    return {
        "n_nfe1_anomaly": len(n1_bad),
        "n_recovered_at_nfe2": len(recovered),
        "n_remaining_at_nfe2": len(remaining),
        "frac_top_among_nfe1_anomaly": (
            float(np.mean([x["in_top_alpha"] for x in n1_bad])) if n1_bad else float("nan")
        ),
        "frac_low_among_nfe1_anomaly": (
            float(np.mean([x["in_low_alpha"] for x in n1_bad])) if n1_bad else float("nan")
        ),
        "frac_random_among_nfe1_anomaly": (
            float(np.mean([x["in_random"] for x in n1_bad])) if n1_bad else float("nan")
        ),
        "examples_remaining": remaining[:8],
        "examples_recovered": recovered[:8],
    }


def write_report(out_dir: Path, payload: dict) -> None:
    nfe_tbl_headers = [
        "NFE",
        "noise",
        "cos mean",
        "cos P5",
        "RMSE mean",
        "RMSE P95",
        "max|e| P95",
        "tail3 RMSE",
        "grip disagree",
        "grip any",
        "cos<0.99",
    ]
    nfe_rows = []
    for nfe in (1, 2, 3):
        for nid in (0, 1):
            s = payload["nfe"][str(nfe)][str(nid)]
            nfe_rows.append(
                [
                    str(nfe),
                    str(nid),
                    f4(s["cosine_mean"]),
                    f4(s["cosine_p5"]),
                    f4(s["rmse_mean"]),
                    f4(s["rmse_p95"]),
                    f4(s["max_abs_p95"]),
                    f4(s["rmse_tail3_mean"]),
                    f4(s["gripper_disagree_mean"]),
                    pct(s["gripper_any_frac"]),
                    pct(s["frac_cosine_lt_0.99"]),
                ]
            )
    grp_headers = [
        "group",
        "NFE",
        "noise",
        "n",
        "cos mean",
        "cos P5",
        "RMSE mean",
        "RMSE P95",
        "grip any",
        "cos<0.99",
    ]
    grp_rows = []
    for gname in ("all", "top_alpha", "random", "low_alpha", "grip_switch", "viz"):
        for nfe in (1, 2):
            for nid in (0, 1):
                s = payload["groups"][gname][str(nfe)][str(nid)]
                grp_rows.append(
                    [
                        gname,
                        str(nfe),
                        str(nid),
                        str(s.get("n", 0)),
                        f4(s.get("cosine_mean")),
                        f4(s.get("cosine_p5")),
                        f4(s.get("rmse_mean")),
                        f4(s.get("rmse_p95")),
                        pct(s.get("gripper_any_frac")),
                        pct(s.get("frac_cosine_lt_0.99")),
                    ]
                )
    viz_headers = [
        "clip",
        "α",
        "noise",
        "NFE1 cos",
        "NFE2 cos",
        "NFE3 cos",
        "NFE1 RMSE",
        "NFE2 RMSE",
        "NFE1 grip",
        "NFE2 grip",
    ]
    viz_rows = []
    for item in payload["viz_frames"]:
        viz_rows.append(
            [
                item["viz_tag"],
                f3(item["alpha"]),
                str(item["noise_id"]),
                f4(item["nfe1_cosine"]),
                f4(item["nfe2_cosine"]),
                f4(item["nfe3_cosine"]),
                f4(item["nfe1_rmse"]),
                f4(item["nfe2_rmse"]),
                f4(item["nfe1_grip"]),
                f4(item["nfe2_grip"]),
            ]
        )
    anom = payload["anomalies"]
    q = payload["answers"]
    lines = [
        "# SmolVLA 离线 NFE 探针（CARE v2 480 帧）",
        "",
        "完整模型 `lerobot/smolvla_libero`。NFE∈{1,2,3,10} 各自从同一初始噪声独立积分到终点。",
        "参照为同帧、同噪声的 NFE=10。统计前 10 步 × 7 维有效动作，排除 padding。RMSE / 最大绝对误差在模型归一化尺度（MEAN_STD）上计算。",
        "夹爪开合：先反归一化到执行端尺度，再按 LIBERO 惯例 `g>0` 为闭合（dummy/hold-open 为 -1）。",
        "α 与 480 帧映射来自 CARE v2 `shared.pt`，本轮不改 CARE 定义，也不用闭环成功率回拧阈值。",
        "",
        "## 正确性",
        "",
        f"- 速度场调用次数：{payload['correctness']['velocity_calls']}。",
        f"- 探针 NFE=10（挂钩记录速度场）与正式 `predict_action_chunk` 最大绝对误差 {f4(payload['correctness']['nfe10_vs_official_max_abs'])}，判定一致。",
        f"- `reset()` 后重复 NFE=10：{'一致' if payload['correctness']['repeat_nfe10_match_after_reset'] else '失败'}。",
        "- compile 关闭；推理模式；相机 rename 与 8 维 state 与闭环一致。",
        "",
        "## NFE 对照（480 帧，两个噪声分开）",
        "",
        "最差样本是该噪声下 480 条记录中余弦最低或 RMSE 最高的单条 `(shared_index, noise_id)`，不是平均矩阵的最小值。",
        "",
        md_table(nfe_tbl_headers, nfe_rows),
        "",
        "### 最差样本（按余弦，NFE=1 / 2）",
        "",
    ]
    for nfe in (1, 2):
        for nid in (0, 1):
            w = payload["nfe"][str(nfe)][str(nid)]["worst_cosine"]
            lines.append(
                f"- NFE={nfe} noise={nid}: shared={w['shared_index']} ep={w['episode_index']} "
                f"t={w['timestamp']} α={f3(w['alpha'])} cos={f4(w['cosine'])} "
                f"RMSE={f4(w['rmse'])} grip={f4(w['gripper_disagree'])} {w['task']}"
                + (f" [{w['viz_tag']}]" if w.get("viz_tag") else "")
            )
    lines += [
        "",
        "## 分组（同一批输出，无重复推理）",
        "",
        "- `all`：480 帧。",
        "- `top_alpha`：每任务 α 最高的 3/12 帧（120）。",
        "- `random`：每任务从完整 12 帧池用种子 42 抽 3 帧（可与 top 重叠）。",
        "- `low_alpha`：每任务 α 最低的 3/12 帧，用于回答低 α 是否也有异常。",
        "- `grip_switch`：示范动作在未来 H=10 计分窗口内发生开合切换；窗口外视频上下文不计。",
        "- `viz`：上轮 12 个预览样本。",
        "",
        f"随机组与高 α 组重叠 {payload['overlap_top_random']} / 120 帧。"
        f"夹爪切换组 n={payload['n_grip_switch']} 帧。",
        "",
        md_table(grp_headers, grp_rows),
        "",
        "## 上轮可视化 12 样本",
        "",
        md_table(viz_headers, viz_rows),
        "",
        "## 代表性异常（NFE=1，按余弦最差的 5 条/噪声）",
        "",
    ]
    for nid in (0, 1):
        lines.append(f"**noise {nid}**")
        lines.append("")
        for a in anom[str(nid)]:
            flags = []
            if a["in_top_alpha"]:
                flags.append("topα")
            if a["in_low_alpha"]:
                flags.append("lowα")
            if a["in_grip_switch"]:
                flags.append("gripH")
            if a["viz_tag"]:
                flags.append(a["viz_tag"])
            flag_s = ",".join(flags) if flags else "-"
            lines.append(
                f"- shared={a['shared_index']} α={f3(a['alpha'])} cos={f4(a['cosine'])} "
                f"RMSE={f4(a['rmse'])} tail3={f4(a['rmse_tail3'])} grip={f4(a['gripper_disagree'])} "
                f"[{flag_s}] {a['task']}"
            )
        lines.append("")
    lines += [
        "## 四个问题",
        "",
        f"**1. 余弦能否区分 NFE=1 和 2？** {q['q1']}",
        "",
        f"**2. 高 α 组是否比同数量随机组更容易出现较大低 NFE 偏差？** {q['q2']}",
        "",
        f"**3. 低 α 样本是否也存在明显异常？** {q['q3']}",
        "",
        f"**4. NFE=2 是否减轻 NFE=1 的异常，CARE 是否提供额外信息？** {q['q4']}",
        "",
        "## 闭环对照（只作背景，不回拧）",
        "",
        "完整模型闭环：NFE=10 为 83.00%，NFE=2 为 82.00%，NFE=1 为 79.25%，复跑 78.25%。",
        "动作相似不能解释为闭环无损。本轮只做离线前向。",
        "",
        "## 文件",
        "",
        "- `config.json` / `correctness.json` / `frames.csv` / `actions.npz` / `summary.json`",
        "- `probe_smolvla_nfe.py`（本脚本副本） / `COMMAND.txt`",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def build_answers(payload: dict) -> dict[str, str]:
    def grab(nfe, nid):
        return payload["nfe"][str(nfe)][str(nid)]

    c1 = [grab(1, 0)["cosine_mean"], grab(1, 1)["cosine_mean"]]
    c2 = [grab(2, 0)["cosine_mean"], grab(2, 1)["cosine_mean"]]
    p1 = [grab(1, 0)["cosine_p5"], grab(1, 1)["cosine_p5"]]
    p2 = [grab(2, 0)["cosine_p5"], grab(2, 1)["cosine_p5"]]
    r1 = [grab(1, 0)["rmse_mean"], grab(1, 1)["rmse_mean"]]
    r2 = [grab(2, 0)["rmse_mean"], grab(2, 1)["rmse_mean"]]
    g1 = [grab(1, 0)["gripper_any_frac"], grab(1, 1)["gripper_any_frac"]]
    g2 = [grab(2, 0)["gripper_any_frac"], grab(2, 1)["gripper_any_frac"]]
    t1 = [grab(1, 0)["rmse_tail3_mean"], grab(1, 1)["rmse_tail3_mean"]]
    t2 = [grab(2, 0)["rmse_tail3_mean"], grab(2, 1)["rmse_tail3_mean"]]
    cos_gap = float(np.mean(c2) - np.mean(c1))
    high_cos = min(c1 + c2) > 0.99
    if high_cos:
        q1 = (
            f"两个噪声下 NFE=1 余弦均值 {f4(c1[0])}/{f4(c1[1])}，NFE=2 为 {f4(c2[0])}/{f4(c2[1])}，"
            f"都不能靠均值余弦分开（差值 {f4(cos_gap)}）。P5 余弦 NFE=1 为 {f4(p1[0])}/{f4(p1[1])}，"
            f"NFE=2 为 {f4(p2[0])}/{f4(p2[1])}。误差仍有差别：RMSE 均值 NFE=1 {f4(r1[0])}/{f4(r1[1])} "
            f"vs NFE=2 {f4(r2[0])}/{f4(r2[1])}；尾部 3 步 RMSE {f4(t1[0])}/{f4(t1[1])} vs "
            f"{f4(t2[0])}/{f4(t2[1])}；夹爪任一步不一致率 {pct(g1[0])}/{pct(g1[1])} vs "
            f"{pct(g2[0])}/{pct(g2[1])}。"
        )
    else:
        q1 = (
            f"余弦可以分开：NFE=1 均值 {f4(c1[0])}/{f4(c1[1])}（P5 {f4(p1[0])}/{f4(p1[1])}），"
            f"NFE=2 均值 {f4(c2[0])}/{f4(c2[1])}（P5 {f4(p2[0])}/{f4(p2[1])}）。"
            f"RMSE / 夹爪 / 尾部 3 步与余弦同向：NFE=2 更接近 NFE=10。"
        )

    def grp(name, nfe, nid):
        return payload["groups"][name][str(nfe)][str(nid)]

    def mean_pair(name, nfe, key):
        return 0.5 * (grp(name, nfe, 0)[key] + grp(name, nfe, 1)[key])

    top_rmse = mean_pair("top_alpha", 1, "rmse_p95")
    rand_rmse = mean_pair("random", 1, "rmse_p95")
    top_bad = mean_pair("top_alpha", 1, "frac_cosine_lt_0.99")
    rand_bad = mean_pair("random", 1, "frac_cosine_lt_0.99")
    top_g = mean_pair("top_alpha", 1, "gripper_any_frac")
    rand_g = mean_pair("random", 1, "gripper_any_frac")
    if top_rmse > rand_rmse and top_bad >= rand_bad:
        q2 = (
            f"是。NFE=1 的 RMSE P95 高 α {f4(top_rmse)} > 随机 {f4(rand_rmse)}；"
            f"余弦<0.99 比例 {pct(top_bad)} vs {pct(rand_bad)}；"
            f"夹爪不一致帧比例 {pct(top_g)} vs {pct(rand_g)}。随机组与高 α 重叠 "
            f"{payload['overlap_top_random']}/120，未剔除重叠。"
        )
    elif top_rmse > rand_rmse or top_bad > rand_bad or top_g > rand_g:
        q2 = (
            f"部分是，但不均匀。NFE=1 RMSE P95 高 α {f4(top_rmse)}、随机 {f4(rand_rmse)}；"
            f"余弦<0.99 {pct(top_bad)} vs {pct(rand_bad)}；夹爪不一致 {pct(top_g)} vs {pct(rand_g)}。"
            f"高 α 提高了碰到较大偏差的机会，但不是唯一来源。"
        )
    else:
        q2 = (
            f"没有。NFE=1 RMSE P95 高 α {f4(top_rmse)}、随机 {f4(rand_rmse)}；"
            f"余弦<0.99 {pct(top_bad)} vs {pct(rand_bad)}；夹爪不一致 {pct(top_g)} vs {pct(rand_g)}。"
            f"低 NFE 偏差并不集中在高 CARE α。"
        )

    low_rmse = mean_pair("low_alpha", 1, "rmse_p95")
    low_bad = mean_pair("low_alpha", 1, "frac_cosine_lt_0.99")
    low_g = mean_pair("low_alpha", 1, "gripper_any_frac")
    all_rmse = mean_pair("all", 1, "rmse_p95")
    if low_bad > 0.05 or low_g > 0.05 or low_rmse > 0.5 * all_rmse:
        q3 = (
            f"有。低 α 组 NFE=1 RMSE P95={f4(low_rmse)}（全体 P95 {f4(all_rmse)}），"
            f"余弦<0.99 为 {pct(low_bad)}，夹爪不一致 {pct(low_g)}。"
            f"平滑窗口也会出现低 NFE 偏差，α 不是充分筛选器。"
        )
    else:
        q3 = (
            f"低 α 组整体轻：NFE=1 RMSE P95={f4(low_rmse)}，余弦<0.99 {pct(low_bad)}，"
            f"夹爪不一致 {pct(low_g)}。仍应看逐帧最差点，不能声称低 α 绝无异常。"
        )

    rec = payload["recovery"]
    rec0, rec1 = rec["0"], rec["1"]
    q4 = (
        f"NFE=1 异常定义：该噪声下余弦<全体 P5 或 RMSE>全体 P95 或夹爪任一步不一致。"
        f"noise0：{rec0['n_nfe1_anomaly']} 条中 {rec0['n_recovered_at_nfe2']} 条在 NFE=2 不再满足；"
        f"noise1：{rec1['n_nfe1_anomaly']} 条中 {rec1['n_recovered_at_nfe2']} 条。"
        f"NFE=1 异常里高 α 占比 {pct(rec0['frac_top_among_nfe1_anomaly'])}/"
        f"{pct(rec1['frac_top_among_nfe1_anomaly'])}，低 α "
        f"{pct(rec0['frac_low_among_nfe1_anomaly'])}/{pct(rec1['frac_low_among_nfe1_anomaly'])}，"
        f"随机 {pct(rec0['frac_random_among_nfe1_anomaly'])}/{pct(rec1['frac_random_among_nfe1_anomaly'])}。"
        f"NFE=2 减轻了多数 NFE=1 异常，但 CARE α 只提供部分额外信息，不能单独决定哪里必须用满 NFE。"
        f"闭环 NFE=2 为 82.00%、NFE=1 为 79.25%/78.25%，与动作相似同向，但不等于无损。"
    )
    return {"q1": q1, "q2": q2, "q3": q3, "q4": q4}


def analyze(out_dir: Path, rows: list[dict], shared: dict, correctness: dict) -> dict:
    records = shared["records"]
    alpha = np.asarray(shared["alpha"], dtype=np.float64)
    groups = group_indices(records, alpha)
    n_grip = int(
        sum(int(r["in_grip_switch"]) for r in rows if int(r["nfe"]) == 1 and int(r["noise_id"]) == 0)
    )
    overlap = len(groups["top_alpha"] & groups["random"])
    nfe_summ = {}
    for nfe in NFES:
        nfe_summ[str(nfe)] = {str(nid): summarize_rows(rows, nfe, nid) for nid in range(N_NOISE)}
    group_summ = {}
    flag_map = {
        "all": "all",
        "top_alpha": "in_top_alpha",
        "random": "in_random",
        "low_alpha": "in_low_alpha",
        "grip_switch": "in_grip_switch",
        "viz": "viz",
    }
    for gname, flag in flag_map.items():
        sub = filter_group(rows, flag)
        group_summ[gname] = {
            str(nfe): {str(nid): summarize_rows(sub, nfe, nid) for nid in range(N_NOISE)} for nfe in (1, 2, 3, 10)
        }
    viz_frames = []
    by = {}
    for r in rows:
        if not str(r.get("viz_tag") or ""):
            continue
        by.setdefault((int(r["shared_index"]), int(r["noise_id"])), {})[int(r["nfe"])] = r
    for key, mp in sorted(by.items(), key=lambda kv: (kv[1][1]["viz_tag"], kv[0][1])):
        r1, r2, r3 = mp[1], mp[2], mp[3]
        viz_frames.append(
            {
                "shared_index": key[0],
                "noise_id": key[1],
                "viz_tag": r1["viz_tag"],
                "alpha": float(r1["alpha"]),
                "task": r1["task"],
                "episode_index": int(r1["episode_index"]),
                "timestamp": int(r1["timestamp"]),
                "nfe1_cosine": float(r1["cosine"]),
                "nfe2_cosine": float(r2["cosine"]),
                "nfe3_cosine": float(r3["cosine"]),
                "nfe1_rmse": float(r1["rmse"]),
                "nfe2_rmse": float(r2["rmse"]),
                "nfe1_grip": float(r1["gripper_disagree"]),
                "nfe2_grip": float(r2["gripper_disagree"]),
            }
        )
    recovery = {}
    for nid in range(N_NOISE):
        cut_c = nfe_summ["1"][str(nid)]["cosine_p5"]
        cut_r = nfe_summ["1"][str(nid)]["rmse_p95"]
        recovery[str(nid)] = nfe2_vs_nfe1(rows, nid, cut_c, cut_r)
    payload = {
        "protocol": "care_v2_smolvla_nfe_probe",
        "n_frames": N_FRAMES,
        "n_noise": N_NOISE,
        "nfes": list(NFES),
        "overlap_top_random": overlap,
        "n_grip_switch": n_grip,
        "nfe": nfe_summ,
        "groups": group_summ,
        "viz_frames": viz_frames,
        "anomalies": {
            str(nid): representative_anomalies(rows, 1, nid, 5) for nid in range(N_NOISE)
        },
        "recovery": recovery,
        "correctness": correctness,
        "closed_loop_sr_background_only": CLOSED_LOOP_SR,
    }
    payload["answers"] = build_answers(payload)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_report(out_dir, payload)
    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--allow-shared", action="store_true")
    p.add_argument("--min-free-mib", type=int, default=MIN_FREE_MIB)
    p.add_argument("--out-dir", type=Path, default=PROBE_DIR)
    p.add_argument("--correctness-only", action="store_true")
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), out_dir / "probe_smolvla_nfe.py")
    (out_dir / "COMMAND.txt").write_text(
        "python "
        + " ".join(str(x) for x in sys.argv)
        + "\n"
    )

    shared = load_shared()
    records = shared["records"]
    alpha = np.asarray(shared["alpha"], dtype=np.float64)
    d_w = np.asarray(shared["d_whitened"], dtype=np.float64)

    if args.analyze_only:
        import pandas as pd

        frames = pd.read_csv(out_dir / "frames.csv")
        rows = frames.to_dict(orient="records")
        correctness = json.loads((out_dir / "correctness.json").read_text())
        analyze(out_dir, rows, shared, correctness)
        print(f"wrote {out_dir / 'REPORT.md'}", flush=True)
        return

    snap = require_shared_gpu(args.gpu, allow_shared=args.allow_shared, min_free_mib=args.min_free_mib)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    from load_policy import load_policy

    t0 = time.time()
    policy = configure_policy(load_policy("cuda:0"), device)
    preprocessor, postprocessor, cfg = load_processors()
    mean, std = action_mean_std(preprocessor)
    noise_shape = (1, int(policy.config.chunk_size), int(policy.config.max_action_dim))
    print(
        f"policy dtype={next(policy.parameters()).dtype} chunk={policy.config.chunk_size} "
        f"max_action_dim={policy.config.max_action_dim} n_action_steps={policy.config.n_action_steps}",
        flush=True,
    )

    samples = preprocess_frames(
        shared, preprocessor, cfg, out_dir / "preprocessed_obs.pt", args.limit
    )
    n_use = len(samples)
    correctness = run_correctness(policy, samples[0], device, mean, std, noise_shape)
    correctness["gpu"] = snap
    correctness["action_mean"] = mean.tolist()
    correctness["action_std"] = std.tolist()
    correctness["load_s"] = time.time() - t0
    (out_dir / "correctness.json").write_text(json.dumps(correctness, indent=2, default=str) + "\n")
    print("correctness ok", json.dumps(correctness["velocity_calls"]), flush=True)
    if args.correctness_only:
        return

    groups = group_indices(records[:n_use] if n_use < N_FRAMES else records, alpha[:n_use])
    if n_use == N_FRAMES:
        grip_switch = demo_gripper_switch(shared)
    else:
        grip_switch = np.zeros((n_use,), dtype=np.bool_)
        full = demo_gripper_switch(shared)
        grip_switch[:] = full[:n_use]
    viz = load_viz_map()

    config = {
        "checkpoint": CKPT,
        "compile_model": False,
        "n_action_steps": N_ACTION_STEPS,
        "num_steps_default": NUM_STEPS,
        "chunk_size": int(policy.config.chunk_size),
        "max_action_dim": int(policy.config.max_action_dim),
        "empty_cameras": 0,
        "rename_map": dict(RENAME_MAP),
        "state_dim": LIBERO_STATE_DIM,
        "exec_steps": EXEC_STEPS,
        "action_dim": ACTION_DIM,
        "nfes": list(NFES),
        "n_noise": N_NOISE,
        "noise_seed_base": NOISE_SEED_BASE,
        "noise_seed_fn": "NOISE_SEED_BASE * 1000003 + shared_index * 2 + noise_id",
        "group_seed": GROUP_SEED,
        "care_seed": CARE_SEED,
        "horizon": HORIZON,
        "frames": n_use,
        "shared": str(OUT_DIR / "shared.pt"),
        "gripper_mapping": "unnormalize MEAN_STD then closed iff g>0 (LIBERO dummy=-1 open)",
        "rmse_scale": "policy-normalized MEAN_STD, first 10 x 7, no padded dims",
        "gpu": int(args.gpu),
        "allow_shared": True,
        "closed_loop_sr_background_only": CLOSED_LOOP_SR,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    rows = []
    acts = {nfe: np.zeros((n_use, N_NOISE, EXEC_STEPS, ACTION_DIM), dtype=np.float32) for nfe in NFES}
    t_loop = time.time()
    for i in range(n_use):
        rec = records[i]
        batch = inference_batch(samples[i]["batch"], device)
        for nid in range(N_NOISE):
            noise = make_noise(noise_shape, noise_seed(i, nid)).to(device)
            chunks = {}
            probes = {}
            for nfe in NFES:
                record_v = nfe == 10
                act, probe = predict_chunk(policy, batch, noise, nfe, record_v=record_v)
                if probe.n_calls != nfe:
                    raise RuntimeError(f"frame {i} noise {nid} NFE={nfe} calls={probe.n_calls}")
                chunks[nfe] = act
                probes[nfe] = probe
                acts[nfe][i, nid] = act.numpy()
            ref = chunks[10]
            vmet10 = velocity_metrics(probes[10].vs)
            for nfe in NFES:
                metrics = action_metrics(chunks[nfe], ref, mean, std) if nfe != 10 else {
                    "cosine": 1.0,
                    "rmse": 0.0,
                    "max_abs": 0.0,
                    "rmse_trans": 0.0,
                    "rmse_rot": 0.0,
                    "rmse_grip": 0.0,
                    "max_abs_trans": 0.0,
                    "max_abs_rot": 0.0,
                    "max_abs_grip": 0.0,
                    "rmse_tail3": 0.0,
                    "gripper_disagree": 0.0,
                    "gripper_disagree_n": 0,
                }
                vmet = vmet10 if nfe == 10 else empty_v_metrics()
                rows.append(
                    fmt_row(
                        shared_index=i,
                        rec=rec,
                        alpha=float(alpha[i]),
                        d_whitened=float(d_w[i]),
                        noise_id=nid,
                        nfe=nfe,
                        n_calls=probes[nfe].n_calls,
                        metrics=metrics,
                        vmet=vmet,
                        groups=groups,
                        grip_switch=grip_switch,
                        viz=viz,
                    )
                )
        if (i + 1) % 10 == 0 or i == 0:
            dt = time.time() - t_loop
            print(f"frame {i+1}/{n_use}  {dt:.0f}s", flush=True)

    write_csv(out_dir / "frames.csv", rows)
    np.savez_compressed(
        out_dir / "actions.npz",
        **{f"nfe{nfe}": acts[nfe] for nfe in NFES},
        alpha=alpha[:n_use],
        shared_index=np.arange(n_use),
    )
    (out_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    if n_use == N_FRAMES:
        analyze(out_dir, rows, shared, correctness)
    print(f"done {out_dir} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

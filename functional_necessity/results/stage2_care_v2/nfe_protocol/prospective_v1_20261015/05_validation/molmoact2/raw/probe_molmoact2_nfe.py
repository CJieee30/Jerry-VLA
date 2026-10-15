#!/usr/bin/env python3
"""MolmoAct2 offline NFE probe on the frozen CARE v2 480-frame set.

Full LeRobot checkpoint (local). NFE in {1,2,3,10}; same observation + generator seed
across NFEs. Hooks action_expert.forward_with_context for velocity counts.
Does not change CARE α or overwrite other stacks' probes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
MOLMO_LIB = HERE.parents[0] / "molmoact2_libero"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MOLMO_LIB))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dump_shared import _load_actions
from protocol import ACTION_DIM, HORIZON, LIBERO_REPO, MOLMO_REPO, MOLMO_ROOT, N_FRAMES, OUT_DIR, hub_snapshot, molmo_parquet_rels

PROBE_DIR = OUT_DIR / "nfe_probe_molmoact2"
NFES = (1, 2, 3, 10)
N_NOISE = 2
EXEC_STEPS = 10
NOISE_SEED_BASE = 1000
GROUP_SEED = 42
MIN_FREE_MIB = 12000
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
    "noise_sha256",
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
    return {"total_mib": int(tot), "used_mib": int(used), "free_mib": int(free), "util": int(util)}


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
            args = subprocess.check_output(["ps", "-o", "args=", "-p", str(pid)], text=True).strip()
        except subprocess.CalledProcessError:
            user, args = "?", ""
        rows.append({"pid": pid, "user": user, "name": name, "mem_mib": mem, "args": args})
    return rows


def require_shared_gpu(gpu: int, *, allow_shared: bool, min_free_mib: int, ignore_eval_drop: bool = False) -> dict[str, Any]:
    if not allow_shared:
        raise SystemExit("Molmo probe requires --allow-shared on a shared GPU")
    occ = _gpu_occupancy(gpu)
    mem = _gpu_memory(gpu)
    evals = [r for r in occ if r.get("user") == "guorj" and "eval_drop.py" in str(r.get("args") or "")]
    if evals and not ignore_eval_drop:
        desc = ", ".join(f"pid={e['pid']}" for e in evals)
        raise SystemExit(f"GPU{gpu} runs eval_drop ({desc}); pick another GPU")
    if int(mem["free_mib"]) < int(min_free_mib):
        raise SystemExit(f"GPU{gpu} free {mem['free_mib']} MiB < {min_free_mib} MiB")
    others = ", ".join(f"{r['user']}:{r['pid']}" for r in occ) or "idle"
    print(f"GPU{gpu} ok free={mem['free_mib']}MiB occupancy=({others})", flush=True)
    return {"gpu": gpu, "memory": mem, "occupancy": occ}


def noise_seed(shared_index: int, noise_id: int) -> int:
    return int(NOISE_SEED_BASE) * 1_000_003 + int(shared_index) * N_NOISE + int(noise_id)


def shared_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def slice_exec(chunk: torch.Tensor) -> torch.Tensor:
    if chunk.ndim == 3:
        chunk = chunk[0]
    return chunk[:EXEC_STEPS, :ACTION_DIM].float().contiguous()


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    fa = F.normalize(_flat(a).float()[None], dim=-1)
    fb = F.normalize(_flat(b).float()[None], dim=-1)
    return float((fa * fb).sum().item())


def _rmse(err: torch.Tensor) -> float:
    return float(torch.sqrt((err.float() ** 2).mean()).item())


def _max_abs(err: torch.Tensor) -> float:
    return float(err.float().abs().max().item())


def load_action_stats():
    from safetensors.torch import load_file
    from load_policy import LEROBOT_CKPT

    path = next(LEROBOT_CKPT.glob("policy_postprocessor_step_1_*.safetensors"))
    d = load_file(path)
    mean = d["action.mean"].float().flatten()[:ACTION_DIM].clone()
    std = d["action.std"].float().flatten()[:ACTION_DIM].clone()
    q01 = d["action.q01"].float().flatten()[:ACTION_DIM].clone()
    q99 = d["action.q99"].float().flatten()[:ACTION_DIM].clone()
    mask = d["action.mask"].float().flatten()[:ACTION_DIM].clone()
    std = torch.clamp(std, min=1e-8)
    return mean, std, q01, q99, mask


def to_libero_physical(normed: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Invert Molmo QUANTILES on masked dims; grip (mask=0) passes through."""
    x = normed.float()
    span = (q99 - q01).clamp(min=1e-8)
    out = x.clone()
    m = mask > 0.5
    out[..., m] = (x[..., m] + 1.0) * 0.5 * span[m] + q01[m]
    if (~m).any():
        out[..., ~m] = x[..., ~m]
    return out


def gripper_closed_physical(phys: torch.Tensor) -> torch.Tensor:
    return phys[..., 6] > 0.0


def action_metrics(pred: torch.Tensor, ref: torch.Tensor, q01, q99, mask) -> dict[str, float]:
    err = pred - ref
    trans, rot, grip = err[:, :3], err[:, 3:6], err[:, 6:7]
    pu = to_libero_physical(pred, q01, q99, mask)
    ru = to_libero_physical(ref, q01, q99, mask)
    disagree = gripper_closed_physical(pu) != gripper_closed_physical(ru)
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


def velocity_metrics(vs: list[torch.Tensor]) -> dict[str, float]:
    toks = [slice_exec(v) for v in vs]
    adj_cos = [_cos(toks[i], toks[i + 1]) for i in range(len(toks) - 1)]
    norms = [float(t.norm().item()) for t in toks]
    adj_rel = [norms[i + 1] / (norms[i] + 1e-12) - 1.0 for i in range(len(norms) - 1)]
    return {
        "v_adj_cos_mean": float(np.mean(adj_cos)),
        "v_adj_cos_min": float(np.min(adj_cos)),
        "v_first_last_cos": _cos(toks[0], toks[-1]),
        "v_adj_norm_rel_mean": float(np.mean(np.abs(adj_rel))),
        "v_first_last_norm_rel": norms[-1] / (norms[0] + 1e-12) - 1.0,
    }


def empty_v_metrics() -> dict[str, float]:
    return {k: float("nan") for k in ("v_adj_cos_mean", "v_adj_cos_min", "v_first_last_cos", "v_adj_norm_rel_mean", "v_first_last_norm_rel")}


def load_shared(path: Path | None = None) -> dict:
    path = path or (OUT_DIR / "shared.pt")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    n = int(blob.get("n", len(blob["records"])))
    if path.resolve() == (OUT_DIR / "shared.pt").resolve():
        if n != N_FRAMES:
            raise SystemExit(f"shared n={n}")
        if abs(float(blob["alpha"].mean()) - 1.0) > 1e-4:
            raise SystemExit(f"alpha mean {float(blob['alpha'].mean())}")
    return blob


def local_global_indices() -> set[int]:
    snap = hub_snapshot(MOLMO_REPO)
    cols = ["index", "episode_index", "frame_index", "task_index", "action"]
    idx: set[int] = set()
    for f in sorted((snap / "data").glob("*/*.parquet")):
        df = pd.read_parquet(f, columns=["index"])
        idx.update(int(x) for x in df["index"].to_numpy())
    return idx


def filter_records(shared: dict, *, local_only: bool, indices: list[int] | None) -> tuple[list[dict], list[int]]:
    records = list(shared["records"])
    pool = list(range(len(records)))
    if indices is not None:
        pool = [int(i) for i in indices]
    if local_only:
        local = local_global_indices()
        pool = [i for i in pool if int(records[i]["global_index"]) in local]
    if not pool:
        raise SystemExit("no records after local/indices filter")
    sub_records = [records[i] for i in pool]
    return sub_records, pool


def _rel_index(dataset, global_index: int) -> int:
    mapping = getattr(dataset, "absolute_to_relative_idx", None) or getattr(getattr(dataset, "reader", None), "absolute_to_relative_idx", None)
    if mapping:
        return int(mapping[int(global_index)])
    return int(global_index)


def load_molmo_stack(device: str):
    import lerobot.policies.molmoact2.modeling_molmoact2 as pol
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors_from_pretrained
    from load_policy import HF_CONFIG, LEROBOT_CKPT, OUT_DIR as MOLMO_OUT, install_loader_patches, load_policy

    install_loader_patches()
    policy = load_policy(device)
    cfg = pol.MolmoAct2Config.from_pretrained(MOLMO_OUT / "policy_config")
    preprocessor, postprocessor = make_molmoact2_pre_post_processors_from_pretrained(
        cfg, str(LEROBOT_CKPT), preprocessor_overrides={"device_processor": {"device": "cpu"}}
    )
    return policy, preprocessor, postprocessor, cfg


def dataset_root() -> Path:
    if (MOLMO_ROOT / "READY").is_file():
        return MOLMO_ROOT
    return hub_snapshot(MOLMO_REPO)


def preprocess_frames(
    shared: dict,
    pool: list[int],
    preprocessor,
    dest: Path,
    meta: dict,
) -> list[dict]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    records = shared["records"]
    n_total = len(records)
    if dest.is_file() and meta.get("limit") is None:
        blob = torch.load(dest, map_location="cpu", weights_only=False)
        ok = int(blob.get("n", 0)) == len(pool) and blob.get("meta") == meta
        if ok:
            print(f"[cache] {dest}", flush=True)
            return blob["samples"]
        print(f"[cache stale] {dest}", flush=True)

    eps = sorted({int(records[i]["episode_index"]) for i in pool})
    root = dataset_root()
    dataset = LeRobotDataset(
        MOLMO_REPO,
        root=str(root),
        episodes=eps,
        revision="main",
        download_videos=False,
    )
    mapping = dataset.absolute_to_relative_idx or {}
    missing = [int(records[i]["global_index"]) for i in pool if int(records[i]["global_index"]) not in mapping]
    if missing:
        raise SystemExit(
            f"{len(missing)} global_index not in dataset root {root} (e.g. {missing[:5]}); run fetch_molmo.py"
        )

    samples = []
    for j, si in enumerate(pool):
        rec = records[si]
        gidx = int(rec["global_index"])
        frame = dataset[_rel_index(dataset, gidx)]
        batch = preprocessor(frame)
        samples.append({"shared_index": si, "global_index": gidx, "batch": _cpu_tensors(batch)})
        if (j + 1) % 20 == 0 or j == 0:
            print(f"preprocess {j+1}/{len(pool)}", flush=True)

    if meta.get("limit") is None and len(pool) == n_total:
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"n": len(samples), "samples": samples, "meta": meta}, dest)
        print(f"wrote {dest}", flush=True)
    return samples


def _cpu_tensors(batch: dict) -> dict:
    return {k: (v.detach().cpu().contiguous() if torch.is_tensor(v) else v) for k, v in batch.items()}


def inference_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if key in INFER_DROP:
            continue
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(int(seed))
    return gen


def noise_reference_tensor(policy, seed: int, device: torch.device) -> torch.Tensor:
    horizon = int(policy._generation_action_horizon())
    max_dim = int(policy._backbone().config.max_action_dim)
    dtype = policy._action_expert().action_embed.weight.dtype
    gen = make_generator(device, seed)
    return torch.randn(1, horizon, max_dim, device=device, dtype=dtype, generator=gen)


def noise_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()


def predict_chunk(policy, batch: dict, seed: int, nfe: int, device: torch.device, *, record_v: bool):
    expert = policy._action_expert()
    orig = expert.forward_with_context
    probe = DenoiseProbe(orig, record=record_v)
    policy.reset()
    expert.forward_with_context = probe
    gen = make_generator(device, seed)
    try:
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch, num_steps=int(nfe), generator=gen)
    finally:
        expert.forward_with_context = orig
    return slice_exec(chunk.detach().cpu()), probe


def demo_gripper_switch(shared: dict, pool: list[int]) -> np.ndarray:
    frames = _load_actions(LIBERO_REPO)
    actions = np.stack(frames["action"].to_numpy()).astype(np.float64)
    episode = frames["episode_index"].to_numpy()
    flags = np.zeros((len(pool),), dtype=np.bool_)
    for j, si in enumerate(pool):
        rec = shared["records"][si]
        start = int(rec["global_index"])
        ep = int(episode[start])
        seq = []
        for t in range(start, min(start + HORIZON, len(actions))):
            if int(episode[t]) != ep:
                break
            seq.append(float(actions[t, 6]))
        if len(seq) >= 2:
            bits = [g > 0.0 for g in seq]
            flags[j] = any(bits[a] != bits[a + 1] for a in range(len(bits) - 1))
    return flags


def group_indices(pool: list[int], records: list[dict], alpha: np.ndarray) -> dict[str, set[int]]:
    by_task: dict[int, list[int]] = defaultdict(list)
    for si in pool:
        by_task[int(records[si]["task_index"])].append(si)
    top, low, rand = [], [], []
    rng = np.random.default_rng(GROUP_SEED)
    for ti in sorted(by_task):
        idxs = by_task[ti]
        ordered = sorted(idxs, key=lambda j: (-float(alpha[j]), int(records[j]["global_index"])))
        k = min(3, len(ordered))
        top.extend(ordered[:k])
        low.extend(ordered[-k:] if k else [])
        pick = rng.choice(np.asarray(sorted(idxs)), size=k, replace=False)
        rand.extend(int(x) for x in pick)
    return {"all": set(pool), "top_alpha": set(top), "random": set(rand), "low_alpha": set(low)}


def run_correctness(policy, sample0: dict, device: torch.device, q01, q99, mask) -> dict:
    batch = inference_batch(sample0["batch"], device)
    seed = noise_seed(int(sample0["shared_index"]), 0)
    ref_noise = noise_reference_tensor(policy, seed, device)
    nhash = noise_sha256(ref_noise)
    counts, chunks = {}, {}
    vmet = {}
    for nfe in NFES:
        act, probe = predict_chunk(policy, batch, seed, nfe, device, record_v=(nfe == 10))
        counts[str(nfe)] = int(probe.n_calls)
        chunks[nfe] = act
        if probe.n_calls != nfe:
            raise RuntimeError(f"NFE={nfe} calls={probe.n_calls}")
        if nfe == 10:
            recorded = act
            vmet = velocity_metrics(probe.vs)

    policy.reset()
    gen = make_generator(device, seed)
    with torch.inference_mode():
        official = slice_exec(policy.predict_action_chunk(batch, num_steps=10, generator=gen).detach().cpu())
    delta = float((recorded - official).abs().max())
    match = bool(torch.allclose(recorded, official, rtol=2e-3, atol=2e-4))

    policy.reset()
    gen2 = make_generator(device, seed)
    with torch.inference_mode():
        official2 = slice_exec(policy.predict_action_chunk(batch, num_steps=10, generator=gen2).detach().cpu())
    reset_ok = bool(torch.allclose(official, official2, rtol=2e-3, atol=2e-4))

    same_seed_n1 = predict_chunk(policy, batch, seed, 1, device, record_v=False)[0]
    same_seed_n1_b = predict_chunk(policy, batch, seed, 1, device, record_v=False)[0]
    noise_repeat = bool(torch.allclose(same_seed_n1, same_seed_n1_b, rtol=1e-4, atol=1e-5))

    dtype_name = str(policy._action_expert().action_embed.weight.dtype)
    report = {
        "velocity_calls": counts,
        "calls_ok": counts == {str(n): n for n in NFES},
        "nfe10_vs_official_max_abs": delta,
        "nfe10_vs_official_match": match,
        "repeat_nfe10_match_after_reset": reset_ok,
        "repeat_nfe1_same_seed": noise_repeat,
        "noise_seed": seed,
        "noise_sha256_fp32": nhash,
        "noise_reference_shape": list(ref_noise.shape),
        "noise_reference_dtype": dtype_name,
        "action_exec_shape": list(official.shape),
        "nfe10_velocity": vmet,
        "nfe1_vs_nfe10": action_metrics(chunks[1], chunks[10], q01, q99, mask),
        "nfe2_vs_nfe10": action_metrics(chunks[2], chunks[10], q01, q99, mask),
        "model_dtype": dtype_name,
        "enable_inference_cuda_graph": False,
        "normalization": "QUANTILES q01/q99 on arm; grip mask=0",
        "gripper_rule": "libero physical g>0 after Molmo inverse quantile on arm",
    }
    if not report["calls_ok"] or not match or not reset_ok or not noise_repeat:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def fmt_row(**kwargs) -> dict:
    return {k: kwargs.get(k, "") for k in CSV_FIELDS}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--allow-shared", action="store_true")
    p.add_argument("--ignore-eval-drop", action="store_true", help="Allow eval_drop on same GPU if memory suffices")
    p.add_argument("--min-free-mib", type=int, default=MIN_FREE_MIB)
    p.add_argument("--out-dir", type=Path, default=PROBE_DIR)
    p.add_argument("--correctness-only", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--local-shards-only", action="store_true")
    p.add_argument("--indices", default=None, help="comma shared_index list")
    p.add_argument("--shared-path", type=Path, default=None)
    p.add_argument("--skip-analyze", action="store_true")
    p.add_argument("--nfes", default=None)
    return p.parse_args()


def main() -> None:
    global NFES
    args = parse_args()
    if args.nfes:
        NFES = tuple(sorted({int(x) for x in args.nfes.split(",")}))
        if 10 not in NFES:
            raise SystemExit("--nfes must include 10")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), out_dir / "probe_molmoact2_nfe.py")
    (out_dir / "COMMAND.txt").write_text("python " + " ".join(str(x) for x in sys.argv) + "\n")

    shared_path = args.shared_path or (OUT_DIR / "shared.pt")
    shared = load_shared(shared_path)
    indices = None if args.indices is None else [int(x) for x in args.indices.split(",")]
    sub_records, pool = filter_records(shared, local_only=args.local_shards_only, indices=indices)
    if args.limit is not None:
        pool = pool[: int(args.limit)]
        sub_records = [shared["records"][i] for i in pool]

    alpha = np.asarray(shared["alpha"], dtype=np.float64).reshape(-1)
    d_w = np.asarray(shared["d_whitened"], dtype=np.float64).reshape(-1)
    meta = {
        "checkpoint": "MolmoAct2-LIBERO-LeRobot",
        "shared_sha256": shared_fingerprint(shared_path),
        "pool": pool,
        "limit": args.limit,
        "local_shards_only": args.local_shards_only,
    }

    if args.correctness_only or not args.skip_analyze:
        pass  # may still run full loop below

    snap = require_shared_gpu(
        args.gpu,
        allow_shared=args.allow_shared,
        min_free_mib=args.min_free_mib,
        ignore_eval_drop=args.ignore_eval_drop,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    t0 = time.time()
    policy, preprocessor, _post, _cfg = load_molmo_stack("cuda:0")
    mean, std, q01, q99, mask = load_action_stats()
    samples = preprocess_frames(shared, pool, preprocessor, out_dir / "preprocessed_obs.pt", meta)
    correctness = run_correctness(policy, samples[0], device, q01, q99, mask)
    correctness["gpu"] = snap
    correctness["load_s"] = time.time() - t0
    correctness["libero_action_mean"] = mean.tolist()
    correctness["libero_action_std"] = std.tolist()
    (out_dir / "correctness.json").write_text(json.dumps(correctness, indent=2) + "\n")
    print("correctness ok", correctness["velocity_calls"], flush=True)
    if args.correctness_only:
        return

    groups = group_indices(pool, shared["records"], alpha)
    grip_switch = demo_gripper_switch(shared, pool)
    n_use = len(pool)
    acts = {nfe: np.zeros((n_use, N_NOISE, EXEC_STEPS, ACTION_DIM), dtype=np.float32) for nfe in NFES}
    rows = []
    noise_hashes: dict[str, str] = {}

    config = {
        "checkpoint": "MolmoAct2-LIBERO-LeRobot",
        "nfes": list(NFES),
        "n_noise": N_NOISE,
        "noise_seed_fn": "NOISE_SEED_BASE * 1000003 + shared_index * 2 + noise_id",
        "noise_sha256": "fp32 bytes of reference randn(1,H,max_dim) with same generator seed",
        "exec_steps": EXEC_STEPS,
        "frames": n_use,
        "pool_shared_index": pool,
        "shared": str(shared_path.resolve()),
        "dataset_root": str(dataset_root()),
        "gpu": int(args.gpu),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    t_loop = time.time()
    for j, si in enumerate(pool):
        rec = shared["records"][si]
        batch = inference_batch(samples[j]["batch"], device)
        for nid in range(N_NOISE):
            seed = noise_seed(si, nid)
            nhash = noise_sha256(noise_reference_tensor(policy, seed, device))
            noise_hashes[f"{si}_{nid}"] = nhash
            chunks = {}
            probes = {}
            for nfe in NFES:
                act, probe = predict_chunk(policy, batch, seed, nfe, device, record_v=(nfe == 10))
                if probe.n_calls != nfe:
                    raise RuntimeError(f"si={si} nfe={nfe} calls={probe.n_calls}")
                chunks[nfe] = act
                probes[nfe] = probe
                acts[nfe][j, nid] = act.numpy()
            ref = chunks[10]
            vmet10 = velocity_metrics(probes[10].vs)
            for nfe in NFES:
                metrics = (
                    {
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
                    if nfe == 10
                    else action_metrics(chunks[nfe], ref, q01, q99, mask)
                )
                vmet = vmet10 if nfe == 10 else empty_v_metrics()
                rows.append(
                    fmt_row(
                        shared_index=si,
                        global_index=int(rec["global_index"]),
                        episode_index=int(rec["episode_index"]),
                        task=rec["task"],
                        task_index=int(rec["task_index"]),
                        frame_in_episode=int(rec["frame_in_episode"]),
                        length=int(rec["length"]),
                        timestamp=int(rec["frame_in_episode"]),
                        noise_id=nid,
                        noise_seed=seed,
                        noise_sha256=nhash,
                        alpha=float(alpha[si]),
                        d_whitened=float(d_w[si]),
                        nfe=nfe,
                        n_velocity_calls=probes[nfe].n_calls,
                        in_all=1,
                        in_top_alpha=int(si in groups["top_alpha"]),
                        in_random=int(si in groups["random"]),
                        in_low_alpha=int(si in groups["low_alpha"]),
                        in_grip_switch=int(bool(grip_switch[j])),
                        viz_tag="",
                        **metrics,
                        **vmet,
                    )
                )
        if (j + 1) % 5 == 0 or j == 0:
            print(f"frame {j+1}/{n_use} {time.time()-t_loop:.0f}s", flush=True)

    write_csv(out_dir / "frames.csv", rows)
    np.savez_compressed(
        out_dir / "actions.npz",
        **{f"nfe{nfe}": acts[nfe] for nfe in NFES},
        alpha=alpha[pool],
        shared_index=np.asarray(pool, dtype=np.int64),
    )
    (out_dir / "noise_hashes.json").write_text(json.dumps(noise_hashes, indent=2) + "\n")
    (out_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ") + "\n")
    print(f"done {out_dir} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU interface checks for a future MolmoAct2 fixed-noise NFE probe (no model load, no inference).

1. Molmo parquet rows at CARE dev indices (local shards only) match lerobot/libero:
   index / episode / frame, demo action windows, in-window gripper switch G.
2. Checkpoint facts the probe depends on: chunk, normalization, gripper mask, dtype,
   CUDA-graph / per-episode-seed defaults.
3. Source facts: predict_action_chunk honours num_steps and generator; noise drawn once.
4. Shard coverage of dev pool, dev holdout and the draft prospective split.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pi05_libero"))

from dump_shared import _episodes, _load_actions  # noqa: E402
from event_care_nfe_diagnostic import demo_gripper_switch  # noqa: E402
from protocol import FN_ROOT, HORIZON, LIBERO_REPO, MOLMO_REPO, OUT_DIR, hub_snapshot  # noqa: E402

CKPT = FN_ROOT / "checkpoints" / "MolmoAct2-LIBERO-LeRobot"
HF_CFG = FN_ROOT / "checkpoints" / "MolmoAct2-LIBERO" / "config.json"
OUT = OUT_DIR / "nfe_protocol" / "molmo_interface_check"


def local_molmo_frames(snap: Path) -> pd.DataFrame:
    files = sorted((snap / "data").glob("*/*.parquet"))
    cols = ["index", "episode_index", "frame_index", "task_index", "action"]
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files]).sort_values("index").set_index("index")


def check_rows(name: str, records: list[dict], molmo: pd.DataFrame, lib_actions: np.ndarray, lib_episode: np.ndarray) -> dict:
    local = [r for r in records if int(r["global_index"]) in molmo.index]
    max_diff, mismatches = 0.0, []
    for r in local:
        g = int(r["global_index"])
        row = molmo.loc[g]
        if int(row["episode_index"]) != int(r["episode_index"]) or int(row["frame_index"]) != int(r["frame_in_episode"]):
            mismatches.append(g)
            continue
        for t in range(g, g + HORIZON):
            if t not in molmo.index or int(lib_episode[t]) != int(r["episode_index"]):
                break
            max_diff = max(max_diff, float(np.abs(np.asarray(molmo.loc[t]["action"]) - lib_actions[t]).max()))
    # G from Molmo demo actions vs lerobot/libero demo actions on the same windows
    m_idx = molmo.index.to_numpy()
    m_act = np.stack(molmo["action"].to_numpy()).astype(np.float64)
    m_ep = molmo["episode_index"].to_numpy()
    pos = {int(g): i for i, g in enumerate(m_idx)}
    sub = [{"global_index": pos[int(r["global_index"])]} for r in local]
    G_m = demo_gripper_switch({"records": sub}, m_act, m_ep) if sub else np.zeros(0, bool)
    G_l = demo_gripper_switch({"records": local}, lib_actions, lib_episode) if local else np.zeros(0, bool)
    grip_vals = np.unique(np.round(m_act[:, 6], 4))
    return {
        "set": name,
        "records": len(records),
        "records_with_local_shard": len(local),
        "index_episode_frame_mismatch": len(mismatches),
        "max_abs_action_diff_vs_libero": max_diff,
        "G_equal": bool(np.array_equal(G_m, G_l)),
        "G1_local": int(G_l.sum()),
        "molmo_gripper_values_local": grip_vals.tolist()[:10],
    }


def coverage(name: str, records: list[dict], meps: pd.DataFrame, local_files: set[tuple[int, int]]) -> dict:
    key = {int(r["episode_index"]): (int(r["data/chunk_index"]), int(r["data/file_index"])) for _, r in meps.iterrows()}
    need = {key[int(r["episode_index"])] for r in records}
    return {"set": name, "episodes": len({int(r["episode_index"]) for r in records}), "molmo_files": len(need),
            "missing_local": len(need - local_files)}


def _randn_uses_generator(src: str) -> bool:
    i = src.find("trajectory = torch.randn(")
    return i >= 0 and "generator=generator" in src[i : i + 400]


def source_facts() -> dict:
    import lerobot

    root = Path(lerobot.__file__).resolve().parent / "policies" / "molmoact2"
    outer = (root / "modeling_molmoact2.py").read_text()
    inner = (root / "molmoact2_hf_model" / "modeling_molmoact2.py").read_text()
    m = re.search(r"def predict_action_chunk\(.*?\n    @torch.no_grad\(\)", outer, re.S)
    body = m.group(0) if m else ""
    return {
        "lerobot_molmoact2_dir": str(root),
        "predict_action_chunk_reads_num_steps": 'kwargs.get("num_steps"' in body,
        "predict_action_chunk_reads_generator": 'kwargs.get("generator")' in body,
        "outer_noise_randn_with_generator": _randn_uses_generator(outer),
        "inner_noise_randn_with_generator": _randn_uses_generator(inner),
        "inner_noise_dtype": "action_expert.action_embed.weight.dtype" if "trajectory_dtype = action_expert.action_embed.weight.dtype" in inner else "unknown",
        "inner_uses_action_cuda_graph_when_enabled": "action_cuda_graph_manager.run_action_flow" in inner,
        "velocity_hook_target": "action_expert.forward_with_context (eager only; CUDA-graph replay bypasses hooks)",
        "rollout_generator_when_per_episode_seed": "_rollout_generator_for_inputs" in body,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snap = hub_snapshot(MOLMO_REPO)
    meps = _episodes(MOLMO_REPO)
    leps = _episodes(LIBERO_REPO)
    ranges_equal = bool(np.array_equal(meps["dataset_from_index"].values, leps["dataset_from_index"].values))
    local_files = {(int(p.parent.name.split("-")[1]), int(p.stem.split("-")[1])) for p in (snap / "data").glob("*/*.parquet")}

    frames = _load_actions(LIBERO_REPO)
    lib_actions = np.stack(frames["action"].to_numpy()).astype(np.float64)
    lib_episode = frames["episode_index"].to_numpy()
    molmo = local_molmo_frames(snap)

    pool = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)["records"]
    hold = torch.load(OUT_DIR / "nfe_probe_care_event_holdout" / "holdout_shared.pt", map_location="cpu", weights_only=False)["records"]
    sets = {"dev_pool_480": pool, "dev_holdout_120": hold}
    draft = OUT_DIR / "nfe_protocol" / "prospective_v0_DRAFT" / "01_split"
    if (draft / "calibration_pool.json").is_file():
        sets["draft_calibration_400"] = json.loads((draft / "calibration_pool.json").read_text())
        sets["draft_validation_480"] = json.loads((draft / "validation_set.json").read_text())

    raw = json.loads((CKPT / "config.json").read_text())
    hf = json.loads(HF_CFG.read_text())
    from safetensors.torch import load_file

    stats = load_file(next(CKPT.glob("policy_postprocessor_step_1_*.safetensors")))
    ckpt = {
        "n_action_steps": raw.get("n_action_steps"),
        "chunk_size": raw.get("chunk_size"),
        "action_mode": raw.get("action_mode"),
        "inference_action_mode": raw.get("inference_action_mode"),
        "normalization_mapping": raw.get("normalization_mapping"),
        "action_norm_mask": stats["action.mask"].tolist(),
        "action_q01": stats["action.q01"].tolist(),
        "action_q99": stats["action.q99"].tolist(),
        "model_dtype": raw.get("model_dtype"),
        "enable_inference_cuda_graph_ckpt": raw.get("enable_inference_cuda_graph"),
        "per_episode_seed": raw.get("per_episode_seed"),
        "rtc_config": raw.get("rtc_config"),
        "hf_flow_matching_num_steps": hf.get("flow_matching_num_steps"),
        "hf_max_action_dim": hf.get("max_action_dim"),
        "hf_max_action_horizon": hf.get("max_action_horizon"),
    }
    out = {
        "label": "CPU interface check on existing dev data; no model loaded",
        "episode_ranges_equal": ranges_equal,
        "local_molmo_files": len(local_files),
        "rows": [check_rows(k, v, molmo, lib_actions, lib_episode) for k, v in sets.items()],
        "coverage": [coverage(k, v, meps, local_files) for k, v in sets.items()],
        "checkpoint": ckpt,
        "source": source_facts(),
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

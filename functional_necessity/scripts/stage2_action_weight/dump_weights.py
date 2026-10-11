#!/usr/bin/env python3
"""Dump Stage-2 action-discontinuity weights. CPU only. Does not score layers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
FN_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FN_ROOT / "scripts" / "molmoact2_libero"))

import torch

from continuity import (
    ACTION_DIM,
    ARM_DIM,
    HORIZON,
    chunk_discontinuity,
    fill_and_mean_normalize,
    spearman,
)

OUT_DIR = FN_ROOT / "results" / "stage2_action_weight"
PI05_CALIB = FN_ROOT / "results" / "calib" / "pi05_v044_libero_64x8_seed42.pt"
SMOL_CALIB = FN_ROOT / "results" / "smolvla_libero" / "calib" / "calib_32_seed42.pt"
MOLMO_CALIB = FN_ROOT / "results" / "molmoact2_libero" / "calib" / "calib_32_seed42.pt"


def _quant_line(stats: dict) -> str:
    q = stats["quantiles_raw"]
    return (
        f"n={stats['n']} filled={stats['n_filled']} ESS={stats['ess']:.2f} "
        f"cv={stats['cv']:.3f} raw p0/25/50/75/90/100="
        + "/".join(f"{v:.4f}" for v in q)
    )


def _from_frames(actions: list[torch.Tensor], pads: list[torch.Tensor | None]):
    raw7, raw6, n_steps = [], [], []
    for action, pad in zip(actions, pads):
        v7, n = chunk_discontinuity(action, pad, dims=ACTION_DIM)
        v6, _ = chunk_discontinuity(action, pad, dims=ARM_DIM)
        raw7.append(v7)
        raw6.append(v6)
        n_steps.append(n)
    w7, s7 = fill_and_mean_normalize(raw7)
    w6, s6 = fill_and_mean_normalize(raw6)
    return {
        "raw7": raw7,
        "raw6": raw6,
        "n_steps": n_steps,
        "weights7": w7,
        "weights6": w6,
        "stats7": s7,
        "stats6": s6,
        "spearman_7_vs_6": spearman(w7, w6),
    }


def dump_pi05() -> dict:
    blob = torch.load(PI05_CALIB, map_location="cpu", weights_only=False)
    batches = blob["calib_data"]
    actions, pads = [], []
    for batch in batches:
        a = batch["action"]
        p = batch.get("action_is_pad")
        bsz = int(a.shape[0])
        for i in range(bsz):
            actions.append(a[i])
            pads.append(None if p is None else p[i])
    packed = _from_frames(actions, pads)
    n_batch, bsz = len(batches), int(batches[0]["action"].shape[0])
    packed["layout"] = "orig_batch"
    packed["n_orig_batches"] = n_batch
    packed["batch_size"] = bsz
    packed["weights7_batched"] = packed["weights7"].reshape(n_batch, bsz)
    packed["weights6_batched"] = packed["weights6"].reshape(n_batch, bsz)
    packed["tensor"] = "50-step normalized chunk already stored with the loss; window = first 10 valid steps"
    return packed


def dump_smol() -> dict:
    blob = torch.load(SMOL_CALIB, map_location="cpu", weights_only=False)
    actions, pads, indices = [], [], []
    for sample in blob["samples"]:
        actions.append(sample["batch"]["action"][0])
        pads.append(sample["batch"].get("action_is_pad", torch.zeros(sample["batch"]["action"].shape[1], dtype=torch.bool))[0])
        indices.append(int(sample["index"]))
    packed = _from_frames(actions, pads)
    packed["layout"] = "sample"
    packed["indices"] = indices
    packed["tensor"] = "50-step normalized chunk already stored with the loss; window = first 10 valid steps"
    return packed


def _molmo_config_and_preprocessor():
    import json
    from load_policy import HF_CONFIG, LEROBOT_CKPT, OUT_DIR, install_loader_patches

    install_loader_patches()
    import lerobot.policies.molmoact2.modeling_molmoact2 as pol
    from lerobot.policies.molmoact2.processor_molmoact2 import (
        make_molmoact2_pre_post_processors_from_pretrained,
    )

    cfg = pol.MolmoAct2Config.from_pretrained(OUT_DIR / "policy_config")
    cfg.device = "cpu"
    cfg.checkpoint_path = str(HF_CONFIG)
    preprocessor, _ = make_molmoact2_pre_post_processors_from_pretrained(
        cfg,
        str(LEROBOT_CKPT),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return cfg, preprocessor


def _molmo_normalize_action(preprocessor, frame) -> torch.Tensor:
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    skip = {"MolmoAct2PackInputsProcessorStep", "DeviceProcessorStep"}
    tr = batch_to_transition(frame)
    for step in preprocessor.steps:
        if type(step).__name__ in skip:
            continue
        tr = step(tr)
    out = transition_to_batch(tr)
    action = out["action"].detach().cpu().float().reshape(-1)[:ACTION_DIM]
    return action


def dump_molmo() -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    blob = torch.load(MOLMO_CALIB, map_location="cpu", weights_only=False)
    _, preprocessor = _molmo_config_and_preprocessor()
    dataset = LeRobotDataset(
        "allenai/MolmoAct2-LIBERO-Dataset",
        episodes=list(range(32)),
        revision="main",
        download_videos=False,
    )
    episodes = dataset.meta.episodes
    from_idx = [int(x) for x in episodes["dataset_from_index"]]
    to_idx = [int(x) for x in episodes["dataset_to_index"]]

    actions, pads, meta, first_diffs = [], [], [], []
    for sample in blob["samples"]:
        index = int(sample["index"])
        ep = int(sample["batch"]["episode_index"])
        stored = sample["batch"]["action"][0, 0, :ACTION_DIM].float().cpu()
        start = index
        end = min(start + HORIZON, to_idx[ep])
        if start < from_idx[ep] or start >= to_idx[ep]:
            raise RuntimeError(f"calib index {index} outside episode {ep} [{from_idx[ep]}, {to_idx[ep]})")
        window = []
        for t in range(start, end):
            window.append(_molmo_normalize_action(preprocessor, dataset[t]))
        window_t = torch.stack(window) if window else torch.zeros(0, ACTION_DIM)
        if window_t.shape[0] >= 1:
            first_diffs.append(float((window_t[0] - stored).abs().max()))
        else:
            first_diffs.append(float("nan"))
        actions.append(window_t)
        pads.append(None)
        meta.append(
            {
                "index": index,
                "episode_index": ep,
                "n_available": int(window_t.shape[0]),
                "episode_end": to_idx[ep],
            }
        )
    packed = _from_frames(actions, pads)
    packed["layout"] = "sample"
    packed["tensor"] = (
        "loss keeps the stored 1-step (1,1,32) action; weights use the same "
        "preprocessor (masked normalizer + clamp, no PackInputs) on the next "
        f"{HORIZON} frames of the same episode"
    )
    packed["first_step_max_abs_diff"] = first_diffs
    packed["first_step_max_abs_diff_max"] = max(first_diffs)
    packed["meta"] = meta
    if packed["first_step_max_abs_diff_max"] > 1e-5:
        raise RuntimeError(
            f"Molmo subsequent-action normalization does not match stored loss action: "
            f"max abs diff {packed['first_step_max_abs_diff_max']}"
        )
    return packed


def _save_stack(name: str, packed: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"weights_{name}.pt"
    payload = {
        "stack": name,
        "horizon": HORIZON,
        "action_dim": ACTION_DIM,
        "arm_dim": ARM_DIM,
        "definition": "mean L2 of adjacent normalized actions over first H valid steps",
        "layout": packed["layout"],
        "tensor": packed["tensor"],
        "weights7": torch.tensor(packed["weights7"], dtype=torch.float32),
        "weights6": torch.tensor(packed["weights6"], dtype=torch.float32),
        "n_steps": torch.tensor(packed["n_steps"], dtype=torch.int64),
        "stats7": packed["stats7"],
        "stats6": packed["stats6"],
        "spearman_7_vs_6": packed["spearman_7_vs_6"],
        "use_7d_on_gpu": bool(packed["spearman_7_vs_6"] >= 0.9),
    }
    if "weights7_batched" in packed:
        payload["weights7_batched"] = torch.tensor(packed["weights7_batched"], dtype=torch.float32)
        payload["weights6_batched"] = torch.tensor(packed["weights6_batched"], dtype=torch.float32)
        payload["n_orig_batches"] = packed["n_orig_batches"]
        payload["batch_size"] = packed["batch_size"]
    if "indices" in packed:
        payload["indices"] = packed["indices"]
    if "meta" in packed:
        payload["meta"] = packed["meta"]
        payload["first_step_max_abs_diff_max"] = packed["first_step_max_abs_diff_max"]
    torch.save(payload, path)
    summary = {
        "stack": name,
        "horizon": HORIZON,
        "tensor": packed["tensor"],
        "stats7": packed["stats7"],
        "stats6": packed["stats6"],
        "spearman_7_vs_6": packed["spearman_7_vs_6"],
        "use_7d_on_gpu": payload["use_7d_on_gpu"],
    }
    if "first_step_max_abs_diff_max" in packed:
        summary["first_step_max_abs_diff_max"] = packed["first_step_max_abs_diff_max"]
    (OUT_DIR / f"weights_{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return path


def main() -> None:
    reports = []
    print("dumping π0.5 weights from stored 50-step chunks", flush=True)
    pi05 = dump_pi05()
    _save_stack("pi05", pi05)
    reports.append(("pi05", pi05))
    print("  7d", _quant_line(pi05["stats7"]), "spearman7vs6", f"{pi05['spearman_7_vs_6']:.4f}", flush=True)

    print("dumping SmolVLA weights from stored 50-step chunks", flush=True)
    smol = dump_smol()
    _save_stack("smolvla", smol)
    reports.append(("smolvla", smol))
    print("  7d", _quant_line(smol["stats7"]), "spearman7vs6", f"{smol['spearman_7_vs_6']:.4f}", flush=True)

    print("dumping Molmo weights from same-episode subsequent normalized actions", flush=True)
    molmo = dump_molmo()
    _save_stack("molmoact2", molmo)
    reports.append(("molmoact2", molmo))
    print(
        "  7d",
        _quant_line(molmo["stats7"]),
        "spearman7vs6",
        f"{molmo['spearman_7_vs_6']:.4f}",
        "first-step max abs diff",
        f"{molmo['first_step_max_abs_diff_max']:.2e}",
        flush=True,
    )

    lines = [
        "# Stage-2 calibration weights (action discontinuity)",
        "",
        "Definition: mean adjacent L2 on **loss-normalized** actions, window H=10.",
        "Weights are then divided by their mean so a uniform field is exactly 1.",
        "GPU scoring uses the 7-d weights if Spearman(7d, arm-6d) ≥ 0.9.",
        "",
        "| Stack | Tensor | n | ESS (7d) | Spearman 7d vs 6d | GPU |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name, packed in reports:
        gpu = "7d" if packed["spearman_7_vs_6"] >= 0.9 else "7d and 6d"
        lines.append(
            f"| {name} | {packed['tensor']} | {packed['stats7']['n']} | "
            f"{packed['stats7']['ess']:.2f} | {packed['spearman_7_vs_6']:.4f} | {gpu} |"
        )
    lines += ["", "## Per-stack 7-d raw discontinuity"]
    for name, packed in reports:
        lines += ["", f"### {name}", "", _quant_line(packed["stats7"]), "", packed["tensor"]]
        if name == "molmoact2":
            lines.append(
                f"First subsequent step vs stored loss action: max abs diff "
                f"{packed['first_step_max_abs_diff_max']:.3e} (must match the same normalizer)."
            )
            n_short = sum(m["n_available"] < 2 for m in packed["meta"])
            lines.append(f"Windows with <2 steps (median-filled): {n_short}/{len(packed['meta'])}.")
    (OUT_DIR / "weights.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_DIR / 'weights.md'}", flush=True)


if __name__ == "__main__":
    main()

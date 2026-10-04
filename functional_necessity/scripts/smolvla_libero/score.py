#!/usr/bin/env python3
"""Nine redundancy scores for lerobot/smolvla_libero VLM layers.

Greedy-PPL is not included. The loss is the continuous flow-matching action
loss with one frozen (noise, time) pair per sample. A layer's score uses that
layer's weights or its residual-stream boundary. The paired action expert is
not ranked; it still reads each VLM layer's K/V.

Formulas match MolmoAct2 / pi0.5. Architecture-only changes: SmolVLA keep-KV
skip, residual hooks on input_layernorm (DecoderLayer.forward is unused), and
Beta(1.5, 1) flow time from the SmolVLA sampler.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from layer_skip import skip_vlm_layers
from load_policy import (
    CALIB_PATH,
    CKPT,
    DATASET_REPO,
    METHODS,
    OUT_DIR,
    load_policy,
    load_preprocessor,
    vlm_blocks,
)

N_SAMPLES = 32
SEED = 42
KS = (4, 8)


def importance_ranking(scores, k: int) -> dict:
    values = [float(x) for x in scores]
    n = len(values)
    keep_order = sorted(range(n), key=lambda i: values[i], reverse=True)
    drop = sorted(keep_order[n - k :])
    return {"importance": values, "redundancy_order": list(reversed(keep_order)), "drop": drop}


def _mean_flow(policy, samples, device) -> torch.Tensor:
    total = None
    for sample in samples:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample["batch"].items()}
        noise = sample["noise"].to(device)
        time = sample["time"].to(device)
        loss, _ = policy.forward(batch, noise=noise, time=time)
        total = loss if total is None else total + loss
    return total / len(samples)


def _ensure_policy_batch(batch: dict) -> dict:
    """Training collate adds the action-chunk batch dim; the processor does not."""
    out = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        tensor = value
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        out[key] = tensor
    action = out["action"]
    if action.ndim == 2:
        out["action"] = action.unsqueeze(0)
        pad = out.get("action_is_pad")
        if torch.is_tensor(pad) and pad.ndim == 1:
            out["action_is_pad"] = pad.unsqueeze(0)
    return out


def _frozen_noise_time(
    action: torch.Tensor,
    generator: torch.Generator,
    action_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = tuple(action.shape)
    if action_dim is not None:
        shape = (*shape[:-1], int(action_dim))
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        generator=generator,
    )
    # Same law as VLAFlowMatching.sample_time: Beta(1.5, 1) * 0.999 + 0.001.
    # Gamma mix so the draw can use the same torch.Generator as the noise.
    z = torch.normal(0.0, 1.0, size=(action.shape[0],), generator=generator)
    u1 = torch.rand(action.shape[0], generator=generator).clamp_min(1e-12)
    u2 = torch.rand(action.shape[0], generator=generator).clamp_min(1e-12)
    gamma_a = z.square() * 0.5 + (-torch.log(u1))
    gamma_b = -torch.log(u2)
    time = (gamma_a / (gamma_a + gamma_b)) * 0.999 + 0.001
    return noise, time.to(dtype=torch.float32)


def build_calib(policy) -> list[dict]:
    if CALIB_PATH.is_file():
        blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
        print(f"calib exists n={len(blob['samples'])}", flush=True)
        return blob["samples"]
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("building a 32-frame LIBERO calibration slice for SmolVLA", flush=True)
    ds_meta = LeRobotDatasetMetadata(DATASET_REPO)
    delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
    dataset = LeRobotDataset(
        DATASET_REPO,
        delta_timestamps=delta_timestamps,
        download_videos=True,
        video_backend="pyav",
        return_uint8=False,
    )
    step = max(1, len(dataset) // N_SAMPLES)
    indices = list(range(0, len(dataset), step))[:N_SAMPLES]
    preprocessor = load_preprocessor("cpu")
    samples = []
    generator = torch.Generator().manual_seed(SEED)
    for index in indices:
        frame = dataset[index]
        batch = _ensure_policy_batch(preprocessor(frame))
        action = batch["action"]
        noise, time = _frozen_noise_time(
            action, generator, action_dim=int(policy.config.max_action_dim)
        )
        cpu_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                cpu_batch[key] = value.detach().cpu()
        samples.append(
            {
                "batch": cpu_batch,
                "noise": noise.cpu(),
                "time": time.cpu(),
                "index": int(index),
            }
        )
        print(
            f"kept frame {index} action={tuple(action.shape)} ({len(samples)}/{N_SAMPLES})",
            flush=True,
        )
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "seed": SEED,
            "n": len(samples),
            "checkpoint": CKPT,
            "dataset": DATASET_REPO,
            "samples": samples,
        },
        CALIB_PATH,
    )
    return samples


def _hooks(blocks, final_norm):
    """Residual boundaries: DecoderLayer.forward is unused, so hook RMSNorm."""
    states = {}
    handles = []

    def make(idx):
        def hook(module, args):
            if args:
                states[idx] = args[0]

        return hook

    for i, block in enumerate(blocks):
        handles.append(block.input_layernorm.register_forward_pre_hook(make(i)))
    handles.append(final_norm.register_forward_pre_hook(make(len(blocks))))
    return states, handles


def _set_vlm_grad(blocks, enabled: bool) -> None:
    for block in blocks:
        for param in block.parameters():
            param.requires_grad_(enabled)


def _chunk_dots(grad, param, mode: str) -> float:
    g = grad.detach().reshape(-1)
    w = param.detach().reshape(-1)
    acc = torch.zeros((), device=g.device, dtype=torch.float32)
    step = 262_144
    for start in range(0, g.numel(), step):
        gf = g[start : start + step].float()
        wf = w[start : start + step].float()
        if mode == "fisher":
            acc = acc + (gf.square() * wf.square()).sum()
        else:
            acc = acc + gf.square().sum()
    return float(acc.item())


def _backward(policy, blocks, sample, device, kind: str, acc):
    for block in blocks:
        for param in block.parameters():
            param.grad = None
    loss = _mean_flow(policy, [sample], device)
    loss.backward()
    for i, block in enumerate(blocks):
        for name, param in block.named_parameters():
            if param.grad is None:
                continue
            if kind == "taylor":
                g = param.grad.detach().reshape(-1)
                w = param.detach().reshape(-1)
                dest = acc[i].get(name)
                if dest is None:
                    dest = torch.zeros(param.shape, dtype=torch.float32)
                    acc[i][name] = dest
                flat = dest.view(-1)
                step = 1_048_576
                for start in range(0, g.numel(), step):
                    flat[start : start + step] += (
                        g[start : start + step].float() * w[start : start + step].float()
                    ).cpu()
            else:
                acc[i] += _chunk_dots(param.grad, param, kind)
            param.grad = None
    return float(loss.detach())


def measure_magnitude(blocks) -> list[float]:
    return [sum(p.detach().float().abs().sum().item() for p in block.parameters()) for block in blocks]


def measure_representation(policy, blocks, final_norm, samples, device, method: str) -> list[float]:
    if method == "cossim":
        acc = [0.0] * len(blocks)
        for sample in tqdm(samples, desc="cossim"):
            states, handles = _hooks(blocks, final_norm)
            try:
                with torch.no_grad():
                    _mean_flow(policy, [sample], device)
                for i in range(len(blocks)):
                    sim = F.cosine_similarity(states[i].float(), states[i + 1].float(), dim=-1)
                    acc[i] += float(sim.mean())
            finally:
                for handle in handles:
                    handle.remove()
        return [1.0 - a / len(samples) for a in acc]
    n_tok = [0.0] * len(blocks)
    sum_in, sum_out, xtx, yty, ytx = ([None] * len(blocks) for _ in range(5))
    for sample in tqdm(samples, desc="cka"):
        states, handles = _hooks(blocks, final_norm)
        try:
            with torch.no_grad():
                _mean_flow(policy, [sample], device)
            for i in range(len(blocks)):
                x = states[i].detach().float().reshape(-1, states[i].shape[-1]).cpu()
                y = states[i + 1].detach().float().reshape(-1, states[i + 1].shape[-1]).cpu()
                n_tok[i] += x.shape[0]
                if sum_in[i] is None:
                    dim = x.shape[1]
                    sum_in[i] = torch.zeros(dim)
                    sum_out[i] = torch.zeros(dim)
                    xtx[i] = torch.zeros(dim, dim)
                    yty[i] = torch.zeros(dim, dim)
                    ytx[i] = torch.zeros(dim, dim)
                sum_in[i] += x.sum(0)
                sum_out[i] += y.sum(0)
                xtx[i] += x.T @ x
                yty[i] += y.T @ y
                ytx[i] += y.T @ x
        finally:
            for handle in handles:
                handle.remove()
    scores = []
    for i in range(len(blocks)):
        n = n_tok[i]
        mean_x, mean_y = sum_in[i] / n, sum_out[i] / n
        xtx_c = xtx[i] - n * torch.outer(mean_x, mean_x)
        yty_c = yty[i] - n * torch.outer(mean_y, mean_y)
        ytx_c = ytx[i] - n * torch.outer(mean_y, mean_x)
        denom = xtx_c.norm() * yty_c.norm()
        cka = float((ytx_c * ytx_c).sum() / denom) if float(denom) > 0 else 0.0
        scores.append(1.0 - cka)
    return scores


def _scoring_mode(policy, blocks, checkpoint: bool) -> None:
    policy.config.gradient_checkpointing = checkpoint
    policy.train()
    for block in blocks:
        if hasattr(block, "dropout"):
            block.dropout.p = 0.0


def measure_ppl(policy, blocks, samples, device) -> list[float]:
    with torch.no_grad():
        baseline = float(_mean_flow(policy, samples, device))
    print(f"PPL baseline {baseline:.6f}", flush=True)
    deltas = []
    for i in tqdm(range(len(blocks)), desc="ppl"):
        with torch.no_grad(), skip_vlm_layers([i]):
            dropped = float(_mean_flow(policy, samples, device))
        deltas.append(dropped - baseline)
        print(f"  L{i:>2} delta={deltas[-1]:+.6f}", flush=True)
    return deltas


def measure_grad(policy, blocks, samples, device, method: str) -> list[float]:
    _set_vlm_grad(blocks, True)
    _scoring_mode(policy, blocks, checkpoint=method != "hessian")
    if method == "hessian":
        return _hessian(policy, blocks, samples, device)
    acc = [{} for _ in blocks] if method == "taylor" else [0.0] * len(blocks)
    for sample in tqdm(samples, desc=method):
        _backward(policy, blocks, sample, device, "igia" if method == "igia" else method, acc)
    if method == "taylor":
        scores = [sum(t.abs().sum().item() for t in bucket.values()) / len(samples) for bucket in acc]
    elif method == "fisher":
        scores = [v / len(samples) for v in acc]
    else:
        scores = list(acc)
    return scores


def _hessian(policy, blocks, samples, device) -> list[float]:
    traces = [0.0] * len(blocks)

    def hvp(loss, params) -> float:
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
        paired = [(p, g) for p, g in zip(params, grads) if g is not None]
        if not paired:
            return 0.0
        probes = [(torch.randint(0, 2, p.shape, device=p.device) * 2 - 1).to(p.dtype) for p, _ in paired]
        gvp = sum((g * z).sum() for (_, g), z in zip(paired, probes))
        hvps = torch.autograd.grad(gvp, [p for p, _ in paired], retain_graph=False, allow_unused=True)
        total = 0.0
        for z, h in zip(probes, hvps):
            if h is not None:
                total += float((z.float() * h.float()).sum())
        return total

    lm_head = getattr(getattr(policy, "model", None), "lm_head", None)
    if lm_head is not None:
        lm_head.to("cpu")
    for param in policy.parameters():
        param.requires_grad_(False)
    policy.config.gradient_checkpointing = True
    policy.train()
    for block in blocks:
        if hasattr(block, "dropout"):
            block.dropout.p = 0.0
    for sample in tqdm(samples, desc="hessian"):
        for i, block in enumerate(blocks):
            params = list(block.parameters())
            for param in params:
                param.requires_grad_(True)
            torch.cuda.empty_cache()
            with sdpa_kernel(SDPBackend.MATH):
                loss = _mean_flow(policy, [sample], device)
                traces[i] += hvp(loss, params)
            for param in params:
                param.requires_grad_(False)
                param.grad = None
            del loss
    return [abs(t / len(samples)) for t in traces]


def measure_gate(policy, blocks, final_norm, samples, device) -> list[float]:
    _set_vlm_grad(blocks, True)
    _scoring_mode(policy, blocks, checkpoint=True)
    rows = []
    for sample in tqdm(samples, desc="gateprobe"):
        states, handles = _hooks(blocks, final_norm)
        loss = _mean_flow(policy, [sample], device)
        for hidden in states.values():
            if hidden.requires_grad:
                hidden.retain_grad()
        loss.backward()
        row = []
        for i in range(len(blocks)):
            h_in, h_out = states.get(i), states.get(i + 1)
            if h_in is None or h_out is None or h_out.grad is None:
                row.append(0.0)
            else:
                row.append(float((h_out.grad.float() * (h_out.float() - h_in.float())).sum()))
        rows.append(row)
        for handle in handles:
            handle.remove()
        for block in blocks:
            for param in block.parameters():
                param.grad = None
    return torch.tensor(rows).abs().mean(0).tolist()


def main() -> None:
    policy = load_policy("cuda:0")
    blocks, final_norm = vlm_blocks(policy)
    print(f"vlm layers {len(blocks)} expert={policy.model.vlm_with_expert.num_expert_layers}", flush=True)
    samples = build_calib(policy)
    with torch.no_grad(), skip_vlm_layers([0]):
        probe = _mean_flow(policy, [samples[0]], torch.device("cuda:0"))
    print(f"keep-KV skip probe loss {float(probe):.6f}", flush=True)
    score_dir = OUT_DIR / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    saved = {}
    for method in METHODS:
        path = score_dir / f"{method}.pt"
        if path.is_file():
            saved[method] = torch.load(path, map_location="cpu", weights_only=False)["scores"]
            print(f"skip existing {method}", flush=True)
            continue
        print(f"==== {method} ====", flush=True)
        try:
            if method == "magnitude":
                scores = measure_magnitude(blocks)
            elif method in {"cossim", "cka"}:
                scores = measure_representation(policy, blocks, final_norm, samples, device, method)
            elif method == "ppl":
                scores = measure_ppl(policy, blocks, samples, device)
            elif method == "gateprobe":
                scores = measure_gate(policy, blocks, final_norm, samples, device)
            else:
                scores = measure_grad(policy, blocks, samples, device, method)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            (score_dir / f"{method}.oom.txt").write_text(str(exc) + "\n")
            print(f"OOM {method}", flush=True)
            continue
        ranks = {f"drop{k}": importance_ranking(scores, k)["drop"] for k in KS}
        ranks["redundancy_order"] = importance_ranking(scores, 1)["redundancy_order"]
        torch.save(
            {"method": method, "scores": scores, **ranks, "n_samples": len(samples)},
            path,
        )
        saved[method] = scores
        print(f"{method} drop4={ranks['drop4']} drop8={ranks['drop8']}", flush=True)
        torch.cuda.empty_cache()
    lines = [
        "# SmolVLA LIBERO redundancy scores",
        "",
        f"Checkpoint: `{CKPT}`.",
        "VLM layers only (16). A dropped layer keeps K/V for the paired action expert and does not update the residual.",
        "Loss: continuous flow-matching, one frozen (noise, time) per sample. Greedy-PPL is not included.",
        "Formulas match MolmoAct2 / pi0.5. Calibration: 32 frames from `lerobot/libero`, seed=42.",
        f"Calibration samples: {len(samples)}.",
        "",
        "| Method | Drop-4 | Drop-8 |",
        "|---|---|---|",
    ]
    for method in METHODS:
        if method not in saved:
            lines.append(f"| {method} | out of memory | out of memory |")
            continue
        report4 = importance_ranking(saved[method], 4)
        report8 = importance_ranking(saved[method], 8)
        lines.append(f"| {method} | `{report4['drop']}` | `{report8['drop']}` |")
    (OUT_DIR / "scores.md").write_text("\n".join(lines) + "\n")
    protocol = {
        "checkpoint": CKPT,
        "dataset": DATASET_REPO,
        "n_samples": len(samples),
        "seed": SEED,
        "n_vlm_layers": len(blocks),
        "skip": "keep-KV identity residual",
        "loss": "flow-matching MSE, frozen (noise, time)",
        "methods": list(METHODS),
        "ks": list(KS),
    }
    (score_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()

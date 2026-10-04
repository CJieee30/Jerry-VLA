#!/usr/bin/env python3
"""Nine redundancy scores for the joint OpenVLA-OFT Llama stack.

Loss is the finetune L1 on the action head. A dropped layer is an identity
block (no Q/K/V/O/MLP). Each calibration sample is one batch: 24GB cannot hold
the v044 batch-8 gradient, so Fisher/IGIA square the per-sample gradient.
"""

from __future__ import annotations

import argparse
import fcntl
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from tqdm import tqdm

from layer_skip import skip_llama_layers
from load_policy import cap_cuda_memory, llama_parts, load_policy, prepare_for_forward, prepare_for_grad
from loss import action_l1, move_batch
from paths import CALIB_PATH, METHODS, SCORE_DIR
from ranking import print_ranking


def empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_samples(limit: int) -> list[dict]:
    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    samples = blob["samples"]
    if limit > 0:
        samples = samples[:limit]
    return samples


def _boundary_hooks(layers, final_norm):
    states = {}
    hooks = []

    def make_pre_hook(idx):
        def hook(module, args, kwargs=None):
            hidden = args[0] if args else None
            if hidden is not None:
                states[idx] = hidden

        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.input_layernorm.register_forward_pre_hook(make_pre_hook(i), with_kwargs=True))
    hooks.append(final_norm.register_forward_pre_hook(make_pre_hook(len(layers)), with_kwargs=True))
    return states, hooks


def measure_magnitude(vla) -> dict:
    layers, _ = llama_parts(vla)
    scores = []
    for layer in layers:
        mag = sum(p.detach().float().abs().sum().item() for p in layer.parameters())
        scores.append(mag)
    return {"method": "magnitude", "scores": scores, "n_samples": 0}


def _mean_loss(vla, action_head, proprio, samples, device) -> float:
    total = 0.0
    with torch.no_grad():
        for sample in samples:
            batch = move_batch(sample, device)
            total += float(action_l1(vla, action_head, proprio, batch).item())
    return total / max(len(samples), 1)


def measure_ppl(vla, action_head, proprio, samples, device, partial_path: Path) -> dict:
    layers, _ = llama_parts(vla)
    prepare_for_forward(vla, action_head, proprio)
    done = {}
    baseline = None
    if partial_path.is_file():
        prev = torch.load(partial_path, map_location="cpu", weights_only=False)
        if prev.get("n_samples") == len(samples):
            done = {int(k): float(v) for k, v in prev.get("deltas", {}).items()}
            baseline = prev.get("baseline_loss")
    if baseline is None:
        print(f"PPL baseline over {len(samples)} samples", flush=True)
        baseline = _mean_loss(vla, action_head, proprio, samples, device)
        print(f"baseline L1 {baseline:.6f}", flush=True)
    deltas = [done.get(i) for i in range(len(layers))]
    for i in tqdm(range(len(layers)), desc="PPL"):
        if deltas[i] is not None:
            continue
        with skip_llama_layers([i]):
            dropped = _mean_loss(vla, action_head, proprio, samples, device)
        deltas[i] = dropped - baseline
        print(f"  L{i:>2} dropped={dropped:.6f} delta={deltas[i]:.6f}", flush=True)
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"n_samples": len(samples), "baseline_loss": baseline, "deltas": {i: deltas[i] for i in range(len(layers)) if deltas[i] is not None}},
            partial_path,
        )
    return {
        "method": "ppl",
        "scores": deltas,
        "baseline_loss": baseline,
        "n_samples": len(samples),
        "paired_flow_matching": False,
    }


def _backward_loss(vla, action_head, proprio, batch) -> torch.Tensor:
    vla.zero_grad(set_to_none=True)
    action_head.zero_grad(set_to_none=True)
    loss = action_l1(vla, action_head, proprio, batch)
    loss.backward()
    return loss


_CHUNK = 262_144


def _chunk_dots(grad, param, mode: str) -> float:
    """Float32 reduction in small tiles so the backward peak can stay put."""
    g = grad.detach().reshape(-1)
    w = param.detach().reshape(-1)
    acc = torch.zeros((), device=g.device, dtype=torch.float32)
    for start in range(0, g.numel(), _CHUNK):
        gf = g[start : start + _CHUNK].float()
        wf = w[start : start + _CHUNK].float()
        if mode == "fisher":
            acc = acc + (gf.square() * wf.square()).sum()
        else:
            acc = acc + gf.square().sum()
    return float(acc.item())


def _taylor_add(grad, param, dest: torch.Tensor) -> None:
    """Add g*w into a CPU accumulator without a full pinned copy."""
    g = grad.detach().reshape(-1)
    w = param.detach().reshape(-1)
    flat = dest.view(-1)
    step = 1_048_576
    for start in range(0, g.numel(), step):
        flat[start : start + step] += (
            g[start : start + step].float() * w[start : start + step].float()
        ).cpu()


def _attach_score_hooks(layers, kind: str):
    """Fold each parameter gradient into the score, then drop it.

    A full Llama-2-7B backward keeps a bf16 gradient the same size as the
    weights (~14GB). That does not fit beside the weights on a 24GB card.
    The post-accumulate hook reduces one parameter at a time.
    """
    state = {"buckets": [{} for _ in layers], "scalars": [0.0] * len(layers)}
    handles = []

    def make_hook(index, name):
        def hook(param):
            grad = param.grad
            if grad is None:
                return
            with torch.no_grad():
                if kind == "taylor":
                    bucket = state["buckets"][index]
                    dest = bucket.get(name)
                    if dest is None:
                        dest = torch.zeros(param.shape, dtype=torch.float32)
                        bucket[name] = dest
                    _taylor_add(grad, param, dest)
                elif kind in ("fisher", "igia"):
                    state["scalars"][index] += _chunk_dots(grad, param, kind)
                else:
                    raise ValueError(kind)
            param.grad = None

        return hook

    for index, layer in enumerate(layers):
        for name, param in layer.named_parameters():
            if param.requires_grad:
                handles.append(param.register_post_accumulate_grad_hook(make_hook(index, name)))
    return handles, state


def _attach_free_grad_hooks(params):
    def hook(param):
        param.grad = None

    handles = []
    for param in params:
        if param.requires_grad:
            handles.append(param.register_post_accumulate_grad_hook(hook))
    return handles


def _remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


def measure_taylor(vla, action_head, proprio, samples, device) -> dict:
    layers, _ = llama_parts(vla)
    prepare_for_grad(vla, action_head, proprio)
    handles, state = _attach_score_hooks(layers, "taylor")
    try:
        for sample in tqdm(samples, desc="Taylor"):
            batch = move_batch(sample, device)
            loss = None
            try:
                loss = _backward_loss(vla, action_head, proprio, batch)
            finally:
                del loss
                vla.zero_grad(set_to_none=True)
                empty_cache()
    finally:
        _remove_hooks(handles)
    n = max(len(samples), 1)
    scores = []
    for bucket in state["buckets"]:
        total = sum(tensor.abs().sum().item() for tensor in bucket.values())
        scores.append(total / n)
    return {"method": "taylor", "scores": scores, "n_samples": len(samples)}


def measure_fisher(vla, action_head, proprio, samples, device) -> dict:
    return _param_grad_method(vla, action_head, proprio, samples, device, "fisher")


def measure_igia(vla, action_head, proprio, samples, device) -> dict:
    return _param_grad_method(vla, action_head, proprio, samples, device, "igia")


def _param_grad_method(vla, action_head, proprio, samples, device, method: str) -> dict:
    layers, _ = llama_parts(vla)
    prepare_for_grad(vla, action_head, proprio)
    handles, state = _attach_score_hooks(layers, method)
    try:
        for sample in tqdm(samples, desc=method):
            batch = move_batch(sample, device)
            loss = None
            try:
                loss = _backward_loss(vla, action_head, proprio, batch)
            finally:
                del loss
                vla.zero_grad(set_to_none=True)
                empty_cache()
    finally:
        _remove_hooks(handles)
    n = max(len(samples), 1)
    totals = state["scalars"]
    if method == "fisher":
        scores = [v / n for v in totals]
    else:
        scores = list(totals)
    return {
        "method": method,
        "scores": scores,
        "n_samples": len(samples),
        "aggregation": "per_sample_grad_then_square",
        "batch_size": 1,
    }


def _rademacher_like(param: torch.Tensor) -> torch.Tensor:
    bits = torch.randint(0, 2, param.shape, device=param.device, dtype=torch.int8)
    return (bits * 2 - 1).to(dtype=param.dtype)


def _hvp_trace(loss, params, retain_loss_graph: bool) -> float:
    active = [p for p in params if p.requires_grad]
    if not active:
        return 0.0
    grads = torch.autograd.grad(loss, active, create_graph=True, retain_graph=True, allow_unused=True)
    paired = [(p, g) for p, g in zip(active, grads) if g is not None]
    del grads
    if not paired:
        return 0.0
    probes = [_rademacher_like(p) for p, _ in paired]
    gvp = sum((g * z).sum() for (_, g), z in zip(paired, probes))
    hvps = torch.autograd.grad(
        gvp, [p for p, _ in paired], retain_graph=retain_loss_graph, allow_unused=True
    )
    total = 0.0
    for z, hvp in zip(probes, hvps):
        if hvp is not None:
            total += (z.float() * hvp.float()).sum().item()
    del gvp, hvps, probes, paired
    return total


def _enable_one_layer(layers, index: int) -> list:
    active = []
    for i, layer in enumerate(layers):
        for param in layer.parameters():
            on = i == index
            param.requires_grad_(on)
            if on:
                active.append(param)
    return active


def measure_hessian(vla, action_head, proprio, samples, device) -> dict:
    layers, _ = llama_parts(vla)
    prepare_for_grad(vla, action_head, proprio)
    traces = [0.0] * len(layers)
    for sample in tqdm(samples, desc="Hessian"):
        batch = move_batch(sample, device)
        for i in range(len(layers)):
            params = _enable_one_layer(layers, i)
            vla.zero_grad(set_to_none=True)
            loss = action_l1(vla, action_head, proprio, batch)
            traces[i] += _hvp_trace(loss, params, retain_loss_graph=False)
            del loss
            vla.zero_grad(set_to_none=True)
            empty_cache()
    n = max(len(samples), 1)
    raw = [t / n for t in traces]
    return {
        "method": "hessian",
        "scores": [abs(t) for t in raw],
        "raw_trace": raw,
        "n_samples": len(samples),
        "hvp": "per_layer_k1",
    }


def measure_gateprobe(vla, action_head, proprio, samples, device) -> dict:
    layers, final_norm = llama_parts(vla)
    prepare_for_grad(vla, action_head, proprio)
    free_hooks = _attach_free_grad_hooks(vla.language_model.parameters())
    rows = []
    none_counts = [0] * len(layers)
    try:
        for sample in tqdm(samples, desc="GateProbe"):
            batch = move_batch(sample, device)
            states, hooks = _boundary_hooks(layers, final_norm)
            vla.zero_grad(set_to_none=True)
            loss = None
            row = [0.0] * len(layers)
            try:
                loss = action_l1(vla, action_head, proprio, batch)
                for hidden in states.values():
                    hidden.retain_grad()
                loss.backward()
                for i in range(len(layers)):
                    h_in = states.get(i)
                    h_out = states.get(i + 1)
                    if h_in is None or h_out is None or h_out.grad is None:
                        none_counts[i] += 1
                        continue
                    residual = (h_out.detach().float() - h_in.detach().float())
                    row[i] = float((h_out.grad.detach().float() * residual).sum().item())
            finally:
                for handle in hooks:
                    handle.remove()
                del states, loss
                vla.zero_grad(set_to_none=True)
                empty_cache()
            rows.append(row)
    finally:
        _remove_hooks(free_hooks)
    raw = torch.tensor(rows, dtype=torch.float32)
    scores = raw.abs().mean(dim=0)
    if all(c == len(samples) for c in none_counts):
        raise RuntimeError(f"GateProbe saw no hidden grads: {none_counts}")
    return {
        "method": "gateprobe",
        "scores": scores.tolist(),
        "n_samples": len(samples),
        "none_hout_grad_counts": none_counts,
    }


def _flatten(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.detach().float().reshape(-1, hidden.shape[-1])


def measure_cossim(vla, action_head, proprio, samples, device) -> dict:
    layers, final_norm = llama_parts(vla)
    prepare_for_forward(vla, action_head, proprio)
    acc = [0.0] * len(layers)
    n = 0
    for sample in tqdm(samples, desc="CosSim"):
        batch = move_batch(sample, device)
        states, hooks = _boundary_hooks(layers, final_norm)
        try:
            with torch.no_grad():
                action_l1(vla, action_head, proprio, batch)
            for i in range(len(layers)):
                sim = F.cosine_similarity(states[i].float(), states[i + 1].float(), dim=-1)
                acc[i] += float(sim.mean().item())
            n += 1
        finally:
            for handle in hooks:
                handle.remove()
    avg = [a / max(n, 1) for a in acc]
    return {
        "method": "cossim",
        "scores": [1.0 - a for a in avg],
        "mean_cos": avg,
        "n_samples": n,
    }


def measure_cka(vla, action_head, proprio, samples, device) -> dict:
    layers, final_norm = llama_parts(vla)
    prepare_for_forward(vla, action_head, proprio)
    n_layers = len(layers)
    n_tok = [0.0] * n_layers
    sum_in = [None] * n_layers
    sum_out = [None] * n_layers
    xtx = [None] * n_layers
    yty = [None] * n_layers
    ytx = [None] * n_layers
    for sample in tqdm(samples, desc="CKA"):
        batch = move_batch(sample, device)
        states, hooks = _boundary_hooks(layers, final_norm)
        try:
            with torch.no_grad():
                action_l1(vla, action_head, proprio, batch)
            for i in range(n_layers):
                x = _flatten(states[i]).cpu()
                y = _flatten(states[i + 1]).cpu()
                n = x.shape[0]
                n_tok[i] += n
                if sum_in[i] is None:
                    dim = x.shape[1]
                    sum_in[i] = torch.zeros(dim)
                    sum_out[i] = torch.zeros(dim)
                    xtx[i] = torch.zeros(dim, dim)
                    yty[i] = torch.zeros(dim, dim)
                    ytx[i] = torch.zeros(dim, dim)
                sum_in[i] += x.sum(dim=0)
                sum_out[i] += y.sum(dim=0)
                xtx[i] += x.T @ x
                yty[i] += y.T @ y
                ytx[i] += y.T @ x
        finally:
            for handle in hooks:
                handle.remove()
            del states
            empty_cache()
    scores = []
    cka_vals = []
    for i in range(n_layers):
        n = n_tok[i]
        mean_x = sum_in[i] / n
        mean_y = sum_out[i] / n
        xtx_c = xtx[i] - n * torch.outer(mean_x, mean_x)
        yty_c = yty[i] - n * torch.outer(mean_y, mean_y)
        ytx_c = ytx[i] - n * torch.outer(mean_y, mean_x)
        numer = (ytx_c * ytx_c).sum()
        denom = xtx_c.norm() * yty_c.norm()
        cka = float((numer / denom).item()) if float(denom) > 0 else 0.0
        cka_vals.append(cka)
        scores.append(1.0 - cka)
    return {"method": "cka", "scores": scores, "cka": cka_vals, "n_samples": len(samples), "n_tokens": n_tok}


MEASURE = {
    "magnitude": measure_magnitude,
    "ppl": measure_ppl,
    "taylor": measure_taylor,
    "fisher": measure_fisher,
    "igia": measure_igia,
    "hessian": measure_hessian,
    "gateprobe": measure_gateprobe,
    "cossim": measure_cossim,
    "cka": measure_cka,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--limit", type=int, default=0, help="Use the first N calibration samples. 0 = all.")
    parser.add_argument("--save-file", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cap_cuda_memory()
    save_path = args.save_file or (SCORE_DIR / f"{args.method}.pt")
    if args.limit == 0 and save_path.is_file():
        print(f"skip existing {save_path}", flush=True)
        return
    lock_path = save_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "a+")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        print(f"skip {args.method}: another process is scoring it", flush=True)
        return
    try:
        _score_held(args, save_path)
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def _score_held(args, save_path: Path) -> None:
    device = torch.device(args.device)
    samples = [] if args.method == "magnitude" else load_samples(args.limit)
    print(f"loading policy for {args.method} n_samples={len(samples)}", flush=True)
    vla, action_head, proprio = load_policy(args.device)
    if args.method == "magnitude":
        result = measure_magnitude(vla)
    elif args.method == "ppl":
        partial = save_path.with_suffix(".partial.pt")
        result = measure_ppl(vla, action_head, proprio, samples, device, partial)
    else:
        result = MEASURE[args.method](vla, action_head, proprio, samples, device)
    result["skip"] = "llama_decoder_identity"
    result["loss"] = "l1_action_head"
    result["checkpoint"] = "openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
    print_ranking(args.method, result["scores"])
    if args.limit == 0:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, save_path)
        print(f"wrote {save_path}", flush=True)
    else:
        print("smoke only; score file not written", flush=True)


if __name__ == "__main__":
    main()

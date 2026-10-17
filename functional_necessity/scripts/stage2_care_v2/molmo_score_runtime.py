"""Shared Molmo scoring runtime: freeze non-VLM, verify checkpointing, one-backward triples."""

from __future__ import annotations

import torch


def mem_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "cuda": True,
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    }


def checkpoint_status(policy) -> dict:
    backbone = policy._backbone()
    transformer = getattr(backbone, "transformer", None)
    vision = getattr(backbone, "vision_backbone", None)
    hf = None
    try:
        hf = policy._hf_model()
    except Exception:
        hf = None
    return {
        "policy_config": bool(getattr(policy.config, "gradient_checkpointing", False)),
        "policy_training": bool(policy.training),
        "transformer": bool(getattr(transformer, "gradient_checkpointing", False)) if transformer is not None else None,
        "vision_backbone": bool(getattr(vision, "gradient_checkpointing", False)) if vision is not None else None,
        "hf_model": bool(getattr(hf, "gradient_checkpointing", False)) if hf is not None else None,
        "effective": bool(
            getattr(policy.config, "gradient_checkpointing", False)
            and policy.training
            and torch.is_grad_enabled()
        ),
    }


def prepare_policy_for_scoring(policy, blocks) -> dict:
    from score import _scoring_mode, _set_vlm_grad

    allowed = {id(p) for b in blocks for p in b.parameters()}
    n_on = n_off = 0
    for p in policy.parameters():
        want = id(p) in allowed
        p.requires_grad_(want)
        if want:
            n_on += 1
        else:
            n_off += 1
    _set_vlm_grad(blocks, True)
    _scoring_mode(policy, blocks, checkpoint=True)
    if hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()
    elif hasattr(policy, "_enable_gradient_checkpointing"):
        policy._enable_gradient_checkpointing()
    lm_head = getattr(getattr(policy, "model", None), "lm_head", None)
    lm_moved = False
    if lm_head is not None:
        lm_head.to("cpu")
        for p in lm_head.parameters():
            p.requires_grad_(False)
        lm_moved = True
    torch.cuda.empty_cache()
    status = checkpoint_status(policy)
    status.update(
        {
            "vlm_trainable_tensors": n_on,
            "frozen_tensors": n_off,
            "lm_head_cpu": lm_moved,
        }
    )
    if not status["effective"]:
        raise RuntimeError(f"gradient checkpointing not effective: {status}")
    return status


def _chunk_fisher(param) -> float:
    if param.grad is None:
        return 0.0
    g = param.grad.detach().reshape(-1)
    w = param.detach().reshape(-1)
    fisher = torch.zeros((), device=g.device, dtype=torch.float32)
    step = 262_144
    for start in range(0, g.numel(), step):
        gf = g[start : start + step].float()
        wf = w[start : start + step].float()
        fisher = fisher + (gf.square() * wf.square()).sum()
    return float(fisher.item())


def _chunk_igia(param) -> float:
    if param.grad is None:
        return 0.0
    g = param.grad.detach().reshape(-1)
    acc = torch.zeros((), device=g.device, dtype=torch.float32)
    step = 262_144
    for start in range(0, g.numel(), step):
        gf = g[start : start + step].float()
        acc = acc + gf.square().sum()
    return float(acc.item())


def score_frame_three(policy, blocks, final_norm, sample, device, *, with_gate: bool = True) -> dict:
    """One backward → IGIA / Fisher / optional GateProbe. Matches original Molmo formulas."""
    from score import _mean_flow

    packed = {
        "batch": {k: v.to(device) if torch.is_tensor(v) else v for k, v in sample["batch"].items()},
        "noise": sample["noise"].to(device),
        "timesteps": sample["timesteps"].to(device),
    }
    states = {}
    hooks = []
    gate_ok = False
    if with_gate:

        def make(idx):
            def hook(module, args):
                if args:
                    states[idx] = args[0]

            return hook

        for i, block in enumerate(blocks):
            hooks.append(block.register_forward_pre_hook(make(i)))
        hooks.append(final_norm.register_forward_pre_hook(make(len(blocks))))

    for block in blocks:
        for p in block.parameters():
            p.grad = None
    loss = _mean_flow(policy, [packed], device)
    if with_gate:
        n_req = 0
        for h in states.values():
            if h.requires_grad:
                h.retain_grad()
                n_req += 1
        gate_ok = n_req > 0
    loss.backward()

    igia, fisher, gate = [], [], []
    n_grad = 0
    for i, block in enumerate(blocks):
        f_acc = i_acc = 0.0
        for param in block.parameters():
            if param.grad is not None:
                n_grad += 1
            f_acc += _chunk_fisher(param)
            i_acc += _chunk_igia(param)
        fisher.append(f_acc)
        igia.append(i_acc)
        if with_gate:
            h_in, h_out = states.get(i), states.get(i + 1)
            if h_in is None or h_out is None or h_out.grad is None:
                gate.append(0.0)
            else:
                inner = float((h_out.grad.float() * (h_out.float() - h_in.float())).sum().item())
                gate.append(abs(inner))

    for h in hooks:
        h.remove()
    for block in blocks:
        for p in block.parameters():
            p.grad = None
    del loss
    del packed
    torch.cuda.empty_cache()
    out = {
        "igia": igia,
        "fisher": fisher,
        "n_params_with_grad": n_grad,
        "gate_hidden_requires_grad": gate_ok,
    }
    if with_gate:
        out["gateprobe"] = gate
    return out

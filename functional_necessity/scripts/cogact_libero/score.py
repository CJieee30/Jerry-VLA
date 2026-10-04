#!/usr/bin/env python3
"""Nine redundancy scores for Dexmal/libero-db-cogact Qwen2.5 LLM layers.

The action loss is CogACT's diffusion MSE on the DiT head. The DiT reads one
cognition token (the last unmasked LLM hidden state), not per-layer K/V, so a
dropped layer is an identity block. The nine formulas match the pi0.5 / Molmo
definitions: high score = keep.

Fisher / IGIA / Taylor share one backward per sample. Each calibration item is
one example with one frozen (noise, timestep), because a 24GB GPU cannot hold
a batch-mean gradient of the 7B LLM. That is (1/N) sum_t of the per-example
gradient, which is the DTR formula when the calibration batch size is 1.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, "/home/guorj/dexbotic")

import torch
import torch.nn.functional as F
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm
from transformers import AutoTokenizer

from dexbotic.model.cogact.cogact_arch import CogACTForCausalLM
from dexbotic.tokenization.tokenization import tokenize_dexbotic

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
CKPT = FN / "checkpoints" / "libero-db-cogact"
OUT = FN / "results" / "cogact_libero"
CALIB = OUT / "calib.pt"
METHODS = (
    "magnitude",
    "cossim",
    "cka",
    "ppl",
    "taylor",
    "fisher",
    "igia",
    "gateprobe",
    "hessian",
)
N_SAMPLES = 32
SEED = 42
KS = (4, 8, 12)
CHUNK = 16
PROMPT = "<image>\nWhat action should the robot take to {task}?"


def importance_ranking(scores, k: int) -> dict:
    values = [float(x) for x in scores]
    n = len(values)
    keep_order = sorted(range(n), key=lambda i: values[i], reverse=True)
    drop = sorted(keep_order[n - k :])
    return {"importance": values, "redundancy_order": list(reversed(keep_order)), "drop": drop}


def llm_parts(model):
    llm = model.model.llm
    return list(llm.layers), llm.norm


def _norm_actions(actions: torch.Tensor, stats: dict) -> torch.Tensor:
    lo = torch.tensor(stats["min"], dtype=torch.float32)
    hi = torch.tensor(stats["max"], dtype=torch.float32)
    actions = actions.clamp(lo, hi)
    return (actions - lo) / (hi - lo + 1e-8) * 2 - 1


def build_calib() -> list[dict]:
    if CALIB.is_file():
        blob = torch.load(CALIB, map_location="cpu", weights_only=False)
        print(f"calib exists n={len(blob['samples'])}", flush=True)
        return blob["samples"]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("building 32-frame LIBERO calibration slice", flush=True)
    dataset = LeRobotDataset("lerobot/libero")
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    tokenizer = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    stats = json.loads((CKPT / "norm_stats.json").read_text())["default"]
    samples = []
    for index in order:
        if len(samples) >= N_SAMPLES:
            break
        frame = dataset[index]
        episode = int(frame["episode_index"])
        actions = [frame["action"].float()]
        cursor = index + 1
        while len(actions) < CHUNK and cursor < len(dataset):
            nxt = dataset[cursor]
            if int(nxt["episode_index"]) != episode:
                break
            actions.append(nxt["action"].float())
            cursor += 1
        while len(actions) < CHUNK:
            actions.append(actions[-1].clone())
        raw = torch.stack(actions, dim=0)
        normed = _norm_actions(raw, stats)
        image = frame["observation.images.image"]
        if image.dtype.is_floating_point:
            image = (image.clamp(0, 1) * 255).to(torch.uint8)
        image = image.permute(1, 2, 0).contiguous()
        task = str(frame["task"]).strip().rstrip("?")
        prompt = PROMPT.format(task=task)
        tokenized = tokenize_dexbotic(
            [[{"from": "human", "value": prompt}, {"from": "gpt", "value": " "}]],
            tokenizer,
            has_image=True,
            chat_template="dexbotic",
        )
        noise = torch.randn(1, CHUNK, 7, generator=generator)
        timestep = torch.randint(0, 100, (1,), generator=generator)
        samples.append(
            {
                "image": image,
                "input_ids": tokenized["input_ids"][0].to(torch.long),
                "actions": normed,
                "noise": noise,
                "timestep": timestep,
                "task": task,
                "index": int(index),
            }
        )
        print(f"kept {len(samples)}/{N_SAMPLES} index={index} task={task[:60]}", flush=True)
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"seed": SEED, "n": len(samples), "prompt": PROMPT, "samples": samples}, CALIB)
    return samples


def _prepare(model, sample, device):
    image = Image.fromarray(sample["image"].numpy())
    images = model.process_images([image]).to(device=device, dtype=torch.bfloat16)
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    actions = sample["actions"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    head = model.model.action_head
    head._frozen_noise = sample["noise"].to(device=device, dtype=torch.float32)
    head._frozen_timestep = sample["timestep"].to(device)
    return input_ids, attention_mask, images, actions


def _install_frozen_loss(model) -> None:
    head = model.model.action_head

    def loss(self, x, z, reduction="mean"):
        noise = self._frozen_noise.to(device=x.device, dtype=x.dtype)
        timestep = self._frozen_timestep.to(device=x.device)
        if noise.shape[0] != x.shape[0]:
            noise = noise.expand(x.shape[0], -1, -1).contiguous()
            timestep = timestep.expand(x.shape[0]).contiguous()
        x_t = self.diffusion.q_sample(x, timestep, noise)
        noise_pred = self.net(x_t, timestep, z)
        value = (noise_pred.float() - noise.float()).square()
        if reduction == "mean":
            return value.mean()
        if reduction == "sum":
            return value.sum()
        return value

    head.loss = loss.__get__(head, head.__class__)


def action_loss(model, sample, device):
    input_ids, attention_mask, images, actions = _prepare(model, sample, device)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images,
        actions=actions,
        repeated_diffusion_steps=1,
        use_cache=False,
    )
    return out.loss


class _IdentitySkip:
    def __init__(self, layers, indices: list[int]):
        self.layers = [layers[i] for i in indices]
        self.original = []

    def __enter__(self):
        for layer in self.layers:
            self.original.append(layer.forward)

            def _identity(hidden_states, *args, _layer=layer, **kwargs):
                del args, kwargs, _layer
                return hidden_states

            layer.forward = _identity
        return self

    def __exit__(self, *exc):
        for layer, forward in zip(self.layers, self.original):
            layer.forward = forward
        return False


def _hooks(layers, final_norm):
    states = {}
    handles = []

    def make(idx):
        def hook(module, args):
            if args:
                states[idx] = args[0]
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make(i)))
    handles.append(final_norm.register_forward_pre_hook(make(len(layers))))
    return states, handles


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
        elif mode == "igia":
            acc = acc + gf.square().sum()
        else:
            raise ValueError(mode)
    return float(acc.item())


def _taylor_add(dest, grad, param):
    g = grad.detach().reshape(-1)
    w = param.detach().reshape(-1)
    if dest is None:
        dest = torch.zeros(param.shape, dtype=torch.float32)
    flat = dest.view(-1)
    step = 1_048_576
    for start in range(0, g.numel(), step):
        flat[start : start + step] += (g[start : start + step].float() * w[start : start + step].float()).cpu()
    return dest


def measure_magnitude(layers) -> list[float]:
    scores = []
    for layer in layers:
        total = 0.0
        for param in layer.parameters():
            total += float(param.detach().cpu().float().abs().sum())
        scores.append(total)
    return scores


def measure_representation(model, layers, final_norm, samples, device, method: str) -> list[float]:
    model.eval()
    if method == "cossim":
        acc = [0.0] * len(layers)
        for sample in tqdm(samples, desc="cossim"):
            states, handles = _hooks(layers, final_norm)
            try:
                with torch.no_grad():
                    action_loss(model, sample, device)
                for i in range(len(layers)):
                    sim = F.cosine_similarity(states[i].float(), states[i + 1].float(), dim=-1)
                    acc[i] += float(sim.mean())
            finally:
                for handle in handles:
                    handle.remove()
        return [1.0 - a / len(samples) for a in acc]

    n_tok = [0.0] * len(layers)
    sum_in, sum_out, xtx, yty, ytx = ([None] * len(layers) for _ in range(5))
    for sample in tqdm(samples, desc="cka"):
        states, handles = _hooks(layers, final_norm)
        try:
            with torch.no_grad():
                action_loss(model, sample, device)
            for i in range(len(layers)):
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
    for i in range(len(layers)):
        n = n_tok[i]
        mean_x, mean_y = sum_in[i] / n, sum_out[i] / n
        xtx_c = xtx[i] - n * torch.outer(mean_x, mean_x)
        yty_c = yty[i] - n * torch.outer(mean_y, mean_y)
        ytx_c = ytx[i] - n * torch.outer(mean_y, mean_x)
        denom = xtx_c.norm() * yty_c.norm()
        cka = float((ytx_c * ytx_c).sum() / denom) if float(denom) > 0 else 0.0
        scores.append(1.0 - cka)
    return scores


def measure_ppl(model, layers, samples, device) -> list[float]:
    model.eval()
    with torch.no_grad():
        baseline = float(sum(action_loss(model, sample, device) for sample in samples) / len(samples))
    print(f"PPL baseline {baseline:.6f}", flush=True)
    deltas = []
    for i in tqdm(range(len(layers)), desc="ppl"):
        with torch.no_grad(), _IdentitySkip(layers, [i]):
            dropped = float(sum(action_loss(model, sample, device) for sample in samples) / len(samples))
        deltas.append(dropped - baseline)
        print(f"  L{i:>2} delta={deltas[-1]:+.6f}", flush=True)
    return deltas


def _scoring_train(model) -> None:
    model.train()
    model.model.action_head.eval()
    model.model.mm_vision_tower.eval()
    if hasattr(model.model.llm, "gradient_checkpointing_enable"):
        model.model.llm.gradient_checkpointing_enable()


def _freeze_except_llm(model, layers) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for layer in layers:
        for param in layer.parameters():
            param.requires_grad_(True)


def measure_grad_trio(model, layers, samples, device) -> dict[str, list[float]]:
    """One layer per backward. A full-LLM grad tensor does not fit in 24GB."""
    for param in model.parameters():
        param.requires_grad_(False)
    _scoring_train(model)
    fisher = [0.0] * len(layers)
    igia = [0.0] * len(layers)
    taylor = [{} for _ in layers]
    for sample in tqdm(samples, desc="taylor/fisher/igia"):
        for i, layer in enumerate(layers):
            params = list(layer.parameters())
            for param in params:
                param.requires_grad_(True)
            loss = action_loss(model, sample, device)
            loss.backward()
            for name, param in layer.named_parameters():
                if param.grad is None:
                    continue
                fisher[i] += _chunk_dots(param.grad, param, "fisher")
                igia[i] += _chunk_dots(param.grad, param, "igia")
                taylor[i][name] = _taylor_add(taylor[i].get(name), param.grad, param)
                param.grad = None
            for param in params:
                param.requires_grad_(False)
            del loss
        torch.cuda.empty_cache()
    n = len(samples)
    return {
        "fisher": [v / n for v in fisher],
        "igia": igia,
        "taylor": [sum(t.abs().sum().item() for t in bucket.values()) / n for bucket in taylor],
    }


def measure_gate(model, layers, final_norm, samples, device) -> list[float]:
    for param in model.parameters():
        param.requires_grad_(False)
    # One leaf keeps the residual stream in the graph without storing every LLM grad.
    layers[0].input_layernorm.weight.requires_grad_(True)
    _scoring_train(model)
    rows = []
    for sample in tqdm(samples, desc="gateprobe"):
        states, handles = _hooks(layers, final_norm)
        loss = action_loss(model, sample, device)
        for hidden in states.values():
            if torch.is_tensor(hidden) and hidden.requires_grad:
                hidden.retain_grad()
        loss.backward()
        row = []
        for i in range(len(layers)):
            h_in, h_out = states.get(i), states.get(i + 1)
            if h_in is None or h_out is None or h_out.grad is None:
                row.append(0.0)
            else:
                row.append(float((h_out.grad.float() * (h_out.float() - h_in.float())).sum()))
        rows.append(row)
        for handle in handles:
            handle.remove()
        for layer in layers:
            for param in layer.parameters():
                param.grad = None
        del loss
    return torch.tensor(rows).abs().mean(0).tolist()


def measure_hessian(model, layers, samples, device) -> list[float]:
    traces = [0.0] * len(layers)

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

    for param in model.parameters():
        param.requires_grad_(False)
    _scoring_train(model)
    for sample in tqdm(samples, desc="hessian"):
        for i, layer in enumerate(layers):
            params = [p for p in layer.parameters() if p.requires_grad or True]
            for param in params:
                param.requires_grad_(True)
            torch.cuda.empty_cache()
            with sdpa_kernel(SDPBackend.MATH):
                loss = action_loss(model, sample, device)
                traces[i] += hvp(loss, params)
            for param in params:
                param.requires_grad_(False)
                param.grad = None
            del loss
    return [abs(t / len(samples)) for t in traces]


def _save(method: str, scores: list[float], n: int) -> None:
    ranks = {f"drop{k}": importance_ranking(scores, k)["drop"] for k in KS}
    path = OUT / "scores" / f"{method}.pt"
    torch.save({"method": method, "scores": scores, **ranks, "n_samples": n}, path)
    print(f"{method} " + " ".join(f"drop{k}={ranks[f'drop{k}']}" for k in KS), flush=True)


def _write_md(saved: dict) -> None:
    lines = [
        "# DB-CogACT LIBERO redundancy scores",
        "",
        "Checkpoint: `Dexmal/libero-db-cogact` (Qwen2.5-7B, 28 layers, DiT-B action head).",
        "A dropped layer is an identity block. The DiT reads the last unmasked hidden state, not per-layer K/V.",
        "Loss: diffusion noise MSE, one frozen (noise, timestep) per sample, `repeated_diffusion_steps=1`.",
        f"Calibration: {N_SAMPLES} frames from `lerobot/libero`, seed {SEED}, chunk {CHUNK}, min-max norm from the checkpoint.",
        "High score = keep. Formulas match pi0.5 / Molmo (Fisher / IGIA square the per-example gradient).",
        "",
        "| Method | Drop-4 | Drop-8 | Drop-12 |",
        "|---|---|---|---|",
    ]
    for method in METHODS:
        if method not in saved:
            lines.append(f"| {method} | out of memory | out of memory | out of memory |")
            continue
        cells = [f"`{importance_ranking(saved[method], k)['drop']}`" for k in KS]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    (OUT / "scores.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    samples = build_calib()
    device = torch.device("cuda")
    print("loading CogACT", flush=True)
    model = CogACTForCausalLM.from_pretrained(
        CKPT,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    tower = model.model.mm_vision_tower
    if not hasattr(tower, "image_processor"):
        from transformers import CLIPImageProcessor

        # Weights are already in the checkpoint. Only attach the processor.
        tower.image_processor = CLIPImageProcessor.from_pretrained(tower.vision_tower_name)
        tower.is_loaded = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    _install_frozen_loss(model)
    layers, final_norm = llm_parts(model)
    print(f"llm layers {len(layers)}", flush=True)
    with torch.no_grad():
        probe = float(action_loss(model, samples[0], device))
    print(f"probe loss {probe:.6f}", flush=True)
    with torch.no_grad(), _IdentitySkip(layers, [0]):
        skipped = float(action_loss(model, samples[0], device))
    print(f"identity skip L0 loss {skipped:.6f}", flush=True)

    score_dir = OUT / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for method in METHODS:
        path = score_dir / f"{method}.pt"
        if path.is_file():
            saved[method] = torch.load(path, map_location="cpu", weights_only=False)["scores"]
            print(f"skip existing {method}", flush=True)
    pending_grad = [m for m in ("taylor", "fisher", "igia") if m not in saved]
    if pending_grad:
        print(f"==== {pending_grad} ====", flush=True)
        try:
            trio = measure_grad_trio(model, layers, samples, device)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            for method in pending_grad:
                (score_dir / f"{method}.oom.txt").write_text(str(exc) + "\n")
            print("OOM taylor/fisher/igia", flush=True)
            trio = {}
        for method, scores in trio.items():
            if method in pending_grad:
                _save(method, scores, len(samples))
                saved[method] = scores
    for method in METHODS:
        if method in saved or method in {"taylor", "fisher", "igia"}:
            continue
        print(f"==== {method} ====", flush=True)
        try:
            if method == "magnitude":
                scores = measure_magnitude(layers)
            elif method in {"cossim", "cka"}:
                scores = measure_representation(model, layers, final_norm, samples, device, method)
            elif method == "ppl":
                scores = measure_ppl(model, layers, samples, device)
            elif method == "gateprobe":
                scores = measure_gate(model, layers, final_norm, samples, device)
            elif method == "hessian":
                scores = measure_hessian(model, layers, samples, device)
            else:
                raise KeyError(method)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            (score_dir / f"{method}.oom.txt").write_text(str(exc) + "\n")
            print(f"OOM {method}", flush=True)
            continue
        _save(method, scores, len(samples))
        saved[method] = scores
    _write_md(saved)
    print("done", flush=True)


if __name__ == "__main__":
    main()

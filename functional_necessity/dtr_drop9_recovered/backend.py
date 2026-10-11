"""OpenPI backend for FN scoring on DTR Drop-9 recovered π0.5.

Loads `s1ghhh/VLADrop_pi05_LIBERO_Drop9Block_GateProbe` with Stage-1
`llm_drop_* = [7,10,11,12,13,14,15,16,17]` on every forward. Metric formulas
match `functional_necessity/scripts` (orig-batch Fisher/IGIA, GateProbe no
h_in fallback, PPL frozen ξ). Does not touch the v044 LeRobot pipeline.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Iterator

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FN_ROOT = HERE.parent
OPENPI_SRC = Path("/home/guorj/VLADrop/models/pi0.5/openpi/src")
DEFAULT_CHECKPOINT = HERE / "checkpoints" / "VLADrop_pi05_LIBERO_Drop9Block_GateProbe"
DEFAULT_CALIB_FILE = HERE / "calib" / "pi05_libero_dropped_calib_64x8_seed42.pt"
DEFAULT_SCORES_DIR = HERE / "results" / "scores"
DEFAULT_RANKS_DIR = HERE / "results" / "ranks"
DEFAULT_SEED = 42
DEFAULT_GRAD_MICROBATCH = 2

STAGE1_DROP = [7, 10, 11, 12, 13, 14, 15, 16, 17]
REMAINING_LAYERS = [0, 1, 2, 3, 4, 5, 6, 8, 9]
N_LAYERS = 18
FN_DROP_K = 3

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# PI0Pytorch.__init__ torch.compile(sample_actions); scoring never calls it.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_common_args(parser: argparse.ArgumentParser, *, need_calib: bool = True) -> argparse.ArgumentParser:
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save-file", type=Path, default=None)
    parser.add_argument("--microbatch-size", type=int, default=None)
    if need_calib:
        parser.add_argument("--calib-file", type=Path, default=DEFAULT_CALIB_FILE)
    return parser


def default_save_file(method: str) -> Path:
    DEFAULT_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_SCORES_DIR / f"{method}.pt"


def _map_tensors(obj: Any, fn) -> Any:
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _map_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_tensors(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_map_tensors(v, fn) for v in obj)
    if hasattr(obj, "images") and hasattr(obj, "state"):
        images = obj.images
        masks = obj.image_masks
        return type(obj)(
            images={k: _map_tensors(v, fn) for k, v in images.items()} if isinstance(images, dict) else images,
            image_masks={k: _map_tensors(v, fn) for k, v in masks.items()} if isinstance(masks, dict) else masks,
            state=_map_tensors(obj.state, fn),
            tokenized_prompt=_map_tensors(obj.tokenized_prompt, fn) if obj.tokenized_prompt is not None else None,
            tokenized_prompt_mask=_map_tensors(obj.tokenized_prompt_mask, fn)
            if obj.tokenized_prompt_mask is not None
            else None,
            token_ar_mask=_map_tensors(obj.token_ar_mask, fn) if obj.token_ar_mask is not None else None,
            token_loss_mask=_map_tensors(obj.token_loss_mask, fn) if obj.token_loss_mask is not None else None,
        )
    return obj


def _split_batch(batch: Any) -> tuple[Any, torch.Tensor]:
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]
    raise TypeError(f"Expected (observation, actions) calib batch, got {type(batch)}")


def _batch_size(batch: Any) -> int:
    _, actions = _split_batch(batch)
    return int(actions.shape[0])


def slice_batch(batch: Any, start: int, end: int) -> tuple[Any, torch.Tensor]:
    obs, actions = _split_batch(batch)
    bsz = int(actions.shape[0])

    def _slice(t: torch.Tensor) -> torch.Tensor:
        if t.ndim > 0 and t.shape[0] == bsz:
            return t[start:end]
        return t

    return _map_tensors(obs, _slice), actions[start:end]


def move_to_device(batch: Any, device: torch.device) -> tuple[Any, torch.Tensor]:
    obs, actions = _split_batch(batch)

    def _to(t: torch.Tensor) -> torch.Tensor:
        return t.to(device, non_blocking=device.type == "cuda")

    return _map_tensors(obs, _to), actions.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")


def load_calib_data(calib_file: Path) -> list[Any]:
    payload = torch.load(calib_file, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "calib_data" in payload:
        data = payload["calib_data"]
    else:
        data = payload
    logging.info("Loaded %d calibration batches from %s", len(data), calib_file)
    if not data:
        raise FileNotFoundError(f"empty calibration file: {calib_file}")
    return data


def calib_num_steps(calib_data: list[Any], microbatch_size: int | None = None) -> int:
    if microbatch_size is None:
        return len(calib_data)
    steps = 0
    for batch in calib_data:
        bsz = _batch_size(batch)
        step = min(microbatch_size, bsz)
        steps += (bsz + step - 1) // step
    return steps


def iter_calib_batches(
    calib_data: list[Any],
    device: torch.device,
    microbatch_size: int | None = None,
) -> Iterator[tuple[Any, torch.Tensor]]:
    for batch in calib_data:
        bsz = _batch_size(batch)
        step = bsz if microbatch_size is None else min(microbatch_size, bsz)
        if step >= bsz:
            yield move_to_device(batch, device)
            continue
        for start in range(0, bsz, step):
            yield move_to_device(slice_batch(batch, start, min(start + step, bsz)), device)


def iter_orig_calib_microbatches(
    calib_data: list[Any],
    device: torch.device,
    microbatch_size: int | None = None,
) -> Iterator[list[tuple[Any, torch.Tensor]]]:
    for batch in calib_data:
        bsz = _batch_size(batch)
        step = bsz if microbatch_size is None else min(microbatch_size, bsz)
        if step >= bsz:
            yield [move_to_device(batch, device)]
            continue
        microbatches = []
        for start in range(0, bsz, step):
            microbatches.append(move_to_device(slice_batch(batch, start, min(start + step, bsz)), device))
        yield microbatches


def cuda_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def paligemma_parts(model):
    core = model.paligemma_with_expert.paligemma.language_model
    return core.layers, core.norm


def freeze_upstream_vision(model) -> None:
    paligemma = model.paligemma_with_expert.paligemma
    n_frozen = 0
    for name, param in paligemma.named_parameters():
        if "language_model" in name:
            continue
        if param.requires_grad:
            param.requires_grad_(False)
            n_frozen += param.numel()
    logging.info("Froze %s upstream vision/projector params", f"{n_frozen:,}")


def prepare_model_for_grad_scoring(model, *, recompute: bool) -> None:
    freeze_upstream_vision(model)
    if recompute:
        model.train()
        model.paligemma_with_expert.paligemma.model.vision_tower.eval()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for parameter-grad scoring")
    else:
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()


def compute_loss(model, batch, noise=None, time=None) -> torch.Tensor:
    obs, actions = _split_batch(batch)
    kwargs = {}
    if noise is not None:
        kwargs["noise"] = noise
    if time is not None:
        kwargs["time"] = time
    if actions.device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model(obs, actions, **kwargs)
    else:
        losses = model(obs, actions, **kwargs)
    if isinstance(losses, (list, tuple)):
        losses = torch.stack(losses)
    return losses.mean()


def orig_batch_accumulate_grads(model, microbatches: list) -> int:
    n_mb = 0
    for batch in microbatches:
        with torch.enable_grad():
            loss = compute_loss(model, batch)
            loss.backward()
        n_mb += 1
        del loss
    return n_mb


def _stage1_lists() -> tuple[list[int], list[int]]:
    return list(STAGE1_DROP), list(STAGE1_DROP)


def load_model(checkpoint: Path, device: torch.device):
    import safetensors.torch
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    ckpt_dir = Path(checkpoint)
    weight_path = ckpt_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"missing {weight_path}")

    drop_attn, drop_mlp = _stage1_lists()
    model_cfg = Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        dtype="bfloat16",
        llm_drop_attn_list=drop_attn,
        llm_drop_mlp_list=drop_mlp,
    )
    object.__setattr__(model_cfg, "dtype", "bfloat16")
    logging.info("Building PI0Pytorch with Stage-1 drop %s", drop_attn)
    model = PI0Pytorch(model_cfg)
    logging.info("Loading weights from %s", weight_path)
    missing, unexpected = safetensors.torch.load_model(model, str(weight_path), strict=False)
    if missing:
        logging.warning("Missing keys (%d): %s", len(missing), missing[:8])
    if unexpected:
        logging.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:8])
    model.to(device)
    pwe = model.paligemma_with_expert
    pwe.llm_drop_attn_list = list(drop_attn)
    pwe.llm_drop_mlp_list = list(drop_mlp)
    logging.info(
        "Stage-1 lists: attn=%s mlp=%s",
        pwe.llm_drop_attn_list,
        pwe.llm_drop_mlp_list,
    )
    return model


@contextmanager
def extra_llm_drop(model, layer_idx: int | Sequence[int]):
    """Temporarily add remaining layer(s) to the OpenPI joint-graph drop lists."""
    layers = [layer_idx] if isinstance(layer_idx, int) else list(layer_idx)
    pwe = model.paligemma_with_expert
    orig_attn = list(pwe.llm_drop_attn_list)
    orig_mlp = list(pwe.llm_drop_mlp_list)
    extra_attn = [i for i in layers if i not in orig_attn]
    extra_mlp = [i for i in layers if i not in orig_mlp]
    if extra_attn:
        pwe.llm_drop_attn_list = orig_attn + extra_attn
    if extra_mlp:
        pwe.llm_drop_mlp_list = orig_mlp + extra_mlp
    try:
        yield
    finally:
        pwe.llm_drop_attn_list = orig_attn
        pwe.llm_drop_mlp_list = orig_mlp


class FlowMatchingReplay:
    """Replay the same (noise, time) pairs so PPL baseline and drop-l share ξ."""

    def __init__(self, model):
        self.model = model
        self.noises: list[torch.Tensor] = []
        self.times: list[torch.Tensor] = []
        self.i = 0

    def capture(self, calib_data, device: torch.device, microbatch_size: int | None = None) -> None:
        self.noises.clear()
        self.times.clear()
        with torch.no_grad():
            for batch in iter_calib_batches(calib_data, device, microbatch_size):
                _, actions = _split_batch(batch)
                noise = self.model.sample_noise(actions.shape, actions.device)
                time = self.model.sample_time(actions.shape[0], actions.device)
                self.noises.append(noise.detach().cpu().contiguous())
                self.times.append(time.detach().cpu().contiguous())
        logging.info("Captured %d flow-matching (noise, time) pairs", len(self.noises))

    def reset(self) -> None:
        self.i = 0

    def step(self) -> None:
        self.i += 1

    def current(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.i >= len(self.noises):
            raise RuntimeError(f"flow replay overflow: i={self.i} n={len(self.noises)}")
        return self.noises[self.i].to(device=device), self.times[self.i].to(device=device)


def save_scores(path: Path, results: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, path)
    logging.info("Saved scores to %s", path)


def importance_ranking(scores: torch.Tensor | list[float], ks=(FN_DROP_K,)) -> dict[str, Any]:
    if torch.is_tensor(scores):
        values = [float(x) for x in scores.detach().cpu().flatten().tolist()]
    else:
        values = [float(x) for x in scores]
    n = len(values)
    keep_order = sorted(range(n), key=lambda i: values[i], reverse=True)
    redundancy_order = list(reversed(keep_order))
    remaining_keep_order = [i for i in keep_order if i in REMAINING_LAYERS]
    remaining_redundancy = list(reversed(remaining_keep_order))
    drops: dict[int, dict[str, list[int]]] = {}
    for k in ks:
        extra = remaining_redundancy[:k]
        keep_remaining = remaining_redundancy[k:]
        drops[int(k)] = {
            "drop": sorted(extra),
            "keep": sorted(keep_remaining),
            "union_drop": sorted(STAGE1_DROP + extra),
        }
    return {
        "importance": values,
        "keep_order": keep_order,
        "redundancy_order": redundancy_order,
        "remaining_keep_order": remaining_keep_order,
        "remaining_redundancy_order": remaining_redundancy,
        "stage1_drop": list(STAGE1_DROP),
        "remaining_layers": list(REMAINING_LAYERS),
        "drop": drops,
    }


def print_ranking(method: str, scores: torch.Tensor | list[float], ks=(FN_DROP_K,)) -> dict[str, Any]:
    report = importance_ranking(scores, ks=ks)
    values = report["importance"]
    print(f"\n{'=' * 70}")
    print(f"{method} importance ({len(values)} PaliGemma layers; high = keep)")
    print(f"{'=' * 70}")
    for i, value in enumerate(values):
        tag = " remaining" if i in REMAINING_LAYERS else " stage1-drop"
        print(f"  L{i:>2}:{tag:14} {value:.6e}")
    print("  remaining keep-order (high→low): " + " > ".join(f"L{i}" for i in report["remaining_keep_order"]))
    print(
        "  remaining redundancy-order (drop first): "
        + " > ".join(f"L{i}" for i in report["remaining_redundancy_order"])
    )
    for k, sets in report["drop"].items():
        print(f"  FN Drop-{k} on remaining: drop {sets['drop']}  keep {sets['keep']}")
        print(f"  union with Stage-1: {sets['union_drop']}")
    return report

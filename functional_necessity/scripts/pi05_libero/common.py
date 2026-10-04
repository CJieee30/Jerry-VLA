"""Shared loading, calibration I/O, and ranking helpers for Stage-2 profiling."""

from __future__ import annotations

import argparse
import logging
import os
import random
import types
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
FN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"
DEFAULT_DATASET = "lerobot/libero"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_SEED = 42
DEFAULT_NUM_BATCHES = 64
DEFAULT_BATCH_SIZE = 8
DEFAULT_DROP_KS = (6, 9, 12, 16)
# 3090 24GB: no_grad CosSim/CKA 能吃下 batch=8；GateProbe 反传在 batch=8 会 OOM。
DEFAULT_GRAD_MICROBATCH = 2
DEFAULT_CALIB_FILE = FN_ROOT / "results" / "calib" / "pi05_v044_libero_64x8_seed42.pt"
DEFAULT_SCORES_DIR = FN_ROOT / "results" / "scores"
DEFAULT_RANKS_DIR = FN_ROOT / "results" / "ranks"


def configure_hf_env() -> None:
    """Prefer the local Hub mirror. Must run before huggingface_hub is imported."""
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)


configure_hf_env()


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()


def _hf_hub_cache() -> Path:
    return _hf_home() / "hub"


def _lerobot_cache_roots() -> tuple[Path, Path]:
    from lerobot.utils.constants import HF_LEROBOT_HOME, HF_LEROBOT_HUB_CACHE

    return HF_LEROBOT_HUB_CACHE, HF_LEROBOT_HOME


def _hub_repo_dir(cache_root: Path, kind: str, repo_id: str) -> Path:
    return cache_root / f"{kind}--{repo_id.replace('/', '--')}"


def _snapshot_from_refs(repo_dir: Path, predicate: Callable[[Path], bool]) -> Path | None:
    snap_parent = repo_dir / "snapshots"
    refs_dir = repo_dir / "refs"
    if refs_dir.is_dir():
        preferred = ("main", "v3.0")
        names = [*(n for n in preferred if (refs_dir / n).is_file())]
        names.extend(sorted(p.name for p in refs_dir.iterdir() if p.is_file() and p.name not in preferred))
        for name in names:
            sha = (refs_dir / name).read_text().strip()
            cand = snap_parent / sha
            if predicate(cand):
                return cand
    if not snap_parent.is_dir():
        return None
    snaps = [p for p in snap_parent.iterdir() if p.is_dir()]
    snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for cand in snaps:
        if predicate(cand):
            return cand
    return None


def _is_lerobot_dataset_root(path: Path) -> bool:
    return path.is_dir() and (path / "meta" / "info.json").is_file()


def _is_pretrained_policy_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    # Hub snapshots such as lerobot/pi05_base may ship weights without config.json.
    return (path / "model.safetensors").is_file() or (path / "model.safetensors.index.json").is_file()


def checkpoint_needs_libero_adapt(checkpoint: str | Path) -> bool:
    """True for general π0.5 base weights that must be scored on LIBERO features."""
    name = Path(str(checkpoint)).name.rstrip("/")
    if name == "pi05_base":
        return True
    text = str(checkpoint).rstrip("/")
    return text.endswith("/pi05_base") or text.endswith("lerobot/pi05_base")


def resolve_dataset_root(repo_id: str, root: str | Path | None = None) -> Path | None:
    """Return a local LeRobot dataset root so metadata loading does not call the Hub."""
    configure_hf_env()
    if root is not None:
        path = Path(root).expanduser().resolve()
        if not _is_lerobot_dataset_root(path):
            raise FileNotFoundError(f"dataset root is missing meta/info.json: {path}")
        return path

    hub_cache, home = _lerobot_cache_roots()
    legacy = home / repo_id
    if _is_lerobot_dataset_root(legacy):
        logging.info("Using local dataset at %s", legacy)
        return legacy

    snap = _snapshot_from_refs(_hub_repo_dir(hub_cache, "datasets", repo_id), _is_lerobot_dataset_root)
    if snap is not None:
        logging.info("Using local dataset snapshot %s", snap)
        return snap

    logging.warning(
        "No local cache for dataset %s under %s or %s; LeRobot may try the Hub",
        repo_id,
        legacy,
        hub_cache,
    )
    return None


def resolve_pretrained_path(checkpoint: str | Path) -> str:
    """Map a Hub id to a local snapshot when present so from_pretrained stays offline."""
    configure_hf_env()
    raw = Path(checkpoint).expanduser()
    if _is_pretrained_policy_root(raw):
        return str(raw.resolve())
    if raw.exists() and raw.is_dir():
        logging.warning("Checkpoint dir %s has no config.json/model.safetensors; using it anyway", raw)
        return str(raw.resolve())

    repo_id = str(checkpoint)
    snap = _snapshot_from_refs(_hub_repo_dir(_hf_hub_cache(), "models", repo_id), _is_pretrained_policy_root)
    if snap is not None:
        logging.info("Using local policy snapshot %s", snap)
        return str(snap)

    logging.warning("No local snapshot for %s; from_pretrained may hit the Hub", repo_id)
    return repo_id


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_common_args(parser: argparse.ArgumentParser, *, need_calib: bool = True) -> argparse.ArgumentParser:
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
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


def build_libero_adapted_pi05_config():
    """LIBERO-shaped PI05Config for scoring `lerobot/pi05_base` on `lerobot/libero`.

    Hub `pi05_base` is a DROID-key checkpoint. DTR / OpenPI score those weights on
    LIBERO by swapping the data interface (2 cameras + 1 empty, 7-DoF action).
    Matches the local training protocol: empty_cameras=1, MEAN_STD, no rename_map.
    """
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.utils.constants import ACTION, OBS_STATE

    return PI05Config(
        compile_model=False,
        gradient_checkpointing=False,
        empty_cameras=1,
        n_action_steps=10,
        dtype="bfloat16",
        input_features={
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
    )


def load_policy_config(checkpoint: str):
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    if checkpoint_needs_libero_adapt(checkpoint):
        config = build_libero_adapted_pi05_config()
        logging.info("Using LIBERO-adapted PI05Config for %s", checkpoint)
        return config

    path = resolve_pretrained_path(checkpoint)
    if (Path(path) / "config.json").is_file():
        config = PI05Config.from_pretrained(path)
    else:
        logging.warning("No config.json at %s; falling back to PI05Config()", path)
        config = PI05Config()
    # v044's saved config has compile_model=True and gradient_checkpointing=True.
    config.compile_model = False
    config.gradient_checkpointing = False
    return config


def load_policy(checkpoint: str, device: str | torch.device):
    """Load a PI05 checkpoint with compile and gradient-checkpointing forced off."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    device = torch.device(device)
    path = resolve_pretrained_path(checkpoint)
    config = load_policy_config(checkpoint)
    config.device = str(device)
    logging.info("Loading policy from %s (compile=False, checkpointing=False)", path)
    policy = PI05Policy.from_pretrained(path, config=config)
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    policy.to(device)
    policy.eval()
    return policy


def freeze_upstream_vision(policy) -> None:
    """Freeze SigLIP + projector. They are upstream of language states, so this
    does not change dL/dh_l or language-layer parameter grads, but drops the
    float32 vision backward graph that OOM'd GateProbe at batch 8 on a 3090.
    """
    paligemma = policy.model.paligemma_with_expert.paligemma
    n_frozen = 0
    for name, param in paligemma.named_parameters():
        if "language_model" in name:
            continue
        if param.requires_grad:
            param.requires_grad_(False)
            n_frozen += param.numel()
    logging.info("Froze %s upstream vision/projector params", f"{n_frozen:,}")


def prepare_policy_for_grad_scoring(policy, *, recompute: bool) -> None:
    freeze_upstream_vision(policy)
    if recompute:
        policy.train()
        # Vision stays eval: it is frozen, and we only want dropout/BN from the language/expert path
        # to match the scoring graph as closely as possible.
        policy.model.paligemma_with_expert.paligemma.model.vision_tower.eval()
        if hasattr(policy.model, "gradient_checkpointing_enable"):
            policy.model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for parameter-grad scoring")
    else:
        policy.eval()
        if hasattr(policy.model, "gradient_checkpointing_disable"):
            policy.model.gradient_checkpointing_disable()


def cuda_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def paligemma_parts(policy):
    core = policy.model._paligemma_language_model_core()
    return core.layers, core.norm


def load_preprocessor(checkpoint: str, device: str = "cpu", dataset_stats=None):
    from lerobot.policies.factory import make_pre_post_processors

    config = load_policy_config(checkpoint)
    if checkpoint_needs_libero_adapt(checkpoint):
        if dataset_stats is None:
            raise ValueError("LIBERO-adapted pi05_base preprocessor requires dataset_stats")
        preprocessor, _ = make_pre_post_processors(
            config,
            pretrained_path=None,
            dataset_stats=dataset_stats,
        )
        return preprocessor

    path = resolve_pretrained_path(checkpoint)
    preprocessor, _ = make_pre_post_processors(
        config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return preprocessor


def preprocess_raw_batch(batch: dict[str, Any], preprocessor, camera_keys: list[str]) -> dict[str, Any]:
    from lerobot.processor.rename_processor import rename_batch_keys

    batch = dict(batch)
    for cam_key in camera_keys:
        if cam_key in batch and torch.is_tensor(batch[cam_key]) and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
    batch = rename_batch_keys(batch, {})
    return preprocessor(batch)


def tensor_to_cpu_store(x: Any, float_dtype: torch.dtype = torch.bfloat16) -> Any:
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.dtype == torch.float32:
            x = x.to(float_dtype)
        return x
    if isinstance(x, dict):
        return {k: tensor_to_cpu_store(v, float_dtype) for k, v in x.items()}
    if isinstance(x, list):
        return [tensor_to_cpu_store(v, float_dtype) for v in x]
    if isinstance(x, tuple):
        return tuple(tensor_to_cpu_store(v, float_dtype) for v in x)
    return x


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            out[key] = value
    return out


def _batch_size(batch: dict[str, Any]) -> int:
    from lerobot.utils.constants import ACTION

    if ACTION in batch and torch.is_tensor(batch[ACTION]):
        return int(batch[ACTION].shape[0])
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    raise KeyError("Cannot infer batch size from calibration batch")


def slice_batch(batch: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    bsz = _batch_size(batch)
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[0] == bsz:
            out[key] = value[start:end]
        elif isinstance(value, list) and len(value) == bsz:
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


def load_calib_data(calib_file: Path) -> list[dict[str, Any]]:
    payload = torch.load(calib_file, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "calib_data" in payload:
        data = payload["calib_data"]
    else:
        data = payload
    logging.info("Loaded %d calibration batches from %s", len(data), calib_file)
    return data


def calib_num_steps(calib_data: list[dict[str, Any]], microbatch_size: int | None = None) -> int:
    if microbatch_size is None:
        return len(calib_data)
    steps = 0
    for batch in calib_data:
        bsz = _batch_size(batch)
        step = min(microbatch_size, bsz)
        steps += (bsz + step - 1) // step
    return steps


def iter_calib_batches(
    calib_data: list[dict[str, Any]],
    device: torch.device,
    microbatch_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    for batch in calib_data:
        bsz = _batch_size(batch)
        step = bsz if microbatch_size is None else min(microbatch_size, bsz)
        if step >= bsz:
            yield move_to_device(batch, device)
            continue
        for start in range(0, bsz, step):
            yield move_to_device(slice_batch(batch, start, min(start + step, bsz)), device)


def iter_orig_calib_microbatches(
    calib_data: list[dict[str, Any]],
    device: torch.device,
    microbatch_size: int | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield the microbatches that reconstruct one original calibration batch.

    Grad-based scores must reduce at this granularity: squaring or taking abs
    after microbatch=2 is not equivalent to doing so on the original batch=8
    mean gradient / inner product.
    """
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


def orig_batch_accumulate_grads(policy, microbatches: list[dict[str, Any]]) -> int:
    """Backward each microbatch without zeroing, reconstructing the orig-batch mean grad.

    After this returns, ``param.grad`` is ``sum_m ∂L_m/∂w``. Divide by the returned
    microbatch count to get ``∂L_orig/∂w`` for the mean-reduced original-batch loss.
    """
    n_mb = 0
    for batch in microbatches:
        with torch.enable_grad():
            loss = compute_loss(policy, batch)
            loss.backward()
        n_mb += 1
        del loss
    return n_mb


class FlowMatchingReplay:
    """Replay the same (noise, time) pairs so PPL baseline and drop-l share ξ."""

    def __init__(self, policy):
        self.policy = policy
        self.noises: list[torch.Tensor] = []
        self.times: list[torch.Tensor] = []
        self.i = 0
        self._orig_sample_noise = None
        self._orig_sample_time = None

    def capture(self, calib_data: list[dict[str, Any]], device: torch.device, microbatch_size: int | None = None) -> None:
        self.noises.clear()
        self.times.clear()
        model = self.policy.model
        with torch.no_grad():
            for batch in iter_calib_batches(calib_data, device, microbatch_size):
                actions = self.policy.prepare_action(batch)
                noise = model.sample_noise(actions.shape, actions.device)
                time = model.sample_time(actions.shape[0], actions.device)
                self.noises.append(noise.detach().cpu().contiguous())
                self.times.append(time.detach().cpu().contiguous())
        logging.info("Captured %d flow-matching (noise, time) pairs", len(self.noises))

    def reset(self) -> None:
        self.i = 0

    def step(self) -> None:
        self.i += 1

    def __enter__(self):
        model = self.policy.model
        self._orig_sample_noise = model.sample_noise
        self._orig_sample_time = model.sample_time
        replay = self

        def sample_noise(_self, shape, device):
            if replay.i >= len(replay.noises):
                raise RuntimeError(f"flow replay overflow: i={replay.i} n={len(replay.noises)}")
            noise = replay.noises[replay.i].to(device=device)
            if tuple(noise.shape) != tuple(shape):
                raise RuntimeError(f"replay noise shape {tuple(noise.shape)} != {tuple(shape)}")
            return noise

        def sample_time(_self, bsize, device):
            if replay.i >= len(replay.times):
                raise RuntimeError(f"flow replay overflow: i={replay.i} n={len(replay.times)}")
            time = replay.times[replay.i].to(device=device)
            if int(time.shape[0]) != int(bsize):
                raise RuntimeError(f"replay time bsize {int(time.shape[0])} != {int(bsize)}")
            return time

        model.sample_noise = types.MethodType(sample_noise, model)
        model.sample_time = types.MethodType(sample_time, model)
        self.reset()
        return self

    def __exit__(self, *exc) -> bool:
        model = self.policy.model
        if self._orig_sample_noise is not None:
            model.sample_noise = self._orig_sample_noise
        if self._orig_sample_time is not None:
            model.sample_time = self._orig_sample_time
        self._orig_sample_noise = None
        self._orig_sample_time = None
        return False


def compute_loss(policy, batch: dict[str, Any]) -> torch.Tensor:
    device_type = batch[next(k for k, v in batch.items() if torch.is_tensor(v))].device.type
    if device_type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch)
    else:
        loss, _ = policy(batch)
    return loss


def save_scores(path: Path, results: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, path)
    logging.info("Saved scores to %s", path)


def importance_ranking(scores: torch.Tensor | list[float], ks=DEFAULT_DROP_KS) -> dict[str, Any]:
    """High score = more important = kept. Redundancy order is the reverse."""
    if torch.is_tensor(scores):
        values = [float(x) for x in scores.detach().cpu().flatten().tolist()]
    else:
        values = [float(x) for x in scores]
    n = len(values)
    keep_order = sorted(range(n), key=lambda i: values[i], reverse=True)
    redundancy_order = list(reversed(keep_order))
    drops: dict[int, dict[str, list[int]]] = {}
    for k in ks:
        if k > n:
            continue
        keep_n = n - k
        drops[int(k)] = {
            "keep": sorted(keep_order[:keep_n]),
            "drop": sorted(keep_order[keep_n:]),
        }
    return {
        "importance": values,
        "keep_order": keep_order,
        "redundancy_order": redundancy_order,
        "drop": drops,
    }


def print_ranking(method: str, scores: torch.Tensor | list[float], ks=DEFAULT_DROP_KS) -> dict[str, Any]:
    report = importance_ranking(scores, ks=ks)
    values = report["importance"]
    n = len(values)
    print(f"\n{'=' * 70}")
    print(f"{method} importance ({n} PaliGemma layers; high = keep)")
    print(f"{'=' * 70}")
    for i, value in enumerate(values):
        print(f"  L{i:>2}: {value:.6e}")
    print("  keep-order (high→low): " + " > ".join(f"L{i}" for i in report["keep_order"]))
    print("  redundancy-order (drop first): " + " > ".join(f"L{i}" for i in report["redundancy_order"]))
    for k, sets in report["drop"].items():
        print(f"  Drop-{k}: drop {sets['drop']}  keep {sets['keep']}")
    return report

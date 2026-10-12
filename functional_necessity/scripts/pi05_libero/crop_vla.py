"""CROP-VLA scout: camera-role visual token selection on π0.5.

Training-free. Vision is encoded once. A Drop-6 keep-KV prefix (installed by
the caller) produces layer-2 language→vision attention. Selected original
embeddings are packed into a shorter prefix; RoPE / action offsets keep the
unpruned position indices. Empty-camera slots are not modified.

LIBERO mapping (lerobot/envs/libero.py):
  agentview_image           → observation.images.image  → global
  robot0_eye_in_hand_image  → observation.images.image2 → wrist
  empty_camera_*            → padded slot               → unchanged
"""

from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.policies.common.vla_utils import (
    clone_past_key_values,
    make_att_2d_masks,
    prepare_attention_masks_4d,
)

TOKENS_PER_VIEW = 256
GRID = 16
SCOUT_LAYER = 2
FISHER_DROP6 = [1, 13, 14, 15, 16, 17]
BUDGETS = {
    "wrist": {"relevance": 96, "diversity": 32},
    "global": {"relevance": 32, "diversity": 96},
}


@dataclass
class CameraSlot:
    index: int
    key: str
    role: str
    start: int
    n_tokens: int
    keep_all: bool


@dataclass
class CropStats:
    full_len: int
    short_len: int
    kv_len: int
    orig_action_offset: int
    short_pad_sum: int
    cameras: list[dict[str, Any]] = field(default_factory=list)
    scout_layer: int = SCOUT_LAYER
    keep_all: bool = False
    timing_ms: dict[str, float] | None = None


def role_for_image_key(key: str) -> str:
    name = key.rsplit(".", 1)[-1]
    if "empty_camera" in key or name.startswith("empty_camera"):
        return "empty"
    if name == "image2":
        return "wrist"
    if name == "image":
        return "global"
    raise ValueError(f"unrecognized image key {key}")


def camera_slots(policy, batch: dict[str, Any], tokens_per_view: int = TOKENS_PER_VIEW) -> list[CameraSlot]:
    keys = list(policy.config.image_features.keys())
    present = [k for k in keys if k in batch]
    missing = [k for k in keys if k not in batch]
    ordered = present + missing
    slots = []
    start = 0
    for i, key in enumerate(ordered):
        role = role_for_image_key(key)
        slots.append(
            CameraSlot(
                index=i,
                key=key,
                role=role,
                start=start,
                n_tokens=tokens_per_view,
                keep_all=role == "empty",
            )
        )
        start += tokens_per_view
    return slots


def greedy_diversity(feats: torch.Tensor, seed: list[int], n_extra: int) -> list[int]:
    """Add tokens with the smallest nearest-neighbor cosine to the seed set."""
    if n_extra <= 0:
        return []
    n = int(feats.shape[0])
    selected = list(seed)
    taken = torch.zeros(n, dtype=torch.bool, device=feats.device)
    taken[selected] = True
    extra: list[int] = []
    for _ in range(n_extra):
        if bool(taken.all()):
            break
        sim = feats @ feats[selected].T
        nearest = sim.max(dim=1).values
        nearest = nearest.masked_fill(taken, 2.0)
        pick = int(nearest.argmin().item())
        extra.append(pick)
        selected.append(pick)
        taken[pick] = True
    return extra


def select_one_camera(
    score: torch.Tensor,
    feats: torch.Tensor,
    *,
    n_rel: int,
    n_div: int,
    valid: torch.Tensor | None = None,
    keep_all: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """Return (keep_sorted, relevance, diversity) local indices in 0..N-1."""
    n = int(score.shape[0])
    if keep_all or n_rel + n_div >= n:
        keep = list(range(n))
        return keep, keep, []
    work = score.detach().float().clone()
    if valid is not None:
        work = work.masked_fill(~valid, -1e9)
    n_rel = min(n_rel, n)
    rel = torch.topk(work, k=n_rel).indices.tolist()
    n_div = min(n_div, n - len(rel))
    div = greedy_diversity(feats, rel, n_div)
    keep = sorted(set(rel) | set(div))
    return keep, rel, div


@contextmanager
def _capture_layer_attention(language_model, layer_idx: int):
    layers = language_model.layers
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"scout layer {layer_idx} out of range for {len(layers)} layers")
    attn = layers[layer_idx].self_attn
    prev_impl = getattr(attn.config, "_attn_implementation", None)
    prev_lm = getattr(language_model.config, "_attn_implementation", None)
    attn.config._attn_implementation = "eager"
    language_model.config._attn_implementation = "eager"
    orig = attn.forward
    box: dict[str, torch.Tensor] = {}

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        if not (isinstance(out, tuple) and len(out) >= 2):
            raise RuntimeError(f"GemmaAttention.forward did not return weights; got {type(out)}")
        weights = out[1]
        if weights is None:
            raise RuntimeError(
                f"layer {layer_idx} attention weights are None; need eager attention for CROP scout"
            )
        box["weights"] = weights
        return out

    attn.forward = wrapped
    try:
        yield box
    finally:
        attn.forward = orig
        if prev_impl is not None:
            attn.config._attn_implementation = prev_impl
        if prev_lm is not None:
            language_model.config._attn_implementation = prev_lm


def _prefix_vlm(model, embs, pad_masks, att_masks, position_ids=None):
    if position_ids is None:
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_2d = make_att_2d_masks(pad_masks, att_masks)
    att_4d = prepare_attention_masks_4d(att_2d)
    model._paligemma_language_model_core().config._attn_implementation = "eager"
    _, past = model.paligemma_with_expert.forward(
        attention_mask=att_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[embs, None],
        use_cache=True,
    )
    return past, int(embs.shape[1])


def _kv_seq_len(past) -> int:
    if past is None:
        return 0
    getter = getattr(past, "get_seq_length", None)
    if callable(getter):
        return int(getter())
    if isinstance(past, (list, tuple)) and past:
        return int(past[0][0].shape[-2])
    raise TypeError(f"unknown past_key_values type {type(past)}")


def _relevance_scores(
    attn_weights: torch.Tensor,
    lang_index: torch.Tensor,
    vis_start: int,
    vis_n: int,
) -> torch.Tensor:
    """Mean over heads and valid language queries. attn: [B, H, Q, K]."""
    w = attn_weights.float()
    if w.ndim == 3:
        w = w.unsqueeze(0)
    if w.ndim != 4:
        raise RuntimeError(f"expected attention [B,H,Q,K], got {tuple(w.shape)}")
    bsz = w.shape[0]
    scores = []
    for b in range(bsz):
        q = lang_index[b].nonzero(as_tuple=False).squeeze(-1)
        if q.numel() == 0:
            scores.append(w.new_zeros(vis_n))
            continue
        chunk = w[b, :, q, vis_start : vis_start + vis_n]
        scores.append(chunk.mean(dim=(0, 1)))
    return torch.stack(scores, dim=0)


def _camera_is_live(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 1:
        return mask.bool()
    return mask.reshape(mask.shape[0], -1)[:, -1].bool()


def _img_mask_1d(mask: torch.Tensor, n_tokens: int) -> torch.Tensor:
    return _camera_is_live(mask)[:, None].expand(mask.shape[0], n_tokens)


def _select_indices(
    slots: list[CameraSlot],
    img_embs: list[torch.Tensor],
    img_masks: list[torch.Tensor],
    attn_weights: torch.Tensor,
    lang_q_full: torch.Tensor,
    n_lang: int,
    *,
    keep_all: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """lang_q_full is bool [B, full_len] marking valid language queries."""
    bsz = int(img_embs[0].shape[0])
    device = img_embs[0].device
    rows: list[torch.Tensor] = []
    meta_b0: list[dict[str, Any]] = []
    lang_start = slots[-1].start + slots[-1].n_tokens
    for b in range(bsz):
        keep: list[int] = []
        cam_meta = []
        for slot, emb, mask in zip(slots, img_embs, img_masks, strict=True):
            local = list(range(slot.n_tokens))
            if slot.keep_all or keep_all:
                chosen, rel, div = local, local, []
            else:
                feats = F.normalize(emb[b].float().reshape(slot.n_tokens, -1), dim=-1)
                live = bool(_camera_is_live(mask)[b].item())
                valid = torch.full((slot.n_tokens,), live, dtype=torch.bool, device=device)
                score = _relevance_scores(
                    attn_weights[b : b + 1], lang_q_full[b : b + 1], slot.start, slot.n_tokens
                )[0]
                budget = BUDGETS[slot.role]
                chosen, rel, div = select_one_camera(
                    score,
                    feats,
                    n_rel=budget["relevance"],
                    n_div=budget["diversity"],
                    valid=valid,
                    keep_all=keep_all,
                )
            keep.extend(slot.start + i for i in chosen)
            if b == 0:
                cam_meta.append(
                    {
                        "key": slot.key,
                        "role": slot.role,
                        "n_keep": len(chosen),
                        "n_full": slot.n_tokens,
                        "relevance": rel,
                        "diversity": div,
                        "keep": chosen,
                    }
                )
        keep.extend(range(lang_start, lang_start + n_lang))
        rows.append(torch.tensor(keep, device=device, dtype=torch.long))
        if b == 0:
            meta_b0 = cam_meta
    n_keep = int(rows[0].numel())
    for row in rows:
        if int(row.numel()) != n_keep:
            raise RuntimeError("CROP keep count differs across the batch; eval uses batch_size=1")
    return torch.stack(rows, dim=0), meta_b0


def _denoise_with_offset(model, prefix_pad_masks, past_key_values, x_t, timestep, prefix_offset):
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = model.embed_suffix(x_t, timestep)
    suffix_len = suffix_pad_masks.shape[1]
    batch_size = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    position_ids = prefix_offset.to(dtype=torch.long)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
    full_att_2d_masks_4d = prepare_attention_masks_4d(full_att_2d_masks)
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    past_key_values = clone_past_key_values(past_key_values)
    outputs_embeds, _ = model.paligemma_with_expert.forward(
        attention_mask=full_att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=[None, suffix_embs],
        use_cache=False,
        adarms_cond=[None, adarms_cond],
    )
    suffix_out = outputs_embeds[1][:, -model.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


class _CudaTimer:
    def __init__(self, enabled: bool):
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.marks: list[tuple[str, torch.cuda.Event]] = []

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.marks.append((name, event))

    def ms(self) -> dict[str, float] | None:
        if not self.enabled or len(self.marks) < 2:
            return None
        torch.cuda.synchronize()
        out: dict[str, float] = {}
        for (name0, ev0), (name1, ev1) in zip(self.marks, self.marks[1:], strict=True):
            out[name1] = float(ev0.elapsed_time(ev1))
        out["total"] = float(self.marks[0][1].elapsed_time(self.marks[-1][1]))
        return out


def sample_actions_crop(
    model,
    images,
    img_masks,
    tokens,
    masks,
    states=None,
    state_masks=None,
    noise=None,
    num_steps=None,
    *,
    keep_all: bool = False,
    scout_layer: int = SCOUT_LAYER,
    drop_layers: list[int] | None = None,
    profile: bool = False,
    **kwargs,
):
    """Scout + select + shortened official prefix + expert. Same noise/NFE as baseline."""
    del kwargs
    if num_steps is None:
        num_steps = model.config.num_inference_steps
    drop_layers = list(drop_layers or [])
    if (not keep_all) and scout_layer in drop_layers:
        raise RuntimeError(
            f"CROP scout layer {scout_layer} is in drop_layers={drop_layers}; it must still run attention"
        )
    if getattr(model, "proprio_history_proj", None) is not None:
        raise RuntimeError("CROP assumes use_proprioceptive_memory=False (cameras then language)")

    bsize = tokens.shape[0]
    device = tokens.device
    if noise is None:
        noise = model.sample_noise(
            (bsize, model.config.chunk_size, model.config.max_action_dim),
            device,
        )

    timer = _CudaTimer(profile)
    timer.mark("start")
    prefix_embs, pad_masks, att_masks = model.embed_prefix(
        images, img_masks, tokens, masks, states, state_masks
    )
    timer.mark("embed")
    n_tok = TOKENS_PER_VIEW
    n_views = len(images)
    n_lang = int(tokens.shape[1])
    if int(prefix_embs.shape[1]) != n_tok * n_views + n_lang:
        raise RuntimeError(
            f"prefix layout {prefix_embs.shape[1]} != {n_tok}*{n_views}+{n_lang}"
        )
    img_embs = [prefix_embs[:, i * n_tok : (i + 1) * n_tok] for i in range(n_views)]
    if int(img_embs[0].shape[1]) != TOKENS_PER_VIEW:
        raise RuntimeError(f"expected {TOKENS_PER_VIEW} tokens per view, got {img_embs[0].shape[1]}")

    slots = []
    start = 0
    live_i = 0
    for i, mask in enumerate(img_masks):
        is_empty = not bool(_camera_is_live(mask).any().item())
        if is_empty:
            role, key = "empty", f"observation.images.empty_camera_{max(i - 2, 0)}"
        elif live_i == 0:
            role, key, live_i = "global", "observation.images.image", 1
        else:
            role, key, live_i = "wrist", "observation.images.image2", live_i + 1
        slots.append(
            CameraSlot(
                index=i,
                key=key,
                role=role,
                start=start,
                n_tokens=n_tok,
                keep_all=role == "empty",
            )
        )
        start += n_tok

    orig_pos = torch.cumsum(pad_masks, dim=1) - 1
    orig_offset = pad_masks.sum(dim=-1)
    att_len = int(prefix_embs.shape[1])

    lang_model = model._paligemma_language_model_core()
    with _capture_layer_attention(lang_model, scout_layer) as box:
        _prefix_vlm(model, prefix_embs, pad_masks, att_masks, orig_pos)
    timer.mark("scout")
    attn_weights = box["weights"]
    lang_start = n_tok * n_views
    lang_q_full = torch.zeros(bsize, att_len, dtype=torch.bool, device=device)
    lang_q_full[:, lang_start:] = masks.bool()

    keep_idx, cam_meta = _select_indices(
        slots,
        img_embs,
        img_masks,
        attn_weights,
        lang_q_full,
        n_lang,
        keep_all=keep_all,
    )
    timer.mark("select")
    gather = keep_idx.unsqueeze(-1).expand(-1, -1, prefix_embs.shape[-1])
    short_embs = torch.gather(prefix_embs, 1, gather)
    short_pad = torch.gather(pad_masks, 1, keep_idx)
    short_att = torch.gather(att_masks, 1, keep_idx)
    short_pos = torch.gather(orig_pos, 1, keep_idx)

    past, short_len = _prefix_vlm(model, short_embs, short_pad, short_att, short_pos)
    timer.mark("prefix")
    kv_len = _kv_seq_len(past)
    if kv_len != short_len:
        raise RuntimeError(f"KV length {kv_len} != packed prefix {short_len}; sequence was not shortened")

    def _body(input_x_t, current_timestep):
        return _denoise_with_offset(
            model, short_pad, past, input_x_t, current_timestep, orig_offset
        )

    actions = euler_integrate(
        _body,
        noise,
        num_steps,
        rtc_processor=model.rtc_processor,
        rtc_enabled=False,
    )
    timer.mark("expert")
    stats = CropStats(
        full_len=att_len,
        short_len=short_len,
        kv_len=kv_len,
        orig_action_offset=int(orig_offset[0].item()),
        short_pad_sum=int(short_pad[0].sum().item()),
        cameras=cam_meta,
        scout_layer=scout_layer,
        keep_all=keep_all,
        timing_ms=timer.ms(),
    )
    model._crop_stats = stats
    return actions


@contextmanager
def crop_visual_tokens(
    policy,
    *,
    enabled: bool = True,
    keep_all: bool = False,
    scout_layer: int = SCOUT_LAYER,
    drop_layers: list[int] | None = None,
    profile: bool = False,
):
    """Replace PI05Pytorch.sample_actions with the CROP scout path."""
    if not enabled:
        yield
        return
    model = policy.model
    orig = model.sample_actions

    def patched(images, img_masks, tokens, masks, states=None, state_masks=None, noise=None, num_steps=None, **kwargs):
        return sample_actions_crop(
            model,
            images,
            img_masks,
            tokens,
            masks,
            states=states,
            state_masks=state_masks,
            noise=noise,
            num_steps=num_steps,
            keep_all=keep_all,
            scout_layer=scout_layer,
            drop_layers=drop_layers,
            profile=profile,
            **kwargs,
        )

    model.sample_actions = patched
    try:
        yield
    finally:
        model.sample_actions = orig


def gpu_occupancy(gpu: int) -> list[dict[str, Any]]:
    """Processes currently using `gpu`, with usernames."""
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
        rows.append({"pid": pid, "user": user, "name": name, "mem_mib": mem})
    return rows


def gpu_is_exclusive(gpu: int, *, allow_users: tuple[str, ...] | None = None) -> tuple[bool, str]:
    """True iff the card has no compute apps. Formal CROP work needs a fully idle GPU."""
    del allow_users
    rows = gpu_occupancy(gpu)
    if not rows:
        return True, "idle"
    desc = ", ".join(f"{r['user']}:{r['pid']}:{r['name']}" for r in rows)
    return False, f"occupied ({desc})"


def require_idle_gpu(gpu: int | None = None, gpus: tuple[int, ...] = (0, 1, 2, 3)) -> int:
    if gpu is not None:
        ok, why = gpu_is_exclusive(gpu)
        if not ok:
            raise SystemExit(
                f"GPU{gpu} is not idle ({why}). CROP eval/efficiency waits for an exclusive card."
            )
        return gpu
    found, note = first_exclusive_gpu(gpus)
    if found is None:
        raise SystemExit("NO_IDLE_GPU: " + note)
    return found


def first_exclusive_gpu(gpus: tuple[int, ...] = (0, 1, 2, 3)) -> tuple[int | None, str]:
    notes = []
    for g in gpus:
        ok, why = gpu_is_exclusive(g)
        notes.append(f"gpu{g} {why}")
        if ok:
            return g, "; ".join(notes)
    return None, "; ".join(notes)


def stats_dict(stats: CropStats) -> dict[str, Any]:
    return asdict(stats)


def save_stats_json(path, stats: CropStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats_dict(stats), indent=2) + "\n")

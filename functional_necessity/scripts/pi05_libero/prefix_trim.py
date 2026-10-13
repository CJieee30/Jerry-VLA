"""Drop invalid prefix slots at inference: empty cameras and right-side language pad.

Training-free. Uses only camera validity masks and the language attention mask.
Does not inspect pixels, token ids, or text. batch=1 only for now.

When enabled, fully-masked cameras are removed before vision encode, and
language is sliced through the last True attention position. Remaining tokens
keep the same RoPE ids they had on the untrimmed path, because padded slots
did not increment `cumsum(pad)`. Action-expert offset is the valid prefix
length *before* any visual-token prune, not the padded physical length and
not the pruned length.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

TOKENS_PER_VIEW = 256


@dataclass
class TrimInfo:
    n_views_in: int
    n_views_out: int
    n_lang_in: int
    n_lang_out: int
    kept_view_indices: list[int]
    dropped_view_indices: list[int]
    untrimmed_physical: int
    trimmed_physical: int
    valid_prefix_len: int
    physical_len: int | None = None
    camera_roles: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def camera_is_live(mask: torch.Tensor) -> torch.Tensor:
    """True where the camera is valid. Mask only; pixels are ignored."""
    if mask.ndim == 1:
        return mask.bool()
    return mask.reshape(mask.shape[0], -1)[:, -1].bool()


def role_for_kept_index(kept_i: int) -> str:
    """Live cameras keep original order: first live=global, second=wrist."""
    if kept_i == 0:
        return "global"
    if kept_i == 1:
        return "wrist"
    return f"live_{kept_i}"


def right_trim_language(tokens: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Keep every language position through the last True attention bit (right pad only)."""
    if tokens.shape[0] != 1 or masks.shape[0] != 1:
        raise RuntimeError("prefix_trim currently supports batch=1 only")
    valid = masks[0].bool()
    if not bool(valid.any().item()):
        raise RuntimeError("language attention mask has no valid positions")
    last = int(valid.nonzero(as_tuple=False)[-1, 0].item())
    n_keep = last + 1
    return tokens[:, :n_keep], masks[:, :n_keep], n_keep


def trim_prefix_inputs(
    images: list[torch.Tensor],
    img_masks: list[torch.Tensor],
    tokens: torch.Tensor,
    masks: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, TrimInfo]:
    """Filter fully-masked cameras and right-pad language. Camera order preserved."""
    if tokens.shape[0] != 1:
        raise RuntimeError("prefix_trim currently supports batch=1 only")
    if len(images) != len(img_masks):
        raise RuntimeError(f"images {len(images)} vs masks {len(img_masks)}")

    kept: list[int] = []
    dropped: list[int] = []
    for i, mask in enumerate(img_masks):
        live = camera_is_live(mask)
        if live.shape[0] != 1:
            raise RuntimeError("prefix_trim currently supports batch=1 only")
        if bool(live.any().item()):
            kept.append(i)
        else:
            dropped.append(i)
    if not kept:
        raise RuntimeError("prefix_trim dropped every camera")

    images_out = [images[i] for i in kept]
    masks_out = [img_masks[i] for i in kept]
    tokens_out, lang_masks_out, n_lang = right_trim_language(tokens, masks)
    n_views_in = len(images)
    n_lang_in = int(tokens.shape[1])
    n_views_out = len(images_out)
    trimmed_physical = n_views_out * TOKENS_PER_VIEW + n_lang
    info = TrimInfo(
        n_views_in=n_views_in,
        n_views_out=n_views_out,
        n_lang_in=n_lang_in,
        n_lang_out=n_lang,
        kept_view_indices=kept,
        dropped_view_indices=dropped,
        untrimmed_physical=n_views_in * TOKENS_PER_VIEW + n_lang_in,
        trimmed_physical=trimmed_physical,
        valid_prefix_len=trimmed_physical,
        camera_roles=[role_for_kept_index(i) for i in range(n_views_out)],
    )
    return images_out, masks_out, tokens_out, lang_masks_out, info


@contextmanager
def prefix_trim(policy, enabled: bool = True):
    """Wrap sample_actions so official inference sees trimmed cameras/language.

    Off: original path. On: trim then the currently installed sample_actions.
    Nested CROP/grid should pass prefix_trim=True into crop_visual_tokens instead
    of stacking this wrapper, so trim runs once.
    """
    if not enabled:
        yield
        return
    model = policy.model
    orig_sample = model.sample_actions
    orig_embed = model.embed_prefix

    def embed_prefix(images, img_masks, tokens, masks, states=None, state_masks=None):
        out = orig_embed(images, img_masks, tokens, masks, states, state_masks)
        info = getattr(model, "_trim_stats", None)
        if info is not None:
            info.physical_len = int(out[0].shape[1])
            info.valid_prefix_len = int(out[1][0].sum().item())
            model._trim_stats = info
        return out

    def sample_actions(images, img_masks, tokens, masks, states=None, state_masks=None, noise=None, num_steps=None, **kwargs):
        images, img_masks, tokens, masks, info = trim_prefix_inputs(images, img_masks, tokens, masks)
        model._trim_stats = info
        return orig_sample(
            images,
            img_masks,
            tokens,
            masks,
            states=states,
            state_masks=state_masks,
            noise=noise,
            num_steps=num_steps,
            **kwargs,
        )

    model.embed_prefix = embed_prefix
    model.sample_actions = sample_actions
    try:
        yield
    finally:
        model.embed_prefix = orig_embed
        model.sample_actions = orig_sample


def cpu_selftest() -> None:
    """Mask-only camera drop and right-pad language. No GPU."""
    b, c, h, w = 1, 3, 8, 8
    images = [torch.full((b, c, h, w), float(i)) for i in range(3)]
    images[2] = torch.full((b, c, h, w), -1.0)
    img_masks = [
        torch.ones(b, dtype=torch.bool),
        torch.ones(b, dtype=torch.bool),
        torch.zeros(b, dtype=torch.bool),
    ]
    tokens = torch.arange(10)[None, :]
    masks = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    out_img, out_m, out_tok, out_lang, info = trim_prefix_inputs(images, img_masks, tokens, masks)
    if info.n_views_out != 2 or info.dropped_view_indices != [2]:
        raise RuntimeError(f"camera trim {info}")
    if info.n_lang_out != 7 or int(out_tok.shape[1]) != 7:
        raise RuntimeError(f"language trim {info.n_lang_out} tok={tuple(out_tok.shape)}")
    if info.valid_prefix_len != 2 * TOKENS_PER_VIEW + 7:
        raise RuntimeError(f"valid_prefix_len {info.valid_prefix_len}")
    if info.untrimmed_physical != 3 * TOKENS_PER_VIEW + 10:
        raise RuntimeError(f"untrimmed_physical {info.untrimmed_physical}")
    if out_img[0] is not images[0] or out_img[1] is not images[1]:
        raise RuntimeError("live cameras were reordered or copied")
    # Pixel values must not drive deletion: live camera of -1 stays; masked random is dropped.
    images_px = [
        torch.full((b, c, h, w), -1.0),
        torch.randn(b, c, h, w),
        torch.randn(b, c, h, w),
    ]
    masks_px = [
        torch.ones(b, dtype=torch.bool),
        torch.ones(b, dtype=torch.bool),
        torch.zeros(b, dtype=torch.bool),
    ]
    _, _, _, _, info_px = trim_prefix_inputs(images_px, masks_px, tokens, masks)
    if info_px.kept_view_indices != [0, 1] or info_px.dropped_view_indices != [2]:
        raise RuntimeError("trim used pixel values instead of masks")
    hole = torch.tensor([[1, 0, 1, 0, 0]], dtype=torch.bool)
    tok_h = torch.arange(5)[None, :]
    _, lang_h, n_h = right_trim_language(tok_h, hole)
    if n_h != 3 or int(lang_h.shape[1]) != 3:
        raise RuntimeError(f"right-trim should keep through last True, got {n_h}")
    try:
        trim_prefix_inputs(images, img_masks, tokens.repeat(2, 1), masks.repeat(2, 1))
        raise RuntimeError("batch>1 should fail")
    except RuntimeError as exc:
        if "batch=1" not in str(exc):
            raise
    print("cpu selftest ok (prefix_trim)", flush=True)

"""Action L1 used by OFT finetune.py (use_l1_regression, proprio, no FiLM)."""

from __future__ import annotations

import torch

from paths import bootstrap


def _constants():
    bootstrap()
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

    return get_current_action_mask, get_next_actions_mask, ACTION_DIM, NUM_ACTIONS_CHUNK


def num_patches(vla) -> int:
    """Vision patches across input images, plus the proprio token. Matches finetune.py."""
    n = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    return n + 1


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            out[key] = value
    return out


def action_l1(vla, action_head, proprio, batch: dict) -> torch.Tensor:
    """L1 between the action head and the normalized ground-truth chunk."""
    get_current_action_mask, get_next_actions_mask, action_dim, n_chunk = _constants()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=batch["input_ids"].is_cuda):
        output = vla(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"].to(dtype=torch.bfloat16),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"],
            proprio_projector=proprio,
            use_film=False,
        )
        last = output.hidden_states[-1]
        text = last[:, num_patches(vla) : -1]
        token_ids = batch["labels"][:, 1:].to(text.device)
        mask = get_current_action_mask(token_ids) | get_next_actions_mask(token_ids)
        bsz = batch["input_ids"].shape[0]
        expected = n_chunk * action_dim
        gathered = text[mask]
        if gathered.shape[0] != bsz * expected:
            raise RuntimeError(
                f"action hidden count {gathered.shape[0]} != batch {bsz} * {expected}"
            )
        actions_h = gathered.reshape(bsz, expected, -1).to(torch.bfloat16)
        pred = action_head.predict_action(actions_h)
        gt = batch["actions"].to(device=pred.device, dtype=pred.dtype)
        loss = torch.nn.functional.l1_loss(gt, pred)
    return loss

"""Load extra_drop = f(task) maps and wrap an OpenPI Policy.

Maps are JSON from build_task_layer_map.py. Lookup order:
1. obs['extra_llm_drop'] if the client sent it
2. (suite, task_id) if present
3. normalized prompt text
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from backend import REMAINING_LAYERS, STAGE1_DROP

LOGGER = logging.getLogger("task_extra_map")
_WS = re.compile(r"\s+")


def normalize_prompt(text: str) -> str:
    return _WS.sub(" ", str(text).strip().lower())


def load_task_extra_map(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if "tasks" not in payload:
        raise ValueError(f"{path} has no tasks[]")
    by_prompt: dict[str, list[int]] = {}
    by_key: dict[tuple[str, int], list[int]] = {}
    for row in payload["tasks"]:
        extra = [int(i) for i in row["extra_drop"]]
        bad = [i for i in extra if i not in REMAINING_LAYERS]
        if bad:
            raise ValueError(f"{path}: extra {bad} not in remaining {REMAINING_LAYERS}")
        by_prompt[normalize_prompt(row["task"])] = extra
        by_key[(str(row["suite"]), int(row["task_id"]))] = extra
    payload["_by_prompt"] = by_prompt
    payload["_by_key"] = by_key
    return payload


def lookup_extra(
    mapping: dict[str, Any],
    *,
    suite: str | None = None,
    task_id: int | None = None,
    prompt: str | None = None,
    explicit: list[int] | None = None,
) -> list[int]:
    if explicit is not None:
        return [int(i) for i in explicit]
    if suite is not None and task_id is not None:
        key = (str(suite), int(task_id))
        if key in mapping["_by_key"]:
            return list(mapping["_by_key"][key])
    if prompt is not None:
        hit = mapping["_by_prompt"].get(normalize_prompt(prompt))
        if hit is not None:
            return list(hit)
    raise KeyError(f"no extra_drop for suite={suite!r} task_id={task_id!r} prompt={prompt!r}")


def union_drop(extra: list[int], stage1: list[int] | None = None) -> list[int]:
    base = list(stage1 if stage1 is not None else STAGE1_DROP)
    return sorted(set(base) | set(int(i) for i in extra))


def paligemma_with_expert(policy: Any):
    model = getattr(policy, "_model", None)
    if model is None:
        raise TypeError(f"{type(policy)} has no _model")
    pwe = getattr(model, "paligemma_with_expert", None)
    if pwe is None:
        raise TypeError(f"{type(model)} has no paligemma_with_expert")
    return pwe


def llm_decoder_layers(pwe: Any):
    """PaliGemma language tower used by sample_actions prefix (use_cache=True)."""
    paligemma = getattr(pwe, "paligemma", None)
    if paligemma is None:
        raise TypeError(f"{type(pwe)} has no paligemma")
    lm = getattr(paligemma, "language_model", None)
    if lm is None:
        raise TypeError(f"{type(paligemma)} has no language_model")
    layers = getattr(lm, "layers", None)
    if layers is None:
        inner = getattr(lm, "model", None)
        layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise TypeError(f"{type(lm)} has no .layers")
    return layers


def dropped_prefix_layers(pwe: Any) -> list[int]:
    return [i for i, layer in enumerate(llm_decoder_layers(pwe)) if layer.drop_attn or layer.drop_mlp]


def apply_llm_drop(pwe: Any, union: list[int]) -> dict[str, Any]:
    """Set joint-graph lists *and* HuggingFace prefix layer flags.

    sample_actions prefix encoding reads GemmaDecoderLayer.drop_attn / drop_mlp,
    which are frozen at module init. Updating only llm_drop_*_list leaves extra
    remaining layers fully active in the KV cache.
    """
    drop = {int(i) for i in union}
    layers = llm_decoder_layers(pwe)
    snapshot = {
        "attn_list": list(pwe.llm_drop_attn_list),
        "mlp_list": list(pwe.llm_drop_mlp_list),
        "flags": [(bool(layer.drop_attn), bool(layer.drop_mlp)) for layer in layers],
    }
    pwe.llm_drop_attn_list = list(union)
    pwe.llm_drop_mlp_list = list(union)
    for i, layer in enumerate(layers):
        on = i in drop
        layer.drop_attn = on
        layer.drop_mlp = on
    return snapshot


def restore_llm_drop(pwe: Any, snapshot: dict[str, Any]) -> None:
    pwe.llm_drop_attn_list = list(snapshot["attn_list"])
    pwe.llm_drop_mlp_list = list(snapshot["mlp_list"])
    for layer, (drop_attn, drop_mlp) in zip(llm_decoder_layers(pwe), snapshot["flags"]):
        layer.drop_attn = drop_attn
        layer.drop_mlp = drop_mlp


class TaskExtraDropPolicy:
    """Keep Stage-1 drop on; add remaining extra layers per request / prompt."""

    def __init__(self, policy: Any, mapping: dict[str, Any], stage1_drop: list[int] | None = None):
        self._policy = policy
        self._mapping = mapping
        self._stage1 = list(stage1_drop if stage1_drop is not None else STAGE1_DROP)
        self._pwe = paligemma_with_expert(policy)
        apply_llm_drop(self._pwe, self._stage1)
        self._last_logged_extra: list[int] | None = None
        LOGGER.info(
            "TaskExtraDropPolicy kind=%s n_tasks=%d stage1=%s prefix_flags=%s",
            mapping.get("kind"),
            len(mapping["tasks"]),
            self._stage1,
            dropped_prefix_layers(self._pwe),
        )

    @property
    def metadata(self) -> dict:
        meta = dict(getattr(self._policy, "metadata", None) or {})
        meta["task_extra_map_kind"] = self._mapping.get("kind")
        meta["stage1_drop"] = list(self._stage1)
        return meta

    def infer(self, obs: dict, **kwargs):
        payload = dict(obs)
        explicit = payload.pop("extra_llm_drop", None)
        extra = lookup_extra(
            self._mapping,
            prompt=payload.get("prompt"),
            explicit=None if explicit is None else [int(i) for i in explicit],
        )
        union = union_drop(extra, self._stage1)
        snapshot = apply_llm_drop(self._pwe, union)
        if extra != self._last_logged_extra:
            LOGGER.info(
                "task extra_drop=%s union=%s prefix_flags=%s",
                extra,
                union,
                dropped_prefix_layers(self._pwe),
            )
            self._last_logged_extra = list(extra)
        try:
            return self._policy.infer(payload, **kwargs)
        finally:
            restore_llm_drop(self._pwe, snapshot)


def wrap_policy_with_task_extra(policy: Any, map_path: Path, stage1_drop: list[int] | None = None):
    mapping = load_task_extra_map(map_path)
    return TaskExtraDropPolicy(policy, mapping, stage1_drop=stage1_drop)

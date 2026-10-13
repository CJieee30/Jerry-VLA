#!/usr/bin/env python3
"""Closed-loop LIBERO eval with DTR Appendix B language-layer skip (no recovery).

Official π0.5 protocol (the 97.5% v044 run): n_action_steps=10, four suites,
10 episodes, seed=1000. Skip is a runtime hook on prefix ``layers[i].forward``;
``sample_actions`` is not replaced. compile_model is forced off so the hook
enters the graph.

Drop indices come from ``results/ranks/ranks.json`` / DROP_INDEX_LOOKUP.md.
Empty ``--drop-layers`` is the full-model anchor (same CLI, compile=false).
"""

import argparse
import json
import logging
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import torch
from termcolor import colored

# Do not enable `from __future__ import annotations` in this file.
# LeRobot's @parser.wrap() reads the function annotation as the config class;
# postponed annotations turn EvalPipelineConfig into a string and draccus crashes.

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_CALIB_FILE,
    DEFAULT_CHECKPOINT,
    DEFAULT_RANKS_DIR,
    FN_ROOT,
    init_logging,
    load_calib_data,
    load_policy,
    move_to_device,
    slice_batch,
)
from layer_skip import dropped_prefix_indices, skip_llm_layers  # noqa: E402
from drop6_extra3_protocol import (  # noqa: E402
    UNIVERSAL_DROP,
    load_eval_map,
    lookup_extra,
    union_drop,
)

DEFAULT_EVAL_DIR = FN_ROOT / "results" / "eval_drop"
DEFAULT_EVAL_TASK_EXTRA_DIR = FN_ROOT / "results" / "eval_task_extra"
LOGGER = logging.getLogger("eval_drop")


def parse_layer_list(raw: str | None) -> list[int]:
    if raw is None or str(raw).strip() in ("", "[]", "none", "None"):
        return []
    text = str(raw).strip().strip("[]")
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def drop_from_ranks(ranks_json: Path, method: str, k: int) -> list[int]:
    payload = json.loads(ranks_json.read_text())
    methods = payload["methods"]
    if method not in methods:
        raise KeyError(f"{method} not in {ranks_json}; have {sorted(methods)}")
    drop_tbl = methods[method]["drop"]
    entry = drop_tbl.get(str(k)) or drop_tbl.get(k)
    if not entry:
        raise KeyError(f"{method} has no drop-{k} in {ranks_json}")
    return [int(i) for i in entry["drop"]]


def resolve_drop_layers(
    *,
    drop_layers: list[int] | None,
    method: str | None,
    k: int | None,
    from_ranks: Path,
) -> tuple[list[int], str, int]:
    if drop_layers:
        return list(drop_layers), method or "manual", int(k or len(drop_layers))
    if method in (None, "", "baseline", "full"):
        return [], "baseline", 0
    if k is None:
        raise ValueError("--method requires --k (or pass --drop-layers)")
    return drop_from_ranks(from_ranks, method, int(k)), method, int(k)


def _parse_ours(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--drop-layers", "--drop_layers", dest="drop_layers", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--from-ranks", type=Path, default=DEFAULT_RANKS_DIR / "ranks.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--render-videos", action="store_true")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calib-file", type=Path, default=DEFAULT_CALIB_FILE)
    parser.add_argument("--task-extra-map", type=Path, default=None)
    parser.add_argument("--predicate-log", type=Path, default=None)
    parser.add_argument(
        "--skip-mode",
        choices=("keep_kv", "no_kv"),
        default="keep_kv",
        help="keep_kv: DTR Appendix B (write K/V). no_kv: identity residual and zero cache slots.",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="CROP-VLA: scout visual tokens by camera role, then rerun a shorter prefix.",
    )
    parser.add_argument(
        "--crop-keep-all",
        action="store_true",
        help="CROP path but keep every visual token (identity check).",
    )
    parser.add_argument("--crop-scout-layer", type=int, default=2)
    parser.add_argument(
        "--crop-budget",
        default="role",
        help="role: wrist 96+32 / global 32+96. shared_mix: both cameras 64+64.",
    )
    parser.add_argument(
        "--crop-selector",
        default="greedy",
        help="greedy: original FPS CROP. lite: one-shot 4-anchor diversity. grid: uniform checkerboard, no scout.",
    )
    parser.add_argument(
        "--prefix-trim",
        action="store_true",
        help="Drop empty cameras before vision encode and right-trim language pad. Requires explicit NFE.",
    )
    ours, rest = parser.parse_known_args(argv)
    return ours, rest


def apply_eval_protocol(cfg, original_argv: list[str]) -> None:
    """Force v044 closed-loop protocol unless the user already overrode a flag."""
    cfg.policy.compile_model = False
    cfg.policy.gradient_checkpointing = False
    if not any(arg.startswith("--policy.n_action_steps") for arg in original_argv):
        cfg.policy.n_action_steps = 10


def write_drop_meta(out_dir: Path, *, method: str, k: int, drop_layers: list[int], extra: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": DEFAULT_CHECKPOINT,
        "method": method,
        "k": k,
        "drop_layers": drop_layers,
        "keep_layers": [i for i in range(18) if i not in set(drop_layers)],
        "recovery": False,
        "skip": "dtr_appendix_b_prefix_kv",
        "n_action_steps": 10,
        "compile_model": False,
        **extra,
    }
    (out_dir / "drop_meta.json").write_text(json.dumps(payload, indent=2) + "\n")


def _action_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm().item())


def run_smoke(ours: argparse.Namespace) -> None:
    """Minute-scale check that prefix skip changes actions (not a no-op)."""
    init_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(ours.checkpoint, device)
    policy.config.n_action_steps = 10
    calib = load_calib_data(ours.calib_file)
    batch = move_to_device(slice_batch(calib[0], 0, 1), device)

    def chunk_with(indices: list[int]) -> torch.Tensor:
        policy.reset()
        torch.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(0)
        keep_kv = ours.skip_mode != "no_kv"
        ctx = skip_llm_layers(policy, indices, keep_kv=keep_kv) if indices else nullcontext()
        with torch.no_grad(), ctx:
            return policy.predict_action_chunk(batch).detach().cpu()

    full = chunk_with([])
    skip17 = chunk_with([17])
    skip0 = chunk_with([0])
    d17 = _action_l2(skip17, full)
    d0 = _action_l2(skip0, full)
    LOGGER.info(
        "smoke skip_mode=%s action-chunk L2 vs full: skip[17]=%.6g  skip[0]=%.6g",
        ours.skip_mode,
        d17,
        d0,
    )
    print(f"smoke skip_mode={ours.skip_mode} skip[17] L2={d17:.6g}")
    print(f"smoke skip_mode={ours.skip_mode} skip[0]  L2={d0:.6g}")
    if d0 <= 1e-5 and d17 <= 1e-5:
        raise SystemExit(
            "skip looks like a no-op: skip[0] and skip[17] both match the full chunk. "
            "Prefix layer.forward hook is not affecting sample_actions."
        )
    if ours.skip_mode == "no_kv" and d17 <= 1e-5:
        raise SystemExit(
            "no_kv skip[17] still matches the full chunk; zero-KV cache write is not reaching the expert."
        )
    if d0 <= d17:
        LOGGER.warning(
            "skip[0] L2 (%.6g) is not larger than skip[17] (%.6g); still hooked if either moved",
            d0,
            d17,
        )
    print("smoke ok: prefix skip changes the action chunk")


def _install_task_extra_run_one(mapping: dict) -> None:
    import lerobot.scripts.lerobot_eval as eval_mod

    orig = eval_mod.run_one

    def patched(task_group, task_id, env, *, policy, **kwargs):
        extra = lookup_extra(mapping, suite=task_group, task_id=task_id)
        union = union_drop(extra)
        LOGGER.info("[%s] task_id=%s extra_drop=%s union=%s", task_group, task_id, extra, union)
        with skip_llm_layers(policy, union):
            flags = dropped_prefix_indices(policy)
            if flags != union:
                raise RuntimeError(f"prefix skip {flags} != union {union} on {task_group} {task_id}")
            tg, tid, metrics = orig(task_group, task_id, env, policy=policy, **kwargs)
        metrics = dict(metrics)
        metrics["extra_drop"] = extra
        metrics["union_drop"] = union
        return tg, tid, metrics

    eval_mod.run_one = patched


def run_libero_eval(
    cfg,
    *,
    drop_layers: list[int],
    method: str,
    k: int,
    render_videos: bool,
    task_extra_map: Path | None = None,
    predicate_log: Path | None = None,
    skip_mode: str = "keep_kv",
    crop: bool = False,
    crop_keep_all: bool = False,
    crop_scout_layer: int = 2,
    crop_budget: str = "role",
    crop_selector: str = "greedy",
    prefix_trim_enabled: bool = False,
) -> None:
    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from lerobot.utils.device_utils import get_safe_torch_device
    from lerobot.utils.random_utils import set_seed

    logging.info(pformat(asdict(cfg)))
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    cfg.output_dir = Path(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    keep_kv = skip_mode != "no_kv"
    skip_name = "dtr_appendix_b_prefix_kv" if keep_kv else "identity_zero_kv"
    logging.info(
        "Drop %s k=%s layers=%s map=%s skip=%s (recovery=false)",
        method,
        k,
        drop_layers,
        task_extra_map,
        skip_name,
    )

    auditor = None
    if predicate_log is not None:
        from predicate_audit import install_predicate_audit

        auditor = install_predicate_audit(n_action_steps=int(cfg.policy.n_action_steps))
        logging.info("Predicate audit log=%s", predicate_log)

    logging.info("Making environment (batch_size=%s).", cfg.eval.batch_size)
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    recording_dir = Path(cfg.output_dir) / "recordings" if cfg.eval.recording else None
    max_episodes_rendered = 10 if render_videos else 0
    videos_dir = Path(cfg.output_dir) / "videos" if render_videos else None

    mapping = None
    if task_extra_map is not None:
        mapping = load_eval_map(task_extra_map)
        if int(cfg.env.max_parallel_tasks) != 1:
            raise ValueError("task-extra maps require --env.max_parallel_tasks=1 so skip hooks are not shared")
        _install_task_extra_run_one(mapping)
        logging.info(
            "Task extra map kind=%s n_tasks=%d U=%s",
            mapping.get("kind"),
            len(mapping["tasks"]),
            UNIVERSAL_DROP,
        )

    write_drop_meta(
        cfg.output_dir,
        method=method,
        k=k,
        drop_layers=drop_layers if mapping is None else list(UNIVERSAL_DROP),
        extra={
            "job_name": cfg.job_name,
            "seed": cfg.seed,
            "num_inference_steps": int(cfg.policy.num_inference_steps),
            "n_episodes_per_task": int(cfg.eval.n_episodes),
            "task_extra_map": None if task_extra_map is None else str(task_extra_map),
            "task_extra_kind": None if mapping is None else mapping.get("kind"),
            "universal_drop": list(UNIVERSAL_DROP) if mapping is not None else drop_layers,
            "per_task_union": mapping is not None,
            "predicate_log": None if predicate_log is None else str(predicate_log),
            "skip": skip_name,
            "skip_mode": skip_mode,
            "keep_kv": keep_kv,
            "crop": bool(crop),
            "crop_keep_all": bool(crop_keep_all),
            "crop_scout_layer": int(crop_scout_layer) if crop else None,
            "crop_budget": crop_budget if crop else None,
            "crop_selector": crop_selector if crop else None,
            "prefix_trim": bool(prefix_trim_enabled),
        },
    )

    amp = torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
    skip_ctx = nullcontext()
    if mapping is None:
        skip_ctx = (
            skip_llm_layers(policy, drop_layers, keep_kv=keep_kv) if drop_layers else nullcontext()
        )
    crop_ctx = nullcontext()
    trim_ctx = nullcontext()
    if crop:
        from crop_vla import crop_visual_tokens, resolve_budgets, resolve_selector

        selector = resolve_selector(crop_selector)
        budgets = None if selector == "grid" else resolve_budgets(crop_budget)
        crop_ctx = crop_visual_tokens(
            policy,
            keep_all=crop_keep_all,
            scout_layer=crop_scout_layer,
            drop_layers=drop_layers,
            budgets=budgets,
            budget_name=crop_budget,
            selector=selector,
            prefix_trim=bool(prefix_trim_enabled),
        )
        logging.info(
            "CROP-VLA enabled keep_all=%s scout_layer=%s drop=%s budget=%s selector=%s trim=%s %s",
            crop_keep_all,
            crop_scout_layer,
            drop_layers,
            crop_budget,
            selector,
            prefix_trim_enabled,
            budgets,
        )
    elif prefix_trim_enabled:
        from prefix_trim import prefix_trim as prefix_trim_cm

        trim_ctx = prefix_trim_cm(policy)
        logging.info("prefix_trim enabled (no visual prune)")
    with torch.no_grad(), amp, skip_ctx, trim_ctx, crop_ctx:
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=max_episodes_rendered,
            videos_dir=videos_dir,
            return_episode_data=False,
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            recording_dir=recording_dir,
            env_features=cfg.env.features if cfg.eval.recording else None,
            recording_repo_id=cfg.eval.recording_repo_id,
            recording_private=cfg.eval.recording_private,
        )
        LOGGER.info("Overall Aggregated Metrics:")
        LOGGER.info(info["overall"])
        for task_group, task_group_info in info.items():
            LOGGER.info("Aggregated Metrics for %s:", task_group)
            LOGGER.info(task_group_info)

    close_envs(envs)
    with open(cfg.output_dir / "eval_info.json", "w") as handle:
        json.dump(info, handle, indent=2)
    if auditor is not None:
        log_path = Path(predicate_log) if predicate_log is not None else cfg.output_dir / "predicate_audit.json"
        auditor.dump(log_path)
    logging.info("End of eval")


def main() -> None:
    init_logging()
    original = sys.argv[1:]
    ours, rest = _parse_ours(original)
    if ours.task_extra_map is not None:
        drop_layers = []
        method = ours.method or Path(ours.task_extra_map).stem
        k = int(ours.k) if ours.k is not None else 9
    else:
        drop_layers, method, k = resolve_drop_layers(
            drop_layers=parse_layer_list(ours.drop_layers),
            method=ours.method,
            k=ours.k,
            from_ranks=ours.from_ranks,
        )
    if ours.smoke:
        run_smoke(ours)
        return

    sys.argv = [sys.argv[0], *rest]
    from lerobot.configs import parser
    from lerobot.configs.eval import EvalPipelineConfig

    def _has_flag(names: tuple[str, ...]) -> bool:
        return any(arg == n or arg.startswith(n + "=") for arg in original for n in names)

    @parser.wrap()
    def _wrapped(cfg: EvalPipelineConfig):
        apply_eval_protocol(cfg, original)
        nfe = int(cfg.policy.num_inference_steps)
        if ours.prefix_trim or ours.crop:
            if not _has_flag(("--policy.num_inference_steps", "--policy.num_inference_steps=")):
                raise SystemExit(
                    "trim/crop eval requires explicit --policy.num_inference_steps "
                    "(do not rely on checkpoint default)"
                )
        if ours.prefix_trim and ours.crop and not _has_flag(("--crop-selector", "--crop-selector=")):
            raise SystemExit("trim+crop eval requires explicit --crop-selector")
        if cfg.output_dir is None:
            if ours.prefix_trim and (ours.crop or ours.crop_keep_all):
                from crop_vla import resolve_selector as _resolve_sel

                selector = _resolve_sel(ours.crop_selector)
                tag = f"eval_trim_{selector}_nfe{nfe}"
                cfg.output_dir = FN_ROOT / "results" / "stage_nfe" / "crop_vla" / tag
            elif ours.prefix_trim:
                tag = f"eval_trim_nfe{nfe}"
                cfg.output_dir = FN_ROOT / "results" / "stage_nfe" / "crop_vla" / tag
            elif ours.crop:
                selector = ours.crop_selector or "greedy"
                if ours.crop_keep_all:
                    tag = "eval_crop_keepall"
                elif selector in ("lite", "crop-lite", "crop_lite"):
                    if ours.crop_budget and ours.crop_budget not in ("role", "crop", ""):
                        tag = f"eval_crop_lite_{ours.crop_budget}"
                    else:
                        tag = "eval_crop_lite"
                elif selector in ("grid", "uniform", "uniform-grid", "uniform_grid"):
                    tag = "eval_crop_grid"
                elif ours.crop_budget and ours.crop_budget not in ("role", "crop", ""):
                    tag = f"eval_crop_{ours.crop_budget}"
                else:
                    tag = "eval_crop"
                cfg.output_dir = FN_ROOT / "results" / "stage_nfe" / "crop_vla" / tag
            elif ours.task_extra_map is not None:
                cfg.output_dir = DEFAULT_EVAL_TASK_EXTRA_DIR / method
            else:
                tag = method if k == 0 else f"k{k}_{method}"
                cfg.output_dir = DEFAULT_EVAL_DIR / tag
        run_libero_eval(
            cfg,
            drop_layers=drop_layers,
            method=method,
            k=k,
            render_videos=ours.render_videos,
            task_extra_map=ours.task_extra_map,
            predicate_log=ours.predicate_log,
            skip_mode=ours.skip_mode,
            crop=ours.crop or ours.crop_keep_all,
            crop_keep_all=ours.crop_keep_all,
            crop_scout_layer=ours.crop_scout_layer,
            crop_budget=ours.crop_budget,
            crop_selector=ours.crop_selector,
            prefix_trim_enabled=bool(ours.prefix_trim),
        )

    _wrapped()


if __name__ == "__main__":
    main()

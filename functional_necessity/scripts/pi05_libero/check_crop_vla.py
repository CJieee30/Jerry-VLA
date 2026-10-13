#!/usr/bin/env python3
"""CROP-VLA offline checks. Camera dump is CPU-only. Model checks wait for an exclusive GPU.

Usage:
  python check_crop_vla.py --cameras-only
  python check_crop_vla.py --gpu N          # identity + prune; refuses if the card is busy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_CALIB_FILE,
    DEFAULT_CHECKPOINT,
    FN_ROOT,
    init_logging,
    load_calib_data,
    load_policy,
    move_to_device,
    slice_batch,
)
from crop_vla import (  # noqa: E402
    FISHER_DROP6,
    GRID,
    SCOUT_LAYER,
    TOKENS_PER_VIEW,
    bind_cuda_device,
    cpu_selftest,
    crop_visual_tokens,
    first_exclusive_gpu,
    greedy_diversity,
    gpu_is_exclusive,
    resolve_budgets,
    select_one_camera,
    stats_dict,
)
from layer_skip import skip_llm_layers  # noqa: E402

OUT = FN_ROOT / "results" / "stage_nfe" / "crop_vla"
DROP = FISHER_DROP6
NFE = 1


def _to_uint8(chw: torch.Tensor) -> np.ndarray:
    x = chw.detach().float().cpu()
    if float(x.min()) < 0:
        x = (x + 1.0) * 0.5
    x = x.clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255).astype(np.uint8)


def save_camera_examples(calib, out_dir: Path, n_tasks: int = 3) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    records = []
    mapping = {
        "observation.images.image": {
            "role": "global",
            "libero": "agentview_image",
            "lerobot": "observation.images.image",
        },
        "observation.images.image2": {
            "role": "wrist",
            "libero": "robot0_eye_in_hand_image",
            "lerobot": "observation.images.image2",
        },
    }
    for batch in calib:
        tasks = batch.get("task") or ["?"] * int(batch["observation.images.image"].shape[0])
        for i in range(int(batch["observation.images.image"].shape[0])):
            text = tasks[i] if isinstance(tasks[i], str) else str(tasks[i])
            short = text.split(",")[0].replace("Task: ", "").strip()[:80]
            if short in seen:
                continue
            seen.add(short)
            stem = f"task{len(records):02d}"
            rec = {"stem": stem, "task": short, "cameras": []}
            images = {}
            for key, meta in mapping.items():
                arr = _to_uint8(batch[key][i])
                path = out_dir / f"{stem}_{meta['role']}.png"
                Image.fromarray(arr).save(path)
                images[meta["role"]] = Image.fromarray(arr)
                rec["cameras"].append({**meta, "file": str(path.relative_to(FN_ROOT)), "shape": list(arr.shape)})
            pair = Image.new(
                "RGB",
                (images["global"].width + images["wrist"].width, max(images["global"].height, images["wrist"].height)),
            )
            pair.paste(images["global"], (0, 0))
            pair.paste(images["wrist"], (images["global"].width, 0))
            pair_path = out_dir / f"{stem}_pair_global_left_wrist_right.png"
            pair.save(pair_path)
            rec["pair"] = str(pair_path.relative_to(FN_ROOT))
            records.append(rec)
            if len(records) >= n_tasks:
                (out_dir / "camera_map.json").write_text(
                    json.dumps(
                        {
                            "note": "LIBERO env maps agentview→image (global) and wrist→image2.",
                            "source": str(DEFAULT_CALIB_FILE),
                            "examples": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                return records
    return records


def overlay_grid(chw: torch.Tensor, cam_meta: dict, path: Path) -> None:
    arr = _to_uint8(chw)
    im = Image.fromarray(arr).convert("RGBA").resize((256, 256), Image.NEAREST)
    draw = ImageDraw.Draw(im, "RGBA")
    cell = 256 // GRID
    rel = set(int(x) for x in cam_meta["relevance"])
    div = set(int(x) for x in cam_meta["diversity"])
    keep = set(int(x) for x in cam_meta["keep"])
    for idx in range(TOKENS_PER_VIEW):
        r, c = divmod(idx, GRID)
        box = [c * cell, r * cell, (c + 1) * cell - 1, (r + 1) * cell - 1]
        if idx in rel and idx not in div:
            draw.rectangle(box, outline=(220, 40, 40, 255), width=2)
        elif idx in div:
            draw.rectangle(box, outline=(40, 90, 230, 255), width=2)
        elif idx not in keep:
            draw.rectangle(box, fill=(0, 0, 0, 120))
    caption = f"{cam_meta['role']} keep={cam_meta['n_keep']} rel={len(rel)} div={len(div)}"
    draw.rectangle([0, 240, 255, 255], fill=(0, 0, 0, 180))
    draw.text((4, 241), caption, fill=(255, 255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").save(path)


def _action_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    x = a[:, :10, :7].float()
    y = b[:, :10, :7].float()
    d = x - y
    return {
        "max_abs": float(d.abs().max()),
        "mae": float(d.abs().mean()),
        "l2": float(d.flatten(1).norm(dim=-1).mean()),
        "finite": bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
    }


# Keep-all re-runs the same Drop-6 prefix + NFE=1 expert on identical embeddings
# and noise. Residual error is bf16/CUDA rounding on O(1) actions. 2e-2 is an
# upper bound (~2% of unit action), not a claim of bit-identity. Do not raise it
# to hide drift.
IDENTITY_MAX_ABS = 2e-2


def _budget_ok(cam: dict, budgets: dict) -> bool:
    if cam["role"] == "empty":
        return cam["n_keep"] == cam["n_full"] == TOKENS_PER_VIEW
    budget = budgets[cam["role"]]
    return (
        cam["n_keep"] == budget["relevance"] + budget["diversity"]
        and len(cam["relevance"]) == budget["relevance"]
        and len(cam["diversity"]) == budget["diversity"]
        and not (set(cam["relevance"]) & set(cam["diversity"]))
    )


def _scout_calls_ok(calls: list[int] | None) -> bool:
    if not calls or len(calls) < SCOUT_LAYER + 1:
        return False
    if calls[SCOUT_LAYER] < 1:
        return False
    return all(c == 0 for c in calls[SCOUT_LAYER + 1 :])


def run_model_checks(
    gpu: int,
    calib,
    n_frames: int,
    *,
    budget_name: str = "role",
) -> dict:
    budgets = resolve_budgets(budget_name)
    device = bind_cuda_device(gpu)
    print(f"bound physical GPU{gpu} -> {device} (current={torch.cuda.current_device()})", flush=True)
    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.n_action_steps = 10
    policy.config.num_inference_steps = NFE
    policy.config.compile_model = False
    policy.eval()
    layers = policy.model._paligemma_language_model_core().layers
    if SCOUT_LAYER in DROP:
        raise RuntimeError("scout layer is dropped")
    attn_mod = type(layers[SCOUT_LAYER].self_attn).__name__
    print(
        f"scout_layer={SCOUT_LAYER} still present ({attn_mod}); drop={DROP}; n_layers={len(layers)}",
        flush=True,
    )

    frames = []
    for batch in calib:
        for i in range(int(batch["observation.images.image"].shape[0])):
            frames.append(slice_batch(batch, i, i + 1))
            if len(frames) >= n_frames:
                break
        if len(frames) >= n_frames:
            break

    identity = []
    pruned = []
    viz_dir = OUT / ("viz" if budget_name == "role" else f"viz_{budget_name}")
    crop_kw = dict(drop_layers=DROP, scout_layer=SCOUT_LAYER, budgets=budgets, budget_name=budget_name)
    with torch.no_grad(), skip_llm_layers(policy, DROP, keep_kv=True):
        for fi, raw in enumerate(frames):
            batch = move_to_device(raw, device)
            gen = torch.Generator(device="cpu")
            gen.manual_seed(1000 + fi)
            noise = torch.randn(
                1,
                int(policy.config.chunk_size),
                int(policy.config.max_action_dim),
                generator=gen,
            ).to(device)
            policy.reset()
            base = policy.predict_action_chunk(batch, noise=noise, num_steps=NFE)
            with crop_visual_tokens(policy, keep_all=True, **crop_kw):
                policy.reset()
                same = policy.predict_action_chunk(batch, noise=noise, num_steps=NFE)
                id_stats = getattr(policy.model, "_crop_stats", None)
            with crop_visual_tokens(policy, keep_all=False, **crop_kw):
                policy.reset()
                crop = policy.predict_action_chunk(batch, noise=noise, num_steps=NFE)
                crop_stats = policy.model._crop_stats
            ident = _action_stats(same, base)
            crop_finite = bool(torch.isfinite(crop[:, :10, :7].float()).all())
            identity.append(ident)
            n_live = sum(1 for c in crop_stats.cameras if c["role"] != "empty")
            expected_short = crop_stats.full_len - (TOKENS_PER_VIEW // 2) * n_live
            offset_ok = id_stats is not None and (
                crop_stats.orig_action_offset == id_stats.orig_action_offset
                and id_stats.orig_action_offset == id_stats.short_pad_sum
                and crop_stats.orig_action_offset > crop_stats.short_pad_sum
            )
            budgets_ok = all(_budget_ok(c, budgets) for c in crop_stats.cameras)
            scout_ok = _scout_calls_ok(crop_stats.scout_layer_calls) and _scout_calls_ok(
                None if id_stats is None else id_stats.scout_layer_calls
            )
            length_ok = (
                crop_stats.short_len == expected_short
                and crop_stats.kv_len == crop_stats.short_len
                and crop_stats.short_len < crop_stats.full_len
                and (id_stats is None or id_stats.short_len == id_stats.full_len)
                and offset_ok
                and budgets_ok
                and scout_ok
                and ident["finite"]
                and crop_finite
            )
            pruned.append(
                {
                    "frame": fi,
                    "task": (raw.get("task") or ["?"])[0] if isinstance(raw.get("task"), list) else None,
                    "vs_baseline_max_abs": _action_stats(crop, base)["max_abs"],
                    "stats": stats_dict(crop_stats),
                    "length_ok": length_ok,
                    "offset_ok": offset_ok,
                    "budgets_ok": budgets_ok,
                    "scout_ok": scout_ok,
                    "crop_finite": crop_finite,
                    "identity_keep_all": ident,
                    "identity_full_len": None if id_stats is None else id_stats.full_len,
                    "identity_short_len": None if id_stats is None else id_stats.short_len,
                    "orig_action_offset": crop_stats.orig_action_offset,
                    "short_pad_sum": crop_stats.short_pad_sum,
                    "scout_layer_calls": crop_stats.scout_layer_calls,
                }
            )
            if fi < 3:
                for cam in crop_stats.cameras:
                    if cam["role"] == "empty":
                        continue
                    key = cam["key"]
                    overlay_grid(raw[key][0], cam, viz_dir / f"frame{fi:02d}_{cam['role']}.png")
            print(
                f"frame {fi}: identity max_abs={ident['max_abs']:.4g}  "
                f"full={crop_stats.full_len} short={crop_stats.short_len} kv={crop_stats.kv_len} "
                f"action_off={crop_stats.orig_action_offset} length_ok={length_ok}",
                flush=True,
            )

    payload = {
        "gpu": gpu,
        "drop": DROP,
        "nfe": NFE,
        "scout_layer": SCOUT_LAYER,
        "budget_name": budget_name,
        "budgets": budgets,
        "identity": identity,
        "pruned": pruned,
        "identity_max_abs_max": max(x["max_abs"] for x in identity),
        "identity_max_abs_limit": IDENTITY_MAX_ABS,
        "all_lengths_ok": all(p["length_ok"] for p in pruned),
    }
    json_name = "offline_check.json" if budget_name == "role" else f"offline_check_{budget_name}.json"
    payload_path = OUT / json_name
    payload_path.write_text(json.dumps(payload, indent=2) + "\n")
    payload["_path"] = str(payload_path)
    return payload


def cpu_select_selftest() -> None:
    torch.manual_seed(0)
    n = TOKENS_PER_VIEW
    feats = F.normalize(torch.randn(n, 8), dim=-1)
    score = torch.linspace(1.0, 0.0, n)
    keep, rel, div = select_one_camera(score, feats, n_rel=96, n_div=32)
    if len(keep) != 128 or keep != sorted(keep):
        raise SystemExit(f"select_one_camera keep={len(keep)} unsorted_or_wrong")
    if len(rel) != 96 or rel != list(range(96)):
        raise SystemExit(f"relevance should be the top-96 scores, got {rel[:8]}...")
    if set(div) & set(rel):
        raise SystemExit("diversity overlapped relevance")
    extra = greedy_diversity(feats, list(range(4)), 3)
    if len(extra) != 3 or any(i < 4 for i in extra):
        raise SystemExit(f"greedy_diversity extra={extra}")
    keep_all, _, _ = select_one_camera(score, feats, n_rel=96, n_div=32, keep_all=True)
    if keep_all != list(range(n)):
        raise SystemExit("keep_all did not return every token")
    mix, mix_rel, mix_div = select_one_camera(score, feats, n_rel=64, n_div=64)
    if len(mix) != 128 or len(mix_rel) != 64 or len(mix_div) != 64 or set(mix_rel) & set(mix_div):
        raise SystemExit(f"shared_mix 64+64 failed keep={len(mix)} rel={len(mix_rel)} div={len(mix_div)}")
    print("cpu select selftest ok", flush=True)


def main() -> None:
    init_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=4)
    parser.add_argument("--budget", default="role", help="role (default) or shared_mix")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cpu_selftest()
    cpu_select_selftest()
    calib = load_calib_data(DEFAULT_CALIB_FILE)
    recs = save_camera_examples(calib, OUT / "cameras")
    print(f"wrote {len(recs)} camera examples under {OUT / 'cameras'}", flush=True)
    for rec in recs:
        cams = ", ".join(f"{c['role']}={c['libero']}→{c['lerobot']}" for c in rec["cameras"])
        print(f"  {rec['stem']}: {rec['task']} | {cams}", flush=True)
    if args.cameras_only:
        return

    gpu = args.gpu
    if gpu is None:
        gpu, note = first_exclusive_gpu()
        print(note, flush=True)
        if gpu is None:
            raise SystemExit(
                "NO_EXCLUSIVE_GPU: skip model checks. Re-run with --gpu N when the card is idle. "
                + note
            )
    else:
        ok, why = gpu_is_exclusive(gpu)
        print(f"gpu{gpu}: {why}", flush=True)
        if not ok:
            raise SystemExit(f"GPU{gpu} is not exclusive ({why}). Will not load the policy.")

    payload = run_model_checks(gpu, calib, args.n_frames, budget_name=args.budget)
    ident = payload["identity_max_abs_max"]
    ok_len = payload["all_lengths_ok"]
    print(
        f"budget={args.budget} identity max_abs over frames={ident:.6g}  prune_length_ok={ok_len}  "
        f"json={payload.get('_path')}",
        flush=True,
    )
    if not all(x.get("finite", False) for x in payload["identity"]):
        raise SystemExit("keep-all or baseline action is non-finite; inspect before eval")
    if ident > IDENTITY_MAX_ABS:
        raise SystemExit(
            f"keep-all path drifted from baseline (max_abs={ident} > {IDENTITY_MAX_ABS}); "
            "inspect before eval"
        )
    if not ok_len:
        raise SystemExit(
            f"pruned prefix/KV/budget/offset/scout-stop check failed; inspect {payload.get('_path')}"
        )
    print("offline checks passed", flush=True)


if __name__ == "__main__":
    main()

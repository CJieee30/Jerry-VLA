#!/usr/bin/env python3
"""Skill-pooled independent Drop-9 ranks (Fisher / GateProbe / IGIA) on v044.

This is the Stage-1 redundancy detector, not Drop-6 + extra-3:
- Full model, no skip(U).
- Same measure_* functions, microbatch=2, scoring seed=42.
- GateProbe uses recompute=False; Fisher/IGIA use recompute=True.
- Calib is the AtomicVLA interior skill pools (32×8), not a new dump.

Control Drop-9 sets are the existing global ranks in results/ranks/ranks.json.
Do not compare these sets to greedy extra-3 {2,7,11}.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging
import time
from collections import Counter

import torch

from common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_GRAD_MICROBATCH,
    DEFAULT_RANKS_DIR,
    DEFAULT_SEED,
    FN_ROOT,
    importance_ranking,
    init_logging,
    load_calib_data,
    load_policy,
    paligemma_parts,
    prepare_policy_for_grad_scoring,
    print_ranking,
    save_scores,
    set_seed,
)
from drop6_extra3_protocol import (
    ATOMIC_SKILLS,
    N_LAYERS,
    dump_json,
    skill_calib_dir,
    wait_for_file,
)
from profiling_fisher import measure_fisher
from profiling_gate import measure_gate
from profiling_igia import measure_igia
from score_drop6_extra3 import maybe_load_done

DEFAULT_OUT_DIR = FN_ROOT / "results" / "drop9_skill"
DEFAULT_SCORES_DIR = FN_ROOT / "results" / "scores" / "skill_drop9"
DROP_K = 9
RANK_KS = (6, 9, 12, 16)


def global_drop_k(method: str, k: int = DROP_K, ranks_json: Path | None = None) -> list[int]:
    path = Path(ranks_json) if ranks_json is not None else DEFAULT_RANKS_DIR / "ranks.json"
    payload = json.loads(path.read_text())
    drop_tbl = payload["methods"][method]["drop"]
    entry = drop_tbl.get(str(k)) or drop_tbl.get(k)
    return [int(i) for i in entry["drop"]]


def vs_global(drop: list[int], glob: list[int]) -> dict:
    a, b = set(int(x) for x in drop), set(int(x) for x in glob)
    inter = a & b
    union = a | b
    return {
        "global_drop": sorted(b),
        "intersection": sorted(inter),
        "symdiff": sorted(a ^ b),
        "jaccard": (len(inter) / len(union)) if union else 1.0,
        "differs": a != b,
    }


def measure(method: str, policy, calib, device, microbatch_size):
    if method == "fisher":
        return measure_fisher(policy, calib, device, microbatch_size)
    if method == "gateprobe":
        return measure_gate(policy, calib, device, microbatch_size)
    if method == "igia":
        return measure_igia(policy, calib, device, microbatch_size)
    raise ValueError(method)


def prepare(method: str, policy) -> None:
    # Match profiling_*.py mains exactly.
    if method == "gateprobe":
        prepare_policy_for_grad_scoring(policy, recompute=False)
    else:
        prepare_policy_for_grad_scoring(policy, recompute=True)


def write_md(payload: dict, path: Path) -> None:
    glob = payload["global_drop9"]
    lines = [
        f"# Skill-aware independent Drop-9 ({payload['method']}, calib seed {payload['calib_seed']})",
        "",
        "Checkpoint: `lerobot/pi05_libero_finetuned_v044`. Full-model Stage-1 "
        "redundancy detector (no skip(U), not extra-3). "
        f"Calib: AtomicVLA verb pools `{payload['calib_dir']}` "
        f"{payload['n_batches']}×{payload['batch_size']}. "
        f"Scoring seed={payload['scoring_seed']}, microbatch={payload['microbatch_size']}.",
        "",
        f"- Global {payload['method']} Drop-9 (64×8 concat, not re-searched): `{glob}`",
        f"- Unique skill Drop-9 sets: **{payload['n_unique_drop9']}**",
        f"- Skills differing from global: `{payload['skills_diff_global']}`",
        "",
        "| Skill | Drop-9 | vs global symdiff | Jaccard | seconds |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in payload["skills"]:
        vs = row["vs_global"]
        lines.append(
            f"| `{row['skill']}` | `{row['drop9']}` | `{vs['symdiff']}` | "
            f"{vs['jaccard']:.3f} | {row['seconds']:.1f} |"
        )
    lines += ["", "## Drop-9 histogram", "", "| Drop-9 | Skills |", "| --- | --- |"]
    for drop, names in payload["histogram"].items():
        lines.append(f"| `{drop}` | {', '.join(f'`{n}`' for n in names)} |")
    lines += ["", f"JSON: `{payload.get('out_json', path.with_suffix('.json'))}`", ""]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("fisher", "gateprobe", "igia"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--microbatch-size", type=int, default=DEFAULT_GRAD_MICROBATCH)
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--calib-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--scores-dir", type=Path, default=None)
    parser.add_argument("--ranks-json", type=Path, default=DEFAULT_RANKS_DIR / "ranks.json")
    parser.add_argument("--wait-timeout", type=float, default=12 * 3600)
    args = parser.parse_args()

    init_logging()
    set_seed(args.seed)
    device = torch.device(args.device)
    calib_dir = Path(args.calib_dir) if args.calib_dir is not None else skill_calib_dir(args.calib_seed)
    out_dir = Path(args.out_dir) if args.out_dir is not None else DEFAULT_OUT_DIR / f"{args.method}_seed{args.calib_seed}"
    scores_dir = (
        Path(args.scores_dir)
        if args.scores_dir is not None
        else DEFAULT_SCORES_DIR / f"seed{args.calib_seed}" / args.method
    )
    per_dir = out_dir / "per_skill"
    per_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    glob9 = global_drop_k(args.method, DROP_K, args.ranks_json)
    logging.info(
        "Skill Drop-9 method=%s calib=%s global_drop9=%s recompute=%s",
        args.method,
        calib_dir,
        glob9,
        args.method != "gateprobe",
    )

    policy = load_policy(args.checkpoint, device)
    layers, _ = paligemma_parts(policy)
    if len(layers) != N_LAYERS:
        raise RuntimeError(f"expected {N_LAYERS} PaliGemma layers, got {len(layers)}")
    prepare(args.method, policy)

    rows = []
    for verb in ATOMIC_SKILLS:
        rec_path = per_dir / f"{verb}.json"
        score_path = scores_dir / f"{verb}.pt"
        done = maybe_load_done(rec_path)
        if done is not None and "drop9" in done:
            logging.info("[skip] %s %s drop9=%s", args.method, verb, done["drop9"])
            rows.append(done)
            continue

        calib_file = wait_for_file(calib_dir / f"{verb}.pt", timeout_s=args.wait_timeout)
        calib = load_calib_data(calib_file)
        logging.info("[%s] %s batches=%d", args.method, verb, len(calib))
        t0 = time.time()
        results = measure(args.method, policy, calib, device, args.microbatch_size)
        scores = results["paligemma"]
        ranking = importance_ranking(scores, ks=RANK_KS)
        drop9 = [int(i) for i in ranking["drop"][DROP_K]["drop"]]
        rec = {
            "skill": verb,
            "method": args.method,
            "calib_seed": args.calib_seed,
            "calib_file": str(calib_file),
            "n_calib_batches": len(calib),
            "scores": [float(x) for x in scores.detach().cpu().flatten().tolist()],
            "redundancy_order": ranking["redundancy_order"],
            "drop": {str(k): v for k, v in ranking["drop"].items()},
            "drop9": drop9,
            "keep9": ranking["drop"][DROP_K]["keep"],
            "vs_global": vs_global(drop9, glob9),
            "aggregation": results.get("aggregation"),
            "n_orig_batches": results.get("n_orig_batches"),
            "seconds": round(time.time() - t0, 3),
            "score_file": str(score_path),
        }
        save_scores(score_path, results)
        dump_json(rec, rec_path)
        rows.append(rec)
        print_ranking(f"{args.method}:{verb}", scores, ks=RANK_KS)
        logging.info("%s %s drop9=%s vs_global_symdiff=%s (%.1fs)", args.method, verb, drop9, rec["vs_global"]["symdiff"], rec["seconds"])
        dump_json(
            {"method": args.method, "n_done": len(rows), "n_total": len(ATOMIC_SKILLS), "skills": rows},
            out_dir / "partial.json",
        )

    hist = Counter(str(row["drop9"]) for row in rows)
    grouped: dict[str, list[str]] = {}
    for drop, _ in hist.most_common():
        grouped[drop] = [row["skill"] for row in rows if str(row["drop9"]) == drop]
    payload = {
        "method": args.method,
        "scope": "skill_independent_drop9",
        "checkpoint": args.checkpoint,
        "calib_dir": str(calib_dir),
        "calib_seed": args.calib_seed,
        "scoring_seed": args.seed,
        "microbatch_size": args.microbatch_size,
        "n_batches": 32,
        "batch_size": 8,
        "skip": "none (full model; Stage-1 independent Drop-9)",
        "global_drop9": glob9,
        "n_unique_drop9": len(hist),
        "skills_diff_global": [row["skill"] for row in rows if row["vs_global"]["differs"]],
        "histogram": grouped,
        "skills": rows,
        "out_json": str(out_dir / "drop9.json"),
        "note": "Not extra-3. Do not compare these drop-9 sets to greedy {2,7,11}.",
    }
    dump_json(payload, out_dir / "drop9.json")
    write_md(payload, out_dir / "drop9.md")
    print((out_dir / "drop9.md").read_text())


if __name__ == "__main__":
    main()

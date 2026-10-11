#!/usr/bin/env python3
"""Collect LEDGE v2 targets: closed-loop drop sets with known SR and the orders that generate them."""

from __future__ import annotations

import json
from pathlib import Path

import torch

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
RES = FN / "results"
AW = RES / "stage2_action_weight"
OUT = RES / "stage3_ledge" / "v2_targets.json"

UNIFORM = ("fisher", "taylor", "gateprobe", "igia", "ppl", "cka", "cossim", "magnitude", "hessian")
CARE = ("fisher", "taylor", "gateprobe", "igia")


def redundancy_order(scores) -> list[int]:
    values = [float(x) for x in scores]
    keep = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return [i for i in reversed(keep) if i != 0]


def _scores(path: Path, key: str):
    if not path.exists():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob.get(key)


def _sr(path: Path) -> tuple[float, int]:
    overall = json.loads(path.read_text())["overall"]
    return float(overall["pc_success"]), int(overall["n_episodes"])


def pi05_sets() -> list[dict]:
    rows = []
    for d in sorted((RES / "eval_drop").iterdir()):
        info, meta = d / "eval_info.json", d / "drop_meta.json"
        if not (info.exists() and meta.exists()):
            continue
        m = json.loads(meta.read_text())
        if m.get("checkpoint") != "lerobot/pi05_libero_finetuned_v044" or m.get("task_extra_map"):
            continue
        sr, n = _sr(info)
        rows.append({"tag": d.name, "drop": sorted(int(i) for i in m.get("drop_layers") or []), "sr": sr, "n": n})
    return rows


def _drop_json_sets(root: Path) -> list[dict]:
    rows = []
    for d in sorted(root.iterdir()):
        info, meta = d / "eval_info.json", d / "drop.json"
        if not (d.is_dir() and info.exists() and meta.exists()):
            continue
        m = json.loads(meta.read_text())
        sr, n = _sr(info)
        rows.append({"tag": d.name, "drop": sorted(int(i) for i in m["drop"]), "sr": sr, "n": n})
    return rows


def orders(stack: str) -> dict[str, list[int]]:
    out = {}
    if stack == "pi05":
        for m in UNIFORM:
            s = _scores(RES / "scores" / f"{m}.pt", "paligemma")
            if s is not None:
                out[f"{m}_uniform"] = redundancy_order(s)
        for m in CARE:
            s = _scores(AW / f"pi05_{m}_action_discontinuity.pt", "paligemma")
            if s is not None:
                out[f"{m}_care"] = redundancy_order(s)
    else:
        root = RES / f"{stack}_libero" / "scores"
        for m in UNIFORM:
            s = _scores(root / f"{m}.pt", "scores")
            if s is not None:
                out[f"{m}_uniform"] = redundancy_order(s)
        for m in CARE:
            s = _scores(AW / f"{stack}_{m}_action_discontinuity.pt", "scores")
            if s is not None:
                out[f"{m}_care"] = redundancy_order(s)
            s = _scores(AW / f"{stack}_{m}_action_discontinuity_arm6.pt", "scores")
            if s is not None:
                out[f"{m}_care_arm6"] = redundancy_order(s)
    return out


def main() -> None:
    stacks = {
        "pi05": pi05_sets(),
        "molmoact2": _drop_json_sets(RES / "molmoact2_libero"),
        "smolvla": _drop_json_sets(RES / "smolvla_libero"),
    }
    anchors = {"pi05": 97.25, "molmoact2": 97.50, "smolvla": 83.00}
    payload = {}
    for stack, sets in stacks.items():
        ords = orders(stack)
        by_set = {}
        for row in sets:
            by_set.setdefault(tuple(row["drop"]), []).append(row)
        hits = {}
        for name, order in ords.items():
            for k in range(1, len(order) + 1):
                key = tuple(sorted(order[:k]))
                if key in by_set:
                    hits.setdefault(name, {})[k] = [r["tag"] for r in by_set[key]]
        payload[stack] = {"anchor": anchors[stack], "sets": sets, "orders": ords, "order_hits": hits}
        print(f"== {stack}: {len(sets)} sets, {len(by_set)} unique")
        for name, h in hits.items():
            print(f"  {name}: " + ", ".join(f"k{k}={'/'.join(t)}" for k, t in sorted(h.items())))
        unmatched = [r["tag"] for r in sets if not any(r["tag"] in t for h in hits.values() for t in h.values())]
        print(f"  not on any order: {unmatched}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1) + "\n")


if __name__ == "__main__":
    main()

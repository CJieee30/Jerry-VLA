#!/usr/bin/env python3
"""Write v044 Drop-6 + extra-3 eval maps: f(task) and same-multiset shuffled."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

from drop6_extra3_protocol import (
    DEFAULT_MAP_DIR,
    DEFAULT_OUT_DIR,
    EVAL_SHUFFLE_SEED,
    dump_json,
    make_eval_map,
    shuffle_eval_map,
)


def load_scored(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if len(payload.get("tasks", [])) != 40:
        raise RuntimeError(f"{path} has {len(payload.get('tasks', []))} tasks")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MAP_DIR)
    parser.add_argument("--shuffle-seed", type=int, default=EVAL_SHUFFLE_SEED)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fisher = load_scored(DEFAULT_OUT_DIR / "fisher" / "extra3.json")
    greedy = load_scored(DEFAULT_OUT_DIR / "greedy_ppl" / "extra3.json")
    jobs = [
        ("map_fisher_extra3.json", "fisher_extra3", "fisher", fisher["tasks"]),
        ("map_greedy_ppl_extra3.json", "greedy_ppl_extra3", "greedy_ppl", greedy["tasks"]),
    ]
    for name, kind, method, tasks in jobs:
        src = make_eval_map(kind=kind, method=method, tasks=tasks)
        dump_json(src, out_dir / name)
        shuf = shuffle_eval_map(src, seed=args.shuffle_seed)
        dump_json(shuf, out_dir / name.replace(".json", "_shuffled.json"))
        print(
            f"{kind}: unique={src['summary']['n_unique_triples']} "
            f"shuffle_same={shuf['n_same_as_source']}/40 -> {out_dir / name}"
        )


if __name__ == "__main__":
    main()

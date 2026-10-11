#!/usr/bin/env python3
"""400-episode LIBERO eval for lerobot/smolvla_libero Drop-k (keep-KV).

Native Drop-k from scores.md: the k lowest VLM-layer scores, including
layer 0 when the metric ranks it redundant. Skip is keep-KV identity residual.
SMOLVLA_DROP_METHOD selects the metric, SMOLVLA_DROP_K selects k (default 4).
Protocol matches the nas=10 baseline: seed=1000, 4 suites × 10 tasks × 10
episodes, relative control, camera rename_map, state dim 8.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layer_skip
from adapt_lerobot_official import (
    CKPT,
    LIBERO_STATE_DIM,
    N_ACTION_STEPS,
    N_EPISODES,
    NUM_STEPS,
    RENAME_MAP,
    SEED,
    SUITES,
    eval_argv,
    install_eval_patches,
)

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
OUT_ROOT = FN / "results" / "smolvla_libero"

# Native Drop-4 from results/smolvla_libero/scores.md (high score = keep).
DROP4 = {
    "magnitude": [0, 3, 7, 8],
    "cossim": [2, 4, 5, 6],
    "cka": [1, 2, 4, 10],
    "ppl": [11, 13, 14, 15],
    "taylor": [12, 13, 14, 15],
    "fisher": [10, 11, 12, 14],
    "igia": [10, 12, 14, 15],
    "gateprobe": [7, 13, 14, 15],
    "hessian": [8, 10, 12, 14],
    # Stage-2 action-discontinuity weighted Fisher; not the Stage-1 set.
    "fisher_action_discontinuity": [10, 12, 13, 14],
    # Same layers as Stage-1 Fisher Drop-4; new output dir so the 69.25 run is not overwritten.
    "fisher_reval": [10, 11, 12, 14],
}

# Native Drop-3. Fisher equals IGIA at k=3, so the IGIA run covers both.
DROP3 = {
    "taylor": [13, 14, 15],
    "igia": [10, 12, 14],
    "gateprobe": [7, 14, 15],
}
SHARED3 = {"igia": ["igia", "fisher"]}

# Native Drop-2. All four sets are distinct (Fisher ≠ IGIA).
DROP2 = {
    "taylor": [14, 15],
    "igia": [12, 14],
    "fisher": [10, 14],
    "gateprobe": [7, 15],
}

# Native Drop-1. [15] covers Taylor / GateProbe / PPL; [14] covers IGIA / Fisher.
DROP1 = {
    "taylor": [15],
    "igia": [14],
    "fisher": [14],
    "gateprobe": [15],
    "ppl": [15],
}
SHARED1 = {
    "taylor": ["taylor", "gateprobe", "ppl"],
    "igia": ["igia", "fisher"],
}

DROPS = {4: DROP4, 3: DROP3, 2: DROP2, 1: DROP1}
SHARED = {4: {}, 3: SHARED3, 2: {}, 1: SHARED1}


def main() -> None:
    method = os.environ.get("SMOLVLA_DROP_METHOD", "").strip().lower()
    k = int(os.environ.get("SMOLVLA_DROP_K", "4"))
    table = DROPS.get(k)
    if table is None or method not in table:
        raise SystemExit(f"no Drop-{k} set for {method!r}; known: {list(table or {})}")
    drop = table[method]
    out = OUT_ROOT / f"drop{k}_{method}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "drop.json").write_text(
        json.dumps(
            {
                "checkpoint": CKPT,
                "method": method,
                "methods": SHARED[k].get(method, [method]),
                "k": k,
                "drop": drop,
                "operator": "keep_kv_identity_residual",
                "exclude_layer0": False,
                "n_action_steps": N_ACTION_STEPS,
                "num_steps": NUM_STEPS,
                "seed": SEED,
                "n_episodes": N_EPISODES,
                "suites": SUITES.split(","),
                "rename_map": RENAME_MAP,
                "state_dim": LIBERO_STATE_DIM,
            },
            indent=2,
        )
        + "\n"
    )
    layer_skip.install()
    layer_skip._DROP = set(drop)
    install_eval_patches()
    sys.argv = eval_argv(str(out), n_episodes=N_EPISODES, task=SUITES)
    from lerobot.scripts.lerobot_eval import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()

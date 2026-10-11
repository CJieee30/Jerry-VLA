#!/usr/bin/env python3
"""LIBERO eval of π0.5 with identity-skipped SigLIP layers.

Language layers stay intact. NFE stays at the checkpoint default (10).
Drop lists are the four lowest Taylor / Fisher scores from
results/stage_nfe/vision_probe_pi05.json. Skip returns the block input
unchanged. This is not keep-KV.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))

DROPS = {
    "fisher": [21, 22, 20, 17],
    "taylor": [22, 21, 23, 20],
}


def _identity(hidden_states, attention_mask=None, **kwargs):
    return hidden_states


def install(drop: set[int]) -> None:
    import lerobot.policies as policies
    import lerobot.policies.factory as factory

    orig = factory.make_policy

    def wrapped(*args, **kwargs):
        policy = orig(*args, **kwargs)
        layers = list(
            policy.model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers
        )
        hit = []
        for i, layer in enumerate(layers):
            if i in drop:
                layer.forward = _identity
                hit.append(i)
        if hit != sorted(drop):
            raise RuntimeError(f"vision skip {hit} != {sorted(drop)} on {len(layers)} layers")
        print(f"vision identity skip {hit} / {len(layers)}", flush=True)
        return policy

    factory.make_policy = wrapped
    policies.make_policy = wrapped


def main() -> None:
    raw = os.environ.get("VISION_DROP", "").strip()
    if raw:
        drop = [int(part) for part in raw.split(",") if part.strip()]
        tag = os.environ["VISION_TAG"].strip()
        method = os.environ.get("VISION_ORDER", "manual").strip()
        k = len(drop)
        lang_raw = os.environ.get("VISION_LANG", "1,13,14,15,16,17").strip()
        if lang_raw.lower() in ("", "none", "baseline"):
            lang = []
        else:
            lang = [int(part) for part in lang_raw.split(",") if part.strip()]
    else:
        method = os.environ.get("VISION_DROP_METHOD", "").strip().lower()
        if method not in DROPS:
            raise SystemExit(f"VISION_DROP_METHOD must be one of {sorted(DROPS)}")
        drop = DROPS[method]
        tag = f"pi05_vision_{method}_drop4"
        k = 4
        lang = []
    nfe = int(os.environ.get("VISION_NFE", "10"))
    n_ep = int(os.environ.get("VISION_N_EPISODES", "10"))
    out = FN / "results" / "stage_nfe" / tag
    if (out / "eval_info.json").is_file():
        raise SystemExit(f"ALREADY_DONE {tag}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "vision_drop.json").write_text(
        json.dumps(
            {
                "checkpoint": "lerobot/pi05_libero_finetuned_v044",
                "module": "siglip",
                "order": method,
                "k": k,
                "drop": drop,
                "operator": "identity_residual",
                "language_drop": lang,
                "language_operator": "dtr_appendix_b_prefix_kv" if lang else None,
                "num_inference_steps": nfe,
                "n_action_steps": 10,
                "n_episodes_per_task": n_ep,
                "seed": 1000,
            },
            indent=2,
        )
        + "\n"
    )
    install(set(drop))
    sys.argv = [
        "eval_drop.py",
        "--output_dir",
        str(out),
        "--job_name",
        tag,
        "--policy.path=lerobot/pi05_libero_finetuned_v044",
        "--policy.n_action_steps=10",
        f"--policy.num_inference_steps={nfe}",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=false",
        "--env.type=libero",
        "--env.task=libero_spatial,libero_object,libero_goal,libero_10",
        "--env.control_mode=relative",
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=1",
        f"--eval.n_episodes={n_ep}",
        "--seed=1000",
    ]
    if lang:
        sys.argv[1:1] = ["--drop-layers", ",".join(str(i) for i in lang), "--method", "fisher", "--k", "6"]
    else:
        sys.argv[1:1] = ["--method", "baseline"]
    import eval_drop

    eval_drop.main()


if __name__ == "__main__":
    main()

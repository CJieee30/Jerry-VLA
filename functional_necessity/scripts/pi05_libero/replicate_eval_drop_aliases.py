#!/usr/bin/env python3
"""Copy unique Drop-3 eval dirs onto methods that share the same drop set.

Closed-loop 400-ep is identical under seed=1000 when drop_layers match, so
Fisher/IGIA/Hessian/Taylor and PPL/GateProbe share one run each († in the table).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from common import FN_ROOT

EVAL_DIR = FN_ROOT / "results" / "eval_drop"

# source tag -> alias tags (same drop_layers as ranks.json k=3)
ALIASES = {
    "k3_fisher": ["k3_taylor", "k3_igia", "k3_hessian"],
    "k3_ppl": ["k3_gateprobe"],
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def replicate_one(src_tag: str, dst_tag: str) -> str:
    src = EVAL_DIR / src_tag
    dst = EVAL_DIR / dst_tag
    src_info = src / "eval_info.json"
    src_meta = src / "drop_meta.json"
    if not src_info.is_file() or not src_meta.is_file():
        return f"skip {dst_tag}: source {src_tag} incomplete"
    if (dst / "eval_info.json").is_file():
        return f"skip {dst_tag}: eval_info.json already exists"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_info, dst / "eval_info.json")
    meta = _load_json(src_meta)
    method = dst_tag.split("_", 1)[1]
    meta["method"] = method
    meta["job_name"] = dst_tag
    meta["shared_eval_from"] = src_tag
    (dst / "drop_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return f"ok {dst_tag} <- {src_tag} drop={meta.get('drop_layers')}"


def main() -> int:
    lines = []
    for src, dsts in ALIASES.items():
        for dst in dsts:
            lines.append(replicate_one(src, dst))
    text = "\n".join(lines)
    print(text)
    return 0 if lines else 1


if __name__ == "__main__":
    sys.exit(main())

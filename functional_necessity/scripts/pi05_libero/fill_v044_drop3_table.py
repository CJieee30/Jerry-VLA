#!/usr/bin/env python3
"""Fill Drop-3 rows in eval_pi05_libero_v044.md once unique eval_info exist."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from common import FN_ROOT

EVAL_DIR = FN_ROOT / "results" / "eval_drop"
MD = FN_ROOT / "main_results" / "eval_pi05_libero_v044.md"

UNIQUE = ["k3_fisher", "k3_ppl", "k3_magnitude", "k3_cossim", "k3_cka"]
ALL_TAGS = [
    "k3_fisher",
    "k3_igia",
    "k3_ppl",
    "k3_gateprobe",
    "k3_hessian",
    "k3_cka",
    "k3_cossim",
    "k3_taylor",
    "k3_magnitude",
]
DISPLAY = {
    "k3_fisher": "Fisher",
    "k3_igia": "IGIA",
    "k3_ppl": "PPL",
    "k3_gateprobe": "GateProbe",
    "k3_hessian": "Hessian",
    "k3_cka": "CKA",
    "k3_cossim": "CosSim",
    "k3_taylor": "Taylor",
    "k3_magnitude": "Magnitude",
}
SHARED = {
    "k3_fisher",
    "k3_igia",
    "k3_hessian",
    "k3_taylor",
    "k3_ppl",
    "k3_gateprobe",
}
SUITE = [
    ("libero_spatial", "Spatial"),
    ("libero_object", "Object"),
    ("libero_goal", "Goal"),
    ("libero_10", "Long"),
]


def _pc(info: dict, key: str) -> float:
    return float(info[key]["pc_success"])


def _fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return f"{x:.1f}"
    return f"{x:.2f}"


def status() -> dict[str, bool]:
    return {tag: (EVAL_DIR / tag / "eval_info.json").is_file() for tag in UNIQUE}


def _row(tag: str, info: dict, *, first: bool) -> str:
    dag = "†" if tag in SHARED else ""
    label = DISPLAY[tag] + dag
    setting = "Drop-3 (3/18)" if first else ""
    cells = [
        setting,
        label,
        _fmt(_pc(info, "libero_spatial")),
        _fmt(_pc(info, "libero_object")),
        _fmt(_pc(info, "libero_goal")),
        _fmt(_pc(info, "libero_10")),
        _fmt(_pc(info, "overall")),
    ]
    return "| " + " | ".join(cells) + " |"


def build_rows() -> tuple[list[str], list[tuple[str, float]]]:
    scored = []
    for tag in ALL_TAGS:
        path = EVAL_DIR / tag / "eval_info.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        info = json.loads(path.read_text())
        scored.append((tag, info, _pc(info, "overall")))
    scored.sort(key=lambda t: (-t[2], ALL_TAGS.index(t[0])))
    rows = []
    for i, (tag, info, _) in enumerate(scored):
        rows.append(_row(tag, info, first=i == 0))
    deltas = [(DISPLAY[tag], avg - 97.25) for tag, _, avg in scored]
    return rows, deltas


def _drop_keep(tag: str) -> tuple[list[int], list[int]]:
    meta = json.loads((EVAL_DIR / tag / "drop_meta.json").read_text())
    drop = [int(i) for i in meta["drop_layers"]]
    keep = [int(i) for i in meta["keep_layers"]]
    return drop, keep


def render_index_table() -> str:
    lines = [
        "### Drop-3",
        "",
        "| Metric | Drop | Keep |",
        "| --- | --- | --- |",
    ]
    for tag in ALL_TAGS:
        drop, keep = _drop_keep(tag)
        dag = "†" if tag in SHARED else ""
        lines.append(f"| {DISPLAY[tag]}{dag} | {drop} | {keep} |")
    return "\n".join(lines)


def fill_md(rows: list[str], deltas: list[tuple[str, float]]) -> None:
    text = MD.read_text()
    text = text.replace(
        "Closed-loop LIBERO results (Drop-6 / Drop-9, no recovery)",
        "Closed-loop LIBERO results (Drop-3 / Drop-6 / Drop-9, no recovery)",
    )
    text = text.replace(
        "All nine Drop-6 sets and all nine independent Drop-9 sets are unique; no † merge.",
        "Drop-3 has five unique drop-sets; Fisher/IGIA/Hessian/Taylor share `[15, 16, 17]` and PPL/GateProbe share `[14, 16, 17]` († one 400-ep). All nine Drop-6 sets and all nine independent Drop-9 sets are unique; no † merge at those k.",
    )
    text = text.replace(
        "Source: `results/eval_drop/{baseline,k6_*,k9_*}/{eval_info.json,drop_meta.json}`.",
        "Source: `results/eval_drop/{baseline,k3_*,k6_*,k9_*}/{eval_info.json,drop_meta.json}`.",
    )
    delta_bits = ", ".join(
        f"{name} {'+' if d >= 0 else ''}{d:.2f}" for name, d in deltas
    )
    if "Drop-3:" not in text.split("Δ Avg.")[1][:400] if "Δ Avg." in text else True:
        text = text.replace(
            "- Drop-6:",
            f"- Drop-3: {delta_bits}.\n- Drop-6:",
            1,
        )
    table_block = "\n".join(rows) + "\n"
    marker = "| Drop-6 (6/18) |"
    if "| Drop-3 (3/18) |" in text:
        text = re.sub(
            r"\| Drop-3 \(3/18\).*\n(?:\|  \|.*\n)+",
            table_block,
            text,
            count=1,
        )
    else:
        text = text.replace(marker, table_block + marker, 1)
    idx = render_index_table() + "\n\n"
    if "### Drop-3" in text:
        text = re.sub(r"### Drop-3\n\n\| Metric \| Drop \| Keep \|.*?(?=\n### Drop-6)", idx, text, flags=re.S)
    else:
        text = text.replace("### Drop-6", idx + "### Drop-6", 1)
    running = "## Drop-3 (running)\n\n"
    if running in text:
        text = text.replace(
            running,
            "",
            1,
        )
        text = re.sub(
            r"tmux: `fn-v044-k3-g\{0,1,2,3\}`.*?\n\n",
            "",
            text,
            count=1,
            flags=re.S,
        )
    MD.write_text(text)
    print(f"updated {MD}")


def main() -> int:
    st = status()
    missing = [t for t, ok in st.items() if not ok]
    if missing:
        print("waiting for unique eval_info:", ", ".join(missing))
        return 2
    from replicate_eval_drop_aliases import main as replicate

    replicate()
    rows, deltas = build_rows()
    fill_md(rows, deltas)
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())

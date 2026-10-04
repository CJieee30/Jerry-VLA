#!/usr/bin/env python3
"""Compare our keep-sets on pi05_base against DTR Table 9 (π0.5 / LIBERO)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# DTR Appendix F / Table 9. Keep blocks; unlisted are dropped.
# https://arxiv.org/html/2606.27755v1
DTR_KEEP = {
    "gateprobe": {
        9: [0, 1, 2, 3, 4, 5, 6, 8, 9],
        12: [0, 1, 2, 5, 6, 8],
        16: [0, 5],
        17: [0],
    },
    "taylor": {
        9: [0, 1, 3, 4, 5, 6, 7, 8, 9],
        12: [0, 3, 4, 5, 6, 8],
        16: [5, 6],
        17: [5],
    },
    "igia": {
        9: [0, 1, 3, 4, 5, 6, 7, 8, 9],
        12: [0, 1, 3, 4, 6, 7],
        16: [0, 7],
        17: [0],
    },
    "fisher": {
        9: [0, 1, 2, 3, 4, 5, 6, 7, 8],
        12: [0, 1, 5, 6, 7, 8],
        16: [0, 5],
        17: [0],
    },
    "hessian": {
        9: [0, 1, 2, 3, 5, 8, 9, 11, 13],
        12: [0, 1, 3, 8, 9, 13],
        16: [0, 1],
        17: [0],
    },
    "ppl": {
        9: [0, 1, 4, 7, 8, 10, 11, 13, 14],
        12: [0, 1, 4, 7, 10, 13],
        16: [0, 13],
        17: [0],
    },
    "magnitude": {
        9: [2, 3, 6, 12, 13, 14, 15, 16, 17],
        12: [2, 13, 14, 15, 16, 17],
        16: [16, 17],
        17: [17],
    },
    "cossim": {
        9: [0, 1, 2, 12, 13, 14, 15, 16, 17],
        12: [0, 1, 14, 15, 16, 17],
        16: [16, 17],
        17: [17],
    },
}


def _jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranks-json",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "ranks_pi05_base" / "ranks.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "ranks_pi05_base" / "DTR_COMPARE.md",
    )
    args = parser.parse_args()
    payload = json.loads(args.ranks_json.read_text())
    methods = payload["methods"]

    lines = [
        "# pi05_base vs DTR Table 9",
        "",
        "DTR scored pretrained π0.5 on LIBERO calibration (64×8, seed 42), then reported keep-sets.",
        "Exact match = same keep set. Jaccard is |A∩B|/|A∪B| over keep indices.",
        "",
        "| Method | k | Ours keep | DTR keep | Match | Jaccard |",
        "|---|---|---|---|---|---|",
    ]
    n_ok = 0
    n_tot = 0
    for name, dtr_by_k in DTR_KEEP.items():
        if name not in methods:
            lines.append(f"| {name} | — | missing | — | skip | — |")
            continue
        for k, dtr_keep in dtr_by_k.items():
            report = methods[name]
            sets = report["drop"].get(str(k)) or report["drop"].get(k)
            if not sets:
                lines.append(f"| {name} | {k} | missing | {dtr_keep} | skip | — |")
                continue
            ours = list(sets["keep"])
            match = ours == dtr_keep
            n_tot += 1
            n_ok += int(match)
            jac = _jaccard(ours, dtr_keep)
            lines.append(
                f"| {name} | Drop-{k} | {ours} | {dtr_keep} | {'yes' if match else 'no'} | {jac:.2f} |"
            )

    lines += [
        "",
        f"**Exact keep-set matches: {n_ok}/{n_tot}**",
        "",
        "Magnitude is data-independent: an exact match means LeRobot `pi05_base` language-layer",
        "weights agree with the OpenPI checkpoint DTR used. Gradient methods can differ slightly",
        "because of BF16, preprocessor, and microbatch reconstruction.",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    print(f"exact matches: {n_ok}/{n_tot}")


if __name__ == "__main__":
    main()

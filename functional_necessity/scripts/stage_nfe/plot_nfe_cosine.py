#!/usr/bin/env python3
"""Publication heatmaps for π0.5 NFE cosine redundancy.

Two figures (PDF + PNG, 300 dpi, fonttype 42):
  1. Velocity field along the 10-step Euler path (v_k vs v_ℓ)
  2. Independent-NFE final actions (NFE i vs NFE j)

Color scale is shared: [0.97, 1.00]. Every measured entry is above 0.97.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
SRC = FN / "results" / "stage_nfe" / "nfe_cosine_pi05.json"
OUT_DIR = FN / "main_results" / "figures"
VMIN, VMAX = 0.97, 1.00

# Sequential warm map: pale cream → paper red. High cosine stays saturated.
CMAP = LinearSegmentedColormap.from_list(
    "nfe_redundancy",
    ["#fff7ec", "#fee8c8", "#fdbb84", "#fc8d59", "#e34a33", "#b30000"],
    N=256,
)


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def _text_color(val: float) -> str:
    norm = (val - VMIN) / (VMAX - VMIN)
    return "white" if norm > 0.62 else "#1a1a1a"


def plot_heatmap(
    mat: np.ndarray,
    xticks: list[str],
    yticks: list[str],
    xlabel: str,
    ylabel: str,
    stem: Path,
    cbar_label: str = "Cosine similarity",
) -> None:
    n = mat.shape[0]
    cell = 0.42 if n <= 6 else 0.36
    fig, ax = plt.subplots(figsize=(1.35 + cell * n + 0.85, 1.15 + cell * n))
    im = ax.imshow(mat, cmap=CMAP, vmin=VMIN, vmax=VMAX, origin="upper", aspect="equal")
    ax.set_xticks(range(n), xticks)
    ax.set_yticks(range(n), yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#444444")
    fmt = ".4f" if n <= 6 else ".3f"
    fs = 7.6 if n <= 6 else 6.2
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                format(mat[i, j], fmt),
                ha="center",
                va="center",
                fontsize=fs,
                color=_text_color(float(mat[i, j])),
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.set_ticks([0.97, 0.98, 0.99, 1.00])
    cbar.ax.tick_params(length=2, labelsize=8)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main() -> None:
    _setup_style()
    payload = json.loads(SRC.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    v = np.asarray(payload["cosine_v_path"], dtype=np.float64)
    v_ticks = [rf"$v_{{{k}}}$" for k in range(v.shape[0])]
    plot_heatmap(
        v,
        v_ticks,
        v_ticks,
        r"Euler step $\ell$",
        r"Euler step $k$",
        OUT_DIR / "nfe_cosine_velocity",
    )

    nfe_order = payload.get("nfe_order")
    act = payload.get("cosine_action_nfe")
    if nfe_order is None or act is None:
        raise SystemExit("JSON missing cosine_action_nfe / nfe_order; re-run probe_nfe_cosine.py")
    a = np.asarray(act, dtype=np.float64)
    a_ticks = [str(n) for n in nfe_order]
    plot_heatmap(
        a,
        a_ticks,
        a_ticks,
        r"NFE $j$",
        r"NFE $i$",
        OUT_DIR / "nfe_cosine_action",
    )
    # Two-panel figure for a paper figure* environment.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.22, 0.90]})
    panels = [
        (axes[0], v, v_ticks, v_ticks, r"Euler step $\ell$", r"Euler step $k$", ".3f", 6.6, r"(a)"),
        (axes[1], a, a_ticks, a_ticks, r"NFE $j$", r"NFE $i$", ".4f", 8.0, r"(b)"),
    ]
    for ax, mat, xt, yt, xl, yl, fmt, fs, tag in panels:
        im = ax.imshow(mat, cmap=CMAP, vmin=VMIN, vmax=VMAX, origin="upper", aspect="equal")
        n = mat.shape[0]
        ax.set_xticks(range(n), xt)
        ax.set_yticks(range(n), yt)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.tick_params(length=0)
        ax.set_title(tag, loc="left", fontsize=11, pad=6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#444444")
        for i in range(n):
            for j in range(n):
                ax.text(
                    j, i, format(mat[i, j], fmt),
                    ha="center", va="center", fontsize=fs,
                    color=_text_color(float(mat[i, j])),
                )
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Cosine similarity")
    cbar.set_ticks([0.97, 0.98, 0.99, 1.00])
    fig.savefig(OUT_DIR / "nfe_cosine_pair.pdf")
    fig.savefig(OUT_DIR / "nfe_cosine_pair.png", dpi=300)
    plt.close(fig)

    print("wrote", OUT_DIR / "nfe_cosine_velocity.pdf")
    print("wrote", OUT_DIR / "nfe_cosine_action.pdf")
    print("wrote", OUT_DIR / "nfe_cosine_pair.pdf")
    print("min_v", float(v.min()), "min_action", float(a.min()))


if __name__ == "__main__":
    main()

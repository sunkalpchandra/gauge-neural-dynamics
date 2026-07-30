"""Figure 2: architecture and objective.

The point the diagram has to make is that only one module is context dependent.
The encoder and decoder never receive the context, so any difference between
contexts must be carried by ``T_c`` -- which is what makes the model falsifiable
rather than merely flexible.  The lower row shows how ``T_c`` is built: a context
embedding supplies coordinates in a learned Lie algebra, and the matrix
exponential turns those coordinates into a group element.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from ..utils.common import FIGURE_DIR
from .style import (
    CAT, CRITICAL, GRID, INK, INK2, MUTED, OURS, OURS_ALT, SURFACE,
    bare, savefig, use_paper_style,
)

SHARED = "#e8eefb"      # context-blind modules
GAUGED = "#fdeee6"      # the one context-dependent module
NEUTRAL = "#f2f1ed"


def box(ax, x, y, w, h, text, fc, ec, fs=7.2, lw=0.8, zorder=3, weight="normal"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.05",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK,
            zorder=zorder + 1, linespacing=1.35, fontweight=weight)


def arrow(ax, p0, p1, color=INK, lw=0.9, style="-|>", rad=0.0, ls="-", zorder=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=7,
                                 color=color, lw=lw, shrinkA=1.5, shrinkB=1.5,
                                 zorder=zorder, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def main(out: Path | None = None) -> Path:
    use_paper_style()
    fig = plt.figure(figsize=(7.0, 2.45))
    ax = fig.add_axes([0.005, 0.0, 0.99, 1.0])
    ax.set_xlim(0, 10.4)
    ax.set_ylim(-0.30, 4.05)
    bare(ax)

    yt, yb = 3.05, 1.28
    h = 0.62

    # ---- top row: the inference / generation path ------------------------
    box(ax, 0.62, yt, 1.02, h, r"$x_c$" "\nactivity", NEUTRAL, MUTED)
    box(ax, 2.10, yt, 1.10, h, r"encoder $E$" "\n(context blind)", SHARED, OURS)
    box(ax, 3.62, yt, 0.86, h, r"$w_c$" "\nobs. frame", NEUTRAL, MUTED)
    box(ax, 5.20, yt, 1.14, h, r"$T_c^{-1}$", GAUGED, OURS_ALT)
    box(ax, 6.62, yt, 0.80, h, r"$z$" "\ncanonical", NEUTRAL, MUTED)
    box(ax, 7.90, yt, 0.90, h, r"$T_b$", GAUGED, OURS_ALT)
    box(ax, 9.28, yt, 1.30, h, r"decoder $D$" "\n(context blind)", SHARED, OURS)

    chain = [(0.62, 1.02), (2.10, 1.10), (3.62, 0.86), (5.20, 1.14),
             (6.62, 0.80), (7.90, 0.90), (9.28, 1.30)]
    for (x0, w0), (x1, w1) in zip(chain[:-1], chain[1:]):
        arrow(ax, (x0 + w0 / 2, yt), (x1 - w1 / 2, yt))
    arrow(ax, (9.28 + 1.30 / 2, yt), (10.30, yt))
    ax.text(10.34, yt, r"$\hat{x}_b$", fontsize=8, va="center", color=INK)

    # ---- bottom row: how T_c is constructed ------------------------------
    box(ax, 0.62, yb, 1.02, h, r"context $c$", NEUTRAL, MUTED)
    box(ax, 2.30, yb, 1.42, h, "context encoder\n" r"$\theta(c)-\theta(c_{\mathrm{ref}})$",
        GAUGED, OURS_ALT)
    box(ax, 4.20, yb, 1.10, h, r"$\theta \in \mathbb{R}^{K}$", NEUTRAL, MUTED)
    box(ax, 6.15, yb, 1.70, h + 0.14, r"$A=\sum_k \theta_k G_k$" "\n\nlearned generators",
        GAUGED, OURS_ALT, fs=7.0)
    box(ax, 8.20, yb, 1.10, h, r"$T=\exp(A)$", GAUGED, OURS_ALT)

    bchain = [(0.62, 1.02), (2.30, 1.42), (4.20, 1.10), (6.15, 1.66), (8.20, 1.10)]
    for (x0, w0), (x1, w1) in zip(bchain[:-1], bchain[1:]):
        arrow(ax, (x0 + w0 / 2, yb), (x1 - w1 / 2, yb))

    # T feeds both gauge blocks in the top row
    for xtarget in (5.20, 7.90):
        arrow(ax, (8.20, yb + h / 2), (xtarget, yt - h / 2), color=OURS_ALT,
              lw=0.8, rad=0.16 if xtarget < 8 else -0.30, ls=(0, (2.5, 1.8)))

    ax.text(9.05, yb, "exactly invertible;\nidentity at " r"$c_{\mathrm{ref}}$",
            fontsize=6.4, color=INK2, ha="left", va="center", linespacing=1.4)

    # ---- objective -------------------------------------------------------
    ax.add_patch(Rectangle((0.10, -0.22), 10.2, 0.72, facecolor="#f7f6f2",
                           edgecolor=GRID, lw=0.6, zorder=1))
    terms = [
        (0.92, r"$\|D(E(x_c))-x_c\|^2$", "reconstruction"),
        (3.05, r"$\|D(T_b T_a^{-1} E(x_a))-x_b\|^2$", "transport"),
        (5.45, r"$\mathrm{Var}_c\,[\,T_c^{-1}E(x_c)\,]$", "invariance"),
        (7.35, r"$\|T_aT_b - T_{a\cdot b}\|$", "group"),
        (9.30, r"$\mathcal{L}_{H_0}(x,z)$", "topology"),
    ]
    for x, formula, name in terms:
        ax.text(x, 0.30, formula, fontsize=6.8, ha="center", va="center", color=INK)
        ax.text(x, 0.00, name, fontsize=6.4, ha="center", va="center", color=INK2)
    for x in (1.95, 4.30, 6.42, 8.36):
        ax.plot([x, x], [-0.12, 0.42], color=GRID, lw=0.6, zorder=2)
    ax.text(0.16, 0.56, "objective", fontsize=6.6, color=MUTED, ha="left", va="bottom")

    # ---- legend ----------------------------------------------------------
    for i, (fc, ec, lab) in enumerate((
        (SHARED, OURS, "shared across contexts"),
        (GAUGED, OURS_ALT, "context dependent"),
    )):
        x0 = 5.55 + i * 2.35
        ax.add_patch(FancyBboxPatch((x0, 3.78), 0.26, 0.17,
                                    boxstyle="round,pad=0.008,rounding_size=0.04",
                                    facecolor=fc, edgecolor=ec, lw=0.8))
        ax.text(x0 + 0.34, 3.865, lab, fontsize=6.6, va="center", color=INK2)

    out = Path(out or FIGURE_DIR / "fig2_architecture.pdf")
    savefig(fig, out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()

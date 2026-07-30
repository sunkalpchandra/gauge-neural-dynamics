"""Figure 1: conceptual overview of gauge neural dynamics.

Three panels, each making one point.

(a) The fixed-coordinate assumption.  A single embedding is fitted to activity
    pooled over contexts, so genuine context differences appear as residual
    scatter around one manifold.
(b) The gauge view.  Context space is the base of a bundle whose fibre is the
    latent manifold; the *same* latent state is expressed in a
    context-dependent frame, and the transition between frames is the learned
    transformation.
(c) The group structure that makes the picture testable.  Composing the
    transformations of two contexts must land on the transformation of their
    composite; the gap is the closure defect we measure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch

from ..utils.common import FIGURE_DIR
from .style import (
    CAT, CRITICAL, GRID, INK, INK2, MUTED, OURS, OURS_ALT, SURFACE,
    bare, panel_label, savefig, use_paper_style,
)


def _ring(n=200, r=1.0, wobble=0.16, seed=0, noise=0.03):
    """A closed, slightly irregular loop -- a stand-in for a low-dimensional
    neural manifold with non-trivial topology."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n)
    rr = r * (1 + wobble * np.cos(3 * t + 0.4))
    xy = np.stack([rr * np.cos(t), rr * np.sin(t)], axis=1)
    return xy + noise * r * rng.normal(size=xy.shape)


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _shear(k, sx=1.0, sy=1.0):
    return np.array([[sx, k], [0.0, sy]])


# ---------------------------------------------------------------------------
def panel_fixed(ax):
    ax.set_title("Fixed coordinate system", pad=5)
    mats = [np.eye(2), _rot(1.05), _shear(0.45, 1.22, 0.78)]
    cols = [CAT[3], CAT[4], CAT[6]]
    labs = ["ctx a", "ctx b", "ctx c"]
    clouds = []
    for M, col, lab in zip(mats, cols, labs):
        pts = _ring(seed=1, noise=0.045) @ M.T
        clouds.append(pts)
        ax.scatter(pts[:, 0], pts[:, 1], s=3.0, color=col, alpha=0.8,
                   linewidths=0, zorder=2, label=lab)
    pooled = np.concatenate(clouds)
    # the single manifold a fixed-coordinate method would fit to the pooled data
    t = np.linspace(0, 2 * np.pi, 240)
    rad = np.percentile(np.linalg.norm(pooled, axis=1), 55)
    ax.plot(rad * np.cos(t), rad * np.sin(t), color=INK, lw=1.5, zorder=4,
            linestyle=(0, (4, 2)))
    ax.annotate("residual read as noise", xy=(-1.02, 0.98), xytext=(0.0, 1.86),
                fontsize=6.6, color=CRITICAL, ha="center",
                arrowprops=dict(arrowstyle="-", color=CRITICAL, lw=0.7,
                                shrinkA=1, shrinkB=2))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3,
              handletextpad=0.12, columnspacing=0.6, markerscale=1.9,
              fontsize=6.4, borderpad=0.0)
    ax.text(0.5, -0.19, "one manifold fitted to pooled activity", ha="center",
            transform=ax.transAxes, fontsize=6.6, color=INK2)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.65, 2.05)
    ax.set_aspect("equal")
    bare(ax)


def panel_bundle(ax):
    ax.set_title("Gauge neural dynamics", pad=5)
    ax.set_xlim(-0.30, 3.42)
    ax.set_ylim(-1.30, 2.55)

    y_base, y_fib = -0.70, 0.95
    ax.annotate("", xy=(3.25, y_base), xytext=(-0.15, y_base),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, mutation_scale=8))
    ax.text(3.30, y_base, r"$\mathcal{C}$", fontsize=9, va="center", color=INK)
    ax.text(1.55, y_base - 0.56, "context space", ha="center", fontsize=6.6, color=INK2)

    base = _ring(r=0.33, seed=1, noise=0.02)
    xs = [0.42, 1.55, 2.68]
    mats = [np.eye(2), _rot(1.05), _shear(0.45, 1.22, 0.78)]
    for x, M, lab in zip(xs, mats, "abc"):
        ax.add_patch(Ellipse((x, y_fib), 0.95, 2.25, facecolor="#f2f1ed",
                             edgecolor=GRID, lw=0.6, zorder=1))
        ax.plot([x, x], [y_base, y_base + 0.13], color=MUTED, lw=0.8, zorder=2)
        ax.scatter([x], [y_base], s=9, color=INK, zorder=3, linewidths=0)
        ax.text(x, y_base - 0.28, f"$c_{lab}$", ha="center", fontsize=7.5, color=INK)
        pts = base @ M.T
        ax.plot(pts[:, 0] + x, pts[:, 1] + y_fib, color=OURS, lw=1.4, zorder=4)
        k = 30
        ax.scatter([pts[k, 0] + x], [pts[k, 1] + y_fib], s=17, color=OURS_ALT,
                   zorder=5, linewidths=0.5, edgecolors=SURFACE)

    for x0, x1, lab in ((xs[0], xs[1], r"$T_{b}T_{a}^{-1}$"), (xs[1], xs[2], r"$T_{c}T_{b}^{-1}$")):
        ax.add_patch(FancyArrowPatch((x0 + 0.30, 1.92), (x1 - 0.30, 1.92),
                                     connectionstyle="arc3,rad=-0.34",
                                     arrowstyle="-|>", mutation_scale=7,
                                     color=INK, lw=0.9, zorder=6))
        ax.text((x0 + x1) / 2, 2.30, lab, ha="center", fontsize=7, color=INK)

    ax.text(xs[0] - 0.44, y_fib + 0.90, r"$\mathcal{M}$", fontsize=8.5, color=INK2)
    ax.text(0.5, -0.055, "one latent state, three coordinate frames",
            ha="center", transform=ax.transAxes, fontsize=6.6, color=INK2)
    bare(ax)


def panel_group(ax):
    ax.set_title("Approximate group structure", pad=5)
    ax.set_xlim(-0.10, 2.10)
    ax.set_ylim(-0.55, 1.95)

    P = {"z": (0.18, 1.42), "b": (1.62, 1.42), "ab": (0.92, 0.45), "c": (1.62, 0.45)}
    ax.scatter(*P["z"], s=26, color=OURS, zorder=6, linewidths=0.5, edgecolors=SURFACE)
    for k in ("b", "c", "ab"):
        ax.scatter(*P[k], s=26, color=OURS_ALT, zorder=6, linewidths=0.5, edgecolors=SURFACE)

    ax.text(P["z"][0], P["z"][1] + 0.17, r"$z$", fontsize=9, ha="center", color=INK)
    ax.text(P["b"][0], P["b"][1] + 0.17, r"$T_b(z)$", fontsize=8, ha="center", color=INK)
    ax.text(P["c"][0] + 0.10, P["c"][1] - 0.04, r"$T_a(T_b(z))$", fontsize=8,
            ha="left", va="center", color=INK)
    ax.text(P["ab"][0] - 0.10, P["ab"][1] - 0.04, r"$T_{a\cdot b}(z)$", fontsize=8,
            ha="right", va="center", color=OURS)

    ax.add_patch(FancyArrowPatch(P["z"], P["b"], arrowstyle="-|>", mutation_scale=8,
                                 color=INK, lw=0.9, shrinkA=5, shrinkB=5, zorder=4))
    ax.text((P["z"][0] + P["b"][0]) / 2, P["z"][1] + 0.13, r"$T_b$", fontsize=8,
            ha="center", color=INK)
    ax.add_patch(FancyArrowPatch(P["b"], P["c"], arrowstyle="-|>", mutation_scale=8,
                                 color=INK, lw=0.9, shrinkA=5, shrinkB=5, zorder=4))
    ax.text(P["b"][0] + 0.07, (P["b"][1] + P["c"][1]) / 2, r"$T_a$", fontsize=8,
            ha="left", va="center", color=INK)
    ax.add_patch(FancyArrowPatch(P["z"], P["ab"], arrowstyle="-|>",
                                 connectionstyle="arc3,rad=-0.22",
                                 mutation_scale=8, color=OURS, lw=1.1,
                                 shrinkA=5, shrinkB=5, zorder=4, linestyle=(0, (3, 2))))

    ax.plot([P["ab"][0], P["c"][0]], [P["ab"][1], P["c"][1]], color=CRITICAL,
            lw=1.6, zorder=5, solid_capstyle="butt")
    ax.text((P["ab"][0] + P["c"][0]) / 2, P["c"][1] - 0.13, "closure defect",
            fontsize=6.6, color=CRITICAL, ha="center", va="top")

    ax.text(0.5, 0.035, r"$T_{a\cdot b}=\exp\,\mathrm{BCH}(\theta_a,\theta_b)$,"
                        "\n"
                        r"$[G_i,G_j]=\sum_k f^k_{ij}G_k + R_{ij}$",
            transform=ax.transAxes, fontsize=7, ha="center", va="bottom", color=INK2,
            linespacing=1.5)
    bare(ax)


def main(out: Path | None = None) -> Path:
    use_paper_style()
    fig = plt.figure(figsize=(7.0, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.60, 1.10], wspace=0.16,
                          left=0.015, right=0.985, top=0.87, bottom=0.03)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_fixed(axes[0])
    panel_bundle(axes[1])
    panel_group(axes[2])
    for ax, L in zip(axes, "abc"):
        panel_label(ax, L, dx=-0.01, dy=1.14)
    out = Path(out or FIGURE_DIR / "fig1_concept.pdf")
    savefig(fig, out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()

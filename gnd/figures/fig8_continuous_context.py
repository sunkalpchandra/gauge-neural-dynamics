r"""Figure 8: the gauge field over context space, and a topological obstruction.

(a)  The learned algebra coordinates as a function of the cue angle, queried on a
     dense grid the model never saw.  A smooth field, not a lookup table.
(b)  Transport ``R^2`` at trained and at entirely held-out cue angles, against the
     two baselines that are even definable for an unobserved condition: leaving
     the latent alone, and re-using the nearest observed angle's transformation.
(c,d) The obstruction.  When the cue family closes into a circle, the required
     group elements ``R(\alpha)`` form a loop of winding number one in
     ``GL(d)^+``, while ``\exp(\sum_k \theta_k(c) G_k)`` is null-homotopic for any
     continuous ``\theta``.  A single chart therefore cannot cover the circle, and
     the failure should be *localised at the antipode of the reference cue* -- the
     branch point.  Panels (c) and (d) show exactly that, and show it disappearing
     when the same architecture is given the universal cover (the unwrapped angle)
     instead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..utils.common import FIGURE_DIR
from .common import load, load_artifacts
from .style import (
    CAT, CRITICAL, GOOD, GRID, INK, INK2, MUTED, OURS, OURS_ALT, SURFACE,
    hairline_grid, panel_label, savefig, use_paper_style,
)

EXP = "exp6_continuous_context"


def _by_angle(rows, part, key):
    sub = [r for r in rows if r.get("part") == part and r.get("context", -1) >= 0]
    if not sub:
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, bool)
    angles = sorted({r["angle_deg"] for r in sub})
    m, s, held = [], [], []
    for a in angles:
        v = [r[key] for r in sub if r["angle_deg"] == a and np.isfinite(r.get(key, np.nan))]
        m.append(np.mean(v) if v else np.nan)
        s.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        held.append(any(r["held_out"] for r in sub if r["angle_deg"] == a))
    return np.array(angles), np.array(m), np.array(s), np.array(held)


def main(out: Path | None = None) -> Path:
    use_paper_style()
    res = load(EXP)
    rows, summary = res["rows"], res["summary"]
    try:
        art = load_artifacts(EXP)
    except FileNotFoundError:
        art = {}

    fig = plt.figure(figsize=(7.0, 2.30))
    gs = fig.add_gridspec(1, 4, wspace=0.48, left=0.065, right=0.99,
                          top=0.80, bottom=0.215)

    # ---------- a: the field over context space ---------------------------
    ax = fig.add_subplot(gs[0, 0])
    if "arc_dense_theta" in art:
        alpha = np.degrees(art["arc_dense_alpha"])
        th = art["arc_dense_theta"]
        order = np.argsort(-np.abs(th).max(0))[:3]
        for i, k in enumerate(order):
            ax.plot(alpha, th[:, k], color=CAT[i], lw=1.2, zorder=3,
                    label=rf"$\theta_{{{int(k) + 1}}}$")
        ang = np.degrees(art["arc_angles"])
        held = np.zeros(len(ang), bool)
        held[art["arc_held_out"].astype(int)] = True
        for i, k in enumerate(order):
            ax.scatter(ang[~held], art["arc_theta"][~held, k], s=9, color=CAT[i],
                       zorder=4, linewidths=0.3, edgecolors=SURFACE)
            ax.scatter(ang[held], art["arc_theta"][held, k], s=16, facecolors="none",
                       edgecolors=CRITICAL, linewidths=0.8, zorder=5)
        ax.legend(fontsize=5.9, loc="upper left", handletextpad=0.3, borderpad=0.1,
                  labelspacing=0.15)
    ax.set_xlabel("cue angle (deg)", labelpad=1.5)
    ax.set_ylabel("algebra coordinate", labelpad=2.0)
    ax.set_title("learned gauge field", fontsize=6.8, pad=3, color=OURS)
    ax.text(0.5, -0.30, "circles: held-out cues", transform=ax.transAxes,
            ha="center", fontsize=5.9, color=CRITICAL)
    hairline_grid(ax, "both")
    panel_label(ax, "a", dx=-0.32, dy=1.24)

    # ---------- b: interpolation to unseen cues ----------------------------
    ax = fig.add_subplot(gs[0, 1])
    groups = [("arc_trained_cues", "trained\ncues"), ("arc_held_out_cues", "held-out\ncues")]
    series = [("transport_r2", "GND", OURS),
              ("transport_r2_nearest_seen", "nearest seen", CAT[1]),
              ("transport_r2_identity", "no transform", MUTED)]
    w = 0.26
    x = np.arange(len(groups))
    for i, (key, lab, col) in enumerate(series):
        v = [summary.get(g, {}).get(key, {}).get("mean", np.nan) for g, _ in groups]
        e = [summary.get(g, {}).get(key, {}).get("sem", 0.0) for g, _ in groups]
        pos = x + (i - 1) * w
        ax.bar(pos, v, width=w * 0.92, color=col, edgecolor=INK if i == 0 else MUTED,
               lw=0.6, zorder=3, label=lab)
        ax.errorbar(pos, v, yerr=e, fmt="none", ecolor=INK2, elinewidth=0.7,
                    capsize=1.3, capthick=0.6, zorder=4)
        for p, vv in zip(pos, v):
            if np.isfinite(vv):
                ax.text(p, vv + 0.02 * (1 if vv >= 0 else -1), f"{vv:.2f}",
                        ha="center", va="bottom" if vv >= 0 else "top",
                        fontsize=5.4, color=INK2)
    ax.axhline(0, color=INK, lw=0.7, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([g[1] for g in groups], fontsize=6.3)
    ax.set_ylabel(r"transport $R^2$", labelpad=2.0)
    ax.set_title("generalisation over\ncontext space", fontsize=6.8, pad=16)
    ax.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
              handletextpad=0.3, borderpad=0.05, columnspacing=0.7, handlelength=1.0)
    hairline_grid(ax)
    panel_label(ax, "b", dx=-0.30, dy=1.24)

    # ---------- c, d: the winding obstruction ------------------------------
    for j, (key, ylab, logy, hline, hlab) in enumerate((
        ("transport_r2", r"transport $R^2$", False, 0.0, "chance"),
        ("gre", "GRE", True, 1.0, "null"),
    )):
        ax = fig.add_subplot(gs[0, 2 + j])
        for part, col, lab in (("circle:circular", CRITICAL, r"circle $S^1$"),
                               ("circle:lifted", GOOD, r"cover $\mathbb{R}$")):
            a, m, s, _ = _by_angle(rows, part, key)
            if len(a) == 0:
                continue
            ax.plot(a, m, "-o", color=col, ms=2.6, lw=1.2, label=lab, zorder=3)
            ax.fill_between(a, m - s, m + s, color=col, alpha=0.16, lw=0, zorder=2)
        ax.axvline(180, color=INK, lw=0.7, linestyle=(0, (2, 2)), zorder=1)
        ax.annotate("branch point", xy=(180, 0.99), xycoords=("data", "axes fraction"),
                    xytext=(-2.5, 0), textcoords="offset points", fontsize=5.7,
                    color=INK2, ha="right", va="top", rotation=90)
        if hline is not None:
            ax.axhline(hline, color=MUTED, lw=0.7, linestyle=(0, (3, 2)), zorder=1)
            # keep the annotation on the opposite side from that panel's legend
            hx, hha = (0.015, "left") if not logy else (0.985, "right")
            ax.text(hx, hline, hlab, transform=ax.get_yaxis_transform(),
                    fontsize=5.7, color=MUTED, va="bottom", ha=hha)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("cue angle (deg)", labelpad=1.5)
        ax.set_ylabel(ylab, labelpad=2.0)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_title("full circle of cues", fontsize=6.8, pad=3)
        ax.legend(fontsize=5.9, loc="lower right" if not logy else "upper left",
                  handletextpad=0.3, borderpad=0.1, labelspacing=0.2, framealpha=0.0)
        hairline_grid(ax, "both")
        panel_label(ax, "cd"[j], dx=-0.30, dy=1.24)

    out = Path(out or FIGURE_DIR / "fig8_continuous_context.pdf")
    savefig(fig, out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()

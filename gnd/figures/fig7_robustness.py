"""Figure 7: robustness and scaling.

Each panel is a sweep over one property of the recording, with the metrics that
respond to it.  The shape matters as much as the level: a method that relies on a
genuine group structure should degrade gracefully, whereas per-context post-hoc
alignment should collapse once each context's estimate becomes noisy.

(a)  Observation noise, cross-context transport ``R^2``.
(b)  Observation noise, geometric recovery error.
(c)  Population size (number of recorded cells).
(d)  Number of behavioural samples.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..utils.common import FIGURE_DIR
from .common import load
from .style import (
    CAT, CRITICAL, GRID, INK, INK2, MUTED, OURS, SURFACE,
    hairline_grid, panel_label, savefig, use_paper_style,
)

EXP = "exp5_robustness"
SERIES = ["GND", "PCA", "Autoencoder", "Procrustes"]      # <= 4 series: one slot each
COLORS = {m: CAT[i] for i, m in enumerate(SERIES)}
MARKERS = {"GND": "o", "PCA": "s", "Autoencoder": "^", "Procrustes": "D"}


def _series(table, sweep, method, key):
    node = table.get(sweep, {})
    xs, ys, es = [], [], []
    for val in sorted(node, key=lambda s: float(s)):
        entry = node[val].get(method, {}).get(key)
        if entry is None:
            continue
        xs.append(float(val))
        ys.append(entry["mean"])
        es.append(entry["sem"])
    return np.array(xs), np.array(ys), np.array(es)


def _sweep_panel(ax, table, sweep, key, xlabel, ylabel, methods, logx=False,
                 hline=None, hlabel=None):
    plotted = []
    for m in methods:
        x, y, e = _series(table, sweep, m, key)
        if len(x) == 0:
            continue
        col = COLORS.get(m, MUTED)
        ax.plot(x, y, "-", color=col, marker=MARKERS.get(m, "o"), ms=3.0,
                lw=1.3 if m == "GND" else 1.0, zorder=4 if m == "GND" else 3,
                label=m + (" (ours)" if m == "GND" else ""))
        ax.fill_between(x, y - e, y + e, color=col, alpha=0.16, lw=0, zorder=2)
        plotted.append(m)
    if hline is not None:
        ax.axhline(hline, color=CRITICAL, lw=0.8, linestyle=(0, (3, 2)), zorder=1)
        if hlabel:
            ax.text(0.985, hline, hlabel, transform=ax.get_yaxis_transform(),
                    fontsize=5.8, color=CRITICAL, va="bottom", ha="right")
    if logx:
        ax.set_xscale("log", base=2)
        x, _, _ = _series(table, sweep, methods[0], key)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v)}" for v in x], fontsize=6.0)
    ax.set_xlabel(xlabel, labelpad=1.5)
    ax.set_ylabel(ylabel, labelpad=2.0)
    hairline_grid(ax, "both")
    return plotted


def main(out: Path | None = None) -> Path:
    use_paper_style()
    res = load(EXP)
    table = res["table"]

    fig = plt.figure(figsize=(7.0, 2.05))
    gs = fig.add_gridspec(1, 4, wspace=0.42, left=0.065, right=0.99,
                          top=0.86, bottom=0.235)

    ax = fig.add_subplot(gs[0, 0])
    plotted = _sweep_panel(ax, table, "noise", "transport_r2",
                           "observation noise ($\\times$ activity s.d.)",
                           r"transport $R^2$", SERIES, hline=0.0, hlabel="chance")
    ax.set_title("noise robustness", fontsize=6.8, pad=3)
    panel_label(ax, "a", dx=-0.30, dy=1.24)

    ax = fig.add_subplot(gs[0, 1])
    _sweep_panel(ax, table, "noise", "gre",
                 "observation noise ($\\times$ activity s.d.)", "GRE", SERIES,
                 hline=1.0, hlabel="no-transformation null")
    ax.set_yscale("log")
    ax.set_title("recovery vs. noise", fontsize=6.8, pad=3)
    panel_label(ax, "b", dx=-0.30, dy=1.24)

    ax = fig.add_subplot(gs[0, 2])
    _sweep_panel(ax, table, "neurons", "transport_r2", "recorded cells",
                 r"transport $R^2$", ["GND", "PCA"], logx=True, hline=0.0)
    ax.set_title("population size", fontsize=6.8, pad=3)
    panel_label(ax, "c", dx=-0.30, dy=1.24)

    ax = fig.add_subplot(gs[0, 3])
    _sweep_panel(ax, table, "samples", "transport_r2", "behavioural samples",
                 r"transport $R^2$", ["GND", "PCA"], logx=True, hline=0.0)
    ax.set_title("sample size", fontsize=6.8, pad=3)
    panel_label(ax, "d", dx=-0.30, dy=1.24)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.035), fontsize=6.3, handletextpad=0.35,
               columnspacing=1.3)

    out = Path(out or FIGURE_DIR / "fig7_robustness.pdf")
    savefig(fig, out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()

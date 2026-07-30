"""Figure 6: ablations.

(a)  Removing one ingredient at a time, measured by cross-context transport
     ``R^2`` -- the quantity the model exists to deliver.  Bars are ordered by
     effect size and shown as a change from the full model, so the reader sees
     directly which components are load-bearing.
(b)  The same ablations on the two geometric metrics, which respond differently:
     a component can be irrelevant for prediction but necessary for recovering the
     right transformation.
(c)  Latent dimension sweep.  Under-parameterising the latent costs recovery;
     over-parameterising costs little, which matters because the true latent
     dimension of a recording is unknown.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..utils.common import FIGURE_DIR
from .common import get, load
from .style import (
    BASELINE_EDGE, BASELINE_FILL, CAT, CRITICAL, GOOD, GRID, INK, INK2, MUTED,
    OURS, OURS_ALT, SURFACE, hairline_grid, panel_label, savefig, use_paper_style,
)

EXP = "exp4_ablations"

# The ablations shown in the main figure, in reading order.
CORE = [
    ("A1: no gauge", "no gauge transformation"),
    ("A2: no group loss", "no group consistency"),
    ("A3: no topology", "no topology term"),
    ("no transport loss", "no transport term"),
    ("no invariance loss", "no invariance term"),
    ("no anchoring", "no identity anchoring"),
    ("unpaired (MMD)", "unpaired alignment"),
    ("gauge=flow", "flow (non-linear) gauge"),
    ("algebra=so", "isometric gauge (so)"),
    ("algebra=sl", "volume-preserving (sl)"),
    ("BCH order 1", "BCH order 1"),
    ("encoder=VAE", "variational encoder"),
    ("K=2 generators", "K = 2 generators"),
    ("K=12 generators", "K = 12 generators"),
]


def main(out: Path | None = None) -> Path:
    use_paper_style()
    res = load(EXP)
    table = res["table"]
    full = {k: get(table, "full", k) for k in ("transport_r2", "gre", "cis", "gcs", "mps")}

    present = [(k, lab) for k, lab in CORE if k in table]
    deltas = [(lab, get(table, k, "transport_r2") - full["transport_r2"],
               get(table, k, "transport_r2", "sem")) for k, lab in present]
    deltas.sort(key=lambda t: t[1])

    fig = plt.figure(figsize=(7.0, 3.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.05, 0.85], wspace=0.52,
                          left=0.215, right=0.985, top=0.90, bottom=0.155)

    # ---------- a: change in transport R^2 --------------------------------
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(deltas))
    vals = [d[1] for d in deltas]
    errs = [d[2] for d in deltas]
    cols = [CRITICAL if v < -0.02 else (GOOD if v > 0.02 else BASELINE_FILL) for v in vals]
    ax.barh(y, vals, height=0.72, color=cols, edgecolor=BASELINE_EDGE, lw=0.5, zorder=3)
    ax.errorbar(vals, y, xerr=errs, fmt="none", ecolor=INK2, elinewidth=0.7,
                capsize=1.4, capthick=0.7, zorder=4)
    ax.axvline(0, color=INK, lw=0.8, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([d[0] for d in deltas], fontsize=6.1)
    ax.set_xlabel(r"change in transport $R^2$ vs. full model")
    ax.set_title(f"full model: $R^2$ = {full['transport_r2']:.3f}", fontsize=6.8, pad=3)
    for yi, v in zip(y, vals):
        ax.text(v + np.sign(v) * 0.012, yi, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=5.6, color=INK2)
    pad = 0.18 * (max(vals) - min(vals) + 1e-6)
    ax.set_xlim(min(vals) - pad * 2.2, max(vals) + pad * 2.2)
    hairline_grid(ax, "x")
    panel_label(ax, "a", dx=-0.55, dy=1.13)

    # ---------- b: geometric metrics --------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    keys = [("gre", "GRE (lower better)", CAT[0]), ("cis", "CIS (higher better)", CAT[1])]
    order = [lab for lab, _, _ in deltas]
    lab2key = {lab: k for k, lab in present}
    yy = np.arange(len(order))
    for i, (key, name, col) in enumerate(keys):
        v = [get(table, lab2key[lab], key) for lab in order]
        e = [get(table, lab2key[lab], key, "sem") for lab in order]
        ax.errorbar(v, yy + (i - 0.5) * 0.24, xerr=e, fmt="o", ms=2.8, color=col,
                    ecolor=col, elinewidth=0.7, capsize=1.2, capthick=0.6,
                    label=name, zorder=3, linestyle="none")
    for key, _, col in keys:
        ax.axvline(full[key], color=col, lw=0.8, linestyle=(0, (3, 2)), zorder=2)
    ax.set_yticks(yy)
    ax.set_yticklabels([])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("metric value")
    ax.legend(fontsize=5.9, loc="lower right", handletextpad=0.25, borderpad=0.15)
    ax.set_title("same rows as (a); dashed = full model", fontsize=6.5, pad=3)
    hairline_grid(ax, "x")
    panel_label(ax, "b", dx=-0.10, dy=1.13)

    # ---------- c: latent dimension sweep ---------------------------------
    ax = fig.add_subplot(gs[0, 2])
    dims, rows = [], {}
    for name in table:
        if name.startswith("A4: latent dim"):
            d = int(name.split()[-1])
            dims.append(d)
            rows[d] = name
    dims.sort()
    for key, col, lab, scale in (("transport_r2", CAT[0], r"transport $R^2$", 1.0),
                                 ("cis", CAT[1], "CIS", 1.0),
                                 ("gre", CAT[2], "GRE", 1.0)):
        v = np.array([get(table, rows[d], key) for d in dims]) * scale
        e = np.array([get(table, rows[d], key, "sem") for d in dims]) * scale
        ax.plot(dims, v, "-o", color=col, ms=2.8, label=lab, zorder=3)
        ax.fill_between(dims, v - e, v + e, color=col, alpha=0.18, lw=0, zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims], fontsize=6.0)
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("metric value")
    ax.axvline(6, color=MUTED, lw=0.7, linestyle=(0, (2, 2)), zorder=1)
    ax.text(6, ax.get_ylim()[1], " default", fontsize=5.8, color=MUTED,
            va="top", ha="left")
    ax.legend(fontsize=5.9, loc="center right", handletextpad=0.3, borderpad=0.15)
    hairline_grid(ax, "both")
    panel_label(ax, "c", dx=-0.30, dy=1.13)

    out = Path(out or FIGURE_DIR / "fig6_ablations.pdf")
    savefig(fig, out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()

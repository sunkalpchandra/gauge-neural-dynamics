"""Figure 5: motor-cortex trajectories across reach conditions.

(a)  Trial-averaged population trajectories in the observed frame, one colour per
     reach direction: the familiar rotational structure, with the condition
     ordering laid out around a ring.
(b)  The same trajectories after the learned gauge is inverted.  If reach
     conditions are the same computation in a rotated frame, they collapse onto
     one canonical trajectory here.
(c)  The strongest baseline for comparison, after its own post-hoc alignment.
(d)  Trajectory alignment error by method, with and without an additional
     optimal rigid alignment.
(e)  Cross-context transport: predicting the population response in another reach
     condition.  Positive values mean the prediction beats that condition's own
     mean; the identity line is the "no transformation" null.
(f)  Test of Eq. (5).  Conjugation preserves eigenvalues, so the rotation
     frequency of the latent dynamics should be shared across conditions while
     the rotation plane turns.  Bars show the coefficient of variation of the top
     rotation frequency across conditions; the dashed line is the same quantity
     measured from a PCA of the recorded activity, i.e. the circuit's own spread.
     Matching that line is the prediction.  A value far *below* it is not better:
     a latent that fails to separate conditions has no frequency spread at all,
     which is why this panel must be read next to (e).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..utils.common import FIGURE_DIR
from .common import bar_panel, get, load, load_artifacts, method_order, project2d, unit_scale
from .style import (
    CAT, CRITICAL, CYCLIC, GRID, INK, INK2, MUTED, OURS, SURFACE,
    bare, hairline_grid, panel_label, savefig, use_paper_style,
)

EXP = "exp3_motor_cortex"
PREFERRED = ("GND", "PCA", "Autoencoder", "VAE", "CCA", "Procrustes", "ManifoldAlign", "UMAP")


def _trial_average(Z, n_time):
    n_trials = Z.shape[0] // n_time
    return Z[: n_trials * n_time].reshape(n_trials, n_time, Z.shape[1], Z.shape[2]).mean(0)


def _traj_panel(ax, Z, n_time, n_prep, angles, extents, title, colored=True):
    A = _trial_average(Z, n_time)                       # (T, C, d)
    P = unit_scale(project2d(A))
    full = np.isclose(extents, extents.max())
    for c in range(P.shape[1]):
        if not full[c]:
            continue
        col = CYCLIC((angles[c] % (2 * np.pi)) / (2 * np.pi)) if colored else MUTED
        ax.plot(P[n_prep:, c, 0], P[n_prep:, c, 1], color=col, lw=1.0, zorder=3)
        ax.plot(P[:n_prep, c, 0], P[:n_prep, c, 1], color=col, lw=0.5, alpha=0.55, zorder=2)
        ax.scatter([P[n_prep, c, 0]], [P[n_prep, c, 1]], s=6, color=col,
                   zorder=4, linewidths=0.3, edgecolors=SURFACE)
    ax.set_title(title, fontsize=6.5, pad=2.5)
    lim = np.percentile(np.abs(P), 99.5) * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    bare(ax)


def main(out: Path | None = None) -> Path:
    use_paper_style()
    res, art = load(EXP), load_artifacts(EXP)
    table = res["table"]
    methods = [m for m in method_order(table, PREFERRED) if m in table]

    n_time = int(art["n_time"])
    n_prep = int(art["n_prep"])
    angles, extents = art["angles"], art["extents"]

    fig = plt.figure(figsize=(7.0, 3.05))
    gs = fig.add_gridspec(2, 3, hspace=0.85, wspace=0.38,
                          left=0.085, right=0.985, top=0.86, bottom=0.21)

    # ---------- a, b, c: trajectories -------------------------------------
    baseline = next((m for m in ("Procrustes", "PCA", "Autoencoder") if f"{m}::z_test" in art), None)
    specs = [("GND::w_test", "observed frame (GND encoder)"),
             ("GND::z_test", "canonical frame (after gauge)")]
    if baseline:
        specs.append((f"{baseline}::z_test", f"{baseline} (post-hoc aligned)"))
    for j, (key, title) in enumerate(specs):
        ax = fig.add_subplot(gs[0, j])
        _traj_panel(ax, art[key], n_time, n_prep, angles, extents, title)
        panel_label(ax, "abc"[j], dx=-0.07, dy=1.34)
        if j == 1:
            ax.set_title(title, fontsize=6.5, pad=2.5, color=OURS)
        te = get(table, "GND" if j < 2 else baseline, "trajectory_alignment_error")
        if j != 0 and np.isfinite(te):
            ax.text(0.5, -0.055, f"alignment error {te:.3f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6.2, color=INK2)
    # direction legend
    ax0 = fig.axes[0]
    for c in np.argsort(angles % (2 * np.pi)):
        if not np.isclose(extents[c], extents.max()):
            continue
    sm = plt.cm.ScalarMappable(cmap=CYCLIC, norm=plt.Normalize(0, 360))
    cb = fig.colorbar(sm, ax=fig.axes[:len(specs)], fraction=0.028, pad=0.012,
                      aspect=14, ticks=[0, 180, 360])
    cb.set_label("reach direction (deg)", fontsize=6.0, labelpad=1.5)
    cb.ax.tick_params(labelsize=5.4, width=0.5, length=1.8)
    cb.outline.set_linewidth(0.4)

    # ---------- d: trajectory alignment error ----------------------------
    ax = fig.add_subplot(gs[1, 0])
    vals = [get(table, m, "trajectory_alignment_error") for m in methods]
    errs = [get(table, m, "trajectory_alignment_error", "sem") for m in methods]
    vals2 = [get(table, m, "trajectory_alignment_error_procrustes") for m in methods]
    x = np.arange(len(methods))
    from .style import method_color, method_edge

    ax.bar(x - 0.19, vals, width=0.36, color=[method_color(m) for m in methods],
           edgecolor=[method_edge(m) for m in methods], lw=0.6, zorder=3,
           label="as learned")
    ax.bar(x + 0.19, vals2, width=0.36, color=SURFACE, hatch="////",
           edgecolor=[method_edge(m) for m in methods], lw=0.6, zorder=3,
           label="+ optimal rigid map")
    ax.errorbar(x - 0.19, vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=0.7,
                capsize=1.4, capthick=0.7, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=42, ha="right")
    ax.set_ylabel("trajectory alignment error")
    ax.legend(fontsize=5.6, loc="upper left", handlelength=1.0, handletextpad=0.3,
              borderpad=0.15, labelspacing=0.2, framealpha=0.0)
    hairline_grid(ax)
    panel_label(ax, "d", dx=-0.30, dy=1.20)

    # ---------- e: transport R^2 ------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    bar_panel(ax, table, methods, "transport_r2", r"cross-context transport $R^2$",
              baseline=0.0, baseline_label="chance", rotation=42, fmt="{:.2f}")
    panel_label(ax, "e", dx=-0.30, dy=1.20)

    # ---------- f: conjugacy of the latent dynamics -----------------------
    ax = fig.add_subplot(gs[1, 2])
    ref = get(table, "GND", "circuit_reference_rotation_frequency_cv")
    bar_panel(ax, table, methods, "rotation_frequency_cv",
              "rotation-frequency CV", rotation=42, fmt="{:.2f}",
              baseline=ref if np.isfinite(ref) else None,
              baseline_label="recorded activity")
    panel_label(ax, "f", dx=-0.30, dy=1.20)

    out = Path(out or FIGURE_DIR / "fig5_motor_cortex.pdf")
    savefig(fig, out, dpi=400)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()

"""Figure 3: hippocampal place-cell remapping and manifold recovery.

Top row: occupancy-normalised rate maps of one representative place cell in
every context.  The field moves coherently with the cue, which is the phenomenon
the model has to explain.

Bottom row: the reference-context latent coloured by the animal's true heading,
with a grey segment to where the same sample lands in every other context.
Perfect alignment collapses the segments to dots; residual misalignment shows up
as a fan.  The first panel is GND's observed-frame latent (before the gauge is
applied), the second its canonical latent (after), and the remainder are the
strongest baselines after their own post-hoc alignment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpecFromSubplotSpec

from ..utils.common import FIGURE_DIR
from ..geometry.metrics import context_invariance_score
from .common import get, load, load_artifacts, project2d, unit_scale
from .style import (
    CYCLIC, INK, INK2, MUTED, OURS, SURFACE, bare, panel_label, rate_cmap,
    savefig, use_paper_style,
)

EXP = "exp1_hippocampus"


def _rate_map(u, r, radius, bins=32, smooth=1.1):
    from scipy.ndimage import gaussian_filter

    edges = np.linspace(-radius, radius, bins + 1)
    occ, _, _ = np.histogram2d(u[:, 0], u[:, 1], bins=[edges, edges])
    tot, _, _ = np.histogram2d(u[:, 0], u[:, 1], bins=[edges, edges], weights=r)
    occ, tot = gaussian_filter(occ, smooth), gaussian_filter(tot, smooth)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(occ > occ.max() * 0.02, tot / occ, np.nan)
    # blank the unvisited corners so the panel shows the arena, not a square
    ctr = 0.5 * (edges[1:] + edges[:-1])
    X, Y = np.meshgrid(ctr, ctr, indexing="ij")
    m[X ** 2 + Y ** 2 > radius ** 2] = np.nan
    return m.T


def _pick_cell(activity, latent, radius, centres=None):
    """An off-centre, high-contrast field.

    Eccentricity is the point: a field sitting at the centre of the arena looks
    the same after a rotation, so the most sharply tuned cell is the worst
    illustration of coherent remapping.
    """
    ecc = np.linalg.norm(centres, axis=1) / radius if centres is not None else None
    # Score by the *worst* context: a cell whose field leaves the arena under the
    # stretch would give an empty panel and illustrate nothing.
    contrast = activity.max(0) - np.median(activity, axis=0)     # (C, cells)
    worst = contrast.min(0)
    best, score = None, -np.inf
    for i in range(activity.shape[2]):
        if ecc is not None and not (0.30 < ecc[i] < 0.65):
            continue
        if worst[i] > score:
            best, score = i, worst[i]
    if best is None:                       # fall back if no cell is in the band
        best = int(np.argmax(worst))
    return int(best)


def main(out: Path | None = None) -> Path:
    use_paper_style()
    res, ds, art = load(EXP), load_artifacts(EXP, "dataset.npz"), load_artifacts(EXP)
    table = res["table"]
    names = [str(s) for s in ds["context_names"]]
    activity, latent = ds["activity"], ds["latent"]
    radius = float(ds["radius"])
    nctx = activity.shape[1]

    fig = plt.figure(figsize=(7.0, 2.75))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.28], hspace=0.42,
                             left=0.045, right=0.985, top=0.90, bottom=0.055)

    # ---------- top: rate maps -------------------------------------------
    top = GridSpecFromSubplotSpec(1, nctx + 1, subplot_spec=outer[0],
                                  width_ratios=[1] * nctx + [0.075], wspace=0.14)
    cell = _pick_cell(activity, latent, radius, ds.get("centres"))
    maps = [_rate_map(latent, activity[:, c, cell], radius) for c in range(nctx)]
    vmin = float(np.nanmin([np.nanmin(m) for m in maps]))
    vmax = float(np.nanmax([np.nanmax(m) for m in maps]))
    cm = rate_cmap()
    for c in range(nctx):
        ax = fig.add_subplot(top[0, c])
        im = ax.imshow(maps[c], origin="lower", cmap=cm, vmin=vmin, vmax=vmax,
                       extent=[-radius, radius, -radius, radius], interpolation="bilinear")
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, color=MUTED, lw=0.6))
        ax.set_title(names[c].replace(": ", ":\n"), fontsize=6.3, pad=2.5, color=INK)
        ax.set_xlim(-radius * 1.03, radius * 1.03)
        ax.set_ylim(-radius * 1.03, radius * 1.03)
        ax.set_aspect("equal")
        bare(ax)
        if c == 0:
            panel_label(ax, "a", dx=-0.10, dy=1.42)
            ax.text(-0.09, 0.5, f"cell {cell}", transform=ax.transAxes, rotation=90,
                    va="center", ha="center", fontsize=6.3, color=INK2)
    cax = fig.add_subplot(top[0, nctx])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("z-scored rate", fontsize=6.0, labelpad=1.5)
    cb.ax.tick_params(labelsize=5.4, width=0.5, length=1.8)
    cb.outline.set_linewidth(0.4)

    # ---------- bottom: latent clouds ------------------------------------
    panels = [("GND", "w_test", "GND: before gauge"), ("GND", "z_test", "GND: after gauge")]
    for m in ("PCA", "Autoencoder", "Procrustes", "CCA"):
        if f"{m}::z_test" in art:
            panels.append((m, "z_test", f"{m}: aligned"))
    panels = panels[:6]

    test_ref = int(art["reference"]) if "reference" in art else 0
    ang = np.arctan2(art["latent_test"][:, 1], art["latent_test"][:, 0])
    n_show = min(len(ang), 700)
    idx = np.linspace(0, len(ang) - 1, n_show).astype(int)

    bot = GridSpecFromSubplotSpec(1, len(panels) + 1, subplot_spec=outer[1],
                                  width_ratios=[1] * len(panels) + [0.075], wspace=0.10)
    for j, (meth, key, title) in enumerate(panels):
        ax = fig.add_subplot(bot[0, j])
        Z = art[f"{meth}::{key}"]                      # (N, C, d)
        # Both GND panels share the projection fitted on the canonical latent, so
        # the "before" panel shows the contexts genuinely fanned out rather than
        # having the spread absorbed into a per-panel basis.
        basis = art.get(f"{meth}::z_test")
        P = unit_scale(project2d(Z, basis_from=basis))
        # Reference-context points, plus a segment to where the same sample lands
        # in every other context. Perfect alignment collapses the segments to
        # dots; residual misalignment shows as a fan.
        ref = test_ref
        sidx = idx[::3]                       # thin the segments, keep all dots
        seg = np.concatenate(
            [np.stack([P[sidx, ref], P[sidx, c]], axis=1)
             for c in range(P.shape[1]) if c != ref], axis=0)
        ax.add_collection(LineCollection(seg, colors=MUTED, linewidths=0.22,
                                         alpha=0.32, zorder=1, rasterized=True))
        sc = ax.scatter(P[idx, ref, 0], P[idx, ref, 1], c=ang[idx], cmap=CYCLIC,
                        s=2.4, alpha=0.9, linewidths=0, vmin=-np.pi, vmax=np.pi,
                        zorder=3, rasterized=True)
        ax.set_title(title, fontsize=6.5, pad=2.5,
                     color=OURS if meth == "GND" else INK)
        # CIS of the latents actually plotted, so the before/after panels differ
        cis = context_invariance_score(Z)["cis"]
        ax.text(0.5, -0.075, f"CIS {cis:.2f}", transform=ax.transAxes, ha="center",
                va="top", fontsize=6.2, color=INK2)
        lim = np.percentile(np.abs(P), 99.0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        bare(ax)
        if j == 0:
            panel_label(ax, "b", dx=-0.10, dy=1.30)
    cax2 = fig.add_subplot(bot[0, len(panels)])
    cb2 = fig.colorbar(sc, cax=cax2, ticks=[-np.pi, 0, np.pi])
    cb2.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=5.4)
    cb2.set_label("true heading", fontsize=6.0, labelpad=1.5)
    cb2.ax.tick_params(width=0.5, length=1.8)
    cb2.outline.set_linewidth(0.4)

    out = Path(out or FIGURE_DIR / "fig3_hippocampus.pdf")
    savefig(fig, out, dpi=400)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()

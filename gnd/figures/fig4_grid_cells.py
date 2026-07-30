"""Figure 4: grid-cell geometry -- toroidal latent and the translation group.

(a,b)  Rate maps over physical space for one grid cell in the reference context
       and in a phase-shifted context: the whole lattice translates coherently.
(c)    The canonical latent, viewed in its leading three principal components:
       a torus.
(d)    Persistent homology of the canonical latent.  A torus has Betti numbers
       (1, 2, 1), and the two long ``H_1`` bars and single long ``H_2`` bar are
       what the barcode should show.
(e,f)  The latent factorises into two circles: the first principal plane carries
       the first grid phase, the second plane the second phase.  Under this
       embedding a phase translation is a rotation in each plane independently --
       an element of the maximal torus of ``SO(4)``.
(g)    Pairwise commutator norms of the learned generators.  For the pure
       translation family the correct answer is an abelian algebra, i.e. a
       commutator matrix that is everywhere near zero.
(h)    Leave-one-context-out linear prediction of the true phase shift from the
       learned algebra coordinates.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..geometry.topology import lifetimes, persistence_diagrams
from ..simulations.grid_cells import build_contexts
from ..utils.common import FIGURE_DIR
from .common import get, load, load_artifacts, project2d, unit_scale
from .style import (
    CAT, CRITICAL, CYCLIC, GRID, INK, INK2, MUTED, OURS, SEQ_BLUE, SURFACE,
    bare, hairline_grid, panel_label, rate_cmap, savefig, use_paper_style,
)

EXP = "exp2_grid_cells"
MAIN_SET = "translation"


def _spatial_map(pos, r, bins=44, smooth=0.9):
    from scipy.ndimage import gaussian_filter

    half = float(np.abs(pos).max())
    edges = np.linspace(-half, half, bins + 1)
    occ, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=[edges, edges])
    tot, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=[edges, edges], weights=r)
    occ, tot = gaussian_filter(occ, smooth), gaussian_filter(tot, smooth)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(occ > occ.max() * 0.03, tot / occ, np.nan)
    return m.T, half


def _match_contexts(names: list[str]):
    """Find the context family whose names match the ones stored in the artefacts.

    The three families differ in length and in their labels, so this is exact;
    guessing instead would silently pair each learned coefficient with the wrong
    ground-truth phase shift.
    """
    for kind in ("translation", "translation+rotation", "all"):
        for n_tr in range(2, 8):
            specs = build_contexts(kind, 0.42, 0.21, n_translations=n_tr)
            if [s.name for s in specs] == list(names):
                return specs
    raise ValueError(f"no context family matches the stored names: {list(names)}")


def _loo_predict(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
    """Leave-one-out linear prediction of ``Y`` from ``X`` (few rows, so LOO)."""
    Xa = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    pred = np.zeros_like(Y, dtype=float)
    for i in range(len(X)):
        keep = np.arange(len(X)) != i
        W, *_ = np.linalg.lstsq(Xa[keep], Y[keep], rcond=None)
        pred[i] = Xa[i] @ W
    ss_res = ((pred - Y) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum() + 1e-12
    return pred, float(1 - ss_res / ss_tot)


def main(out: Path | None = None) -> Path:
    use_paper_style()
    res, art = load(EXP), load_artifacts(EXP)
    tables = res["tables"]
    table = tables.get(MAIN_SET, next(iter(tables.values())))

    fig = plt.figure(figsize=(7.0, 2.95))
    gs = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.40,
                          left=0.055, right=0.985, top=0.90, bottom=0.10)

    pos, act = art["position"], art["activity_full"]
    names = [str(s) for s in art["context_names"]]
    cell = int(np.argmax(act[:, 0, :].std(0)))
    shifted = min(1, act.shape[1] - 1)

    # ---------- a, b: spatial rate maps ----------------------------------
    maps = [_spatial_map(pos, act[:, c, cell])[0] for c in (0, shifted)]
    half = _spatial_map(pos, act[:, 0, cell])[1]
    vmin = float(np.nanmin([np.nanmin(m) for m in maps]))
    vmax = float(np.nanmax([np.nanmax(m) for m in maps]))
    for j, (c, m) in enumerate(zip((0, shifted), maps)):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(m, origin="lower", cmap=rate_cmap(), vmin=vmin, vmax=vmax,
                       extent=[-half, half, -half, half], interpolation="bilinear")
        ax.set_title(names[c], fontsize=6.4, pad=2.5)
        ax.set_aspect("equal")
        bare(ax)
        panel_label(ax, "ab"[j], dx=-0.06, dy=1.26)
    cb = fig.colorbar(im, ax=fig.axes[:2], fraction=0.035, pad=0.012, aspect=14)
    cb.set_label("z-scored rate", fontsize=6.0, labelpad=1.5)
    cb.ax.tick_params(labelsize=5.4, width=0.5, length=1.8)
    cb.outline.set_linewidth(0.4)

    Z = art["GND::z_test"]
    phase = art["phase_test"]
    n = min(len(phase), 2500)
    idx = np.linspace(0, len(phase) - 1, n).astype(int)

    # ---------- c: the torus, in three principal components --------------
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    from sklearn.decomposition import PCA

    flat = Z.reshape(-1, Z.shape[-1])
    P3 = PCA(n_components=3).fit_transform(flat - flat.mean(0)).reshape(*Z.shape[:-1], 3)
    P3 = P3 / (np.sqrt((P3.reshape(-1, 3) ** 2).sum(1).mean()) + 1e-12)
    ax.scatter(P3[idx, 0, 0], P3[idx, 0, 1], P3[idx, 0, 2], c=phase[idx, 0],
               cmap=CYCLIC, s=1.6, alpha=0.85, linewidths=0, rasterized=True)
    ax.set_title("canonical latent (PC1-3)", fontsize=6.4, pad=-2, color=OURS)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor(GRID)
        pane.line.set_color(GRID)
    ax.view_init(elev=26, azim=38)
    panel_label(ax, "c", dx=-0.02, dy=1.16)

    # ---------- d: barcode ------------------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    dgms = persistence_diagrams(Z[:, 0], maxdim=2, n_points=420, seed=0)
    y = 0
    yticks, ylabels = [], []
    for d, (dgm, col) in enumerate(zip(dgms, [MUTED, CAT[0], CAT[1]])):
        life = np.sort(lifetimes(dgm))[::-1][:10]
        births = np.asarray(dgm)[np.argsort(-(lifetimes(dgm)))][:10, 0]
        y0 = y
        for b, l in zip(births, life):
            ax.plot([b, b + l], [y, y], color=col, lw=1.5, solid_capstyle="butt")
            y += 1
        yticks.append((y0 + y - 1) / 2)
        ylabels.append(f"$H_{d}$")
        y += 1.6
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("filtration value", labelpad=1.5)
    frac = table.get("betti_correct", {}).get("mean", float("nan"))
    ax.set_title(r"persistence of $z$" + (f"  ($\\beta$=(1,2,1) in {frac:.0%} of runs)"
                                          if np.isfinite(frac) else ""),
                 fontsize=6.4, pad=3)
    ax.invert_yaxis()
    hairline_grid(ax, "x")
    panel_label(ax, "d", dx=-0.16, dy=1.20)

    # ---------- e, f: the two circles -------------------------------------
    P = PCA(n_components=min(4, Z.shape[-1])).fit_transform(flat - flat.mean(0))
    P = (P / (np.sqrt((P ** 2).sum(1).mean()) + 1e-12)).reshape(*Z.shape[:-1], -1)
    for j, (dims, ph) in enumerate((((0, 1), 0), ((2, 3), 1))):
        ax = fig.add_subplot(gs[1, j])
        d0, d1 = dims
        if d1 >= P.shape[-1]:
            bare(ax)
            continue
        sc = ax.scatter(P[idx, 0, d0], P[idx, 0, d1], c=phase[idx, ph], cmap=CYCLIC,
                        s=1.6, alpha=0.85, linewidths=0, vmin=-np.pi, vmax=np.pi,
                        rasterized=True)
        ax.set_title(f"PC{d0+1}-PC{d1+1}, coloured by $\\theta_{ph+1}$",
                     fontsize=6.4, pad=2.5)
        ax.set_aspect("equal")
        bare(ax)
        panel_label(ax, "ef"[j], dx=-0.06, dy=1.20)
    cb2 = fig.colorbar(sc, ax=[fig.axes[-2], fig.axes[-1]], fraction=0.035,
                       pad=0.012, aspect=14, ticks=[-np.pi, 0, np.pi])
    cb2.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=5.4)
    cb2.set_label("grid phase", fontsize=6.0, labelpad=1.5)
    cb2.ax.tick_params(width=0.5, length=1.8)
    cb2.outline.set_linewidth(0.4)

    # ---------- g: commutator matrix --------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    G = art["GND::generators"]
    K = G.shape[0]
    comm = np.einsum("iab,jbc->ijac", G, G) - np.einsum("jab,ibc->ijac", G, G)
    mag = np.linalg.norm(comm.reshape(K, K, -1), axis=2)
    im3 = ax.imshow(mag, cmap=SEQ_BLUE, vmin=0.0, vmax=max(mag.max(), 1e-3))
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([str(i + 1) for i in range(K)], fontsize=5.6)
    ax.set_yticklabels([str(i + 1) for i in range(K)], fontsize=5.6)
    ax.set_title(r"$\|[G_i,G_j]\|_F$" f"  (mean {get(table,'GND','abelianness'):.3f})",
                 fontsize=6.4, pad=3)
    ax.set_xlabel("generator $j$", labelpad=1.0)
    ax.set_ylabel("generator $i$", labelpad=1.0)
    cb3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.03)
    cb3.ax.tick_params(labelsize=5.4, width=0.5, length=1.8)
    cb3.outline.set_linewidth(0.4)
    panel_label(ax, "g", dx=-0.22, dy=1.20)

    # ---------- h: recovered vs true phase shift ---------------------------
    ax = fig.add_subplot(gs[1, 3])
    # Rebuild the ground-truth context specs by matching the names saved with the
    # artefacts, rather than assuming which context family was run.
    specs = _match_contexts(names)
    true = np.array([[s.group_params["delta1"], s.group_params["delta2"]] for s in specs])
    theta = art["GND::theta"]
    k = min(len(true), len(theta))
    pred, r2 = _loo_predict(theta[:k], true[:k])
    lim = np.abs(true[:k]).max() * 1.25 + 0.2
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.7, linestyle=(0, (3, 2)), zorder=1)
    for i, (col, lab) in enumerate(zip((CAT[0], CAT[1]), (r"$\delta_1$", r"$\delta_2$"))):
        ax.scatter(true[:k, i], pred[:, i], s=16, color=col, label=lab,
                   linewidths=0.4, edgecolors=SURFACE, zorder=3)
    ax.set_xlabel("true phase shift (rad)", labelpad=1.5)
    ax.set_ylabel("predicted (LOO)", labelpad=1.5)
    ax.set_title(f"algebra recovery  $R^2$ = {r2:.2f}", fontsize=6.4, pad=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=6.0, handletextpad=0.2, borderpad=0.1)
    hairline_grid(ax, "both")
    panel_label(ax, "h", dx=-0.30, dy=1.20)

    out = Path(out or FIGURE_DIR / "fig4_grid_cells.pdf")
    savefig(fig, out, dpi=400)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()

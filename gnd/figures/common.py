"""Helpers shared by the results figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.common import RESULTS_DIR, load_json


def load(exp: str) -> dict:
    p = RESULTS_DIR / exp / "results.json"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}; run the experiment first")
    return load_json(p)


def load_artifacts(exp: str, name: str = "artifacts.npz") -> dict:
    p = RESULTS_DIR / exp / name
    if not p.exists():
        raise FileNotFoundError(f"missing {p}; run the experiment first")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def get(table: dict, method: str, key: str, field: str = "mean") -> float:
    try:
        return float(table[method][key][field])
    except (KeyError, TypeError):
        return float("nan")


def project2d(X: np.ndarray, basis_from: np.ndarray | None = None) -> np.ndarray:
    """Project to the leading two principal components of ``basis_from``.

    The basis is always fitted on the pooled data of the panel, so a method is
    never given a per-context projection that would hide misalignment.
    """
    from sklearn.decomposition import PCA

    src = X if basis_from is None else basis_from
    src = np.asarray(src, float).reshape(-1, src.shape[-1])
    p = PCA(n_components=2).fit(src - src.mean(0))
    flat = np.asarray(X, float).reshape(-1, X.shape[-1])
    out = p.transform(flat - src.mean(0))
    return out.reshape(*X.shape[:-1], 2)


def unit_scale(X: np.ndarray) -> np.ndarray:
    """Rescale to unit RMS radius so panels are visually comparable."""
    X = np.asarray(X, float)
    c = X.reshape(-1, X.shape[-1]).mean(0)
    r = np.sqrt(((X.reshape(-1, X.shape[-1]) - c) ** 2).sum(1).mean()) + 1e-12
    return (X - c) / r


def method_order(table: dict, preferred: tuple[str, ...]) -> list[str]:
    """Preferred methods first, then whatever else is present."""
    present = [m for m in preferred if m in table]
    return present + sorted(set(table) - set(present))


def bar_panel(ax, table, methods, key, ylabel, fmt="{:.2f}", baseline=None,
              baseline_label=None, rotation=38, annotate=True, ylim=None):
    """Method-comparison bars: identity comes from the axis, colour only marks ours."""
    from .style import (
        BASELINE_EDGE, CRITICAL, INK, INK2, MUTED, annotate_bars, hairline_grid,
        method_color, method_edge,
    )

    vals = [get(table, m, key) for m in methods]
    errs = [get(table, m, key, "sem") for m in methods]
    x = np.arange(len(methods))
    bars = ax.bar(x, vals, width=0.72,
                  color=[method_color(m) for m in methods],
                  edgecolor=[method_edge(m) for m in methods], linewidth=0.6, zorder=3)
    ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=INK2, elinewidth=0.7,
                capsize=1.6, capthick=0.7, zorder=4)
    if baseline is not None:
        ax.axhline(baseline, color=CRITICAL, lw=0.8, linestyle=(0, (3, 2)), zorder=2)
        if baseline_label:
            ax.text(len(methods) - 0.45, baseline, baseline_label, fontsize=6.0,
                    color=CRITICAL, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=rotation, ha="right")
    ax.set_ylabel(ylabel)
    hairline_grid(ax)
    if ylim:
        ax.set_ylim(*ylim)
    if annotate:
        annotate_bars(ax, bars, vals, fmt=fmt, size=5.6)
    return bars, vals

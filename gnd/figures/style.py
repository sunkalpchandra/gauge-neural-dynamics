"""Shared figure style: palette, colour roles, and matplotlib defaults.

Design decisions, and why
-------------------------
*Categorical colour is not asked to carry method identity.*  There are ten
methods in the comparison, which is more than any palette can separate for
colour-blind readers.  In every comparison panel the method name is therefore on
the category axis or a direct label, and colour is used only to separate "ours"
from "baselines".  Sweeps, which have at most four series, use the first four
validated categorical slots and always carry a legend.

*Rate maps use a single-hue sequential ramp* (blue, light to dark), not a rainbow:
firing rate is a magnitude, and a multi-hue ramp invents boundaries where the data
has none.

*Phase, cue angle and reach direction use a cyclic ramp.*  These variables live on
a circle, so a sequential ramp would place an artificial seam at an arbitrary
point; ``twilight`` is perceptually uniform and closes on itself.

The categorical slots below pass the adjacent-pair colour-vision gates on the
light surface (worst adjacent CVD dE 9.1, worst adjacent normal-vision dE 19.6).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# -- validated categorical slots (light surface) -----------------------------
CAT = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# -- chrome and ink ---------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# -- role colours ------------------------------------------------------------
OURS = CAT[0]
OURS_ALT = CAT[1]
OURS_ALT2 = CAT[2]
BASELINE_FILL = "#d5d3cb"
BASELINE_EDGE = "#898781"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

OUR_METHODS = ("GND", "GND-flow", "GND-unpaired", "GND-gl", "full")


def method_color(name: str) -> str:
    """Blue for the full model, orange/aqua for its variants, grey for baselines."""
    if name in ("GND", "full"):
        return OURS
    if name in ("GND-flow", "GND-gl"):
        return OURS_ALT
    if name == "GND-unpaired":
        return OURS_ALT2
    return BASELINE_FILL


def method_edge(name: str) -> str:
    return INK if name in OUR_METHODS else BASELINE_EDGE


# -- ramps -------------------------------------------------------------------
_BLUE_STEPS = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_ORANGE_STEPS = ["#fdf4ef", "#fbdccc", "#f7bc9d", "#f19b6d", "#eb6834", "#c8501f", "#9c3d17", "#702b10"]
SEQ_BLUE = LinearSegmentedColormap.from_list("gnd_blue", _BLUE_STEPS)
SEQ_ORANGE = LinearSegmentedColormap.from_list("gnd_orange", _ORANGE_STEPS)
DIVERGING = LinearSegmentedColormap.from_list(
    "gnd_div", ["#0d366b", "#3987e5", "#cde2fb", "#f0efec", "#f7c0c0", "#e34948", "#8c1f1f"]
)
CYCLIC = plt.get_cmap("twilight_shifted")


def rate_cmap(bad: str = "#f4f3ef"):
    cm = SEQ_BLUE.copy()
    cm.set_bad(bad)
    return cm


# -- matplotlib defaults ------------------------------------------------------
def use_paper_style(base_size: float = 8.0) -> None:
    """Vector-PDF publication defaults matching the paper's Times body text."""
    mpl.rcParams.update({
        "pdf.fonttype": 42,          # embed TrueType, keep text selectable
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": base_size,
        "axes.titlesize": base_size + 0.5,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "figure.titlesize": base_size + 1.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.3,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def panel_label(ax, letter: str, dx: float = -0.16, dy: float = 1.06, size: float = 9.5) -> None:
    # Axes3D.text takes (x, y, z, s); text2D is the flat-overlay equivalent.
    fn = getattr(ax, "text2D", ax.text)
    fn(dx, dy, letter, transform=ax.transAxes, fontsize=size,
       fontweight="bold", va="top", ha="left", color=INK)


def hairline_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def bare(ax) -> None:
    """Strip a panel down to the mark -- for manifold scatter panels."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def savefig(fig, path, preview_dir: str | None = None, **kw) -> None:
    """Write the vector PDF, and optionally a raster preview for visual checks."""
    import os
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", **kw)
    print(f"  wrote {path}")
    preview_dir = preview_dir or os.environ.get("GND_FIGURE_PREVIEW")
    if preview_dir:
        pv = Path(preview_dir)
        pv.mkdir(parents=True, exist_ok=True)
        raster = {**kw, "dpi": 190}          # caller may already have set dpi
        fig.savefig(pv / (path.stem + ".png"), format="png", **raster)


def annotate_bars(ax, bars, values, fmt="{:.2f}", dy=0.012, rotation=0, size=None):
    """Direct labels on bars -- the relief for sub-3:1 fills, and it lets the
    reader recover exact values without a table."""
    span = max(abs(v) for v in values) or 1.0
    for b, v in zip(bars, values):
        if not np.isfinite(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + np.sign(v or 1) * dy * span,
                fmt.format(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=size or mpl.rcParams["font.size"] - 2,
                color=INK2, rotation=rotation)

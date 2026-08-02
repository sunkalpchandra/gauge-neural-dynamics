#!/usr/bin/env python
"""Generate every table and every in-text number in the paper from results JSON.

Nothing numeric is typed into ``main.tex``.  Tables are written to
``paper/generated/*.tex`` and in-text values are emitted as LaTeX macros in
``paper/generated/numbers.tex``, so a stale or missing experiment produces a
loud failure here rather than a wrong number in the PDF.

Usage
-----
    python scripts/make_tables.py               # all available experiments
    python scripts/make_tables.py --strict      # fail if anything is missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnd.utils.common import RESULTS_DIR, load_json  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "paper" / "generated"

MISSING: list[str] = []

# Which generated files each experiment owns.  When an experiment is refused its
# tables must be *removed*, not merely left alone: they are tracked in git, so
# leaving them would typeset the previous run's numbers under a caption claiming
# to describe the current one, while this script reports the experiment missing.
OWNED_TABLES = {
    "exp1_hippocampus": ("table_hippocampus.tex", "table_hippocampus_per_context.tex"),
    "exp2_grid_cells": ("table_grid.tex", "table_grid_families.tex"),
    "exp3_motor_cortex": ("table_motor.tex",),
    "exp4_ablations": ("table_ablations.tex",),
    "exp5_robustness": ("table_noise.tex",),
    "exp6_continuous_context": ("table_continuous.tex",),
}


def invalidate(exp: str) -> None:
    """Delete the tables an unusable experiment owns."""
    for name in OWNED_TABLES.get(exp, ()):
        p = OUT / name
        if p.exists():
            p.unlink()
            print(f"  removed stale paper/generated/{name} ({exp} unusable)")


# ---------------------------------------------------------------------------
def load(exp: str) -> dict | None:
    """Load a results file, tolerating the per-seed checkpoints.

    A checkpoint written mid-run carries ``complete: False`` and has no
    aggregated table yet; we say so rather than crashing, so that a partial
    build still tells you what is and is not final.  A ``--quick`` pilot is
    refused for the same reason: its numbers are real but are produced from a
    single seed and a fraction of the epochs, and nothing downstream could tell
    them apart from a finished sweep.
    """
    p = RESULTS_DIR / exp / "results.json"
    if not p.exists():
        MISSING.append(f"results for {exp}")
        invalidate(exp)
        return None
    res = load_json(p)
    if not res.get("complete", True):
        # Refuse partial data outright.  Reporting it but returning it anyway
        # would let half-finished numbers reach the paper, which is precisely
        # what this whole generation path exists to prevent.
        MISSING.append(f"{exp} is incomplete (partial checkpoint); refusing to use it")
        invalidate(exp)
        return None
    if res.get("args", {}).get("quick"):
        MISSING.append(f"{exp} is a --quick pilot run, not a full sweep; refusing to use it")
        invalidate(exp)
        return None
    if "table" not in res and "tables" not in res and "summary" not in res:
        MISSING.append(f"{exp} has no aggregated table yet")
        invalidate(exp)
        return None
    return res


def cell(table: dict, method: str, key: str, digits: int = 3, best: bool = False) -> str:
    try:
        e = table[method][key]
    except (KeyError, TypeError):
        return "--"
    if e is None or not np.isfinite(e["mean"]):
        return "--"
    s = f"{e['mean']:.{digits}f}\\,\\tiny{{$\\pm$\\,{e['sem']:.{digits}f}}}"
    return f"\\textbf{{{s}}}" if best else s


def val(table: dict, method: str, key: str) -> float:
    try:
        return float(table[method][key]["mean"])
    except (KeyError, TypeError):
        return float("nan")


def sem(table: dict, method: str, key: str) -> float:
    try:
        return float(table[method][key]["sem"])
    except (KeyError, TypeError):
        return float("nan")


def fmt(m: float, s: float | None = None, digits: int = 3) -> str:
    if not np.isfinite(m):
        return "--"
    if s is None or not np.isfinite(s):
        return f"{m:.{digits}f}"
    return f"${m:.{digits}f} \\pm {s:.{digits}f}$"


def best_of(table: dict, methods, key: str, higher_is_better: bool) -> tuple[str, float]:
    vals = [(m, val(table, m, key)) for m in methods]
    vals = [(m, v) for m, v in vals if np.isfinite(v)]
    if not vals:
        return "--", float("nan")
    return (max if higher_is_better else min)(vals, key=lambda t: t[1])


def artefact_realised_abelianness(exp: str) -> str:
    """Realised abelianness from an experiment's saved first-seed artefacts.

    A fallback only, for runs that predate the metric: rather than repeat several
    hours of compute we recompute it from the stored generators and coefficients.
    It is a single-seed diagnostic and is labelled as such wherever quoted.
    Prefer :func:`realised_abelian` , which uses the per-seed values when the
    experiment has been re-run since.
    """
    from gnd.geometry.metrics import realised_abelianness

    p = RESULTS_DIR / exp / "artifacts.npz"
    if not p.exists():
        return "--"
    with np.load(p, allow_pickle=False) as z:
        if "GND::generators" not in z.files or "GND::theta" not in z.files:
            return "--"
        return fmt(realised_abelianness(z["GND::generators"], z["GND::theta"]), None)


def realised_abelian(table: dict, exp: str, method: str = "GND") -> str:
    """Realised abelianness over seeds, falling back to the first-seed artefact."""
    m = val(table, method, "realised_abelianness")
    if np.isfinite(m):
        return fmt(m, sem(table, method, "realised_abelianness"))
    return artefact_realised_abelianness(exp)


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body)
    print(f"  wrote paper/generated/{name}")


# ---------------------------------------------------------------------------
METRIC_COLUMNS = [
    ("transport_r2", r"Transport $R^2\!\uparrow$", 3, True),
    ("cis", r"CIS$\uparrow$", 3, True),
    ("gcs", r"GCS$\uparrow$", 3, True),
    ("gre", r"GRE$\downarrow$", 3, False),
    ("mps", r"MPS$\uparrow$", 3, True),
    ("algebra_recovery_r2", r"Alg.\ $R^2\!\uparrow$", 3, True),
    ("transform_magnitude", r"$\|T_c-\mathrm{id}\|$", 2, None),
]

ROW_ORDER = ("GND", "GND-flow", "GND-unpaired", "GND-gl", "PCA", "UMAP",
             "Autoencoder", "VAE", "CCA", "Procrustes", "ManifoldAlign")


def comparison_table(table: dict, caption: str, label: str,
                     columns=METRIC_COLUMNS, rows=ROW_ORDER) -> str:
    methods = [m for m in rows if m in table]
    extra = sorted(set(table) - set(methods))
    methods += extra
    if not methods:
        return ""
    best = {}
    for key, _, _, hib in columns:
        if hib is None:
            continue
        best[key] = best_of(table, methods, key, hib)[0]

    n = len(columns)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{tab:" + label + r"}",
        r"\vspace{2pt}",
        r"\small",
        r"\begin{tabular}{l" + "c" * n + r"}",
        r"\toprule",
        "Method & " + " & ".join(c[1] for c in columns) + r" \\",
        r"\midrule",
    ]
    for i, m in enumerate(methods):
        if m == "PCA" and i > 0:
            lines.append(r"\midrule")
        label_m = r"\textbf{" + m + r"}" if m.startswith("GND") else m
        if m == "GND":
            label_m = r"\textbf{GND (ours)}"
        cells = [cell(table, m, k, d, best.get(k) == m) for k, _, d, _ in columns]
        lines.append(label_m + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def do_exp1(macros: dict) -> None:
    res = load("exp1_hippocampus")
    if res is None:
        return
    t = res["table"]
    n_seeds = max((v.get("_n_seeds", 0) for v in t.values()), default=0)
    write("table_hippocampus.tex", comparison_table(
        t,
        "\\textbf{Experiment 1: hippocampal place-cell remapping.} "
        f"Mean $\\pm$ s.e.m.\\ over {n_seeds} seeds, all on held-out samples. "
        "Transport $R^2$ is the accuracy of predicting the population response in "
        "another environment; GRE is normalised so that $1.0$ is the score of "
        "assuming no transformation at all. The last column reports how far the "
        "recovered transformations actually move the latent, and must be read "
        "together with GCS: a family collapsed onto the identity composes "
        "perfectly while explaining nothing.",
        "hippocampus"))
    bl = ("PCA", "UMAP", "Autoencoder", "VAE", "CCA", "Procrustes", "ManifoldAlign")
    bname, bval = best_of(t, bl, "transport_r2", True)
    gname, gval = best_of(t, bl, "gre", False)
    macros.update({
        "hipSeeds": str(n_seeds),
        "hipGNDtransport": fmt(val(t, "GND", "transport_r2"), sem(t, "GND", "transport_r2")),
        "hipGNDcis": fmt(val(t, "GND", "cis"), sem(t, "GND", "cis")),
        "hipGNDgcs": fmt(val(t, "GND", "gcs"), sem(t, "GND", "gcs")),
        "hipGNDgre": fmt(val(t, "GND", "gre"), sem(t, "GND", "gre")),
        "hipGNDmps": fmt(val(t, "GND", "mps"), sem(t, "GND", "mps")),
        "hipGNDclosure": fmt(val(t, "GND", "closure_defect"), sem(t, "GND", "closure_defect"), 2),
        "hipGNDleak": fmt(val(t, "GND", "context_leakage"), sem(t, "GND", "context_leakage")),
        "hipGNDalg": fmt(val(t, "GND", "algebra_recovery_r2"), sem(t, "GND", "algebra_recovery_r2")),
        "hipFlowTransport": fmt(val(t, "GND-flow", "transport_r2"), sem(t, "GND-flow", "transport_r2")),
        "hipFlowGre": fmt(val(t, "GND-flow", "gre"), sem(t, "GND-flow", "gre")),
        "hipFlowMps": fmt(val(t, "GND-flow", "mps"), sem(t, "GND-flow", "mps")),
        "hipFlowGcs": fmt(val(t, "GND-flow", "gcs"), sem(t, "GND-flow", "gcs")),
        "hipUnpairedTransport": fmt(val(t, "GND-unpaired", "transport_r2"),
                                    sem(t, "GND-unpaired", "transport_r2")),
        "hipUnpairedCis": fmt(val(t, "GND-unpaired", "cis"), sem(t, "GND-unpaired", "cis")),
        "hipBestBaseName": bname,
        "hipBestBaseTransport": fmt(bval, sem(t, bname, "transport_r2")),
        "hipBestBaseGreName": gname,
        "hipBestBaseGre": fmt(gval, sem(t, gname, "gre")),
        "hipMAgcs": fmt(val(t, "ManifoldAlign", "gcs"), sem(t, "ManifoldAlign", "gcs")),
        "hipMAmag": fmt(val(t, "ManifoldAlign", "transform_magnitude"),
                        sem(t, "ManifoldAlign", "transform_magnitude"), 2),
        "hipGNDmag": fmt(val(t, "GND", "transform_magnitude"),
                         sem(t, "GND", "transform_magnitude"), 2),
        "hipRealisedAbelian": realised_abelian(t, "exp1_hippocampus"),
    })
    # Recovery on the deliberately non-affine morph context, where the linear
    # gauge is expected to fail and the flow gauge to do better.  Quoted
    # separately because the overall GRE averages over four affine contexts too.
    def morph_gre(method):
        sub = [r for r in res["rows"] if r.get("method") == method]
        key = next((k for k in (sub[0] if sub else {}) if k.startswith("gre::") and "morph" in k), None)
        if key is None:
            return float("nan"), float("nan")
        v = np.array([r[key] for r in sub if np.isfinite(r.get(key, np.nan))])
        if v.size == 0:
            return float("nan"), float("nan")
        return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0

    macros["hipGNDmorph"] = fmt(*morph_gre("GND"))
    macros["hipFlowMorph"] = fmt(*morph_gre("GND-flow"))
    macros["hipPCAmorph"] = fmt(*morph_gre("PCA"))

    # per-context recovery, including the deliberately non-affine morph
    rows = [r for r in res["rows"] if r.get("method") == "GND"]
    keys = sorted({k for r in rows for k in r if k.startswith("gre::")})
    if keys:
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\caption{\textbf{Per-context recovery error in Experiment 1.} "
                 r"The final context is a radially graded twist, which is a "
                 r"diffeomorphism but not an affine map, and so lies outside the "
                 r"reach of the linear gauge by construction; the flow gauge "
                 r"recovers it.}",
                 r"\label{tab:hip-per-context}", r"\vspace{2pt}", r"\small",
                 r"\begin{tabular}{l" + "c" * len(keys) + r"}", r"\toprule",
                 "Model & " + " & ".join(k.split("::")[1].replace("&", "\\&") for k in keys) + r" \\",
                 r"\midrule"]
        for m in ("GND", "GND-flow", "PCA"):
            sub = [r for r in res["rows"] if r.get("method") == m]
            if not sub:
                continue
            vals = []
            for k in keys:
                v = [r[k] for r in sub if np.isfinite(r.get(k, np.nan))]
                vals.append(f"{np.mean(v):.2f}" if v else "--")
            lines.append((r"\textbf{GND}" if m == "GND" else m) + " & " + " & ".join(vals) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        write("table_hippocampus_per_context.tex", "\n".join(lines))


def do_exp2(macros: dict) -> None:
    res = load("exp2_grid_cells")
    if res is None:
        return
    tabs = res.get("tables", {})
    main = tabs.get("translation", {})
    n_seeds = max((v.get("_n_seeds", 0) for v in main.values()), default=0)
    write("table_grid.tex", comparison_table(
        main,
        "\\textbf{Experiment 2: grid-cell phase translations.} "
        f"Mean $\\pm$ s.e.m.\\ over {n_seeds} seeds. The context family is the "
        "abelian translation group $T^2$, which acts on the $\\mathbb{R}^4$ torus "
        "embedding as a pair of independent plane rotations, so the correct "
        "algebra is commuting and the correct Betti numbers are $(1,2,1)$.",
        "grid"))

    # the three context families side by side, for the GND rows only
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{\textbf{Group structure across grid-cell context families.} "
             r"Adding the $60^\circ$ lattice automorphism makes the group "
             r"non-abelian, so a non-zero commutator is then the correct answer. "
             r"The rescaling family is a negative control: a non-integer gain "
             r"change is well defined on the universal cover but not on the torus, "
             r"so no latent gauge transformation of the population manifold can "
             r"express it.}",
             r"\label{tab:grid-families}", r"\vspace{2pt}", r"\small",
             r"\begin{tabular}{llcccc}", r"\toprule",
             r"Context family & Model & GCS$\uparrow$ & "
             r"$\overline{\|[G_i,G_j]\|}$ & GRE$\downarrow$ & Transport $R^2\!\uparrow$ \\",
             r"\midrule"]
    pretty = {"translation": r"$T^2$ (abelian)",
              "translation+rotation": r"$T^2\rtimes Z_6$",
              "all": r"$+$ rescaling (control)"}
    for fam in ("translation", "translation+rotation", "all"):
        tb = tabs.get(fam)
        if not tb:
            continue
        first = True
        for m in ("GND", "GND-gl", "GND-flow"):
            if m not in tb:
                continue
            lines.append(
                (pretty.get(fam, fam) if first else "") + " & " +
                (r"\textbf{GND}" if m == "GND" else m) + " & " +
                cell(tb, m, "gcs") + " & " + cell(tb, m, "abelianness", 3) + " & " +
                cell(tb, m, "gre", 2) + " & " + cell(tb, m, "transport_r2") + r" \\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]
    write("table_grid_families.tex", "\n".join(lines))

    rot = tabs.get("translation+rotation", {})
    ctrl = tabs.get("all", {})

    # A seed counts as a failed fit when its final *training* loss exceeds
    # LOSS_TOL times the median across seeds for the same method.  The criterion
    # reads the training objective only, never the evaluation metrics being
    # compared, so it cannot be tuned to produce a favourable comparison.  On
    # this sweep every seed lies within 1.06x of its own median except one, at
    # 1.63x, so the threshold sits in an empty gap rather than on a boundary.
    LOSS_TOL = 1.25

    def _tr(method: str) -> list[dict]:
        return sorted([r for r in res["rows"]
                       if r.get("context_set") == "translation"
                       and r.get("method") == method and "error" not in r],
                      key=lambda r: r["seed"])

    def _split(method: str) -> tuple[list[int], list[int], float]:
        rows = _tr(method)
        loss = np.array([r.get("final_loss", np.nan) for r in rows], float)
        if not rows or not np.isfinite(loss).all():
            return [r["seed"] for r in rows], [], float("nan")
        med = np.median(loss)
        ok = [r["seed"] for r, l in zip(rows, loss) if l <= LOSS_TOL * med]
        bad = [r["seed"] for r, l in zip(rows, loss) if l > LOSS_TOL * med]
        worst = float(loss.max() / med) if med > 0 else float("nan")
        return ok, bad, worst

    gnd_ok, gnd_bad, gnd_worst = _split("GND")
    gl_ok, _, _ = _split("GND-gl")
    both = sorted(set(gnd_ok) & set(gl_ok))

    def _over(method: str, seeds: list[int], key: str) -> str:
        v = np.array([r.get(key, np.nan) for r in _tr(method) if r["seed"] in seeds], float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return "--"
        return fmt(float(v.mean()),
                   float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None)

    n_tr = len(_tr("GND"))
    macros.update({
        "gridLossTol": f"{LOSS_TOL:g}",
        "gridConvergedSeeds": f"{len(both)} of {n_tr}" if n_tr else "--",
        "gridFailedSeeds": f"{len(gnd_bad)} of {n_tr}" if n_tr else "--",
        "gridLossRatio": (f"{gnd_worst:.1f}" if np.isfinite(gnd_worst) else "--"),
        "gridGNDgreConv": _over("GND", both, "gre"),
        "gridGLgreConv": _over("GND-gl", both, "gre"),
        "gridGNDtransportConv": _over("GND", both, "transport_r2"),
        "gridGLtransportConv": _over("GND-gl", both, "transport_r2"),
    })

    # The flow gauge is bimodal on the non-abelian family -- it beats the linear
    # gauge on some seeds and diverges on others -- so the mean alone misdescribes
    # it.  Count the seeds instead, paired within seed.
    def _rot(method: str) -> list[dict]:
        return sorted([r for r in res["rows"]
                       if r.get("context_set") == "translation+rotation"
                       and r.get("method") == method and "error" not in r],
                      key=lambda r: r["seed"])

    paired = [(a.get("gre"), b.get("gre")) for a, b in zip(_rot("GND"), _rot("GND-flow"))]
    paired = [(a, b) for a, b in paired
              if isinstance(a, (int, float)) and isinstance(b, (int, float))]
    flow_better = sum(1 for a, b in paired if b < a)

    macros.update({
        "gridRotGcs": fmt(val(rot, "GND", "gcs"), sem(rot, "GND", "gcs")),
        "gridRotTransport": fmt(val(rot, "GND", "transport_r2"),
                                sem(rot, "GND", "transport_r2")),
        # No artefact fallback here: the saved archive is the *translation*
        # family's, so falling back to it would quote the abelian family's number
        # under a non-abelian label.
        "gridRotRealisedAbelian": fmt(val(rot, "GND", "realised_abelianness"),
                                      sem(rot, "GND", "realised_abelianness")),
        "gridRotFlowTransport": fmt(val(rot, "GND-flow", "transport_r2"),
                                    sem(rot, "GND-flow", "transport_r2")),
        "gridRotFlowBetterSeeds": (f"{flow_better} of {len(paired)}" if paired else "--"),
    })
    macros.update({
        "gridSeeds": str(n_seeds),
        "gridGNDgcs": fmt(val(main, "GND", "gcs"), sem(main, "GND", "gcs")),
        "gridGNDabelian": fmt(val(main, "GND", "abelianness"), sem(main, "GND", "abelianness")),
        "gridRealisedAbelian": realised_abelian(main, "exp2_grid_cells"),
        "gridGLgcs": fmt(val(main, "GND-gl", "gcs"), sem(main, "GND-gl", "gcs")),
        "gridPCAgre": fmt(val(main, "PCA", "gre"), sem(main, "PCA", "gre")),
        "gridPCAgcs": fmt(val(main, "PCA", "gcs"), sem(main, "PCA", "gcs")),
        "gridPCAbetti": (f"{val(main, 'PCA', 'betti_correct'):.0%}".replace("%", r"\%")
                         if np.isfinite(val(main, "PCA", "betti_correct")) else "--"),
        "gridGNDcisMain": fmt(val(main, "GND", "cis"), sem(main, "GND", "cis")),
        "gridGNDgre": fmt(val(main, "GND", "gre"), sem(main, "GND", "gre")),
        "gridGNDcis": fmt(val(main, "GND", "cis"), sem(main, "GND", "cis")),
        "gridGNDtransport": fmt(val(main, "GND", "transport_r2"), sem(main, "GND", "transport_r2")),
        "gridGNDalg": fmt(val(main, "GND", "algebra_recovery_r2"),
                          sem(main, "GND", "algebra_recovery_r2")),
        "gridGNDbetti": f"{val(main, 'GND', 'betti_correct'):.0%}".replace("%", r"\%"),
        "gridGLgre": fmt(val(main, "GND-gl", "gre"), sem(main, "GND-gl", "gre")),
        "gridGLabelian": fmt(val(main, "GND-gl", "abelianness"), sem(main, "GND-gl", "abelianness")),
        "gridRotAbelian": fmt(val(rot, "GND", "abelianness"), sem(rot, "GND", "abelianness")),
        "gridRotGre": fmt(val(rot, "GND", "gre"), sem(rot, "GND", "gre")),
        "gridRotFlowGre": fmt(val(rot, "GND-flow", "gre"), sem(rot, "GND-flow", "gre")),
        "gridCtrlGre": fmt(val(ctrl, "GND", "gre"), sem(ctrl, "GND", "gre"), 2),
        "gridCtrlTransport": fmt(val(ctrl, "GND", "transport_r2"), sem(ctrl, "GND", "transport_r2")),
        "gridBestBaseGre": fmt(*(lambda n: (val(main, n, "gre"), sem(main, n, "gre")))(
            best_of(main, ("PCA", "UMAP", "Autoencoder", "VAE", "CCA", "Procrustes",
                           "ManifoldAlign"), "gre", False)[0])),
        "gridBestBaseGreName": best_of(main, ("PCA", "UMAP", "Autoencoder", "VAE", "CCA",
                                              "Procrustes", "ManifoldAlign"), "gre", False)[0],
        "gridBestBaseGcs": fmt(*(lambda n: (val(main, n, "gcs"), sem(main, n, "gcs")))(
            best_of(main, ("PCA", "UMAP", "Autoencoder", "VAE", "CCA", "Procrustes",
                           "ManifoldAlign"), "gcs", True)[0])),
    })


def do_exp3(macros: dict) -> None:
    res = load("exp3_motor_cortex")
    if res is None:
        return
    t = res["table"]
    n_seeds = max((v.get("_n_seeds", 0) for v in t.values()), default=0)
    cols = [
        ("transport_r2", r"Transport $R^2\!\uparrow$", 3, True),
        ("trajectory_alignment_error", r"Traj.\ err.$\downarrow$", 3, False),
        ("cis", r"CIS$\uparrow$", 3, True),
        ("gcs", r"GCS$\uparrow$", 3, True),
        ("rotation_frequency_cv", r"Freq.\ CV$\downarrow$", 3, False),
        ("rotation_plane_angle_deg", r"Plane angle (deg)", 1, None),
    ]
    write("table_motor.tex", comparison_table(
        t,
        "\\textbf{Experiment 3: motor-cortex reach conditions.} "
        f"Mean $\\pm$ s.e.m.\\ over {n_seeds} seeds. Trajectory error is the "
        "residual between trial-averaged canonical trajectories of different reach "
        "conditions. Frequency CV tests Eq.~\\eqref{eq:conjugacy}: conjugation "
        "preserves eigenvalues, so rotation frequency should be shared across "
        "conditions while the rotation plane turns.",
        "motor", columns=cols))
    bl = ("PCA", "UMAP", "Autoencoder", "VAE", "CCA", "Procrustes", "ManifoldAlign")
    bname, bval = best_of(t, bl, "transport_r2", True)
    tname, tval = best_of(t, bl, "trajectory_alignment_error", False)
    macros.update({
        "motSeeds": str(n_seeds),
        "motGNDtransport": fmt(val(t, "GND", "transport_r2"), sem(t, "GND", "transport_r2")),
        "motGNDtraj": fmt(val(t, "GND", "trajectory_alignment_error"),
                          sem(t, "GND", "trajectory_alignment_error")),
        "motGNDcis": fmt(val(t, "GND", "cis"), sem(t, "GND", "cis")),
        "motGNDgcs": fmt(val(t, "GND", "gcs"), sem(t, "GND", "gcs")),
        "motGNDfreqcv": fmt(val(t, "GND", "rotation_frequency_cv"),
                            sem(t, "GND", "rotation_frequency_cv")),
        "motGNDplane": fmt(val(t, "GND", "rotation_plane_angle_deg"),
                           sem(t, "GND", "rotation_plane_angle_deg"), 1),
        "motRefFreqcv": fmt(val(t, "GND", "circuit_reference_rotation_frequency_cv"),
                            sem(t, "GND", "circuit_reference_rotation_frequency_cv")),
        "motCircuitEquiv": fmt(val(t, "GND", "circuit_equivariance_residual_mean"),
                               sem(t, "GND", "circuit_equivariance_residual_mean")),
        "motBestBaseName": bname,
        "motBestBaseTransport": fmt(bval, sem(t, bname, "transport_r2")),
        "motBestBaseTrajName": tname,
        "motBestBaseTraj": fmt(tval, sem(t, tname, "trajectory_alignment_error")),
        "motGNDgre": fmt(val(t, "GND", "gre"), sem(t, "GND", "gre")),
        # The rotation-frequency CV is minimised trivially by a latent that fails
        # to separate conditions at all, so what matters is distance from the
        # circuit's own spread, not smallness.
        "motGNDfreqdev": fmt(abs(val(t, "GND", "rotation_frequency_cv")
                                 - val(t, "GND", "circuit_reference_rotation_frequency_cv")), None),
        "motMAfreqcv": fmt(val(t, "ManifoldAlign", "rotation_frequency_cv"),
                           sem(t, "ManifoldAlign", "rotation_frequency_cv")),
        "motMAtransport": fmt(val(t, "ManifoldAlign", "transport_r2"),
                              sem(t, "ManifoldAlign", "transport_r2")),
    })


def do_exp4(macros: dict) -> None:
    res = load("exp4_ablations")
    if res is None:
        return
    t = res.get("table", {})
    full = val(t, "full", "transport_r2")
    order = ["full", "A1: no gauge", "A2: no group loss", "A2a: no closure only",
             "A3: no topology", "no transport loss", "no invariance loss",
             "no anchoring", "unpaired (MMD)", "gauge=flow", "algebra=so",
             "algebra=sl", "algebra=se", "BCH order 1", "BCH order 3",
             "encoder=VAE", "K=2 generators", "K=4 generators", "K=12 generators"]
    order = [o for o in order if o in t] + [k for k in sorted(t)
                                            if k not in order and not k.startswith("A4")]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{\textbf{Ablations.} Each row removes or replaces one "
             r"ingredient and is refitted from scratch on the place-cell "
             r"simulation; $\Delta$ is the change in transport $R^2$ relative to "
             r"the full model. Mean $\pm$ s.e.m.\ over "
             + str(max((v.get("_n_seeds", 0) for v in t.values()), default=0))
             + r" seeds. The latent-dimension sweep is shown in "
             r"Fig.~\ref{fig:ablations}c.}",
             r"\label{tab:ablations}", r"\vspace{2pt}", r"\small",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Configuration & Transport $R^2$ & $\Delta$ & CIS & GCS & GRE \\",
             r"\midrule"]
    for k in order:
        v = val(t, k, "transport_r2")
        d = v - full
        dstr = "--" if k == "full" or not np.isfinite(d) else f"${d:+.3f}$"
        name = r"\textbf{full model}" if k == "full" else k.replace("&", "\\&")
        lines.append(f"{name} & {cell(t, k, 'transport_r2')} & {dstr} & "
                     f"{cell(t, k, 'cis')} & {cell(t, k, 'gcs')} & {cell(t, k, 'gre')} \\\\")
        if k == "full":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table_ablations.tex", "\n".join(lines))
    macros.update({
        "ablSeeds": str(max((v.get("_n_seeds", 0) for v in t.values()), default=0)),
        "ablFullTransport": fmt(full, sem(t, "full", "transport_r2")),
        "ablNoGauge": fmt(val(t, "A1: no gauge", "transport_r2"),
                          sem(t, "A1: no gauge", "transport_r2")),
        "ablNoGaugeDelta": fmt(val(t, "A1: no gauge", "transport_r2") - full, None),
        "ablNoGroupDelta": fmt(val(t, "A2: no group loss", "transport_r2") - full, None),
        "ablNoGroupGcs": fmt(val(t, "A2: no group loss", "gcs"),
                             sem(t, "A2: no group loss", "gcs")),
        "ablNoTopoDelta": fmt(val(t, "A3: no topology", "transport_r2") - full, None),
        "ablNoTopoMps": fmt(val(t, "A3: no topology", "mps"), sem(t, "A3: no topology", "mps")),
        "ablFullMps": fmt(val(t, "full", "mps"), sem(t, "full", "mps")),
        "ablNoTransportDelta": fmt(val(t, "no transport loss", "transport_r2") - full, None),
        "ablNoInvDelta": fmt(val(t, "no invariance loss", "transport_r2") - full, None),
        "ablBchOneGcs": fmt(val(t, "BCH order 1", "gcs"), sem(t, "BCH order 1", "gcs")),
        "ablDimTwo": fmt(val(t, "A4: latent dim 2", "transport_r2"),
                         sem(t, "A4: latent dim 2", "transport_r2")),
        "ablDimTwenty": fmt(val(t, "A4: latent dim 20", "transport_r2"),
                            sem(t, "A4: latent dim 20", "transport_r2")),
        "ablDimTwoGre": fmt(val(t, "A4: latent dim 2", "gre"), sem(t, "A4: latent dim 2", "gre")),
        "ablFullGre": fmt(val(t, "full", "gre"), sem(t, "full", "gre")),
        "ablFullGcs": fmt(val(t, "full", "gcs"), sem(t, "full", "gcs")),
        # The no-gauge ablation is the sharpest demonstration that GCS has to be
        # read beside the transformation magnitude: doing nothing composes
        # perfectly.
        "ablNoGaugeGre": fmt(val(t, "A1: no gauge", "gre"), sem(t, "A1: no gauge", "gre"), 2),
        "ablNoGaugeGcs": fmt(val(t, "A1: no gauge", "gcs"), sem(t, "A1: no gauge", "gcs")),
        "ablNoGaugeMag": fmt(val(t, "A1: no gauge", "transform_magnitude"),
                             sem(t, "A1: no gauge", "transform_magnitude")),
        "ablNoTransportGre": fmt(val(t, "no transport loss", "gre"),
                                 sem(t, "no transport loss", "gre"), 2),
        "ablNoTransportGcs": fmt(val(t, "no transport loss", "gcs"),
                                 sem(t, "no transport loss", "gcs")),
        "ablNoTransportMag": fmt(val(t, "no transport loss", "transform_magnitude"),
                                 sem(t, "no transport loss", "transform_magnitude"), 2),
        "ablNoAnchorDelta": fmt(val(t, "no anchoring", "transport_r2") - full, None),
        "ablNoAnchorGre": fmt(val(t, "no anchoring", "gre"), sem(t, "no anchoring", "gre")),
    })


def do_exp5(macros: dict) -> None:
    res = load("exp5_robustness")
    if res is None:
        return
    t = res.get("table", {})

    def g(sweep, value, method, key):
        try:
            e = t[sweep][str(value)][method][key]
            return float(e["mean"]), float(e["sem"])
        except (KeyError, TypeError):
            return float("nan"), float("nan")

    noise_vals = sorted(t.get("noise", {}), key=float)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{\textbf{Noise robustness.} Cross-context transport $R^2$ "
             r"as observation noise grows, in units of the activity standard "
             r"deviation. The Poisson row draws spike counts instead of adding "
             r"Gaussian noise. Mean $\pm$ s.e.m.\ over "
             + str(len(res["args"]["seeds"])) + r" seeds.}",
             r"\label{tab:noise}", r"\vspace{2pt}", r"\small",
             r"\begin{tabular}{l" + "c" * (len(noise_vals) + 1) + r"}", r"\toprule",
             r"Method & " + " & ".join(f"${float(v):g}$" for v in noise_vals)
             + r" & Poisson \\", r"\midrule"]
    for m in ("GND", "PCA", "Autoencoder", "Procrustes"):
        row = []
        for v in noise_vals:
            mu, se = g("noise", v, m, "transport_r2")
            row.append("--" if not np.isfinite(mu) else
                       f"{mu:.3f}\\,\\tiny{{$\\pm$\\,{se:.3f}}}")
        mu, se = g("poisson", 0.25, m, "transport_r2")
        row.append("--" if not np.isfinite(mu) else f"{mu:.3f}\\,\\tiny{{$\\pm$\\,{se:.3f}}}")
        lines.append((r"\textbf{GND}" if m == "GND" else m) + " & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table_noise.tex", "\n".join(lines))

    lo, hi = (noise_vals[0], noise_vals[-1]) if noise_vals else (None, None)
    if lo is not None:
        macros.update({
            "robSeeds": str(len(res["args"]["seeds"])),
            "robNoiseLo": f"{float(lo):g}",
            "robNoiseHi": f"{float(hi):g}",
            "robGNDlo": fmt(*g("noise", lo, "GND", "transport_r2")),
            "robGNDhi": fmt(*g("noise", hi, "GND", "transport_r2")),
            "robPCAhi": fmt(*g("noise", hi, "PCA", "transport_r2")),
            "robGNDpoisson": fmt(*g("poisson", 0.25, "GND", "transport_r2")),
        })
    ncounts = sorted(t.get("neurons", {}), key=float)
    if ncounts:
        macros.update({
            "robNeuronsLo": f"{int(float(ncounts[0]))}",
            "robNeuronsHi": f"{int(float(ncounts[-1]))}",
            "robGNDneuronsLo": fmt(*g("neurons", ncounts[0], "GND", "transport_r2")),
            "robGNDneuronsHi": fmt(*g("neurons", ncounts[-1], "GND", "transport_r2")),
        })
    scounts = sorted(t.get("samples", {}), key=float)
    if scounts:
        macros.update({
            "robSamplesLo": f"{int(float(scounts[0]))}",
            "robGNDsamplesLo": fmt(*g("samples", scounts[0], "GND", "transport_r2")),
        })


def do_exp6(macros: dict) -> None:
    res = load("exp6_continuous_context")
    if res is None:
        return
    s = res.get("summary", {})
    rows = res.get("rows", [])

    def g(group, key):
        try:
            e = s[group][key]
            return float(e["mean"]), float(e["sem"])
        except (KeyError, TypeError):
            return float("nan"), float("nan")

    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{\textbf{Generalisation over context space, and the "
             r"winding obstruction.} Top: cue angles held out entirely from "
             r"training are predicted from the cue value alone. Bottom: when the "
             r"cue family closes into a circle, a single-chart field cannot "
             r"represent the required winding; giving the same architecture the "
             r"universal cover removes the problem. Mean $\pm$ s.e.m.\ over "
             + str(len(res["args"]["seeds"])) + r" seeds.}",
             r"\label{tab:continuous}", r"\vspace{2pt}", r"\small",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"Setting & Transport $R^2\!\uparrow$ & GRE$\downarrow$ & "
             r"Transport $R^2$, no transform \\", r"\midrule"]
    for grp, name in (("arc_trained_cues", "trained cue angles"),
                      ("arc_held_out_cues", r"\textbf{held-out cue angles}")):
        a = fmt(*g(grp, "transport_r2"))
        b = fmt(*g(grp, "gre"))
        c = fmt(*g(grp, "transport_r2_identity"))
        lines.append(f"{name} & {a} & {b} & {c} \\\\")
    lines.append(r"\midrule")
    for grp, name in (("circle_circular", r"full circle, context space $= S^1$"),
                      ("circle_lifted", r"full circle, universal cover $\mathbb{R}$")):
        a = fmt(*g(grp, "transport_r2"))
        b = fmt(*g(grp, "gre"), digits=2)
        lines.append(f"{name} & {a} & {b} & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("table_continuous.tex", "\n".join(lines))

    def at_angle(part, key, target=180.0):
        v = [r[key] for r in rows if r.get("part") == part
             and abs(r.get("angle_deg", -1) - target) < 1e-6 and np.isfinite(r.get(key, np.nan))]
        return (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))) if len(v) > 1 \
            else ((float(np.mean(v)), float("nan")) if v else (float("nan"), float("nan")))

    macros.update({
        "contSeeds": str(len(res["args"]["seeds"])),
        "contTrained": fmt(*g("arc_trained_cues", "transport_r2")),
        "contHeld": fmt(*g("arc_held_out_cues", "transport_r2")),
        "contHeldGre": fmt(*g("arc_held_out_cues", "gre")),
        "contIdentity": fmt(*g("arc_held_out_cues", "transport_r2_identity")),
        "contNearest": fmt(*g("arc_held_out_cues", "transport_r2_nearest_seen")),
        "contCirc": fmt(*g("circle_circular", "transport_r2")),
        "contLift": fmt(*g("circle_lifted", "transport_r2")),
        "contCircAtPi": fmt(*at_angle("circle:circular", "transport_r2")),
        "contLiftAtPi": fmt(*at_angle("circle:lifted", "transport_r2")),
        "contCircGreAtPi": fmt(*at_angle("circle:circular", "gre"), digits=2),
        "contLiftGreAtPi": fmt(*at_angle("circle:lifted", "gre"), digits=2),
    })


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="also exit non-zero if any macro has no value "
                         "(a missing experiment always exits non-zero)")
    args = ap.parse_args(argv)

    macros: dict[str, str] = {}
    for fn in (do_exp1, do_exp2, do_exp3, do_exp4, do_exp5, do_exp6):
        fn(macros)

    bad = sorted(k for k, v in macros.items() if v == "--")
    body = ["% Auto-generated by scripts/make_tables.py -- do not edit.",
            "% Every numeric value quoted in the paper text is defined here and",
            "% comes directly from results/*/results.json.", ""]
    bad_names = [k for k in macros if not k.isalpha()]
    if bad_names:                       # TeX control words are letters only
        raise ValueError(f"macro names must be alphabetic: {bad_names}")
    for k in sorted(macros):
        body.append(f"\\newcommand{{\\{k}}}{{{macros[k]}}}")
    body.append("")
    write("numbers.tex", "\n".join(body))

    print(f"\n{len(macros)} macros written")
    if bad:
        print(f"WARNING: {len(bad)} macro(s) have no value: {', '.join(bad)}")
    if MISSING:
        # A missing experiment is the loud failure this generation path exists
        # to produce; exiting 0 here would let it pass unnoticed through
        # run_all.py and CI.
        print("MISSING: " + "; ".join(MISSING))
        return 1
    if args.strict and bad:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

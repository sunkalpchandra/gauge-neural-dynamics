#!/usr/bin/env python
"""Check every directional claim the paper's grid-cell section makes.

``make_tables.py`` guarantees that no *number* in the paper is stale.  It cannot
guarantee that the *sentences around them* still hold: "the highest of the three
experiments", "PCA reaches GRE X against our Y", "the flow gauge does better" are
comparisons, and re-running an experiment can falsify one while every macro still
resolves.  Section 5.2 was in fact written against a one-seed pilot, and one of
its claims was reversed by the five-seed data.

Each assertion below is a prose claim restated as an inequality between numbers
the paper quotes.  A failure means the prose needs changing, not the number.

Comparisons between two methods are additionally checked **per seed and
jackknifed**: if dropping any single seed reverses the direction of the mean, the
claim is reported FRAG and treated as a failure.  A five-seed mean can be carried
entirely by one outlier -- a second claim in this section was, before this check
existed -- and a comparison that survives only with that seed in it is not
something to write a sentence around.

Exits non-zero if any claim fails or is fragile.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

R = ROOT / "results"


def load(exp):
    with open(R / exp / "results.json") as fh:
        return json.load(fh)


G2 = load("exp2_grid_cells")


def v(tab, method, key):
    try:
        return float(tab[method][key]["mean"])
    except (KeyError, TypeError):
        return float("nan")


def series(fam, method, key):
    """Per-seed values, ordered by seed."""
    rows = sorted([r for r in G2["rows"]
                   if r.get("context_set") == fam and r.get("method") == method
                   and "error" not in r],
                  key=lambda r: r["seed"])
    return np.array([r.get(key, np.nan) for r in rows], float)


g = G2["tables"]
main, rot, ctrl = g["translation"], g["translation+rotation"], g["all"]
hip = load("exp1_hippocampus")["table"]
mot = load("exp3_motor_cortex")["table"]

checks = []


def claim(text, ok, detail):
    checks.append(("OK  " if ok else "FAIL", text, detail))


def claim_paired(text, a, b, a_greater, detail="", mask=None):
    """``mean(a) > mean(b)`` (or ``<``), checked per seed and jackknifed.

    ``a`` and ``b`` are per-seed arrays over the same seeds.  ``mask`` restricts
    the comparison to a subset of seeds -- used where one side has a fit that did
    not converge, so that the comparison is not decided by a failed run in either
    direction.  Passing a mask is a claim about converged fits and the prose has
    to say so.
    """
    keep = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        keep &= mask
    a, b = a[keep], b[keep]
    if a.size == 0:
        checks.append(("FAIL", text, "no paired seeds"))
        return
    want = (lambda d: d > 0) if a_greater else (lambda d: d < 0)
    full = a.mean() - b.mean()
    wins = int(sum(want(x - y) for x, y in zip(a, b)))
    flips = [i for i in range(a.size)
             if not want(np.delete(a, i).mean() - np.delete(b, i).mean())]
    note = (f"{detail + '; ' if detail else ''}"
            f"mean difference {full:+.3f}, holds on {wins}/{a.size} seeds individually")
    if not want(full):
        checks.append(("FAIL", text, note))
    elif flips:
        checks.append(("FRAG", text,
                       f"{note} -- but dropping seed {flips[0]} alone reverses it "
                       f"({np.delete(a, flips[0]).mean():.3f} vs "
                       f"{np.delete(b, flips[0]).mean():.3f})"))
    else:
        checks.append(("OK  ", text, note))


# ---------------------------------------------------------------------------
# comparisons across experiments -- different simulations, so not seed-paired
# ---------------------------------------------------------------------------
claim("GCS for grid is the highest of the three experiments",
      v(main, "GND", "gcs") > v(hip, "GND", "gcs") and v(main, "GND", "gcs") > v(mot, "GND", "gcs"),
      f"grid {v(main,'GND','gcs'):.3f} vs hip {v(hip,'GND','gcs'):.3f} "
      f"vs motor {v(mot,'GND','gcs'):.3f}")

from gnd.geometry.metrics import realised_abelianness  # noqa: E402

with np.load(R / "exp1_hippocampus" / "artifacts.npz", allow_pickle=False) as z:
    hip_realised = float(realised_abelianness(z["GND::generators"], z["GND::theta"]))
claim("realised algebra very nearly commutes, and by less than the place-cell one",
      v(main, "GND", "realised_abelianness") < hip_realised,
      f"grid {v(main,'GND','realised_abelianness'):.3f} (5 seeds) vs hip {hip_realised:.3f} "
      "(1 seed, recomputed post hoc -- exp1 predates the metric)")

claim("basis commutator is large enough to 'read as a failed prediction'",
      v(main, "GND", "abelianness") > 0.2,
      f"basis abelianness {v(main,'GND','abelianness'):.3f}, "
      f"realised {v(main,'GND','realised_abelianness'):.3f}")

claim("Betti (1,2,1) recovered in most runs",
      v(main, "GND", "betti_correct") >= 0.5,
      f"betti_correct {v(main,'GND','betti_correct'):.2f}")

# ---------------------------------------------------------------------------
# GND against the baselines, on the abelian family
# ---------------------------------------------------------------------------
claim_paired("PCA beats GND on GRE (the stated concession)",
             series("translation", "PCA", "gre"), series("translation", "GND", "gre"),
             a_greater=False)

claim_paired("but GND beats PCA on GCS",
             series("translation", "GND", "gcs"), series("translation", "PCA", "gcs"),
             a_greater=True)

claim_paired("and GND finds the toroidal homology more often than PCA",
             series("translation", "GND", "betti_correct"),
             series("translation", "PCA", "betti_correct"), a_greater=True)

# ---------------------------------------------------------------------------
# the isometric restriction against the unrestricted gl variant
# ---------------------------------------------------------------------------
LOSS_TOL = 1.25          # must match scripts/make_tables.py


def converged(method, fam="translation"):
    loss = series(fam, method, "final_loss")
    return np.isfinite(loss) & (loss <= LOSS_TOL * np.median(loss))


both = converged("GND") & converged("GND-gl")

claim_paired("isometric gauge composes better than the unrestricted one",
             series("translation", "GND", "gcs"), series("translation", "GND-gl", "gcs"),
             a_greater=True)

for key, lbl in (("gre", "GRE"), ("transport_r2", "transport R^2")):
    a, b = series("translation", "GND", key), series("translation", "GND-gl", key)
    d = float(np.abs(a[both] - b[both]).max())
    claim(f"...and is indistinguishable from it on {lbl} where both converge",
          d < 0.01,
          f"largest per-seed |difference| over the {int(both.sum())} converged "
          f"seeds: {d:.4f}")

loss = series("translation", "GND", "final_loss")
failed = ~converged("GND")
claim("the seeds flagged by training loss are exactly the ones that fail",
      int(failed.sum()) == 1 and series("translation", "GND", "transport_r2")[failed][0] < 0,
      f"final training loss / median: {np.round(loss / np.median(loss), 2).tolist()}; "
      f"transport R^2 of the flagged seed(s): "
      f"{np.round(series('translation','GND','transport_r2')[failed], 3).tolist()}")

# ---------------------------------------------------------------------------
# the two supplementary families
# ---------------------------------------------------------------------------
claim_paired("the automorphism costs the linear gauge composition consistency",
             series("translation+rotation", "GND", "gcs"),
             series("translation", "GND", "gcs"), a_greater=False)

claim_paired("the realised commutator norm rises with the lattice automorphism",
             series("translation+rotation", "GND", "realised_abelianness"),
             series("translation", "GND", "realised_abelianness"), a_greater=True)

fg = series("translation+rotation", "GND-flow", "gre")
lg = series("translation+rotation", "GND", "gre")
n_better = int((fg < lg).sum())
claim("the flow gauge beats the linear one on some seeds and diverges on others",
      0 < n_better < fg.size,
      f"flow GRE per seed {np.round(fg, 3).tolist()} vs "
      f"linear {np.round(lg, 3).tolist()} -> flow better on {n_better} of {fg.size}")

claim("...and is worse on average, which is what the prose says",
      fg.mean() > lg.mean()
      and series("translation+rotation", "GND-flow", "transport_r2").mean()
      < series("translation+rotation", "GND", "transport_r2").mean(),
      f"mean GRE {fg.mean():.3f} vs {lg.mean():.3f} -- deliberately not jackknifed: "
      "the prose says this mean is carried by the diverging seeds")

# Compared on the seeds where the abelian fit converged: the abelian mean is
# dragged down by its one failed seed, which would otherwise flatter the control.
ctrl_mask = converged("GND", "translation") & converged("GND", "all")

claim_paired("the rescaling control fails: transport R^2 falls",
             series("all", "GND", "transport_r2"),
             series("translation", "GND", "transport_r2"), a_greater=False,
             mask=ctrl_mask, detail="converged seeds only")

claim_paired("the rescaling control's GRE is worse than the abelian family's",
             series("all", "GND", "gre"), series("translation", "GND", "gre"),
             a_greater=True, mask=ctrl_mask, detail="converged seeds only")

# ---------------------------------------------------------------------------
bad = 0
for status, text, detail in checks:
    print(f"[{status}] {text}\n         {detail}")
    bad += status != "OK  "
print(f"\n{len(checks) - bad}/{len(checks)} claims hold")
if bad:
    print("FRAG = the direction of the mean is reversed by dropping a single seed.")
raise SystemExit(1 if bad else 0)

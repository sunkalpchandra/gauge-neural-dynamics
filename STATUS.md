# Project status

Written for whoever picks this up next. It says what is finished, what is not,
what is known to be shaky, and what I would do first.

Last updated 2026-08-01, after the Experiment 2 sweep and a correctness pass over
the gauge field; see `git log` for the exact commit.

---

## What changed in this pass

**Experiment 2 had never actually been run.** `results/exp2_grid_cells/results.json`
held a one-seed `--quick` pilot marked incomplete, so fifteen macros the paper
cites were undefined and `artifacts.npz` was missing. The five-seed sweep over all
three context families has now been run and is in `results/`.

**The flow gauge's composition law was wrong.** Two errors that cancel each other
only on antisymmetric algebras:

- `FlowGaugeField._jacobian_action` computed the *vector*-Jacobian product
  `J^T w` where the Lie bracket needs `J w`. A single reverse-mode `grad` with
  `grad_outputs=w` gives the transpose; getting `J w` needs the double-backward
  now in place.
- `FlowGaugeField.compose` added `+½[a,b]`. The structure constants are those of
  the *vector-field* bracket, and for a linear field `[V_A,V_B] = V_{-[A,B]}`, so
  BCH in matrix coordinates requires `−½[a,b]`. The third-order term is
  unaffected: both of its brackets flip sign and the flips cancel.

On an sl(2) triple the composition error now falls as `θ²` — 2.6e-2, 6.3e-3,
1.6e-3, 3.9e-4 at θ-scale 0.2, 0.1, 0.05, 0.025 — and beats dropping the bracket
entirely by 8x to 60x. Before the fix it fell only as `θ` and was *worse* than
dropping the bracket. The previous suite could not see this: every algebra it
tested was antisymmetric or abelian, which is exactly the case where the two
errors cancel. `tests/test_gauge.py` now tests a non-antisymmetric one.

**Rotation frequencies were `1/dt` too large.** `fit_linear_dynamics` already
divides by `dt`, and `dynamics_spectrum` divided by it again — a factor of 100 at
the 10 ms motor step. Nothing the paper quotes changes (it quotes coefficients of
variation, which are ratios), but `rotation_frequency_hz_mean` in
`results/exp3_motor_cortex/results.json` is 100x too large until that experiment
is re-run.

**A `--quick` run could silently overwrite the results the paper is built from.**
That is how the Experiment 2 pilot got there: `run_all.py --quick` is the
documented quick start *and* a CI step, and it wrote to the same `results/`
directory with the same file names, with nothing downstream able to tell a pilot
from a finished sweep. Now `--quick` writes to `results/quick` and
`figures/quick`, and both `make_tables.py` and the figure loader refuse a pilot
outright.

---

## What is finished

| Experiment | Seeds | State | Headline |
|---|---|---|---|
| 1. Hippocampus | 5 | **final** | transport R² 0.698 vs 0.637 best baseline; GRE 0.108 |
| 2. Grid cells | 5, all three families | **final** | GCS 0.999; realised abelianness 0.137; Betti (1,2,1) in 4 of 5 runs |
| 3. Motor cortex | 5 | **final** | transport R² 0.876; every baseline negative |
| 4. Ablations | 3 | **final** | no-gauge costs −0.608; no-transport costs −0.813 |
| 5. Robustness | 3 | **final** | leads at every noise level, 0.802 → 0.327 |
| 6. Continuous context | 5 | **final** | held-out cues 0.688 ± 0.002 = trained cues |

Tests: 54, all passing, warning-free. The paper compiles with every macro defined
and no LaTeX error, but the body is **10 pages against a 9-page limit** — see
item 10 below.

Check the current state yourself rather than trusting that table:

```bash
.venv/bin/python scripts/check_macros.py        # names any missing result, by experiment
.venv/bin/python scripts/check_grid_claims.py   # re-checks section 5.2's comparisons
bash scripts/build_paper.sh                     # fails on missing inputs or >9 pages
```

`check_grid_claims.py` is new and exists because `make_tables.py` guarantees that
no *number* is stale but cannot guarantee the *sentences around them* still hold.
"The highest of the three experiments", "PCA reaches GRE X against our Y", "the
flow gauge does better" are comparisons; every macro can resolve while a claim is
false. It restates all seventeen of §5.2's comparisons as inequalities, and for
method-against-method comparisons it also **jackknifes**: if dropping any single
seed reverses the direction of the mean, the claim is reported FRAG and fails.

That check earned its place immediately. Three of the section's claims were
distorted by the one non-converged seed of item 4:

- "the flow gauge does better" on the rotation family — *reversed* by the
  five-seed data, and never supported by any run in this repository;
- "at some cost in chart quality" for the isometric restriction — the entire
  apparent cost was that one seed; on the four converged seeds the two variants
  agree to 0.0004 on GRE and 0.0023 on transport $R^2$, and the isometric gauge
  composes better. The restriction costs stability, not accuracy, and §5.2 now
  says that instead;
- "transport $R^2$ falls" for the rescaling control — true, but the mean-only
  version was fragile, because the abelian baseline it falls *from* was itself
  dragged down by the failed seed. Compared on converged seeds the control is
  lower on 4 of 4, by 0.364.

If you re-run an experiment, run this too, and extend it for the sections it does
not yet cover — §5.1 and §5.3 make comparisons of the same kind.

---

## Results that are stale because of the fix, and were not re-run

Re-running these would change numbers the paper quotes, and in one case bears on
a stated conclusion. That is an author's call, not a maintenance one, so the code
is fixed and the results are left alone and named here.

| Result | Why it is stale |
|---|---|
| `exp1_hippocampus`, `GND-flow` row | `algebra="gl"`, so nothing cancels: composed with the wrong bracket throughout training and evaluation |
| `exp4_ablations`, `gauge=flow` row | same |
| `exp3_motor_cortex`, `rotation_frequency_hz_mean` / `frequencies_hz` | 100x too large; the quoted CV and plane angle are unaffected |

Experiment 2's own flow rows were produced with the corrected code and are
current.

This bears on the negative result at `main.tex:668` — "the flow gauge ... recovers
the underlying geometry worse on every recovery measure" — and on the old item 5
of this file, which flagged that claim as possibly an optimisation failure. **It
was not an optimisation failure; the composition law was wrong.** To size the
effect without disturbing the stored results, all five seeds of Experiment 1's
flow row were recomputed in a scratch directory. Re-running seed 0 at the
pre-fix commit reproduced the stored row to within 0.003 on every metric, so this
is a like-for-like comparison:

| metric | GND (linear gauge) | GND-flow, stored | GND-flow, corrected |
|---|---|---|---|
| transport $R^2$ | $0.698 \pm 0.006$ | $0.731 \pm 0.013$ | $0.742 \pm 0.012$ |
| GRE | $0.108 \pm 0.010$ | $0.250 \pm 0.081$ | $0.186 \pm 0.058$ |
| GCS | $0.909 \pm 0.003$ | $0.852 \pm 0.006$ | $0.804 \pm 0.006$ |
| MPS | $0.918 \pm 0.017$ | $0.734 \pm 0.008$ | $0.741 \pm 0.009$ |

**The conclusion survives**: the corrected flow gauge still fits better
(transport $R^2$ higher on 5 of 5 seeds) and still recovers worse on GRE, GCS and
MPS. But the GRE gap narrows by about 45%, and paired within seed the corrected
flow gauge is now *better* than the linear one on GRE for 2 of 5 seeds, so the
"on every recovery measure" phrasing is stronger than the seed-level evidence
supports. Deciding whether to re-run Experiments 1 and 4 and restate that
paragraph is the first thing on the list below.

---

## Things I would not trust without checking

These are real weaknesses, not modesty.

1. **GRE is chart-dependent.** It fits an affine map from canonical latent to
   ground-truth coordinate on the reference context and freezes it. Where the
   ground-truth coordinate is not an affine function of the latent — the motor
   simulation especially — the chart extrapolates badly and the number is not
   meaningful. The paper says so and leads with transport R² there instead. A
   better recovery metric for that case is an open problem; the chart-free
   leave-one-context-out algebra recovery is a partial answer and needs more
   contexts than Experiment 1 has.

2. **The Betti gap threshold (1.5) is the one tuned constant.** At the sample
   sizes used, a disc gives spurious `H_1` gaps around 1.1 and a torus a true
   two-bar gap around 1.9, so 1.5 separates them. The separation for `H_1` on a
   torus is marginal for any gap-based estimator. `persistence_gap` is reported
   alongside so a marginal call is visible, but a shuffled-null significance test
   would be better.

3. **Abelianness has two versions and only one is meaningful.** `abelianness()`
   takes commutators of the learned *basis*, which is the wrong question: the
   basis spans K directions while the contexts may exercise a much smaller
   subgroup. Use `realised_abelianness()`, which measures the elements the
   contexts actually produce. Experiment 2 now reports it per seed (0.137 ± 0.027
   against a basis measure of 0.475). Experiment 1 predates the metric and is
   still recomputed post hoc from its artefacts in `make_tables.py`, as a
   single-seed diagnostic; `make_tables.realised_abelian` prefers the per-seed
   value and falls back to the artefact, so re-running Experiment 1 upgrades it
   automatically.

4. **One seed in five fails to converge on the grid translation family.** Seed 2
   of the isometric gauge lands in a bad optimum: transport $R^2$ −0.547 against
   about 0.945 for the other four, GRE 0.957 against about 0.436, Betti (1,1,0)
   against (1,2,1), and a final training loss 1.63x the median where every other
   seed of either variant is within 1.06x. That single seed is why Table 2
   reports transport $R^2$ 0.646 ± 0.298 for GND while the unrestricted `gl`
   variant, which converges on all five, reports 0.944 ± 0.000. Nothing has been
   excluded from the table. Where §5.2 compares converged fits it says so, and a
   fit is flagged as failed on the **training loss alone** — above 1.25x the
   median for that model, a threshold sitting in the empty gap between 1.06 and
   1.63 — never on the evaluation measures being compared. This matters because
   the contaminated mean silently distorted three of the section's comparisons:
   see below.

5. **`algebra_recovery_r2` is undefined for Experiment 1** — it needs more
   linearly-acting contexts than the five-context place-cell design provides, so
   it silently returns NaN — and on the grid translation family it is strongly
   negative for GND (−7.7) while PCA scores +0.58. That inversion is consistent
   with the section's own concession that for a single grid module PCA lands on
   nearly the right chart, but it is the one table cell where a baseline beats
   the method on a recovery measure by a wide margin, and it deserves a second
   look before the table is published. Related: panel (h) of Figure 4 is titled
   "algebra recovery $R^2$" and shows −28.08, which is *not* the table's
   `algebra_recovery_r2`. The figure regresses the two true phase shifts on all
   six raw coefficients (`fig4_grid_cells._loo_predict`), which with five
   contexts fits seven parameters to four leave-one-out training points and
   cannot be meaningful; the metric first reduces the coefficients to their
   leading principal directions (`metrics.algebra_recovery_r2`). Two different
   estimators under one label — the figure's should use the metric.

6. **These are simulations.** Nothing here is evidence about recorded data.

7. **GCS and holonomy both reward a gauge that barely transforms.** The
   composition residual is `O(θ²)` and its normaliser `O(θ)`, so both improve as
   the transformations shrink. This is not hypothetical: in
   `results/exp3_motor_cortex`, ManifoldAlign scores GCS 0.974 against GND's
   0.971 — the best of any method — with transformation magnitude 0.075 against
   GND's 0.741. It wins the headline metric by nearly not transforming. The paper
   does report GCS next to transformation magnitude, which is the right guard,
   and the GCS docstring that claimed otherwise has been corrected. **Holonomy has
   no such guard anywhere**, is normalised by `‖z_probe‖` rather than by
   transformation size, and is unbounded above: stored values run from 0.004
   (ManifoldAlign, by collapse) to 4.9e+99 (UMAP, where the order-2 BCH truncation
   is extrapolated far outside its radius of convergence). A mean over 40 loops of
   a quantity that reaches 1e99 is not a statistic. Holonomy is a stored
   diagnostic and is not quoted in the paper; do not start quoting it without
   renormalising by transformation size and bounding the composition.

8. **`realised_abelianness` returns 0.0 — its best value — for a collapsed
   gauge.** With fewer than two contexts having `‖A_c‖ > 1e-8` the commutator loop
   never executes and the function returns 0.0, i.e. "perfectly abelian".
   Verified on a genuinely non-abelian so(3) basis: an all-zero `theta`, a single
   non-zero context, and a set of `theta` spanning only one direction all return
   exactly 0.000, while a genuine two-dimensional family returns 0.695. This is
   the grid-cell section's headline claim, so read it together with the
   transformation magnitude and context-decoding accuracy, which do detect
   collapse (for the grid runs they do: magnitude 1.19, CIS 0.999). Fixing it
   means deciding what the metric should return when the realised group is too
   small to have a commutator — a design decision, not a bug fix, which is why it
   is documented rather than changed.

9. **Two stored diagnostics do not mean what their scale suggests.**
   `spectral_error` has no null-model normalisation although
   `induced_recovery_error` three lines below it does, so a model that learns
   `T_c = I` does not score 1 (it scores 0.17 against a true 10° rotation and
   0.77 against 45°, where `induced_recovery_error` correctly returns 1.000).
   `readout_r2_nonlinear` is an in-sample R²: the random-Fourier ridge is fit and
   scored on the same points, and `evaluate.py` never passes `fit_index`, so the
   non-linear chart behind `gre_nonlinear_chart` is fitted on the test split it is
   then applied within. It is saturated and non-discriminative (0.998 GND, 0.995
   PCA, 0.994 UMAP). Neither is quoted in the paper; neither should be.

10. **The paper body is over the page limit.** `build_paper.sh` reports 10 body
    pages against a limit of 9. This is not new and is not caused by the
    Experiment 2 numbers: reverting to the pre-existing prose and rebuilding gives
    10 pages as well. Trimming is an author's decision, so the build will keep
    exiting non-zero until it is made — or until `GND_PAGE_LIMIT` is set to what
    the venue actually allows.

---

## What I would do next, in order

1. **Decide what to do about the corrected flow gauge.** Either re-run
   Experiment 1 and Experiment 4 and restate the negative result at
   `main.tex:668` from the new numbers, or say in the paper that the flow
   comparison was computed with a composition law that has since been corrected.
   The five-seed evidence above is enough to decide which. Note that Experiment
   2's rotation family, which *was* run with the corrected code, shows the flow
   gauge is bimodal rather than uniformly worse — better than the linear gauge on
   2 of 5 seeds and divergent on the other 3 — and §5.2 now says so. That
   paragraph previously claimed the flow gauge "does better" on the rotation
   family, which no run in this repository ever supported; the five-seed data
   contradicts it on the mean, so the paragraph was rewritten from the numbers
   and `scripts/check_grid_claims.py` now guards it.

2. **Fix the page count**, or set `GND_PAGE_LIMIT` to the real venue limit.

3. **Public data.** The two predictions in the discussion are testable without
   fitting this model at all. The conjugacy prediction (Eq. 5: rotation
   frequencies shared across reach conditions, eigenplanes rotating) can be
   checked on any centre-out reaching dataset. The winding prediction needs a
   cyclic cue manipulation, which is rarer, but grid-cell data with environment
   rotations would do.

4. **A multi-chart gauge.** Proposition 1 says a single chart cannot cover a
   cyclic context variable, and the experiment confirms the failure appears at
   the branch point. The fix is a bundle atlas: two or more charts with learned
   transition functions on the overlap. That is the natural next version of the
   model and would make the gauge language do real work rather than being an
   analogy.

5. **Rate remapping.** Currently outside the framework by construction, since it
   is a change in `f` rather than in `T_c`. Modelling it would mean a gauge
   acting on observation space as well, which is a genuine extension rather than
   a tweak.

6. **Replace the paper style file.** `paper/neurips_workshop.sty` is my
   reimplementation of the NeurIPS layout, written because the official file for
   the target year was not available. Drop in the official `neurips_<year>.sty`
   and change one `\usepackage` line; nothing else should need to move.

---

## How the repository is wired

The thing worth knowing: **no number in the paper is typed by hand.**
`scripts/make_tables.py` reads `results/*/results.json` and writes both the tables
and `paper/generated/numbers.tex`, a file of LaTeX macros. `main.tex` cites those
macros. So:

- a missing experiment produces a named failure, not a stale number;
- partial checkpoints and `--quick` pilots are refused outright rather than
  half-used, and the tables an unusable experiment owns are *deleted* rather than
  left to be typeset under a caption describing a run that no longer exists;
- `make_tables.py` exits non-zero when an experiment is missing, `run_all.py`
  propagates the failure instead of always returning 0, and CI runs
  `check_macros.py`;
- `build_paper.sh` fails on missing inputs, undefined citations or references,
  bibtex errors, or a body over nine pages. (The citation and reference checks
  used to be dead code: their pattern required `undefined` immediately after the
  key, and pdflatex writes `Citation \`key' on page 1 undefined`.)

If you change an experiment, re-run `make_tables.py` and the numbers in the prose
update themselves. If you find yourself typing a number into `main.tex`, add a
macro instead.

The `.npz` artefacts under `results/` are tracked deliberately. They hold the
first-seed latents, generators and coefficients each figure needs, so
`python -m gnd.figures.make_all` works on a fresh clone without repeating the
sweep. They now carry a `provenance_seed` key; archives written before this pass
do not, and an archive without one cannot be checked against the `results.json`
beside it.

**On reproducing Experiment 2.** The sweep here was run as fifteen independent
`(context set, seed)` processes and merged, because `run_seed` re-seeds from the
seed before it builds anything and each job therefore computes exactly what the
sequential driver would. That was checked rather than assumed: a two-seed
sequential run and two one-seed runs agree bitwise on 803 of 856 numeric fields,
with every GND, GND-gl and baseline headline metric identical except
ManifoldAlign's transport $R^2$, which differs by 2.6e-3. The plain
`python -m gnd.experiments.exp2_grid_cells --seeds 0 1 2 3 4` gives the same
thing in one process, in about four hours.

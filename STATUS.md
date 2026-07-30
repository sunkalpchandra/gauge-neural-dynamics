# Project status

Written for whoever picks this up next. It says what is finished, what is not,
what is known to be shaky, and what I would do first.

Last updated after the five-seed sweep; see `git log` for the exact commit.

---

## What is finished

| Experiment | Seeds | State | Headline |
|---|---|---|---|
| 1. Hippocampus | 5 | **final** | transport R² 0.698 vs 0.637 best baseline; GRE 0.108 |
| 2. Grid cells | 5 (main family) | **main family final**, two supplementary families were still running at time of writing | GCS 0.999; Betti (1,2,1) in most runs |
| 3. Motor cortex | 5 | **final** | transport R² 0.876; every baseline negative |
| 4. Ablations | 3 | **final** | no-gauge costs −0.608; no-transport costs −0.813 |
| 5. Robustness | 3 | **final** | leads at every noise level, 0.802 → 0.327 |
| 6. Continuous context | 5 | **final** | held-out cues 0.688 ± 0.002 = trained cues |

Tests: 51, all passing, warning-free. Paper body is 9 pages and compiles.

Check the current state yourself rather than trusting this table:

```bash
.venv/bin/python scripts/check_macros.py     # names any missing result, by experiment
bash scripts/build_paper.sh                  # fails on missing inputs or >9 pages
```

---

## What is not finished

**Experiment 2's supplementary context families.** The `translation` family (the
one carrying the baseline comparison, Table 2 and Figure 4) is complete with five
seeds. The `translation+rotation` and `all` families feed
Table `tab:grid-families` and five in-text macros. If `check_macros.py` reports
`gridRotGre`, `gridCtrlGre` and friends as missing, that run did not finish;
re-run just that experiment:

```bash
.venv/bin/python -m gnd.experiments.exp2_grid_cells --seeds 0 1 2 3 4
```

It checkpoints after every seed, so it is safe to interrupt.

**`results/exp2_grid_cells/artifacts.npz`** may be absent. Figure 4 needs it. It
is written by the run above.

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
   contexts actually produce. The two disagree substantially (0.451 vs 0.128 on
   the grid data). The completed runs predate the second one, so it is recomputed
   post hoc from artefacts in `make_tables.py` and is a **single-seed**
   diagnostic. Re-running the experiments would put it in the tables properly.

4. **`algebra_recovery_r2` is undefined for Experiment 1.** It needs more
   linearly-acting contexts than the five-context place-cell design provides, so
   it silently returns NaN there. It works for Experiments 2 and 3.

5. **The flow gauge underperforms** the linear one on every recovery measure
   despite being a strict generalisation. I believe this is real — expressiveness
   costs identifiability — but it could also be an optimisation failure. The
   structure constants are cached and refreshed every 25 steps, and RK4 with six
   steps may be too coarse. Worth ruling out before leaning on the claim.

6. **These are simulations.** Nothing here is evidence about recorded data.

---

## What I would do next, in order

1. **Public data.** The two predictions in the discussion are testable without
   fitting this model at all. The conjugacy prediction (Eq. 5: rotation
   frequencies shared across reach conditions, eigenplanes rotating) can be
   checked on any centre-out reaching dataset. The winding prediction needs a
   cyclic cue manipulation, which is rarer, but grid-cell data with environment
   rotations would do.

2. **A multi-chart gauge.** Proposition 1 says a single chart cannot cover a
   cyclic context variable, and the experiment confirms the failure appears at
   the branch point. The fix is a bundle atlas: two or more charts with learned
   transition functions on the overlap. That is the natural next version of the
   model and would make the gauge language do real work rather than being an
   analogy.

3. **Rate remapping.** Currently outside the framework by construction, since it
   is a change in `f` rather than in `T_c`. Modelling it would mean a gauge
   acting on observation space as well, which is a genuine extension rather than
   a tweak.

4. **Replace the paper style file.** `paper/neurips_workshop.sty` is my
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
- partial checkpoints are refused outright rather than half-used;
- `build_paper.sh` fails on missing inputs, undefined citations, or a body over
  nine pages.

If you change an experiment, re-run `make_tables.py` and the numbers in the prose
update themselves. If you find yourself typing a number into `main.tex`, add a
macro instead.

The `.npz` artefacts under `results/` are tracked deliberately. They hold the
first-seed latents, generators and coefficients each figure needs, so
`python -m gnd.figures.make_all` works on a fresh clone without repeating the
sweep.

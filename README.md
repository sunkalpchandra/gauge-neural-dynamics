# Gauge Neural Dynamics (GND)

[![tests](https://github.com/sunkalpchandra/gauge-neural-dynamics/actions/workflows/tests.yml/badge.svg)](https://github.com/sunkalpchandra/gauge-neural-dynamics/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Learning context-dependent coordinate systems in biological neural representations.**

Neural population activity is usually summarised by fitting one low-dimensional
manifold to recordings pooled across conditions. But the same circuit re-expresses
what appears to be the same computation in different coordinates when the context
changes — place fields rotate with a cue card, grid phases shift coherently between
environments, motor cortical trajectories for different reach directions look like
rotated copies of one another.

This repository implements and tests a framework in which that is taken literally.
An invariant latent state `z` is observed through a **shared** tuning map `f` after
a **context-dependent** transformation,

```
x_c = f( T_c(z) ),     T_c = exp( Σ_k θ_k(c) G_k )
```

where the generators `G_k` span a *learned* finite-dimensional Lie algebra and
`θ` maps the observable context variable to coordinates in it. Identity and exact
invertibility hold by construction; **closure under composition does not, and is
measured** rather than assumed.

Because the encoder and decoder never see the context, every context effect must be
expressible as a latent transformation. That makes the framework falsifiable, and
two of the experimental conditions here are built so that no such transformation
exists — the model is expected to fail on them, and does.

**Picking this up?** Read [STATUS.md](STATUS.md) first: what is final, what is
not, which results I would not trust without re-checking, and what I would do
next.

---

## Quick start

```bash
git clone https://github.com/sunkalpchandra/gauge-neural-dynamics.git
cd gauge-neural-dynamics
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# check the whole pipeline end-to-end in a few minutes
.venv/bin/python scripts/run_all.py --quick
```

`--quick` writes to `results/quick/` and `figures/quick/`, never to the tracked
results and figures the paper is built from, so a smoke test cannot be mistaken
for -- or overwrite -- a real sweep.

Full reproduction (five seeds; a few hours on eight CPU cores):

```bash
.venv/bin/python scripts/run_all.py
```

That runs every experiment, regenerates every figure, regenerates every table and
in-text number, and builds the PDF. The stages can also be run separately with
`--stage experiments|figures|tables|paper`.

Everything runs on **CPU**. The models are small, and the gauge field relies on
`torch.matrix_exp` and `torch.linalg.lstsq`, neither of which is implemented for
Apple's MPS backend.

---

## Repository layout

```
gnd/
  models/
    gauge_field.py      Lie-algebra gauge: generators, BCH composition, closure defect,
                        matrix-exponential and Neural-ODE (flow) variants
    encoder.py          shared, context-blind encoders (MLP and variational)
    decoder.py          shared, context-blind decoder
    neural_dynamics.py  latent vector field, push-forward, conjugacy diagnostics
    gnd.py              the full model and its objective
    train.py            training loop
    baselines.py        PCA, UMAP, AE, VAE, CCA, Procrustes, manifold alignment
  simulations/
    hippocampus.py      place cells; rotations, affine distortion, non-affine morph,
                        and a continuous cue-rotation family
    grid_cells.py       one grid module; toroidal latent, translation / lattice-rotation /
                        rescaling context families
    motor_cortex.py     rate RNN trained by BPTT on centre-out reaching
  geometry/
    metrics.py          GCS, CIS, GRE, MPS + holonomy, algebra recovery, diagnostics
    manifold.py         alignment, gauge fixing, neighbourhood preservation
    topology.py         persistent homology, Betti estimation, differentiable H0 loss
  experiments/          exp1..exp6 drivers + the shared evaluation harness
  figures/              figure scripts and the shared style system
scripts/
  run_all.py            one entry point for everything
  make_tables.py        generates every table and every in-text number from results
  build_paper.sh        compiles the paper and fails on placeholders or LaTeX errors
paper/                  main.tex, appendix.tex, references.bib, generated/
results/                results.json + artefacts per experiment (created by the runs)
figures/                vector PDFs (created by the figure scripts)
tests/                  correctness tests for the mathematical machinery
```

---

## Experiments

| | What it asks | Key comparison |
|---|---|---|
| **1. Hippocampus** | Can a learned gauge recover cue-driven remapping? | 7 baselines, 5 contexts incl. a non-affine morph |
| **2. Grid cells** | Does the learned algebra reproduce the group acting on the torus? | abelian `T²` vs non-abelian `T²⋊Z₆` vs a rescaling control |
| **3. Motor cortex** | Are reach conditions the same computation in a rotated frame? | 16 conditions; tests the conjugacy prediction |
| **4. Ablations** | Which ingredients are load-bearing? | 19 configurations + latent-dimension sweep |
| **5. Robustness** | How does it degrade? | noise, Poisson counts, population size, sample size |
| **6. Continuous context** | Does the model learn a *field* over context space? | held-out cue angles; the winding obstruction |

Run one at a time, e.g.:

```bash
.venv/bin/python -m gnd.experiments.exp1_hippocampus --seeds 0 1 2 3 4
.venv/bin/python -m gnd.experiments.exp2_grid_cells --context-sets translation
.venv/bin/python -m gnd.experiments.exp4_ablations --only "A1: no gauge" "A3: no topology"
.venv/bin/python -m gnd.experiments.exp5_robustness --sweeps noise
```

Every driver accepts `--quick` for a fast smoke run and `--help` for its options.

---

## Metrics

Four metrics are introduced, all defined so that they apply **unchanged** to
methods that have no explicit transformation model. For such a baseline an affine
map is fitted per context on the training split, its matrix logarithm taken, and a
`K`-dimensional algebra extracted by PCA; every metric then runs through the same
code path used for GND. All values are reported on held-out samples.

- **Gauge Consistency Score** — how nearly the transformations compose, using the
  truncated BCH series with least-squares structure constants. Always reported next
  to a **transformation magnitude**, because a family collapsed onto the identity
  composes perfectly while explaining nothing.
- **Context Invariance Score** — intraclass-correlation form; cannot be gamed by
  collapse, since a constant latent scores 0. Reported with context-decoding
  accuracy.
- **Geometric Recovery Error** — normalised so that `1.0` is the score of "no
  transformation at all". Uses a ridge-regularised chart fitted on the reference
  context only. A chart-free variant is also reported for cases where the
  ground-truth coordinate is not affine in the latent.
- **Manifold Preservation Score** — bottleneck distance between persistence
  diagrams, with Betti numbers estimated from the largest gap in the barcode.

---

## Reproducibility

- Every RNG (Python, NumPy, PyTorch) is seeded per run; deterministic kernels are
  requested. Results are reported as mean ± s.e.m. over five seeds.
- **No number in the paper is typed by hand.** `scripts/make_tables.py` reads
  `results/*/results.json` and writes both the tables and a `numbers.tex` of LaTeX
  macros; `main.tex` contains no literal results. A missing experiment therefore
  produces a loud failure at build time rather than a stale number in the PDF:
  `make_tables.py` exits non-zero and *deletes* the tables that experiment owns,
  rather than leaving the previous run's numbers to be typeset under a caption
  describing a run that no longer exists.
- A partial checkpoint or a `--quick` pilot is refused by both the table
  generator and the figure scripts, so a smoke run cannot be mistaken for a
  finished sweep.
- `scripts/build_paper.sh` exits non-zero on any LaTeX error, undefined citation or
  reference, bibtex failure, remaining placeholder, or a body over the page limit.

---

## Building the paper

```bash
.venv/bin/python scripts/make_tables.py     # tables + in-text numbers from results
.venv/bin/python -m gnd.figures.make_all    # all figures to figures/*.pdf
bash scripts/build_paper.sh                 # -> paper/main.pdf
```

You need a LaTeX installation. TinyTeX works on macOS and Linux and needs no
`sudo`; its installer symlinks the binaries into `~/.local/bin`, which
`build_paper.sh` finds on its own:

```bash
curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh
tlmgr install natbib geometry xcolor hyperref booktabs amsfonts caption \
              microtype times psnfss units rsfs multirow placeins
```

`paper/neurips_workshop.sty` is an independent reimplementation of the published
NeurIPS layout, used because the official style file for the target year is not
bundled here. It reproduces the same page geometry, fonts and heading style, so
page counts match. **To use the official file**, drop `neurips_<year>.sty` into
`paper/` and change the `\usepackage` line in `main.tex`; nothing else needs to
change.

---

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

The tests check the parts where a silent bug would invalidate a scientific claim:
that the gauge is exactly invertible and exactly the identity at the reference
context; that the flow gauge reduces to the linear gauge when its non-linearity is
switched off; that structure constants and the closure defect are correct on
algebras with known answers (`so(3)`, and a commuting pair); that BCH composition
is exact for a genuinely abelian family and second-order accurate on a
*non-antisymmetric* one, which is the only configuration in which a sign error in
the bracket is visible; that the fitted rotation frequency of a planted
oscillation comes back in hertz; that persistent homology returns the right Betti
numbers for a circle, a torus and a disc; and that each metric responds in the
right direction to a controlled perturbation.

---

## Scope and limitations

A fuller and blunter account is in [STATUS.md](STATUS.md).

This work is on **simulations**, built from published phenomenology and, for the
motor case, from a network trained on a task rather than hand-designed. No claim
here is a claim about recorded data.

Three limitations are intrinsic rather than incidental:

- **Rate remapping** — context-dependent firing-rate changes with fields held fixed
  — is a change in `f`, not in `T_c`, and is outside the model by construction. The
  same applies to global remapping modelled as a permutation of cell identity, which
  is a gauge transformation in observation space rather than latent space.
- **Grid rescaling** by a non-integer gain is well defined on the universal cover
  but is not a map on the torus at all, so no latent gauge transformation of the
  population manifold can express it. It is included as a negative control.
- **Chart dependence** of the recovery error is a genuine methodological weakness
  where the ground-truth coordinate is not an affine function of the latent, as in
  the motor simulation. It is reported as such rather than quoted as if meaningful.

---

## Citation

```bibtex
@inproceedings{gnd2026,
  title     = {Gauge Neural Dynamics: Learning Context-Dependent Coordinate Systems
               in Biological Neural Representations},
  author    = {Anonymous},
  booktitle = {NeurIPS Workshop on Symmetry and Geometry in Neural Representations},
  year      = {2026},
}
```

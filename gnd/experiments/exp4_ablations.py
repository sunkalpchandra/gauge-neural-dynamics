"""Experiment 4: ablations.

Each ablation removes or changes exactly one ingredient of the full model and is
re-fitted from scratch on the place-cell simulation of Experiment 1.  The five
ablations required by the study design are

A1  no gauge transformation at all (``T_c = id``) -- this *is* the "no-context
    latent model" baseline, and isolates the contribution of the gauge;
A2  no group-consistency constraint (``w_group = w_closure = 0``);
A3  no topology regularisation (``w_topology = 0``);
A4  latent dimension swept over ``{2, 3, 4, 6, 8, 12, 20}``;
A5  noise robustness -- run separately in :mod:`gnd.experiments.exp5_robustness`,
    together with the population- and sample-size scaling.

We additionally ablate the pieces whose necessity is a genuine open question:
the choice of ambient algebra, the BCH truncation order, identity anchoring
(gauge fixing), the transport and invariance terms, the number of generators,
and the paired-versus-correspondence-free alignment mode.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..models.gnd import GNDConfig
from ..simulations.hippocampus import simulate_place_cells
from ..utils.common import RESULTS_DIR, aggregate_seeds, save_json, set_seed
from .common import HEADLINE_KEYS, aggregate_table, checkpoint, run_gnd
from .exp1_hippocampus import build_ground_truth


def ablation_grid(base: GNDConfig) -> dict[str, dict]:
    """Name -> config overrides.  ``full`` is the unmodified model."""
    grid: dict[str, dict] = {
        "full": {},
        # A1: remove the gauge transformation (= no-context latent model)
        "A1: no gauge": {"gauge": "none", "w_group": 0.0, "w_closure": 0.0},
        # A2: remove the group-consistency constraints
        "A2: no group loss": {"w_group": 0.0, "w_closure": 0.0},
        "A2a: no closure only": {"w_closure": 0.0},
        # A3: remove topology preservation
        "A3: no topology": {"w_topology": 0.0},
        # further architectural questions
        "algebra=so": {"algebra": "so"},
        "algebra=sl": {"algebra": "sl"},
        "algebra=se": {"algebra": "se"},
        "gauge=flow": {"gauge": "flow"},
        "BCH order 1": {"bch_order": 1},
        "BCH order 3": {"bch_order": 3},
        "no anchoring": {"anchor": False},
        "no transport loss": {"w_transport": 0.0},
        "no invariance loss": {"w_invariance": 0.0},
        "unpaired (MMD)": {"alignment_mode": "mmd"},
        "encoder=VAE": {"encoder": "vae", "w_kl": 1e-3},
        "K=2 generators": {"n_generators": 2},
        "K=4 generators": {"n_generators": 4},
        "K=12 generators": {"n_generators": 12},
    }
    for d in (2, 3, 4, 6, 8, 12, 20):
        grid[f"A4: latent dim {d}"] = {"n_latent": d}
    return grid


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-cells", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--n-latent", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(RESULTS_DIR / "exp4_ablations"))
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these ablation names")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.seeds, args.n_samples, args.epochs, args.n_cells = [0], 800, 40, 50

    base = GNDConfig(
        n_latent=args.n_latent, hidden=192, depth=3, n_generators=6, algebra="gl",
        epochs=args.epochs, batch_size=256, lr=2e-3,
        w_group=0.2, w_closure=0.05, w_topology=0.5,
    )
    grid = ablation_grid(base)
    if args.only:
        grid = {k: v for k, v in grid.items() if k in set(args.only)}
    if args.quick:
        grid = {k: grid[k] for k in list(grid)[:5]}

    rows = []
    for seed in args.seeds:
        set_seed(seed)
        ds = simulate_place_cells(
            n_cells=args.n_cells, n_samples=args.n_samples, noise=args.noise,
            morph_context=True, seed=seed,
        ).standardise()
        train, test = ds.split(0.8, seed=seed)
        gt = build_ground_truth(test)
        for name, ov in grid.items():
            cfg = replace(base, seed=seed, **ov)
            res, _ = run_gnd(
                train, test, gt, cfg, name, args.device, False,
                topology_points=min(400, test.n_samples),
            )
            res.update({"seed": seed, "ablation": name, **{f"cfg_{k}": v for k, v in ov.items()}})
            rows.append(res)
            checkpoint(Path(args.out), {"rows": rows, "args": vars(args), "complete": False})
            print(
                f"  [seed {seed}] {name:<22} GCS={res['gcs']:.3f} CIS={res['cis']:.3f} "
                f"GRE={res['gre']:.3f} MPS={res['mps']:.3f} "
                f"transportR2={res['transport_r2']:.3f} ({res['fit_seconds']:.0f}s)"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = {}
    for name in grid:
        sub = [r for r in rows if r.get("ablation") == name]
        if sub:
            agg = aggregate_seeds(sub)
            table[name] = {k: agg[k] for k in HEADLINE_KEYS if k in agg}
            table[name]["_n_seeds"] = len(sub)
    save_json({"rows": rows, "table": table, "args": vars(args)}, out / "results.json")
    print(f"\nwrote {out/'results.json'}")
    return {"rows": rows, "table": table}


if __name__ == "__main__":
    main()

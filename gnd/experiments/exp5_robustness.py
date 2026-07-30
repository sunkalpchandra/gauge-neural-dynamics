"""Experiment 5: robustness and scaling.

Three sweeps on the place-cell simulation, each with the full evaluation suite:

``noise``    observation-noise level from 0 to 0.8 of the activity s.d., plus a
             Poisson-count condition, comparing GND with the three strongest
             baselines.  This is ablation A5 of the study design.
``neurons``  population size from 25 to 400 cells -- the regime that matters for
             applying the method to real recordings.
``samples``  number of behavioural samples from 500 to 4000.

The point of the sweeps is not only the mean level but the *shape*: a method
that relies on a genuine group structure should degrade gracefully, whereas
post-hoc alignment should collapse once the per-context estimates become noisy.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..models.gnd import GNDConfig
from ..simulations.hippocampus import simulate_place_cells
from ..utils.common import RESULTS_DIR, aggregate_seeds, save_json, set_seed
from .common import HEADLINE_KEYS, run_baseline, run_gnd
from .exp1_hippocampus import build_ground_truth

SWEEP_BASELINES = ("pca", "autoencoder", "procrustes")


def _run_point(cfg, train, test, gt, device, baselines, seed, ae_epochs) -> list[dict]:
    rows = []
    res, _ = run_gnd(train, test, gt, cfg, "GND", device, False, topology_points=min(400, test.n_samples))
    rows.append(res)
    for kind in baselines:
        try:
            r, _ = run_baseline(
                kind, train, test, gt, cfg.n_latent, seed, device,
                topology_points=min(400, test.n_samples),
                n_generators=cfg.n_generators, ae_epochs=ae_epochs,
            )
            rows.append(r)
        except Exception as exc:
            rows.append({"method": kind, "error": str(exc)})
    return rows


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--n-latent", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(RESULTS_DIR / "exp5_robustness"))
    ap.add_argument("--sweeps", nargs="*", default=["noise", "neurons", "samples"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.seeds, args.epochs = [0], 40

    base = GNDConfig(
        n_latent=args.n_latent, hidden=192, depth=3, n_generators=6, algebra="gl",
        epochs=args.epochs, batch_size=256, lr=2e-3,
        w_group=0.2, w_closure=0.05, w_topology=0.5,
    )

    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8] if not args.quick else [0.1, 0.4]
    neuron_counts = [25, 50, 100, 200, 400] if not args.quick else [50, 100]
    sample_counts = [500, 1000, 2000, 4000] if not args.quick else [600, 1200]

    rows: list[dict] = []

    def sweep(tag: str, points: list, make, baselines):
        for val in points:
            for seed in args.seeds:
                set_seed(seed)
                ds = make(val, seed).standardise()
                train, test = ds.split(0.8, seed=seed)
                gt = build_ground_truth(test)
                cfg = replace(base, seed=seed)
                for r in _run_point(cfg, train, test, gt, args.device, baselines, seed,
                                    ae_epochs=max(args.epochs, 200)):
                    r.update({"sweep": tag, "value": val, "seed": seed})
                    rows.append(r)
                got = [r for r in rows if r.get("sweep") == tag and r.get("value") == val and r.get("seed") == seed]
                g = next((r for r in got if r.get("method") == "GND"), {})
                print(
                    f"  [{tag}={val} seed {seed}] GND GCS={g.get('gcs', float('nan')):.3f} "
                    f"CIS={g.get('cis', float('nan')):.3f} GRE={g.get('gre', float('nan')):.3f} "
                    f"transportR2={g.get('transport_r2', float('nan')):.3f}"
                )

    if "noise" in args.sweeps:
        print("=== noise sweep ===")
        sweep("noise", noise_levels,
              lambda v, s: simulate_place_cells(n_cells=100, n_samples=2000, noise=v, seed=s),
              SWEEP_BASELINES)
        print("=== poisson noise ===")
        for seed in args.seeds:
            set_seed(seed)
            ds = simulate_place_cells(
                n_cells=100, n_samples=2000, noise=0.25, noise_kind="poisson", seed=seed
            ).standardise()
            train, test = ds.split(0.8, seed=seed)
            gt = build_ground_truth(test)
            for r in _run_point(replace(base, seed=seed), train, test, gt, args.device,
                                SWEEP_BASELINES, seed, max(args.epochs, 200)):
                r.update({"sweep": "poisson", "value": 0.25, "seed": seed})
                rows.append(r)

    if "neurons" in args.sweeps:
        print("=== population-size sweep ===")
        sweep("neurons", neuron_counts,
              lambda v, s: simulate_place_cells(n_cells=v, n_samples=2000, noise=0.15, seed=s),
              ("pca",))

    if "samples" in args.sweeps:
        print("=== sample-size sweep ===")
        sweep("samples", sample_counts,
              lambda v, s: simulate_place_cells(n_cells=100, n_samples=v, noise=0.15, seed=s),
              ("pca",))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table: dict = {}
    for tag in sorted({r["sweep"] for r in rows}):
        table[tag] = {}
        for val in sorted({r["value"] for r in rows if r["sweep"] == tag}):
            table[tag][str(val)] = {}
            for method in sorted({r.get("method") for r in rows
                                  if r["sweep"] == tag and r["value"] == val and "method" in r}):
                sub = [r for r in rows if r["sweep"] == tag and r["value"] == val
                       and r.get("method") == method and "error" not in r]
                if not sub:
                    continue
                agg = aggregate_seeds(sub)
                table[tag][str(val)][method] = {k: agg[k] for k in HEADLINE_KEYS if k in agg}
    save_json({"rows": rows, "table": table, "args": vars(args)}, out / "results.json")
    print(f"\nwrote {out/'results.json'}")
    return {"rows": rows, "table": table}


if __name__ == "__main__":
    main()

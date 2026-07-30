"""Experiment 1: hippocampal place-cell remapping.

Question.  When the same population of place cells is recorded in several
environments whose relationship is a coordinate transformation of space, can a
learned gauge transformation recover that transformation, and does it do so
better than the manifold methods currently used to compare neural populations
across conditions?

Protocol.  Five contexts -- familiar, two rotations, an affine stretch-and-
shear, and a deliberately non-affine radial morph -- share one bank of tuning
curves.  GND and seven baselines are fitted on 80% of the foraging trajectory
and evaluated on the remaining 20%.  Two GND variants are also fitted: a flow
(diffeomorphism) gauge, which should be the only model able to handle the morph
context, and a correspondence-free variant that replaces the paired invariance
term by distribution matching.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..models.gnd import GNDConfig
from ..simulations.base import ContextualDataset
from ..simulations.hippocampus import simulate_place_cells
from ..utils.common import RESULTS_DIR, save_json
from .common import DEFAULT_BASELINES, HEADLINE_KEYS, aggregate_table, run_seed
from .evaluate import GroundTruth


def build_ground_truth(test: ContextualDataset) -> GroundTruth:
    """Ground truth for place cells: allocentric position in the plane.

    The true context action is affine on position, so both the data-space
    recovery error and the conjugation-invariant spectral error are available
    (the latter only for the purely linear contexts).
    """
    u = test.latent
    per_ctx = np.stack([s.apply(u) for s in test.contexts])
    mats = {
        c: s.matrix
        for c, s in enumerate(test.contexts)
        if s.is_affine and s.matrix is not None and np.allclose(s.offset, 0.0)
    }
    return GroundTruth(
        coords=u,
        coords_per_context=per_ctx,
        matrices=mats,
        embedding=u,                 # the arena is a disc: contractible, beta = (1, 0)
        expected_betti=[1, 0],
        maxdim=1,
    )


def default_config(n_latent: int = 6) -> GNDConfig:
    return GNDConfig(
        n_latent=n_latent,
        hidden=192,
        depth=3,
        n_generators=6,
        algebra="gl",
        epochs=300,
        batch_size=256,
        lr=2e-3,
        w_recon=1.0,
        w_transport=1.0,
        w_invariance=1.0,
        w_group=0.2,
        w_closure=0.05,
        w_topology=0.5,
    )


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-cells", type=int, default=120)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--n-latent", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(RESULTS_DIR / "exp1_hippocampus"))
    ap.add_argument("--baselines", nargs="*", default=list(DEFAULT_BASELINES))
    ap.add_argument("--quick", action="store_true", help="small run for smoke testing")
    args = ap.parse_args(argv)

    if args.quick:
        args.seeds, args.n_samples, args.epochs, args.n_cells = [0], 900, 60, 60

    cfg = default_config(args.n_latent)
    cfg.epochs = args.epochs

    def build(seed: int):
        return simulate_place_cells(
            n_cells=args.n_cells, n_samples=args.n_samples, noise=args.noise,
            morph_context=True, seed=seed,
        )

    variants = {
        "GND-flow": {"gauge": "flow", "flow_nonlinearity": 0.5},
        "GND-unpaired": {"alignment_mode": "mmd"},
    }

    rows, first_art = [], None
    for seed in args.seeds:
        print(f"=== exp1 seed {seed} ===")
        r, art = run_seed(
            build, build_ground_truth, cfg, seed, variants=variants,
            baselines=tuple(args.baselines), device=args.device,
            topology_points=min(500, args.n_samples // 5),
            verbose=(seed == args.seeds[0]),
            ae_epochs=min(args.epochs, 200),
        )
        rows += r
        if first_art is None:
            first_art = art
        for row in r:
            if "error" in row:
                continue
            print(
                f"  {row['method']:<14} GCS={row.get('gcs', float('nan')):.3f} "
                f"CIS={row.get('cis', float('nan')):.3f} GRE={row.get('gre', float('nan')):.3f} "
                f"MPS={row.get('mps', float('nan')):.3f} "
                f"transportR2={row.get('transport_r2', float('nan')):.3f}"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = aggregate_table(rows, HEADLINE_KEYS)
    save_json({"rows": rows, "table": table, "args": vars(args)}, out / "results.json")
    _save_artifacts(out, first_art, rows)
    print(f"\nwrote {out/'results.json'}")
    return {"rows": rows, "table": table}


def _save_artifacts(out: Path, art: dict, rows: list[dict]) -> None:
    """Persist what the figure scripts need (first seed only)."""
    ds, test = art["dataset"], art["test"]
    payload = {
        "latent_test": test.latent,
        "context_names": np.array([s.name for s in test.contexts]),
        "centres": ds.meta["centres"],
        "sigma": ds.meta["sigma"],
        "radius": ds.meta["radius"],
    }
    for method in ("GND", "GND-flow", "GND-unpaired", "PCA", "UMAP", "Autoencoder", "VAE", "CCA",
                   "Procrustes", "ManifoldAlign"):
        if method in art:
            for key in ("w_test", "z_test", "theta", "generators", "matrices"):
                v = art[method].get(key)
                if v is not None:
                    payload[f"{method}::{key}"] = np.asarray(v)
    np.savez_compressed(out / "artifacts.npz", **payload)
    # raw activity for the rate-map figure, stored separately (it is large)
    np.savez_compressed(
        out / "dataset.npz",
        activity=ds.activity.astype(np.float32),
        latent=ds.latent,
        radius=ds.meta["radius"],
        centres=ds.meta["centres"],
        context_names=np.array([s.name for s in ds.contexts]),
    )


if __name__ == "__main__":
    main()

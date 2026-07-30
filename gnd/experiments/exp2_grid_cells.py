"""Experiment 2: grid-cell geometry and the group structure of remapping.

Question.  A single grid module's population manifold is a torus.  Do the
learned gauge transformations reproduce the *group* that acts on that torus --
the abelian translation group ``T^2``, and the non-abelian extension
``T^2 x| Z_6`` obtained by adding the 60-degree lattice rotation -- and does the
learned latent still have the homology of a torus?

Three context sets are run.

``translation``            pure phase shifts.  Prediction: the learned algebra
                           is *abelian*, i.e. the commutators of its generators
                           vanish, and composition is exact at BCH order 1.
``translation+rotation``   adds the 60-degree lattice automorphism, an integer
                           matrix and hence a genuine torus map.  The group is
                           non-abelian, so a non-zero commutator is now the
                           correct answer.
``all``                    adds a non-integer grid rescaling.  This is a
                           *negative control*: rescaling is well defined on the
                           universal cover but not on the torus, so the theory
                           predicts that no latent gauge transformation of the
                           population manifold can express it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..geometry.metrics import abelianness, lie_closure_defect
from ..models.gnd import GNDConfig
from ..simulations.base import ContextualDataset
from ..simulations.grid_cells import block_rotation, simulate_grid_cells, torus_embedding
from ..utils.common import RESULTS_DIR, save_json
from .common import DEFAULT_BASELINES, HEADLINE_KEYS, aggregate_table, run_seed
from .evaluate import GroundTruth


def build_ground_truth(test: ContextualDataset) -> GroundTruth:
    r"""Ground truth for grid cells: the ``R^4`` torus embedding of the phase.

    The phase pair is embedded as
    ``(cos t1, sin t1, cos t2, sin t2)/sqrt(2)``, under which a phase
    translation acts as the block rotation ``R(d1) (+) R(d2)`` -- an element of
    the maximal torus of ``SO(4)``.  That is the concrete prediction the
    spectral recovery error tests.

    The context action is applied to the *wrapped* phase.  For translations and
    for the integer lattice rotation this is exactly equivalent to acting on the
    universal cover; for the non-integer rescaling it is not, and no
    well-defined torus action exists at all.  The resulting large recovery error
    for that context is the intended outcome of the negative control, not an
    artefact.
    """
    th = test.latent
    coords = torus_embedding(th)
    per_ctx = np.stack([torus_embedding(_wrap(s.apply(th))) for s in test.contexts])
    mats = {}
    for c, s in enumerate(test.contexts):
        gp = s.group_params
        if gp.get("rot60", 0.0) == 0.0 and gp.get("scale", 1.0) == 1.0:
            mats[c] = block_rotation(np.array([gp["delta1"], gp["delta2"]]))
    return GroundTruth(
        coords=coords,
        coords_per_context=per_ctx,
        matrices=mats,
        embedding=coords,
        expected_betti=[1, 2, 1],
        maxdim=2,
    )


def _wrap(x: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(x) + np.pi, 2 * np.pi) - np.pi


def default_config(n_latent: int = 6) -> GNDConfig:
    """Configuration for the toroidal case.

    The gauge group is restricted to the *isometries* of the latent
    (``algebra="so"``, so that ``exp(A)`` is orthogonal).  This is not a
    convenience: a torus is not preserved by a general linear map, so on a
    periodic manifold the isometry group is the only linear group that can act
    on it at all.  Section 5 reports the unrestricted ``gl`` variant as an
    ablation, and it is markedly worse -- as the geometry predicts.
    """
    return GNDConfig(
        n_latent=n_latent,
        hidden=192,
        depth=3,
        n_generators=6,
        algebra="so",
        epochs=400,
        batch_size=256,
        lr=2e-3,
        w_group=0.2,
        w_closure=0.05,
        w_topology=0.5,
    )


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-cells", type=int, default=100)
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--n-latent", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(RESULTS_DIR / "exp2_grid_cells"))
    ap.add_argument("--context-sets", nargs="*", default=["translation", "translation+rotation", "all"])
    ap.add_argument("--main-context-set", default="translation")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.seeds, args.n_samples, args.epochs, args.n_cells = [0], 1200, 80, 60

    cfg = default_config(args.n_latent)
    cfg.epochs = args.epochs

    # The abelian translation case is the theoretically exact one and carries
    # the baseline comparison.  Adding the lattice automorphism makes the group
    # non-abelian and, on the torus embedding, non-linear -- so the flow gauge
    # is run alongside the linear one there.  The rescaling set is the negative
    # control and needs no baselines.
    variants_by_set = {
        "translation": {"GND-gl": {"algebra": "gl"}},
        "translation+rotation": {"GND-flow": {"gauge": "flow"}},
        "all": {"GND-flow": {"gauge": "flow"}},
    }

    all_rows, tables, first_art = [], {}, {}
    for ctx_kind in args.context_sets:
        rows = []
        for seed in args.seeds:
            print(f"=== exp2 [{ctx_kind}] seed {seed} ===")

            def build(s, kind=ctx_kind):
                return simulate_grid_cells(
                    n_cells=args.n_cells, n_samples=args.n_samples, noise=args.noise,
                    context_kind=kind, seed=s,
                )

            baselines = tuple(DEFAULT_BASELINES) if ctx_kind == args.main_context_set else ()
            r, art = run_seed(
                build, build_ground_truth, cfg, seed,
                variants=variants_by_set.get(ctx_kind),
                baselines=baselines, device=args.device,
                topology_points=min(400, args.n_samples // 6),
                verbose=(seed == args.seeds[0] and ctx_kind == args.main_context_set),
                ae_epochs=min(args.epochs, 200),
            )
            for row in r:
                row["context_set"] = ctx_kind
                if "error" in row:
                    continue
                print(
                    f"  {row['method']:<14} GCS={row.get('gcs', float('nan')):.3f} "
                    f"CIS={row.get('cis', float('nan')):.3f} GRE={row.get('gre', float('nan')):.3f} "
                    f"abelian={row.get('abelianness', float('nan')):.3f} "
                    f"betti={row.get('betti_learned_str')}"
                )
            rows += r
            if ctx_kind == args.main_context_set and not first_art:
                first_art = art
        all_rows += rows
        tables[ctx_kind] = aggregate_table(rows, HEADLINE_KEYS)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json({"rows": all_rows, "tables": tables, "args": vars(args)}, out / "results.json")
    if first_art:
        _save_artifacts(out, first_art)
    print(f"\nwrote {out/'results.json'}")
    return {"rows": all_rows, "tables": tables}


def _save_artifacts(out: Path, art: dict) -> None:
    ds, test = art["dataset"], art["test"]
    payload = {
        "phase_test": test.latent,
        "torus_test": torus_embedding(test.latent),
        "context_names": np.array([s.name for s in test.contexts]),
        "position": ds.meta["position"],
        "activity_full": ds.activity[:, :, :12].astype(np.float32),
        "phase_full": ds.latent,
        "K": ds.meta["K"],
    }
    for method in ("GND", "PCA", "UMAP", "Autoencoder", "VAE", "CCA", "Procrustes", "ManifoldAlign"):
        if method in art:
            for key in ("w_test", "z_test", "theta", "generators", "matrices"):
                v = art[method].get(key)
                if v is not None:
                    payload[f"{method}::{key}"] = np.asarray(v)
    np.savez_compressed(out / "artifacts.npz", **payload)


if __name__ == "__main__":
    main()

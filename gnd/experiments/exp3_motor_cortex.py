"""Experiment 3: motor-cortex population dynamics across reach conditions.

Question.  Reaches to different targets are, in an obvious sense, the same
computation performed in a rotated frame.  Do the population trajectories for
different conditions relate by a learned gauge transformation, and does the
model reproduce the prediction of Eq. (5) -- that conjugation leaves the
*eigenvalues* of the latent dynamics unchanged while rotating their
eigenplanes?

Setup.  An RNN is trained by BPTT to produce bell-shaped hand velocity toward
each of eight targets at two reach extents.  No symmetry is imposed on the
recurrent weights, so the circuit's own departure from exact equivariance
(:func:`gnd.simulations.motor_cortex.equivariance_defect`) provides an empirical
ceiling on what any latent-isometry model can recover.  Contexts are the sixteen
reach conditions; the ground-truth action on the reach plane is
``e_c R(phi_c)``, a two-parameter abelian group ``SO(2) x R_+``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..models.gnd import GNDConfig
from ..models.neural_dynamics import dynamics_spectrum, fit_linear_dynamics, rotational_plane_angle
from ..simulations.base import ContextualDataset
from ..simulations.motor_cortex import ReachTaskConfig, equivariance_defect, simulate_motor_cortex
from ..utils.common import RESULTS_DIR, save_json
from .common import (DEFAULT_BASELINES, HEADLINE_KEYS, aggregate_table, checkpoint,
                     run_seed)
from .evaluate import GroundTruth


def build_ground_truth(test: ContextualDataset) -> GroundTruth:
    """Ground truth for reaching: the commanded hand velocity in the reach plane.

    The canonical computational variable is the movement being prepared and
    executed, expressed in the reference condition's frame; the true context
    action is ``e_c R(phi_c)`` acting on that two-dimensional vector.  Because
    the transformation is linear on this coordinate, both the data-space
    recovery error and the conjugation-invariant spectral error apply.
    """
    Y = test.meta["target_output"]                       # (n_cond, T, 2)
    t_idx = test.time_index
    ref = test.reference
    coords = Y[ref][t_idx]                               # (N, 2)
    per_ctx = np.stack([Y[c][t_idx] for c in range(test.n_contexts)])
    mats = {c: np.asarray(s.matrix, float) for c, s in enumerate(test.contexts)}
    emb = np.concatenate([coords, (t_idx / max(t_idx.max(), 1))[:, None]], axis=1)
    return GroundTruth(
        coords=coords,
        coords_per_context=per_ctx,
        matrices=mats,
        embedding=emb,                                   # a smooth arc: beta = (1, 0)
        expected_betti=[1, 0],
        maxdim=1,
    )


def default_config(n_latent: int = 8) -> GNDConfig:
    return GNDConfig(
        n_latent=n_latent,
        hidden=192,
        depth=3,
        n_generators=6,
        algebra="gl",
        epochs=250,
        batch_size=256,
        lr=2e-3,
        w_group=0.2,
        w_closure=0.05,
        w_topology=0.3,
        dynamics="linear",
        w_dynamics=0.1,
        # Sixteen contexts give 240 ordered pairs; sampling only eight per step
        # leaves most pairs almost unseen, so the transport term is widened here.
        n_transport_pairs=20,
        n_group_pairs=16,
    )


# ---------------------------------------------------------------------------
def trajectory_alignment(z: np.ndarray, test: ContextualDataset) -> dict:
    """Residual after aligning per-context trajectories to the reference.

    Trial-averaged trajectories are compared directly (the canonical latents
    should already coincide) and after an optimal orthogonal Procrustes map
    (the residual any rigid-alignment method could not remove).
    """
    from scipy.linalg import orthogonal_procrustes

    T = test.meta["n_time"]
    n_trials = z.shape[0] // T
    Z = z[: n_trials * T].reshape(n_trials, T, z.shape[1], z.shape[2]).mean(0)   # (T, C, d)
    ref = test.reference
    direct, procr = [], []
    scale = np.linalg.norm(Z[:, ref] - Z[:, ref].mean(0)) + 1e-12
    for c in range(test.n_contexts):
        if c == ref:
            continue
        direct.append(np.linalg.norm(Z[:, c] - Z[:, ref]) / scale)
        R, _ = orthogonal_procrustes(Z[:, c], Z[:, ref])
        procr.append(np.linalg.norm(Z[:, c] @ R - Z[:, ref]) / scale)
    return {
        "trajectory_alignment_error": float(np.mean(direct)),
        "trajectory_alignment_error_procrustes": float(np.mean(procr)),
    }


def dynamics_conjugacy(z_obs: np.ndarray, test: ContextualDataset, dt: float) -> dict:
    r"""Test the prediction ``F_c = M_c F M_c^{-1}`` of Eq. (5).

    Linear dynamics are fitted separately to each context's *observed-frame*
    latent trajectory.  Conjugate matrices share a spectrum, so the rotation
    frequencies should agree across contexts while the dominant rotation plane
    turns with the reach direction.  We report the coefficient of variation of
    the top rotation frequency across contexts and the mean principal angle
    between dominant planes.
    """
    T = test.meta["n_time"]
    n_prep = test.meta["n_prep"]
    n_trials = z_obs.shape[0] // T
    Z = z_obs[: n_trials * T].reshape(n_trials, T, z_obs.shape[1], z_obs.shape[2]).mean(0)
    As, freqs = [], []
    for c in range(test.n_contexts):
        A = fit_linear_dynamics(Z[n_prep:, c], dt=dt)
        As.append(A)
        freqs.append(dynamics_spectrum(A)["top_frequency_hz"])
    freqs = np.array(freqs)
    ref = test.reference
    angles = [rotational_plane_angle(As[ref], As[c]) for c in range(test.n_contexts) if c != ref]
    return {
        "rotation_frequency_hz_mean": float(freqs.mean()),
        "rotation_frequency_cv": float(freqs.std(ddof=1) / (np.abs(freqs.mean()) + 1e-12)),
        "rotation_plane_angle_deg": float(np.mean(angles)),
    }


# ---------------------------------------------------------------------------
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-recorded", type=int, default=100)
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--n-units", type=int, default=128)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--n-latent", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--rnn-steps", type=int, default=1500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(RESULTS_DIR / "exp3_motor_cortex"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.seeds, args.n_trials, args.epochs, args.rnn_steps = [0], 4, 50, 250

    cfg = default_config(args.n_latent)
    cfg.epochs = args.epochs

    task = ReachTaskConfig(n_units=args.n_units, train_steps=args.rnn_steps)

    def build(seed: int):
        return simulate_motor_cortex(
            n_recorded=args.n_recorded, n_trials=args.n_trials, cfg=task,
            noise=args.noise, seed=seed, verbose=False,
        )

    rows, first_art = [], None
    for seed in args.seeds:
        print(f"=== exp3 seed {seed} ===")
        r, art = run_seed(
            build, build_ground_truth, cfg, seed, variants=None,
            baselines=tuple(DEFAULT_BASELINES), device=args.device,
            topology_points=400, verbose=(seed == args.seeds[0]),
            ae_epochs=min(args.epochs, 200),
        )
        test = art["test"]
        dt = test.meta["dt"]
        ceil = equivariance_defect(art["dataset"])
        # Reference point for the conjugacy prediction: the same analysis run on
        # a plain PCA of the recorded activity, i.e. the circuit's own spread of
        # rotation frequencies across conditions.  No latent model can be
        # expected to beat the circuit itself.
        from sklearn.decomposition import PCA
        _p = PCA(n_components=min(8, test.n_neurons)).fit(test.activity.reshape(-1, test.n_neurons))
        _w = _p.transform(test.activity.reshape(-1, test.n_neurons)).reshape(
            test.n_samples, test.n_contexts, -1)
        ceil.update({f"reference_{k}": v for k, v in dynamics_conjugacy(_w, test, dt).items()})
        for row in r:
            if "error" in row:
                continue
            m = row["method"]
            if m in art and "z_test" in art[m]:
                row.update(trajectory_alignment(art[m]["z_test"], test))
                row.update(dynamics_conjugacy(art[m]["w_test"], test, dt))
            row.update({f"circuit_{k}": v for k, v in ceil.items()})
            print(
                f"  {m:<14} GCS={row.get('gcs', float('nan')):.3f} "
                f"CIS={row.get('cis', float('nan')):.3f} GRE={row.get('gre', float('nan')):.3f} "
                f"trajErr={row.get('trajectory_alignment_error', float('nan')):.3f} "
                f"freqCV={row.get('rotation_frequency_cv', float('nan')):.3f}"
            )
        rows += r
        if first_art is None:
            first_art = art
            first_art["rnn_ceiling"] = ceil
        checkpoint(Path(args.out), {"rows": rows,
                                    "table": aggregate_table(rows, HEADLINE_KEYS),
                                    "args": vars(args), "complete": False})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = HEADLINE_KEYS + (
        "trajectory_alignment_error", "trajectory_alignment_error_procrustes",
        "rotation_frequency_hz_mean", "rotation_frequency_cv", "rotation_plane_angle_deg",
        "circuit_equivariance_residual_mean",
        "circuit_reference_rotation_frequency_cv",
        "circuit_reference_rotation_plane_angle_deg",
        "circuit_reference_rotation_frequency_hz_mean",
    )
    table = aggregate_table(rows, keys)
    save_json({"rows": rows, "table": table, "args": vars(args), "complete": True},
              out / "results.json")
    _save_artifacts(out, first_art, args.seeds[0])
    print(f"\nwrote {out/'results.json'}")
    return {"rows": rows, "table": table}


def _save_artifacts(out: Path, art: dict, seed: int) -> None:
    """Persist what Figure 5 needs, stamped with the seed that produced it."""
    ds, test = art["dataset"], art["test"]
    payload = {
        "provenance_seed": np.array(seed),
        "context_names": np.array([s.name for s in test.contexts]),
        "angles": ds.meta["angles"],
        "extents": ds.meta["extents"],
        "n_time": test.meta["n_time"],
        "n_prep": test.meta["n_prep"],
        "dt": test.meta["dt"],
        "time_index_test": test.time_index,
        "unit_rates": ds.meta["unit_rates"],
        "target_output": ds.meta["target_output"],
        "rnn_loss_curve": np.array(ds.meta["rnn_loss_curve"]),
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

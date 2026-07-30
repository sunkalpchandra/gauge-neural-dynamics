"""Shared experiment driver: fit every method, evaluate them identically, save.

All three experiments follow the same protocol.

1. Simulate a paired multi-context dataset and z-score it with statistics
   pooled over contexts (per-context scaling would erase part of the effect).
2. Split into train / test along the sample axis (along whole trials for the
   motor simulation).
3. Fit GND and every baseline on the *training* split only.
4. Evaluate every method on the *held-out* split with the identical metric
   suite of :mod:`gnd.experiments.evaluate`.
5. Repeat over seeds; report mean and standard error.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import torch

from ..geometry.metrics import GaugeSummary
from ..models.baselines import BASELINES, BaselineResult
from ..models.gnd import GNDConfig
from ..models.train import fit_gnd
from ..simulations.base import ContextualDataset
from ..utils.common import save_json, set_seed
from .evaluate import (
    GroundTruth,
    canonical_from_model,
    evaluate,
    gauge_summary_from_baseline,
    gauge_summary_from_model,
)

DEFAULT_BASELINES = ("pca", "umap", "autoencoder", "vae", "cca", "procrustes", "manifold_align")


# ---------------------------------------------------------------------------
def run_gnd(
    train: ContextualDataset,
    test: ContextualDataset,
    gt: GroundTruth | None,
    cfg: GNDConfig,
    name: str = "GND",
    device: str = "cpu",
    verbose: bool = False,
    topology_points: int = 500,
    keep_artifacts: bool = False,
) -> tuple[dict, dict]:
    t0 = time.time()
    model, hist = fit_gnd(train, cfg, device=device, verbose=verbose)
    gauge = gauge_summary_from_model(model, test, device=device)
    w_tr, _ = canonical_from_model(model, train, device=device)
    w_te, z_te = canonical_from_model(model, test, device=device)

    # native generative pathway
    t = test.tensors(device)
    theta = model.coefficients(t["context_features"], t["context_features"][test.reference])
    with torch.no_grad():
        native_rec = model.decode(
            model.encode(t["activity"].reshape(-1, test.n_neurons))[0]
        ).cpu().numpy().reshape(test.n_samples, test.n_contexts, -1)
        native_tr = np.stack(
            [
                model.transport(t["activity"][:, test.reference], theta[test.reference], theta[c]).cpu().numpy()
                for c in range(test.n_contexts)
            ],
            axis=1,
        )

    res = evaluate(
        name, w_tr, w_te, z_te, gauge, train, test, gt,
        native_recon=native_rec, native_transport=native_tr,
        topology_points=topology_points, seed=cfg.seed,
    )
    res["fit_seconds"] = time.time() - t0
    res["n_parameters"] = hist["n_parameters"]
    res["final_loss"] = hist["total"][-1] if hist.get("total") else float("nan")
    art = {
        "w_test": w_te, "z_test": z_te, "theta": gauge.coefficients,
        "generators": gauge.generators, "matrices": gauge.matrices,
        "history": {k: v for k, v in hist.items() if isinstance(v, list)},
    }
    if keep_artifacts:
        art["model"] = model
    return res, art


def run_baseline(
    kind: str,
    train: ContextualDataset,
    test: ContextualDataset,
    gt: GroundTruth | None,
    n_latent: int,
    seed: int,
    device: str = "cpu",
    topology_points: int = 500,
    n_generators: int = 6,
    ae_epochs: int = 200,
) -> tuple[dict, dict]:
    t0 = time.time()
    res_b: BaselineResult = BASELINES[kind](
        train, test, n_latent=n_latent, seed=seed, device=device, epochs=ae_epochs
    )
    gauge = gauge_summary_from_baseline(res_b, n_generators=n_generators)
    z_te = res_b.canonical("test")
    res = evaluate(
        res_b.name, res_b.w_train, res_b.w_test, z_te, gauge, train, test, gt,
        native_recon=res_b.recon_test, native_transport=None,
        topology_points=topology_points, seed=seed,
    )
    res["fit_seconds"] = time.time() - t0
    res.update({f"gauge_{k}": v for k, v in gauge.meta.items() if isinstance(v, (int, float))})
    return res, {"w_test": res_b.w_test, "z_test": z_te, "matrices": res_b.matrices}


# ---------------------------------------------------------------------------
def run_seed(
    build_dataset,
    build_ground_truth,
    gnd_cfg: GNDConfig,
    seed: int,
    variants: dict[str, dict] | None = None,
    baselines: tuple[str, ...] = DEFAULT_BASELINES,
    device: str = "cpu",
    split_frac: float = 0.8,
    topology_points: int = 500,
    verbose: bool = False,
    ae_epochs: int = 200,
    keep_artifacts: bool = False,
) -> tuple[list[dict], dict]:
    """Fit and evaluate every method for one seed."""
    set_seed(seed)
    ds = build_dataset(seed).standardise()
    train, test = ds.split(split_frac, seed=seed)
    gt = build_ground_truth(test)

    rows, arts = [], {"dataset": ds, "train": train, "test": test, "ground_truth": gt}
    cfg = replace(gnd_cfg, seed=seed)
    r, a = run_gnd(train, test, gt, cfg, "GND", device, verbose, topology_points, keep_artifacts)
    rows.append({**r, "seed": seed})
    arts["GND"] = a

    for name, overrides in (variants or {}).items():
        cfg_v = replace(cfg, **overrides)
        r, a = run_gnd(train, test, gt, cfg_v, name, device, False, topology_points, keep_artifacts)
        rows.append({**r, "seed": seed, **{f"cfg_{k}": v for k, v in overrides.items()}})
        arts[name] = a

    for kind in baselines:
        try:
            r, a = run_baseline(
                kind, train, test, gt, cfg.n_latent, seed, device, topology_points,
                n_generators=cfg.n_generators, ae_epochs=ae_epochs,
            )
            rows.append({**r, "seed": seed})
            arts[r["method"]] = a
        except Exception as exc:                       # a failed baseline must not kill the run
            rows.append({"method": kind, "seed": seed, "error": str(exc)})
            print(f"  !! baseline {kind} failed: {type(exc).__name__}: {exc}")
    return rows, arts


def checkpoint(out, payload: dict, name: str = "results.json") -> None:
    """Write partial results after every seed.

    Without this a multi-hour sweep produces nothing at all if it is interrupted,
    and the seeds already computed are lost.  Writing through a temporary file
    keeps the on-disk copy readable at every moment.
    """
    from pathlib import Path

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (name + ".tmp")
    save_json(payload, tmp)
    tmp.replace(out / name)


def aggregate_table(rows: list[dict], keys: tuple[str, ...]) -> dict:
    """Group rows by ``method`` and reduce each metric to mean / sem / n."""
    from ..utils.common import aggregate_seeds

    out: dict[str, dict] = {}
    for method in sorted({r["method"] for r in rows if "method" in r}):
        sub = [r for r in rows if r.get("method") == method and "error" not in r]
        if not sub:
            continue
        agg = aggregate_seeds(sub)
        out[method] = {k: agg[k] for k in keys if k in agg}
        out[method]["_n_seeds"] = len(sub)
    return out


HEADLINE_KEYS = (
    "gcs", "cis", "gre", "mps", "mps_cross_context", "gre_nonlinear_chart",
    "transform_magnitude", "readout_r2_nonlinear",
    "gauge_logm_imag_residual", "gauge_algebra_reconstruction_error",
    "transport_r2", "recon_r2", "transport_r2_native", "recon_r2_native",
    "composition_error", "closure_defect", "abelianness", "realised_abelianness",
    "holonomy",
    "context_leakage", "context_decoding_acc", "spectral_error", "induced_recovery_error",
    "algebra_recovery_r2",
    "alignment_error", "readout_r2", "knn_overlap", "trustworthiness", "continuity",
    "betti_correct", "cross_context_bottleneck", "ground_truth_bottleneck",
    "fit_seconds", "n_parameters",
)

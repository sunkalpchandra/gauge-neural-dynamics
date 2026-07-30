"""Uniform evaluation harness.

Every method -- GND and each baseline -- is reduced to the same three objects
before any metric is computed:

    w        (N, C, d)   observed-frame latents on the held-out split
    gauge    GaugeSummary the estimated context transformations
    readout  callable     a common non-linear latent -> activity map

The common readout is fitted on the *training* split from pooled
(observed-frame latent, activity) pairs and is used for the headline transport
score, so that cross-method differences reflect the latent geometry and the
transformation rather than decoder capacity.  Methods that also have a native
decoder additionally report ``*_native`` numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..geometry.metrics import (
    GaugeSummary,
    algebra_recovery_r2,
    bch_compose,
    context_decodability,
    context_invariance_score,
    gauge_consistency_score,
    geometric_recovery_error,
    holonomy_defect,
    manifold_preservation_score,
    neighbourhood_preservation,
    post_hoc_algebra,
    r2_score_matrix,
    spectral_recovery_error,
    structure_constants,
)
from ..models.baselines import BaselineResult, RandomFeatureReadout
from ..models.gauge_field import FlowGaugeField, IdentityGauge, LinearGaugeField
from ..models.gnd import GaugeNeuralDynamics
from ..simulations.base import ContextualDataset


# ---------------------------------------------------------------------------
# ground truth bundle
# ---------------------------------------------------------------------------
@dataclass
class GroundTruth:
    """Everything the recovery and topology metrics need from a simulation."""

    coords: np.ndarray                       # (N, m) reference-context coordinates
    coords_per_context: np.ndarray           # (C, N, m) after the true action
    matrices: dict = field(default_factory=dict)   # {c: true linear action or None}
    embedding: np.ndarray | None = None      # point cloud whose topology is the target
    expected_betti: list[int] | None = None
    maxdim: int = 1


# ---------------------------------------------------------------------------
# building a GaugeSummary
# ---------------------------------------------------------------------------
def gauge_summary_from_model(
    model: GaugeNeuralDynamics, dataset: ContextualDataset, device: str = "cpu", n_probe: int = 256
) -> GaugeSummary:
    """Extract generators, coefficients and group elements from a fitted GND."""
    t = dataset.tensors(device)
    ctx, ref = t["context_features"], t["context_features"][dataset.reference]
    with torch.no_grad():
        theta = model.coefficients(ctx, ref).cpu().numpy()

    g = model.gauge
    if isinstance(g, LinearGaugeField):
        with torch.no_grad():
            G = g.bank.generators().cpu().numpy()
            M = g.matrices(torch.as_tensor(theta, dtype=torch.float32)).cpu().numpy()
        f, _ = structure_constants(G)
        return GaugeSummary(
            generators=G,
            coefficients=theta,
            matrices=M,
            affine=g.affine,
            compose_fn=lambda a, b: bch_compose(a, b, f, g.bch_order),
            meta={"gauge": "linear", "algebra": g.bank.algebra},
        )

    if isinstance(g, IdentityGauge):
        d = model.cfg.n_latent
        M = np.stack([np.eye(d)] * dataset.n_contexts)
        return GaugeSummary(
            generators=np.zeros((1, d, d)), coefficients=np.zeros((dataset.n_contexts, 1)),
            matrices=M, affine=False, meta={"gauge": "none"},
        )

    # flow gauge: transformation is non-linear, so act through the model
    def apply(z, th):
        with torch.no_grad():
            zt = torch.as_tensor(np.asarray(z, float), dtype=torch.float32, device=device)
            tt = torch.as_tensor(np.asarray(th, float), dtype=torch.float32, device=device)
            return g.transform(zt, tt.expand(zt.shape[0], -1)).cpu().numpy()

    probe = torch.randn(min(n_probe, 128), model.cfg.n_latent, device=device)
    f_t, res = g.structure_constants(probe)
    f_np = f_t.detach().cpu().numpy()
    K = g.n_generators
    off = ~torch.eye(K, dtype=torch.bool)
    closure = float(res[off].mean())
    abelian = float(g.commutator_norm(probe))
    return GaugeSummary(
        generators=None,
        coefficients=theta,
        matrices=None,
        affine=False,
        nonlinear_apply=apply,
        compose_fn=lambda a, b: bch_compose(a, b, f_np, g.bch_order),
        meta={"gauge": "flow", "closure_defect": closure, "abelianness": abelian},
    )


def gauge_summary_from_baseline(res: BaselineResult, n_generators: int = 6) -> GaugeSummary:
    """Post-hoc algebra fitted to a baseline's estimated transformations."""
    return post_hoc_algebra(res.matrices, n_generators)


# ---------------------------------------------------------------------------
# canonical latents
# ---------------------------------------------------------------------------
def canonical_from_model(model: GaugeNeuralDynamics, dataset: ContextualDataset, device: str = "cpu"):
    t = dataset.tensors(device)
    w, z, theta = model.latents(t["activity"], t["context_features"], t["context_features"][dataset.reference])
    return w.cpu().numpy(), z.cpu().numpy()


def canonical_from_summary(w: np.ndarray, gauge: GaugeSummary) -> np.ndarray:
    """``z_c = T_c^{-1} w_c`` for a linear summary."""
    if gauge.matrices is None:
        raise ValueError("non-linear gauge: use the model's own canonicalisation")
    inv = np.stack([np.linalg.pinv(M) for M in gauge.matrices])
    if gauge.affine:
        raise NotImplementedError("affine baselines are not used")
    return np.einsum("cab,ncb->nca", inv, w)


# ---------------------------------------------------------------------------
# the evaluation itself
# ---------------------------------------------------------------------------
def evaluate(
    name: str,
    w_train: np.ndarray,
    w_test: np.ndarray,
    z_test: np.ndarray,
    gauge: GaugeSummary,
    train: ContextualDataset,
    test: ContextualDataset,
    gt: GroundTruth | None,
    native_recon: np.ndarray | None = None,
    native_transport: np.ndarray | None = None,
    topology_points: int = 500,
    seed: int = 0,
    readout_features: int = 768,
) -> dict:
    """Compute the full metric suite for one method."""
    C, ref = test.n_contexts, test.reference
    out: dict[str, float] = {"method": name, "n_latent": int(w_test.shape[-1])}

    # -- common readout (fit on train, applied on test) --------------------
    rr = RandomFeatureReadout(n_features=readout_features, ridge=1e-3, seed=seed).fit(
        w_train.reshape(-1, w_train.shape[-1]), train.activity.reshape(-1, train.n_neurons)
    )
    rec = rr.predict(w_test.reshape(-1, w_test.shape[-1])).reshape(test.n_samples, C, -1)
    out["recon_r2"] = r2_score_matrix(rec, test.activity)

    trs = []
    for c in range(C):
        if c == ref:
            continue
        w_pred = gauge.apply(z_test[:, ref], c)
        pred = rr.predict(w_pred)
        r2 = r2_score_matrix(pred, test.activity[:, c])
        out[f"transport_r2_ctx{c}"] = r2
        trs.append(r2)
    out["transport_r2"] = float(np.mean(trs)) if trs else float("nan")

    if native_recon is not None:
        out["recon_r2_native"] = r2_score_matrix(native_recon, test.activity)
    if native_transport is not None:
        vals = [
            r2_score_matrix(native_transport[:, c], test.activity[:, c]) for c in range(C) if c != ref
        ]
        out["transport_r2_native"] = float(np.mean(vals)) if vals else float("nan")

    # -- metric 1: gauge consistency ---------------------------------------
    probe = z_test[: min(512, len(z_test)), ref]
    out.update(gauge_consistency_score(gauge, probe, seed=seed))
    out.update(holonomy_defect(gauge, probe, seed=seed))

    # -- metric 2: context invariance --------------------------------------
    out.update(context_invariance_score(z_test))
    out.update(context_decodability(z_test, seed=seed))

    # -- metric 3: geometric recovery --------------------------------------
    if gt is not None:
        out.update(
            geometric_recovery_error(
                z_test, gauge, gt.coords, gt.coords_per_context, reference=ref,
                context_names=[s.name for s in test.contexts],
            )
        )
        out.update(spectral_recovery_error(gauge, z_test, gt.coords, gt.matrices, reference=ref))
        out.update(algebra_recovery_r2(gauge, gt.matrices, reference=ref))

    # -- metric 4: manifold preservation -----------------------------------
    mps = manifold_preservation_score(
        z_test,
        true_latent=None,
        maxdim=gt.maxdim if gt else 1,
        n_points=topology_points,
        seed=seed,
        expected_betti=gt.expected_betti if gt else None,
        true_embedding=gt.embedding if gt else None,
    )
    out.update({k: v for k, v in mps.items() if not isinstance(v, list)})
    out["betti_learned_str"] = str(mps.get("betti_learned"))
    if gt is not None and gt.embedding is not None:
        out.update(neighbourhood_preservation(z_test[:, ref], gt.embedding))
    return out


def summarise_ablation_flags(cfg) -> dict:
    return {
        "gauge": cfg.gauge,
        "algebra": cfg.algebra,
        "n_latent": cfg.n_latent,
        "w_group": cfg.w_group,
        "w_topology": cfg.w_topology,
        "w_closure": cfg.w_closure,
        "alignment_mode": cfg.alignment_mode,
        "encoder": cfg.encoder,
    }

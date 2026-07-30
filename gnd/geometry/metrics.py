r"""Metrics for context-dependent neural geometry.

Four metrics are introduced by the paper, all defined so that they can be
applied identically to GND and to every baseline:

``GCS``  **Gauge Consistency Score** -- how nearly the learned transformations
        compose, i.e. how nearly they form a group.
``CIS``  **Context Invariance Score** -- how stable the canonical latent is
        across contexts, in intraclass-correlation form.
``GRE``  **Geometric Recovery Error** -- distance between the learned and the
        true transformation, computed modulo the unavoidable conjugation
        (gauge) freedom and normalised by the size of the true transformation.
``MPS``  **Manifold Preservation Score** -- agreement of persistent-homology
        diagrams between the learned latent and the true manifold.

Fairness of comparison.  A baseline such as PCA has no parametric map from
context to transformation, so the transformations are estimated post hoc: an
affine map is fitted per context on the *training* split, its matrix logarithm
taken, and a ``K``-dimensional algebra extracted by PCA
(:func:`post_hoc_algebra`).  Every metric is then computed from
``(generators, coefficients)`` by exactly the same code path used for GND, and
all reported values are on a held-out split.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm, logm

from .manifold import (
    apply_readout,
    fit_latent_readout,
    knn_overlap,
    trustworthiness_continuity,
)
from .topology import betti_numbers, diagram_distance, persistence_diagrams, persistence_gap


# ---------------------------------------------------------------------------
# a common description of a learned gauge
# ---------------------------------------------------------------------------
@dataclass
class GaugeSummary:
    """Everything the metrics need about a (learned or post-hoc) gauge.

    ``generators`` are ``(K, m, m)`` algebra basis elements, ``coefficients``
    are ``(C, K)`` per-context coordinates in that basis, and ``matrices`` are
    the resulting ``(C, m, m)`` group elements.  ``affine`` marks homogeneous
    coordinates.  ``nonlinear_apply`` is supplied instead of matrices by the
    flow gauge: a callable ``(z, theta) -> z'``.
    """

    generators: np.ndarray | None
    coefficients: np.ndarray | None
    matrices: np.ndarray | None
    affine: bool = False
    nonlinear_apply: object | None = None
    compose_fn: object | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_contexts(self) -> int:
        if self.coefficients is not None:
            return self.coefficients.shape[0]
        return self.matrices.shape[0]

    def apply(self, z: np.ndarray, c: int) -> np.ndarray:
        if self.nonlinear_apply is not None:
            return self.nonlinear_apply(z, self.coefficients[c])
        return apply_matrix(self.matrices[c], z, self.affine)

    def apply_theta(self, z: np.ndarray, theta: np.ndarray) -> np.ndarray:
        if self.nonlinear_apply is not None:
            return self.nonlinear_apply(z, theta)
        A = np.einsum("k,kab->ab", theta, self.generators)
        return apply_matrix(expm(A), z, self.affine)


def apply_matrix(M: np.ndarray, z: np.ndarray, affine: bool) -> np.ndarray:
    z = np.asarray(z, float)
    if affine:
        z1 = np.concatenate([z, np.ones((z.shape[0], 1))], axis=1)
        out = z1 @ np.asarray(M, float).T
        return out[:, :-1] / np.clip(out[:, -1:], 1e-6, None)
    return z @ np.asarray(M, float).T


# ---------------------------------------------------------------------------
# post-hoc algebra extraction (used for every baseline)
# ---------------------------------------------------------------------------
def real_logm(M: np.ndarray) -> tuple[np.ndarray, float]:
    """Principal real matrix logarithm and the size of the discarded imaginary part.

    A large imaginary residual signals that the transformation is outside the
    range of the principal branch (e.g. a rotation by more than pi), in which
    case the extracted algebra element is not unique.  We report the residual
    rather than hiding it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = logm(np.asarray(M, float))
    L = np.asarray(L)
    resid = float(np.abs(L.imag).max()) if np.iscomplexobj(L) else 0.0
    return np.real(L), resid


def post_hoc_algebra(matrices: np.ndarray, n_generators: int) -> GaugeSummary:
    """Fit a ``K``-dimensional algebra to a set of estimated group elements.

    Takes matrix logarithms, extracts the leading ``K`` principal directions of
    the resulting elements of ``gl(d)``, and re-expresses every context in that
    basis.  This is the fairest way to give a non-parametric baseline the same
    group-theoretic machinery GND learns.
    """
    Ms = np.asarray(matrices, float)
    C, m, _ = Ms.shape
    logs, resids = [], []
    for M in Ms:
        L, r = real_logm(M)
        logs.append(L.reshape(-1))
        resids.append(r)
    L = np.stack(logs)                                   # (C, m^2)
    K = int(min(n_generators, max(1, C - 1), L.shape[1]))
    mu = np.zeros(L.shape[1])                            # no centring: 0 is the identity
    U, S, Vt = np.linalg.svd(L - mu, full_matrices=False)
    G = Vt[:K]                                           # orthonormal in Frobenius
    theta = (L - mu) @ G.T
    G = G.reshape(K, m, m)
    nrm = np.linalg.norm(G.reshape(K, -1), axis=1)
    nrm = np.clip(nrm, 1e-8, None)
    G = G / nrm[:, None, None]
    theta = theta * nrm[None, :]
    recon = np.stack([expm(np.einsum("k,kab->ab", t, G)) for t in theta])
    return GaugeSummary(
        generators=G,
        coefficients=theta,
        matrices=recon,
        affine=False,
        meta={
            "logm_imag_residual": float(np.max(resids)) if resids else 0.0,
            "algebra_reconstruction_error": float(
                np.linalg.norm(recon - Ms) / (np.linalg.norm(Ms) + 1e-12)
            ),
            "n_generators": K,
        },
    )


# ---------------------------------------------------------------------------
# Lie-algebra diagnostics
# ---------------------------------------------------------------------------
def structure_constants(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares structure constants and normalised commutator residuals."""
    G = np.asarray(G, float)
    K = G.shape[0]
    Gf = G.reshape(K, -1)
    comm = np.einsum("iab,jbc->ijac", G, G) - np.einsum("jab,ibc->ijac", G, G)
    Cf = comm.reshape(K * K, -1)
    sol, *_ = np.linalg.lstsq(Gf.T, Cf.T, rcond=None)    # (K, K^2)
    f = sol.T.reshape(K, K, K)
    recon = np.einsum("ijk,kab->ijab", f, G)
    num = np.linalg.norm((comm - recon).reshape(K, K, -1), axis=2)
    den = np.clip(np.linalg.norm(comm.reshape(K, K, -1), axis=2), 1e-8, None)
    return f, num / den


def lie_closure_defect(G: np.ndarray) -> float:
    """Mean normalised commutator residual over non-degenerate pairs."""
    G = np.asarray(G, float)
    K = G.shape[0]
    f, res = structure_constants(G)
    comm = np.einsum("iab,jbc->ijac", G, G) - np.einsum("jab,ibc->ijac", G, G)
    mag = np.linalg.norm(comm.reshape(K, K, -1), axis=2)
    mask = (~np.eye(K, dtype=bool)) & (mag > 1e-4)
    return float(res[mask].mean()) if mask.any() else 0.0


def abelianness(G: np.ndarray) -> float:
    """Mean ``||[G_i, G_j]||_F`` for unit-normalised generators.

    Near zero for a commuting algebra -- the prediction for pure grid-cell
    phase translations.
    """
    G = np.asarray(G, float)
    K = G.shape[0]
    if K < 2:
        return 0.0
    comm = np.einsum("iab,jbc->ijac", G, G) - np.einsum("jab,ibc->ijac", G, G)
    mag = np.linalg.norm(comm.reshape(K, K, -1), axis=2)
    off = ~np.eye(K, dtype=bool)
    return float(mag[off].mean())


def bch_compose(a: np.ndarray, b: np.ndarray, f: np.ndarray, order: int = 2) -> np.ndarray:
    out = a + b
    if order >= 2:
        ab = np.einsum("ijk,i,j->k", f, a, b)
        out = out + 0.5 * ab
        if order >= 3:
            out = out + (1 / 12) * (
                np.einsum("ijk,i,j->k", f, a, ab) - np.einsum("ijk,i,j->k", f, b, ab)
            )
    return out


# ---------------------------------------------------------------------------
# metric 1: Gauge Consistency Score
# ---------------------------------------------------------------------------
def gauge_consistency_score(
    gauge: GaugeSummary, z_probe: np.ndarray, bch_order: int = 2, max_pairs: int = 64, seed: int = 0
) -> dict:
    r"""How nearly do the learned transformations compose?

    For ordered context pairs ``(a, b)`` we compare the *realised* composition
    ``T_a(T_b(z))`` against the *predicted* one ``T_{a.b}(z)``, where the
    composed coefficients come from the truncated BCH series with the
    least-squares structure constants of the learned algebra:

        eps(a, b) = || T_a T_b z - T_{a.b} z || / || T_a T_b z - z || .

    The denominator is how far the composite map actually moves the probe
    points, so a transformation family collapsing to the identity scores badly
    rather than perfectly.  ``GCS = max(0, 1 - mean eps)``.
    """
    if gauge.coefficients is None:
        return {"gcs": float("nan"), "composition_error": float("nan")}
    th = gauge.coefficients
    G = gauge.generators
    compose = gauge.compose_fn
    if compose is None:
        if G is None:
            return {"gcs": float("nan"), "composition_error": float("nan")}
        f, _ = structure_constants(G)
        compose = lambda a, b: bch_compose(a, b, f, bch_order)  # noqa: E731

    C = th.shape[0]
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for a in range(C) for b in range(C) if a != b]
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]

    errs = []
    for a, b in pairs:
        zb = gauge.apply_theta(z_probe, th[b])
        zab = gauge.apply_theta(zb, th[a])
        zc = gauge.apply_theta(z_probe, compose(th[a], th[b]))
        num = np.sqrt(((zab - zc) ** 2).sum(1).mean())
        den = np.sqrt(((zab - z_probe) ** 2).sum(1).mean()) + 1e-9
        errs.append(num / den)
    eps = float(np.mean(errs))
    # How far the transformations actually move the latent.  GCS must always be
    # read together with this: a family collapsed onto the identity is
    # (trivially) closed under composition and would otherwise score highly
    # while explaining nothing.
    disp = [
        np.sqrt(((gauge.apply_theta(z_probe, th[c]) - z_probe) ** 2).sum(1).mean())
        for c in range(C)
    ]
    scale = np.sqrt(((z_probe - z_probe.mean(0)) ** 2).sum(1).mean()) + 1e-9
    return {
        "gcs": float(max(0.0, 1.0 - eps)),
        "composition_error": eps,
        "transform_magnitude": float(np.mean(disp) / scale),
        "closure_defect": lie_closure_defect(G) if G is not None else gauge.meta.get("closure_defect", float("nan")),
        "abelianness": abelianness(G) if G is not None else gauge.meta.get("abelianness", float("nan")),
    }


def holonomy_defect(
    gauge: GaugeSummary, z_probe: np.ndarray, bch_order: int = 2, max_loops: int = 40, seed: int = 0
) -> dict:
    r"""Curvature of the learned connection over context space.

    For a triple ``(a, b, c)`` the transition elements are
    ``g_{ab} = T_b T_a^{-1}`` with BCH coefficients
    ``tau_{ab} = theta_b . (-theta_a)``.  Transporting a latent state around
    the closed loop ``a -> b -> c -> a`` returns it to its start if and only if
    the learned family is closed; the residual displacement is the holonomy and
    is a direct, measurable prediction of path-dependent remapping.
    """
    if gauge.coefficients is None:
        return {"holonomy": float("nan")}
    th = gauge.coefficients
    compose = gauge.compose_fn
    if compose is None:
        if gauge.generators is None:
            return {"holonomy": float("nan")}
        f, _ = structure_constants(gauge.generators)
        compose = lambda a, b: bch_compose(a, b, f, bch_order)  # noqa: E731
    C = th.shape[0]
    if C < 3:
        return {"holonomy": float("nan")}
    rng = np.random.default_rng(seed)
    triples = [(a, b, c) for a in range(C) for b in range(C) for c in range(C) if len({a, b, c}) == 3]
    if len(triples) > max_loops:
        triples = [triples[i] for i in rng.choice(len(triples), max_loops, replace=False)]

    scale = np.sqrt((z_probe ** 2).sum(1).mean()) + 1e-9
    vals = []
    for a, b, c in triples:
        tau_ab = compose(th[b], -th[a])
        tau_bc = compose(th[c], -th[b])
        tau_ca = compose(th[a], -th[c])
        z = gauge.apply_theta(z_probe, tau_ab)
        z = gauge.apply_theta(z, tau_bc)
        z = gauge.apply_theta(z, tau_ca)
        vals.append(np.sqrt(((z - z_probe) ** 2).sum(1).mean()) / scale)
    return {"holonomy": float(np.mean(vals)), "holonomy_max": float(np.max(vals))}


# ---------------------------------------------------------------------------
# metric 2: Context Invariance Score
# ---------------------------------------------------------------------------
def context_invariance_score(z: np.ndarray) -> dict:
    r"""Intraclass-correlation form of context invariance.

    With ``z`` of shape ``(N, C, d)`` (sample, context, latent),

        within  = E_n tr Cov_c[ z_{n,c} ],
        between = tr Cov_n[ mean_c z_{n,c} ],
        CIS     = between / (between + within),

    which is 1 when the canonical latent is identical across contexts and 0
    when context differences dominate.  Because both terms are traces of
    covariances in the same space, CIS is invariant to a global rescaling of
    the latent but *not* to collapse: a constant latent has zero between-
    variance and hence CIS = 0, so the metric cannot be gamed.
    """
    z = np.asarray(z, float)
    N, C, d = z.shape
    zbar = z.mean(1)
    within = float(((z - zbar[:, None]) ** 2).sum(-1).mean())
    between = float(((zbar - zbar.mean(0)) ** 2).sum(-1).mean())
    raw = float(between / (between + within + 1e-12))
    # Chance correction.  As the context-dependent component grows without bound
    # the raw ratio tends to 1/C, not to 0, because averaging C contexts shrinks
    # the noise in the per-sample mean by exactly that factor.  Without the
    # correction the metric would not be comparable between experiments with
    # different numbers of contexts (5 here, 16 for the reach conditions).  A
    # collapsed latent still scores 0, since it has no between-sample variance.
    chance = 1.0 / C
    return {
        "cis": float(max(0.0, (raw - chance) / (1.0 - chance + 1e-12))),
        "cis_raw": raw,
        "cis_chance": chance,
        "within_context_var": within,
        "between_sample_var": between,
        "alignment_error": float(np.sqrt(within) / (np.sqrt(between) + 1e-12)),
    }


def context_decodability(z: np.ndarray, seed: int = 0, max_n: int = 3000) -> dict:
    """Cross-validated accuracy of decoding the context label from ``z``.

    Chance is ``1/C``.  Perfect invariance means chance-level decoding, so we
    report the *leakage* ``(acc - chance) / (1 - chance)`` in ``[0, 1]``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    z = np.asarray(z, float)
    N, C, d = z.shape
    X = z.reshape(N * C, d)
    y = np.tile(np.arange(C), (N, 1)).reshape(-1)
    if len(X) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), max_n, replace=False)
        X, y = X[idx], y[idx]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acc = float(cross_val_score(clf, X, y, cv=3, n_jobs=1).mean())
    chance = 1.0 / C
    return {
        "context_decoding_acc": acc,
        "context_leakage": float(max(0.0, (acc - chance) / (1 - chance + 1e-12))),
    }


# ---------------------------------------------------------------------------
# metric 3: Geometric Recovery Error
# ---------------------------------------------------------------------------
def geometric_recovery_error(
    z_canonical: np.ndarray,
    gauge: GaugeSummary,
    true_coords: np.ndarray,
    true_coords_per_context: np.ndarray,
    reference: int = 0,
    fit_index: np.ndarray | None = None,
    context_names: list[str] | None = None,
) -> dict:
    r"""Distance between the learned and the true context transformation.

    The generative model is invariant under a change of latent frame, so we
    first fit a *single* affine readout ``phi`` from the canonical latent to
    the ground-truth coordinate, using the reference context only, and then
    hold it fixed.  Recovery is the failure of the intertwining relation
    ``phi o T_c = T*_c o phi``:

        GRE_c = || phi(T_c(z)) - T*_c(u) || / || T*_c(u) - u || .

    The denominator makes ``GRE = 1`` the score of the null model "no
    transformation at all", so values below 1 are the only meaningful ones.

    Parameters
    ----------
    true_coords:
        ``(N, m)`` ground-truth coordinates in the reference context.  Each
        simulation supplies the representation in which its transformations act
        naturally: allocentric position for place cells, the ``R^4`` torus
        embedding for grid cells, the reach plane for motor cortex.
    true_coords_per_context:
        ``(C, N, m)`` the same coordinates after the *true* context action.
    """
    z = np.asarray(z_canonical, float)
    u = np.asarray(true_coords, float)
    U = np.asarray(true_coords_per_context, float)
    N, C = z.shape[0], z.shape[1]
    fit_index = np.arange(N) if fit_index is None else fit_index
    readout = fit_latent_readout(z[fit_index, reference], u[fit_index])

    # A second, non-linear chart.  Fitted on the reference context only and
    # then frozen, it cannot absorb any context effect, but it removes the
    # advantage GND would otherwise enjoy from producing a latent in which the
    # true coordinate happens to be an affine function of the embedding.
    from ..models.baselines import RandomFeatureReadout

    rff = RandomFeatureReadout(n_features=384, ridge=1e-3, seed=0).fit(z[fit_index, reference], u[fit_index])
    rff_r2 = 1.0 - ((rff.predict(z[fit_index, reference]) - u[fit_index]) ** 2).sum() / (
        ((u[fit_index] - u[fit_index].mean(0)) ** 2).sum() + 1e-12
    )

    per_ctx, raw, raw_nl = {}, [], []
    for c in range(C):
        if c == reference:
            continue
        u_true = U[c]
        wc = gauge.apply(z[:, reference], c)
        den = np.sqrt(((u_true - u) ** 2).sum(1).mean()) + 1e-12
        e = np.sqrt(((apply_readout(readout, wc) - u_true) ** 2).sum(1).mean()) / den
        e_nl = np.sqrt(((rff.predict(wc) - u_true) ** 2).sum(1).mean()) / den
        key = f"gre_ctx{c}" if context_names is None else f"gre::{context_names[c]}"
        per_ctx[key] = float(e)
        raw.append(e)
        raw_nl.append(e_nl)
    return {
        "gre": float(np.mean(raw)) if raw else float("nan"),
        "gre_nonlinear_chart": float(np.mean(raw_nl)) if raw_nl else float("nan"),
        "readout_r2": readout["r2"],
        "readout_r2_nonlinear": float(rff_r2),
        **per_ctx,
    }


def spectral_recovery_error(
    gauge: GaugeSummary,
    z_canonical: np.ndarray,
    true_coords: np.ndarray,
    true_matrices: dict[int, np.ndarray],
    reference: int = 0,
) -> dict:
    """Conjugation-invariant recovery: compare eigenvalue spectra.

    The induced action on ground-truth coordinates is ``Phi M_c Phi^+``; its
    eigenvalues are compared with those of the true matrix ``M*_c``.  This
    complements :func:`geometric_recovery_error` because eigenvalues are
    invariant under any change of latent frame, so no readout has to be
    trusted.  Only contexts whose true action is linear on ``true_coords`` are
    included.
    """
    if gauge.matrices is None or not true_matrices:
        return {"spectral_error": float("nan"), "induced_recovery_error": float("nan")}
    z = np.asarray(z_canonical, float)
    readout = fit_latent_readout(z[:, reference], np.asarray(true_coords, float))
    Phi = readout["Phi"]
    Pinv = np.linalg.pinv(Phi)
    errs, ind_errs = [], []
    for c, Mtrue in true_matrices.items():
        if c == reference or Mtrue is None:
            continue
        M = gauge.matrices[c]
        if gauge.affine:
            M = M[:-1, :-1]
        induced = Phi @ M @ Pinv
        Mt = np.asarray(Mtrue, float)
        ev_h = np.sort_complex(np.linalg.eigvals(induced))
        ev_t = np.sort_complex(np.linalg.eigvals(Mt))
        k = min(len(ev_h), len(ev_t))
        errs.append(np.abs(ev_h[:k] - ev_t[:k]).mean() / (np.abs(ev_t[:k]).mean() + 1e-12))
        # Chart-free counterpart of GRE: compare the whole induced matrix, again
        # normalised so that 1 is the score of "no transformation".
        ind_errs.append(
            np.linalg.norm(induced - Mt) / (np.linalg.norm(Mt - np.eye(len(Mt))) + 1e-12)
        )
    return {
        "spectral_error": float(np.mean(errs)) if errs else float("nan"),
        "induced_recovery_error": float(np.mean(ind_errs)) if ind_errs else float("nan"),
    }


def algebra_recovery_r2(
    gauge: GaugeSummary, true_matrices: dict[int, np.ndarray], reference: int = 0
) -> dict:
    r"""Does the learned algebra coordinate linearly encode the true one?

    The Lie algebra is a vector space, so if a model has recovered the group
    structure then its coefficients ``theta_c`` should be an invertible linear
    image of the true algebra coordinates ``vec(log M*_c)``.  We regress the
    latter on the former and report a **leave-one-context-out** coefficient of
    determination, which is essential here: with only a handful of contexts an
    in-sample linear fit would be near-perfect for any ``theta`` at all.

    This metric is completely free of the chart ambiguity that affects
    :func:`geometric_recovery_error` -- it never maps the latent anywhere -- and
    is therefore the recovery measure we rely on when the ground-truth
    coordinate is not an affine function of the latent.
    """
    if gauge.coefficients is None or len(true_matrices) < 4:
        return {"algebra_recovery_r2": float("nan")}
    ctxs = sorted(c for c, M in true_matrices.items() if M is not None)
    if len(ctxs) < 4:
        return {"algebra_recovery_r2": float("nan")}
    Y = np.stack([real_logm(np.asarray(true_matrices[c], float))[0].reshape(-1) for c in ctxs])
    X = gauge.coefficients[ctxs]
    n = len(X)
    # With few contexts, K + 1 free coefficients would leave too little data for a
    # meaningful leave-one-out fit, so the predictor is first reduced to its
    # leading principal directions.  This *tightens* the test: the learned
    # coefficients must encode the true algebra coordinate in a low-dimensional
    # linear subspace rather than anywhere in R^K.
    r = max(1, min(X.shape[1], n - 3))
    if r < X.shape[1]:
        Xc = X - X.mean(0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        X = Xc @ Vt[:r].T
    X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    n, p = X.shape
    if n <= p + 1:                            # still not enough contexts
        return {"algebra_recovery_r2": float("nan")}
    preds = np.zeros_like(Y)
    for i in range(n):
        keep = np.arange(n) != i
        W, *_ = np.linalg.lstsq(X[keep], Y[keep], rcond=None)
        preds[i] = X[i] @ W
    ss_res = ((preds - Y) ** 2).sum()
    ss_tot = ((Y - Y.mean(0)) ** 2).sum() + 1e-12
    return {"algebra_recovery_r2": float(1.0 - ss_res / ss_tot)}


# ---------------------------------------------------------------------------
# metric 4: Manifold Preservation Score
# ---------------------------------------------------------------------------
def manifold_preservation_score(
    z_canonical: np.ndarray,
    true_latent: np.ndarray | None = None,
    maxdim: int = 1,
    n_points: int = 500,
    seed: int = 0,
    expected_betti: list[int] | None = None,
    true_embedding: np.ndarray | None = None,
) -> dict:
    r"""Topological agreement between the learned latent and the true manifold.

    Two things are measured.  (i) *Cross-context* preservation: persistence
    diagrams of the canonical latent computed separately in each context should
    agree, since the gauge hypothesis says they are the same manifold.
    (ii) *Ground-truth* preservation: they should also agree with the diagram of
    the true latent manifold.  We summarise both as

        MPS = 1 / (1 + mean_d W_2(diagram_d, reference diagram_d)) ,

    which lies in ``(0, 1]`` and equals 1 for an exact match.
    """
    z = np.asarray(z_canonical, float)
    N, C, d = z.shape
    dgms = [persistence_diagrams(z[:, c], maxdim=maxdim, n_points=n_points, seed=seed) for c in range(C)]

    cross = []
    for a in range(C):
        for b in range(a + 1, C):
            for k in range(maxdim + 1):
                cross.append(diagram_distance(dgms[a][k], dgms[b][k], "bottleneck"))
    out = {
        "mps_cross_context": float(1.0 / (1.0 + np.mean(cross))) if cross else float("nan"),
        "cross_context_bottleneck": float(np.mean(cross)) if cross else float("nan"),
    }

    target = true_embedding if true_embedding is not None else true_latent
    if target is not None:
        dgm_true = persistence_diagrams(np.asarray(target, float), maxdim=maxdim, n_points=n_points, seed=seed)
        gt = []
        for c in range(C):
            for k in range(maxdim + 1):
                gt.append(diagram_distance(dgms[c][k], dgm_true[k], "bottleneck"))
        out["mps"] = float(1.0 / (1.0 + np.mean(gt)))
        out["ground_truth_bottleneck"] = float(np.mean(gt))
        out["betti_true_estimated"] = betti_numbers(dgm_true)
    else:
        out["mps"] = out["mps_cross_context"]

    bl = betti_numbers(dgms[0])
    out["betti_learned"] = bl
    # How decisive the Betti call was, so a marginal one is visible in the table.
    out["betti_gap"] = float(np.min([persistence_gap(dgms[0][k], bl[k])
                                     for k in range(len(bl)) if bl[k] > 0] or [0.0]))
    if expected_betti is not None:
        k = min(len(bl), len(expected_betti))
        out["betti_correct"] = float(all(bl[i] == expected_betti[i] for i in range(k)))
        out["betti_expected"] = list(expected_betti)
    return out


def neighbourhood_preservation(z: np.ndarray, u: np.ndarray, k: int = 15) -> dict:
    """Local geometry: k-NN overlap, trustworthiness and continuity."""
    z, u = np.asarray(z, float), np.asarray(u, float)
    n = min(len(z), 1200)
    idx = np.random.default_rng(0).choice(len(z), n, replace=False)
    t, c = trustworthiness_continuity(u[idx], z[idx], k=k)
    return {
        "knn_overlap": knn_overlap(u[idx], z[idx], k=k),
        "trustworthiness": t,
        "continuity": c,
    }


# ---------------------------------------------------------------------------
# reconstruction / transport helpers shared with the baselines
# ---------------------------------------------------------------------------
def r2_score_matrix(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean(0, keepdims=True)) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-12))

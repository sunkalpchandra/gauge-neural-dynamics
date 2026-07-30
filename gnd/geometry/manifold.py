"""Manifold-geometry utilities: alignment, neighbourhood preservation,
intrinsic dimension, and the gauge-fixing regression used by the recovery
metrics.

Gauge fixing
------------
The generative model ``x_c = f(T_c(z))`` is invariant under
``z -> S z, T_c -> S T_c S^{-1}, f -> f o S^{-1}`` for invertible ``S``, so a
learned latent frame is only ever determined up to conjugation.  Comparing a
learned transformation to a ground-truth one therefore requires either
(a) fitting the frame change explicitly, or (b) comparing conjugation-invariant
quantities.  :func:`fit_latent_readout` implements (a) -- a single affine map
from canonical latent to ground-truth generative coordinates, fitted *once* on
the reference context and then held fixed -- and :func:`induced_linear_action`
implements the push-forward needed for (b).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------
def procrustes_align(A: np.ndarray, B: np.ndarray, scaling: bool = True, centre: bool = True):
    """Best ``A -> B`` map restricted to rotation (+ optional scale, shift).

    Returns ``(A_aligned, info)``.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    muA = A.mean(0) if centre else np.zeros(A.shape[1])
    muB = B.mean(0) if centre else np.zeros(B.shape[1])
    A0, B0 = A - muA, B - muB
    d = max(A0.shape[1], B0.shape[1])
    A0 = np.pad(A0, ((0, 0), (0, d - A0.shape[1])))
    B0 = np.pad(B0, ((0, 0), (0, d - B0.shape[1])))
    R, sc = orthogonal_procrustes(A0, B0)
    s = sc / (A0 ** 2).sum() if scaling and (A0 ** 2).sum() > 0 else 1.0
    out = s * (A0 @ R) + muB
    return out[:, : B.shape[1]], {"R": R, "scale": float(s), "muA": muA, "muB": muB}


def affine_align(A: np.ndarray, B: np.ndarray):
    """Unconstrained least-squares affine map ``A -> B``."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    X = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
    W, *_ = np.linalg.lstsq(X, B, rcond=None)
    return X @ W, {"W": W[:-1].T, "b": W[-1]}


def cca_align(A: np.ndarray, B: np.ndarray, n_components: int | None = None, seed: int = 0):
    """Canonical correlation alignment; returns projections and correlations."""
    n = n_components or min(A.shape[1], B.shape[1])
    n = max(1, min(n, A.shape[1], B.shape[1], A.shape[0] - 1))
    cca = CCA(n_components=n, max_iter=1000)
    Xa, Xb = cca.fit_transform(np.asarray(A, float), np.asarray(B, float))
    corr = np.array(
        [np.corrcoef(Xa[:, i], Xb[:, i])[0, 1] for i in range(n)]
    )
    return Xa, Xb, {"correlations": corr, "model": cca}


def normalised_alignment_error(A: np.ndarray, B: np.ndarray, kind: str = "procrustes") -> float:
    """Residual of the best ``A -> B`` map, normalised by ``||B - mean(B)||``.

    ``0`` is a perfect match; ``1`` is no better than predicting the mean.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    if kind == "procrustes":
        Ahat, _ = procrustes_align(A, B)
    elif kind == "affine":
        Ahat, _ = affine_align(A, B)
    elif kind == "identity":
        Ahat = A[:, : B.shape[1]]
    else:
        raise ValueError(kind)
    den = np.linalg.norm(B - B.mean(0, keepdims=True)) + 1e-12
    return float(np.linalg.norm(Ahat - B) / den)


# ---------------------------------------------------------------------------
# gauge fixing / readout
# ---------------------------------------------------------------------------
def fit_latent_readout(z: np.ndarray, u: np.ndarray, ridge: float = 1e-3) -> dict:
    """Affine readout ``phi: z -> u`` from canonical latent to ground truth.

    Fitted once, on the reference context only, and then applied unchanged to
    every context.  Because it is context independent it cannot absorb any
    context transformation -- it only removes the global frame ambiguity.

    ``ridge`` is a *relative* penalty (scaled by ``tr(Z^T Z)/d``) and is not
    cosmetic.  When the latent dimension exceeds that of the ground-truth
    coordinate, an unpenalised fit puts weights of order ``1/sigma`` on the
    low-variance latent directions.  The readout is then accurate on the
    reference context but wildly extrapolating on the transported latents it is
    subsequently applied to, which would penalise a method for having a rich
    latent rather than for getting the transformation wrong.
    """
    z, u = np.asarray(z, float), np.asarray(u, float)
    mu, sd = z.mean(0), z.std(0) + 1e-9
    Zs = (z - mu) / sd
    d = Zs.shape[1]
    G = Zs.T @ Zs
    lam = ridge * np.trace(G) / max(d, 1)
    W = np.linalg.solve(G + lam * np.eye(d), Zs.T @ (u - u.mean(0)))
    Phi = (W / sd[:, None]).T
    b = u.mean(0) - Phi @ mu
    pred = z @ Phi.T + b
    ss_res = ((pred - u) ** 2).sum()
    ss_tot = ((u - u.mean(0)) ** 2).sum() + 1e-12
    return {"Phi": Phi, "b": b, "r2": float(1 - ss_res / ss_tot)}


def apply_readout(readout: dict, z: np.ndarray) -> np.ndarray:
    return np.asarray(z, float) @ readout["Phi"].T + readout["b"]


def induced_linear_action(readout: dict, M: np.ndarray) -> np.ndarray:
    """Push a latent linear map ``M`` through the readout: ``Phi M Phi^+``.

    Conjugation-invariant spectra of the result can be compared directly with
    the ground-truth transformation's spectrum.
    """
    Phi = readout["Phi"]
    return Phi @ np.asarray(M, float) @ np.linalg.pinv(Phi)


# ---------------------------------------------------------------------------
# neighbourhood structure
# ---------------------------------------------------------------------------
def knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def knn_overlap(X: np.ndarray, Z: np.ndarray, k: int = 15) -> float:
    """Fraction of ``k`` nearest neighbours shared between two embeddings."""
    a, b = knn_indices(np.asarray(X, float), k), knn_indices(np.asarray(Z, float), k)
    return float(np.mean([len(set(ai) & set(bi)) / k for ai, bi in zip(a, b)]))


def trustworthiness_continuity(X: np.ndarray, Z: np.ndarray, k: int = 15) -> tuple[float, float]:
    """Trustworthiness (no false neighbours) and continuity (no lost ones)."""
    from sklearn.manifold import trustworthiness

    t = float(trustworthiness(X, Z, n_neighbors=k))
    c = float(trustworthiness(Z, X, n_neighbors=k))
    return t, c


def geodesic_distances(X: np.ndarray, k: int = 12) -> np.ndarray:
    """Graph geodesics on a symmetrised k-NN graph (Isomap-style)."""
    from scipy.sparse.csgraph import shortest_path
    from sklearn.neighbors import kneighbors_graph

    G = kneighbors_graph(X, n_neighbors=k, mode="distance")
    G = G.maximum(G.T)
    return shortest_path(G, directed=False)


def intrinsic_dimension(X: np.ndarray, fraction: float = 0.9) -> float:
    """TwoNN maximum-likelihood intrinsic-dimension estimate (Facco et al. 2017)."""
    X = np.asarray(X, float)
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 1e-12
    mu = np.sort(r2[ok] / r1[ok])
    n = len(mu)
    keep = int(fraction * n)
    mu = mu[:keep]
    F = np.arange(1, keep + 1) / n
    x, y = np.log(mu), -np.log(1 - F)
    slope = float((x @ y) / (x @ x + 1e-12))
    return slope


def participation_ratio(X: np.ndarray) -> float:
    """``(sum lambda)^2 / sum lambda^2`` -- a smooth effective dimensionality."""
    X = np.asarray(X, float)
    C = np.cov(X - X.mean(0), rowvar=False)
    ev = np.linalg.eigvalsh(np.atleast_2d(C))
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (np.sum(ev ** 2) + 1e-12))


def variance_explained(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    from sklearn.decomposition import PCA

    n = min(n_components, X.shape[1], X.shape[0])
    return PCA(n_components=n).fit(X).explained_variance_ratio_

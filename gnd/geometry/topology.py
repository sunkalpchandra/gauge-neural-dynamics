"""Topological tools: persistent homology, Betti-number estimation, and a
differentiable 0-dimensional persistence loss used during training.

Persistent homology is the natural instrument here because the claim under test
is that *the same* latent manifold underlies every context.  Two manifolds that
differ only by a diffeomorphism have identical homology, so agreement of
persistence diagrams across contexts is a necessary condition for the gauge
hypothesis, and disagreement is direct evidence against it.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree


# ---------------------------------------------------------------------------
# persistent homology
# ---------------------------------------------------------------------------
def subsample(X: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Uniform subsample; persistent homology in H2 is O(n^3)-ish in practice."""
    if X.shape[0] <= n:
        return X
    rng = np.random.default_rng(seed)
    return X[rng.choice(X.shape[0], n, replace=False)]


def persistence_diagrams(
    X: np.ndarray,
    maxdim: int = 2,
    n_points: int = 700,
    seed: int = 0,
    normalise: bool = True,
    metric: str = "euclidean",
) -> list[np.ndarray]:
    """Vietoris--Rips persistence diagrams of a point cloud.

    ``normalise`` divides all filtration values by the mean pairwise distance,
    which makes diagrams comparable across latent spaces of different scale --
    essential when comparing GND against PCA or UMAP embeddings.
    """
    from ripser import ripser

    Xs = subsample(np.asarray(X, dtype=float), n_points, seed)
    if normalise:
        from scipy.spatial.distance import pdist

        scale = float(np.mean(pdist(Xs))) or 1.0
        Xs = Xs / scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ripser(Xs, maxdim=maxdim, metric=metric)

    # The essential class (an infinite bar, always present in H_0) is given the
    # diameter of the cloud as its death time.  Capping it at the largest *finite*
    # death instead -- the obvious alternative -- would make the essential bar
    # indistinguishable from the longest noise bar and destroy the gap that Betti
    # estimation relies on.
    from scipy.spatial.distance import pdist

    diameter = float(np.max(pdist(Xs))) if len(Xs) > 1 else 1.0
    dgms = []
    for d in res["dgms"]:
        d = np.asarray(d, dtype=float)
        if d.size:
            infinite = ~np.isfinite(d[:, 1])
            if infinite.any():
                d = d.copy()
                d[infinite, 1] = diameter
        dgms.append(d)
    return dgms


def lifetimes(dgm: np.ndarray) -> np.ndarray:
    if dgm is None or len(dgm) == 0:
        return np.zeros(0)
    return np.asarray(dgm)[:, 1] - np.asarray(dgm)[:, 0]


def betti_numbers(
    dgms: list[np.ndarray],
    max_betti: int = 6,
    min_gap: float = 1.5,
    ratio: float | None = None,
) -> list[int]:
    """Estimate Betti numbers from the largest multiplicative gap in the barcode.

    Sorting the lifetimes of dimension ``d`` in decreasing order, the estimate is
    the index ``k <= max_betti`` maximising ``l_k / l_{k+1}`` -- the standard
    "find the gap separating signal bars from the noise band" heuristic.

    The gap must exceed ``min_gap``, and this matters: without it the rule would
    always return at least one feature, so a contractible cloud such as a disc
    would be reported as having a one-dimensional hole.  The default of 1.5 sits
    between the two cases that bracket it at the sample sizes used here -- a disc,
    whose largest spurious ``H_1`` gap is about 1.1, and a torus, whose true
    two-bar gap is about 1.9 -- and :func:`persistence_gap` reports the achieved
    value so that a marginal call is visible rather than hidden. When no gap is decisive
    we return ``0`` in dimensions ``>= 1`` (no significant feature) and ``1`` in
    dimension ``0`` (a non-empty cloud always has at least one component).
    :func:`persistence_gap` reports how decisive the chosen gap was and should be
    quoted alongside.

    ``ratio`` switches to plain thresholding at ``ratio * max_lifetime``; it is
    retained for diagnostics and is not used for reported values.
    """
    out = []
    for d, dgm in enumerate(dgms):
        life = np.sort(lifetimes(dgm))[::-1]
        life = life[life > 0]
        if life.size == 0:
            out.append(0)
            continue
        if ratio is not None:
            out.append(int((life >= ratio * life.max()).sum()))
            continue
        default = 1 if d == 0 else 0
        cand = min(max_betti, life.size - 1)
        if cand < 1:
            out.append(default)
            continue
        gaps = np.array([life[k - 1] / max(life[k], 1e-12) for k in range(1, cand + 1)])
        best = int(np.argmax(gaps))
        out.append(best + 1 if gaps[best] >= min_gap else default)
    return out


def persistence_gap(dgm: np.ndarray, k: int) -> float:
    """Ratio between the ``k``-th and ``(k+1)``-th longest bars.

    Large values mean the Betti number ``k`` is unambiguous; values near 1 mean
    the estimate rests on an arbitrary threshold.
    """
    life = np.sort(lifetimes(dgm))[::-1]
    if life.size <= k:
        return float("inf") if life.size == k and k > 0 else 0.0
    if k == 0:
        return 0.0
    denom = life[k] if life[k] > 1e-12 else 1e-12
    return float(life[k - 1] / denom)


def trim_diagram(dgm: np.ndarray, min_life_frac: float = 0.01, max_bars: int = 40) -> np.ndarray:
    """Drop the noise band before comparing diagrams.

    Optimal-matching distances sum over *all* bars, so a diagram with hundreds
    of near-zero ``H_0`` bars would swamp the signal and make the matching
    cubic in the sample size.  We keep the ``max_bars`` longest bars that reach
    ``min_life_frac`` of the longest lifetime; points removed this way lie
    within ``min_life_frac`` of the diagonal and contribute negligibly.
    """
    d = np.asarray(dgm, dtype=float).reshape(-1, 2)
    if d.size == 0:
        return d
    life = d[:, 1] - d[:, 0]
    keep = life >= min_life_frac * max(life.max(), 1e-12)
    d = d[keep]
    if len(d) > max_bars:
        d = d[np.argsort(-(d[:, 1] - d[:, 0]))[:max_bars]]
    return d


def diagram_distance(
    dgm_a: np.ndarray,
    dgm_b: np.ndarray,
    kind: str = "bottleneck",
    min_life_frac: float = 0.01,
    max_bars: int = 40,
) -> float:
    """Bottleneck or Wasserstein distance between two trimmed diagrams.

    Bottleneck is the default: it is the largest discrepancy in the persistence
    plane and, unlike the summed Wasserstein cost, does not grow with the
    number of bars, so it is comparable across embeddings of different sizes.
    """
    from persim import bottleneck, wasserstein

    a = trim_diagram(dgm_a, min_life_frac, max_bars)
    b = trim_diagram(dgm_b, min_life_frac, max_bars)
    if a.size == 0 and b.size == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "wasserstein":
            return float(wasserstein(a, b))
        return float(bottleneck(a, b))


def topology_distance_profile(
    X: np.ndarray, Y: np.ndarray, maxdim: int = 1, n_points: int = 500, seed: int = 0
) -> dict:
    """Per-dimension diagram distances between two point clouds."""
    da = persistence_diagrams(X, maxdim=maxdim, n_points=n_points, seed=seed)
    db = persistence_diagrams(Y, maxdim=maxdim, n_points=n_points, seed=seed)
    out = {}
    for d in range(maxdim + 1):
        out[f"wasserstein_H{d}"] = diagram_distance(da[d], db[d], "wasserstein")
        out[f"bottleneck_H{d}"] = diagram_distance(da[d], db[d], "bottleneck")
    return out


# ---------------------------------------------------------------------------
# differentiable topological signature loss (Moor et al., 2020)
# ---------------------------------------------------------------------------
def _pairwise(x: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, x, p=2)


def _mst_edges(D: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the minimum spanning tree edges of a distance matrix.

    The MST edges are exactly the 0-dimensional persistence pairs of the
    Vietoris--Rips filtration, so matching their *lengths* between input and
    latent space matches the ``H_0`` persistence signature.
    """
    d = D.detach().cpu().numpy()
    d = np.maximum(d, d.T)
    mst = minimum_spanning_tree(d).tocoo()
    return mst.row, mst.col


class TopologicalSignatureLoss(torch.nn.Module):
    r"""Symmetric ``H_0`` persistence-signature loss.

    With ``pi_X`` the MST edge set of the input batch and ``pi_Z`` that of the
    latent batch,

        L = || D_X[pi_X] - D_Z[pi_X] ||^2 + || D_Z[pi_Z] - D_X[pi_Z] ||^2 ,

    both distance matrices being scale-normalised first.  Gradients flow
    through the selected distances; the (piecewise constant) edge selection is
    treated as a constant, following Moor et al. (2020).
    """

    def __init__(self, normalise: bool = True):
        super().__init__()
        self.normalise = normalise

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        Dx, Dz = _pairwise(x), _pairwise(z)
        if self.normalise:
            Dx = Dx / Dx.max().clamp_min(1e-8)
            Dz = Dz / Dz.max().clamp_min(1e-8)
        rx, cx = _mst_edges(Dx)
        rz, cz = _mst_edges(Dz)
        lx = ((Dx[rx, cx] - Dz[rx, cx]) ** 2).sum()
        lz = ((Dz[rz, cz] - Dx[rz, cz]) ** 2).sum()
        n = max(len(rx) + len(rz), 1)
        return (lx + lz) / n


# ---------------------------------------------------------------------------
# summaries used in figures
# ---------------------------------------------------------------------------
def barcode_summary(dgms: list[np.ndarray], top: int = 6) -> dict:
    """Longest ``top`` lifetimes per homological dimension."""
    return {f"H{d}": np.sort(lifetimes(dgm))[::-1][:top].tolist() for d, dgm in enumerate(dgms)}

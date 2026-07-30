"""Simulated entorhinal grid-cell module with toroidal latent geometry.

Biological setting
------------------
Grid cells of a single module fire on a common triangular lattice and differ
only in their spatial phase (Hafting et al., 2005).  Because the lattice is
periodic, the population activity of one module is a function of the
two-dimensional *phase* alone, so the population manifold is a torus -- a
prediction confirmed directly in recorded data (Gardner et al., 2022).

Generative model
----------------
Let ``k_1, k_2, k_3`` be three wave vectors 60 degrees apart with common
magnitude ``4 pi / (sqrt(3) lambda)``.  Since ``k_3 = k_2 - k_1``, the rate of
cell ``i`` at position ``u`` depends on ``u`` only through the phase pair

    theta = K u,     K = [k_1; k_2] in R^{2x2},

and is ``2 pi``-periodic in each component.  We therefore treat the *phase*
as the generative latent and write

    x_c = f( T_c(theta) ),      f(theta + 2 pi n) = f(theta)  for n in Z^2.

Context actions
---------------
* **Translation.**  Translating the environment by ``d`` shifts every phase by
  ``K d``: ``T(theta) = theta + delta``.  These form the abelian group ``T^2``
  and *do* descend to the latent torus, so they are exactly representable as a
  latent gauge transformation.  Coherent phase shifts across environments are
  observed experimentally (Fyhn et al., 2007).
* **Lattice rotation by 60 degrees.**  ``T(theta) = M theta`` with
  ``M = K R_{60} K^{-1}`` an *integer* matrix; it is a genuine torus
  automorphism and generates ``Z_6``.  Together with translations it gives the
  non-abelian group ``T^2 x| Z_6``.
* **Rescaling (grid gain change).**  ``T(theta) = s theta`` with non-integer
  ``s`` (Barry et al., 2007) is well defined on the universal cover ``R^2`` but
  *not* on the torus ``R^2 / 2 pi Z^2``.  We include it as a deliberate
  negative control: the theory predicts it cannot be captured by any latent
  gauge transformation of the population manifold, and the experiments confirm
  this.
"""

from __future__ import annotations

import numpy as np

from .base import ContextSpec, ContextualDataset, add_observation_noise


def wave_vectors(spacing: float, orientation: float) -> np.ndarray:
    """Three hexagonal wave vectors, shape (3, 2).

    ``k_3 = k_2 - k_1`` holds exactly, which is what makes the single-module
    population activity a function of two phases only.
    """
    k = 4.0 * np.pi / (np.sqrt(3.0) * spacing)
    angles = orientation + np.array([0.0, np.pi / 3.0, 2.0 * np.pi / 3.0])
    return k * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def grid_rates(
    theta: np.ndarray,
    phases: np.ndarray,
    peak: np.ndarray,
    sharpness: float = 4.0,
    baseline: float = 0.05,
) -> np.ndarray:
    """Grid firing rates as a function of the phase pair ``theta`` (N, 2).

    Uses the standard three-cosine construction passed through an exponential
    non-linearity, which produces sharply peaked hexagonal fields:

        g_i = exp( beta * ( mean_j cos(psi_j) - 1 ) ),
        psi_1 = th1 - ph1,  psi_2 = th2 - ph2,  psi_3 = psi_2 - psi_1 .
    """
    d = theta[:, None, :] - phases[None, :, :]           # (N, cells, 2)
    c1, c2 = np.cos(d[..., 0]), np.cos(d[..., 1])
    c3 = np.cos(d[..., 1] - d[..., 0])
    g = (c1 + c2 + c3) / 3.0
    return peak[None, :] * np.exp(sharpness * (g - 1.0)) + baseline


def _integer_rotation_matrix(spacing: float, orientation: float) -> np.ndarray:
    """``M = K R_60 K^{-1}``; exactly integer for a hexagonal lattice."""
    K = wave_vectors(spacing, orientation)[:2]           # (2, 2)
    c, s = np.cos(np.pi / 3.0), np.sin(np.pi / 3.0)
    R = np.array([[c, -s], [s, c]])
    M = K @ R @ np.linalg.inv(K)
    return np.round(M)                                   # snap tiny numerical error


def build_contexts(
    kind: str,
    spacing: float,
    orientation: float,
    n_translations: int = 4,
    rng: np.random.Generator | None = None,
) -> list[ContextSpec]:
    """Assemble the context set.

    ``kind`` is one of ``"translation"`` (abelian ``T^2``),
    ``"translation+rotation"`` (non-abelian ``T^2 x| Z_6``) or ``"all"``
    (adds the non-integer rescaling negative control).
    """
    rng = rng or np.random.default_rng(0)
    specs: list[ContextSpec] = [
        ContextSpec(
            name="phase 0 (reference)",
            features=np.zeros(0),
            matrix=np.eye(2),
            offset=np.zeros(2),
            group_params={"delta1": 0.0, "delta2": 0.0, "rot60": 0.0, "scale": 1.0},
        )
    ]
    # Deterministic, well spread phase offsets (avoids near-degenerate draws).
    base = np.array([[0.7, 0.35], [-0.55, 0.9], [0.25, -0.8], [1.15, 1.05],
                     [-1.0, -0.45], [0.45, 1.4]]) * np.pi
    for i in range(n_translations):
        d = base[i % len(base)]
        specs.append(
            ContextSpec(
                name=f"translation {i + 1}",
                features=np.zeros(0),
                matrix=np.eye(2),
                offset=d,
                group_params={"delta1": float(d[0]), "delta2": float(d[1]), "rot60": 0.0, "scale": 1.0},
            )
        )

    if kind in ("translation+rotation", "all"):
        M = _integer_rotation_matrix(spacing, orientation)
        specs.append(
            ContextSpec(
                name="lattice rotation 60deg",
                features=np.zeros(0),
                matrix=M,
                offset=np.zeros(2),
                group_params={"delta1": 0.0, "delta2": 0.0, "rot60": 1.0, "scale": 1.0},
            )
        )
        d = base[0] * 0.6
        specs.append(
            ContextSpec(
                name="rotation + translation",
                features=np.zeros(0),
                matrix=M,
                offset=d,
                group_params={"delta1": float(d[0]), "delta2": float(d[1]), "rot60": 1.0, "scale": 1.0},
            )
        )

    if kind == "all":
        s = 1.37                                          # non-integer on purpose
        specs.append(
            ContextSpec(
                name="grid rescaling (control)",
                features=np.zeros(0),
                matrix=s * np.eye(2),
                offset=np.zeros(2),
                group_params={"delta1": 0.0, "delta2": 0.0, "rot60": 0.0, "scale": s},
            )
        )

    k = len(specs)
    for i, spec in enumerate(specs):
        f = np.zeros(k)
        f[i] = 1.0
        spec.features = f
    return specs


def simulate_grid_cells(
    n_cells: int = 100,
    n_samples: int = 3000,
    n_modules: int = 1,
    spacing: float = 0.42,
    orientation: float = 0.21,
    module_spacing_ratio: float = 1.42,
    sharpness: float = 4.0,
    peak_rate: float = 15.0,
    rate_jitter: float = 0.25,
    baseline: float = 0.05,
    noise: float = 0.15,
    noise_kind: str = "gaussian",
    arena: float = 1.5,
    context_kind: str = "translation+rotation",
    n_translations: int = 4,
    phase_sampling: str = "trajectory",
    seed: int = 0,
) -> ContextualDataset:
    """Build a paired multi-context grid-cell dataset.

    ``phase_sampling='trajectory'`` runs a foraging random walk through a square
    arena of side ``2*arena`` and reads off the induced phases, giving the
    correlated sampling of real recordings; ``'uniform'`` samples phases
    directly and uniformly on the torus.

    With ``n_modules > 1`` the population manifold is a product of tori; the
    reported homology is then that of ``T^{2 n_modules}``.  All experiments in
    the paper use a single module, matching Gardner et al. (2022).
    """
    rng = np.random.default_rng(seed)

    Ks = []
    for m in range(n_modules):
        lam = spacing * (module_spacing_ratio ** m)
        Ks.append(wave_vectors(lam, orientation + 0.11 * m)[:2])

    if phase_sampling == "trajectory":
        u = _random_walk_square(n_samples, arena, rng)
        theta_ref = u @ Ks[0].T
    else:
        theta_ref = rng.uniform(-np.pi, np.pi, size=(n_samples, 2))
        u = theta_ref @ np.linalg.inv(Ks[0]).T

    contexts = build_contexts(context_kind, spacing, orientation, n_translations, rng)

    cells_per_module = [n_cells // n_modules] * n_modules
    cells_per_module[0] += n_cells - sum(cells_per_module)
    phases, peaks = [], []
    for m in range(n_modules):
        nm = cells_per_module[m]
        phases.append(rng.uniform(-np.pi, np.pi, size=(nm, 2)))
        peaks.append(peak_rate * np.exp(rng.normal(0, rate_jitter, size=nm)))

    acts = []
    for spec in contexts:
        theta_c = spec.apply(theta_ref)                    # action on module-0 phase
        per_mod = []
        for m in range(n_modules):
            if m == 0:
                th = theta_c
            else:
                # Other modules see the same *physical* transformation, mapped
                # through their own lattice: theta_m = K_m K_0^{-1} theta_0.
                th = theta_c @ (Ks[m] @ np.linalg.inv(Ks[0])).T
            per_mod.append(grid_rates(th, phases[m], peaks[m], sharpness, baseline))
        acts.append(np.concatenate(per_mod, axis=1))
    activity = np.stack(acts, axis=1)
    activity = add_observation_noise(activity, noise, noise_kind, rng)

    ctx_feats = np.stack([s.features for s in contexts], axis=0)
    # Ground-truth latent is the *wrapped* phase: that is what the population
    # manifold actually encodes.
    wrapped = np.mod(theta_ref + np.pi, 2 * np.pi) - np.pi
    return ContextualDataset(
        activity=activity.astype(np.float32),
        latent=wrapped.astype(np.float32),
        context_features=ctx_feats.astype(np.float32),
        contexts=contexts,
        reference=0,
        meta={
            "simulation": "grid_cells",
            "K": Ks[0],
            "phases": phases[0],
            "spacing": spacing,
            "orientation": orientation,
            "position": u,
            "latent_topology": "torus",
            "true_latent_dim": 2,
            "betti_true": [1, 2, 1] if n_modules == 1 else None,
            "n_modules": n_modules,
        },
    )


def _random_walk_square(n: int, half: float, rng: np.random.Generator) -> np.ndarray:
    pos = np.zeros((n, 2))
    p = np.zeros(2)
    v = rng.normal(0, 1, 2)
    v /= np.linalg.norm(v) + 1e-9
    speed = 0.05 * half
    for t in range(n):
        v = v + rng.normal(0, 0.5, 2)
        v /= np.linalg.norm(v) + 1e-9
        p = p + speed * v
        for k in range(2):
            if abs(p[k]) > half:
                p[k] = np.sign(p[k]) * (2 * half - abs(p[k]))
                v[k] = -v[k]
        pos[t] = p
    return pos


def torus_embedding(theta: np.ndarray) -> np.ndarray:
    """Canonical isometric-in-spirit embedding ``T^2 -> R^4``.

    Under this embedding a phase translation ``theta -> theta + delta`` acts as
    the block rotation ``R(delta_1) (+) R(delta_2)``, i.e. an element of the
    maximal torus of ``SO(4)``.  This is the concrete sense in which the
    translation group is *linearly* representable in a four-dimensional latent
    space, and is the prediction tested in Experiment 2.
    """
    return np.stack(
        [np.cos(theta[:, 0]), np.sin(theta[:, 0]), np.cos(theta[:, 1]), np.sin(theta[:, 1])],
        axis=1,
    ) / np.sqrt(2.0)


def block_rotation(delta: np.ndarray) -> np.ndarray:
    """The 4x4 matrix implementing a phase translation in the R^4 embedding."""
    out = np.zeros((4, 4))
    for j in range(2):
        c, s = np.cos(delta[j]), np.sin(delta[j])
        out[2 * j: 2 * j + 2, 2 * j: 2 * j + 2] = [[c, -s], [s, c]]
    return out


def spatial_rate_map(
    dataset: ContextualDataset, cell: int, context: int, bins: int = 40, smooth: float = 0.8
):
    """Rate map over *physical space* for a grid cell (for figure panels)."""
    from scipy.ndimage import gaussian_filter

    u = dataset.meta["position"]
    half = np.abs(u).max()
    r = dataset.activity[:, context, cell]
    edges = np.linspace(-half, half, bins + 1)
    occ, _, _ = np.histogram2d(u[:, 0], u[:, 1], bins=[edges, edges])
    tot, _, _ = np.histogram2d(u[:, 0], u[:, 1], bins=[edges, edges], weights=r)
    if smooth > 0:
        occ, tot = gaussian_filter(occ, smooth), gaussian_filter(tot, smooth)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(occ > 1e-6, tot / occ, np.nan)
    return m.T, np.array([-half, half, -half, half])

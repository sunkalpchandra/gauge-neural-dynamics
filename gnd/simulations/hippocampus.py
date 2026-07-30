"""Simulated hippocampal place-cell population with context-dependent remapping.

Biological setting
------------------
A rodent forages in a cylindrical arena.  A population of ``n_cells`` CA1
place cells tiles the arena with Gaussian firing fields

    r_i(u) = A_i * exp( -||u - mu_i||^2 / (2 sigma_i^2) ) + b_i .

Rotating the polarising cue card rotates the entire place-cell map coherently
(Muller & Kubie, 1987; Bostock et al., 1991), stretching the arena deforms it
(O'Keefe & Burgess, 1996), and morphing between two familiar shapes produces a
graded, non-affine distortion (Wills et al., 2005; Leutgeb et al., 2005).

Generative model
----------------
We take the *generative latent* to be the animal's allocentric position
``u in R^2`` and the *observation map* ``f`` to be the fixed bank of tuning
curves above.  A context ``c`` acts by transforming the coordinate frame in
which the map is expressed:

    x_c = f( T_c(u) ).

Because ``f`` is shared, all context differences are carried by ``T_c`` --
exactly the hypothesis the paper tests.  The reference context has
``T_0 = id``.

The default context set contains four affine contexts, whose ground truth
transformations generate a subgroup of the affine group ``Aff(2)``, plus one
deliberately *non-affine* morph context that lies outside the reach of the
linear gauge model and is used to probe its failure mode.
"""

from __future__ import annotations

import numpy as np

from .base import ContextSpec, ContextualDataset, add_observation_noise


# ---------------------------------------------------------------------------
# tuning curves
# ---------------------------------------------------------------------------
def _sunflower_centres(n: int, radius: float, rng: np.random.Generator, jitter: float) -> np.ndarray:
    """Approximately uniform points on a disc (Vogel / sunflower spiral).

    Gives a much more even tiling than rejection sampling at these population
    sizes, which keeps the population-vector geometry well conditioned.
    """
    idx = np.arange(1, n + 1, dtype=float)
    r = radius * np.sqrt(idx / n)
    theta = idx * np.pi * (3.0 - np.sqrt(5.0))
    pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    pts = pts + rng.normal(0.0, jitter * radius, size=pts.shape)
    return pts


def place_field_rates(
    u: np.ndarray,
    centres: np.ndarray,
    sigma: np.ndarray,
    peak: np.ndarray,
    baseline: float = 0.0,
) -> np.ndarray:
    """Gaussian place fields evaluated at positions ``u`` (N, 2)."""
    d2 = ((u[:, None, :] - centres[None, :, :]) ** 2).sum(-1)  # (N, cells)
    return peak[None, :] * np.exp(-d2 / (2.0 * sigma[None, :] ** 2)) + baseline


# ---------------------------------------------------------------------------
# ground-truth context actions
# ---------------------------------------------------------------------------
def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _shear_scale(sx: float, sy: float, k: float, theta: float = 0.0) -> np.ndarray:
    return _rot(theta) @ np.array([[sx, k], [0.0, sy]])


def default_contexts(radius: float, morph: bool = True) -> list[ContextSpec]:
    """Five contexts: identity, two rotations, an affine distortion, a morph.

    The one-hot ``features`` vector is the only context information the model
    receives -- it must infer the geometry, not read it off.
    """
    specs: list[ContextSpec] = []
    definitions = [
        ("A: familiar", np.eye(2), np.zeros(2), {"theta": 0.0, "det": 1.0}),
        ("B: cue rotated 60deg", _rot(np.pi / 3), np.zeros(2), {"theta": np.pi / 3, "det": 1.0}),
        (
            "C: rotated -40deg + shift",
            _rot(-0.7),
            np.array([0.10, -0.06]) * radius,
            {"theta": -0.7, "det": 1.0},
        ),
        (
            "D: stretched + sheared",
            _shear_scale(1.25, 0.80, 0.22),
            np.zeros(2),
            {"theta": 0.0, "det": 1.0},
        ),
    ]
    for name, M, b, gp in definitions:
        gp = dict(gp)
        gp["det"] = float(np.linalg.det(M))
        specs.append(ContextSpec(name=name, features=np.zeros(0), matrix=M, offset=b, group_params=gp))

    if morph:
        def _morph(u: np.ndarray, R: float = radius) -> np.ndarray:
            """Radially graded twist: rotation angle grows with eccentricity.

            This is a diffeomorphism of the disc that is *not* an affine map,
            mimicking the graded distortion seen in morph-box experiments.
            """
            r = np.linalg.norm(u, axis=1, keepdims=True)
            ang = 0.9 * (r / R) ** 2
            c, s = np.cos(ang), np.sin(ang)
            return np.concatenate([c * u[:, :1] - s * u[:, 1:], s * u[:, :1] + c * u[:, 1:]], axis=1)

        specs.append(
            ContextSpec(
                name="E: non-affine morph",
                features=np.zeros(0),
                warp=_morph,
                group_params={"nonaffine": 1.0},
            )
        )

    # one-hot identity codes
    k = len(specs)
    for i, spec in enumerate(specs):
        f = np.zeros(k)
        f[i] = 1.0
        spec.features = f
    return specs


# ---------------------------------------------------------------------------
# dataset builder
# ---------------------------------------------------------------------------
def rotation_family_contexts(
    n_rotations: int = 12,
    span: float = 2 * np.pi,
    feature_mode: str = "circular",
) -> list[ContextSpec]:
    r"""A one-parameter family of cue rotations with an *observable* cue variable.

    The context variable handed to the model is the cue itself, not an arbitrary
    label, so contexts become points in a continuous context space.  Two things
    then become testable that discrete labels cannot express: whether the learned
    gauge field *interpolates* to cue angles never seen in training, and whether
    the field can be defined globally at all.

    ``span`` sets the extent of the family.  ``span < 2 pi`` gives a
    contractible arc.  ``span = 2 pi`` closes the loop, and then the geometry
    bites: the required group elements ``R(alpha)`` trace a loop of winding
    number one in ``GL(d)^+``, whereas ``exp(sum_k theta_k(c) G_k)`` with a
    continuous ``theta`` is always null-homotopic.  No single-chart field of this
    form can therefore cover the whole circle.

    ``feature_mode`` selects the domain.  ``"circular"`` presents
    ``(cos alpha, sin alpha)``, i.e. context space *is* the circle, and the
    obstruction applies.  ``"lifted"`` presents the unwrapped angle
    ``alpha / 2 pi``, i.e. the universal cover, on which the same architecture is
    unobstructed.  Comparing the two isolates the topological effect from every
    other difference.
    """
    if n_rotations < 2:
        raise ValueError("need at least two rotations")
    closed = abs(span - 2 * np.pi) < 1e-6
    alphas = (
        np.arange(n_rotations) * span / n_rotations
        if closed
        else np.linspace(-span / 2, span / 2, n_rotations)
    )
    specs = []
    for a in alphas:
        if feature_mode == "circular":
            f = np.array([np.cos(a), np.sin(a)])
        elif feature_mode == "lifted":
            f = np.array([a / (2 * np.pi)])
        else:
            raise ValueError(f"unknown feature_mode {feature_mode!r}")
        specs.append(
            ContextSpec(
                name=f"cue {np.degrees(a):+.0f}deg",
                features=f,
                matrix=_rot(a),
                offset=np.zeros(2),
                group_params={"theta": float(a), "det": 1.0},
            )
        )
    return specs


def simulate_place_cells(
    n_cells: int = 120,
    n_samples: int = 3000,
    radius: float = 1.0,
    field_sigma: float = 0.22,
    sigma_jitter: float = 0.25,
    peak_rate: float = 12.0,
    rate_jitter: float = 0.3,
    baseline: float = 0.2,
    centre_jitter: float = 0.03,
    noise: float = 0.15,
    noise_kind: str = "gaussian",
    trajectory: bool = True,
    morph_context: bool = True,
    rate_remap: float = 0.0,
    context_mode: str = "discrete",
    n_rotations: int = 12,
    context_span: float = 2 * np.pi,
    context_feature_mode: str = "circular",
    seed: int = 0,
) -> ContextualDataset:
    """Build a paired multi-context place-cell dataset.

    Parameters
    ----------
    trajectory:
        If ``True`` sample positions from a smooth, boundary-reflecting random
        walk (a foraging trajectory); otherwise sample i.i.d. uniformly on the
        disc.  Trajectory sampling produces the temporally correlated,
        non-uniform occupancy typical of real recordings.
    rate_remap:
        Fraction of per-context multiplicative gain jitter applied to each
        cell's peak rate.  ``0`` gives pure "coherent" remapping; positive
        values superimpose *rate remapping*, which the shared-``f`` assumption
        does not cover and which therefore acts as model mis-specification.
    noise:
        Observation noise level; see :func:`gnd.simulations.base.add_observation_noise`.
    """
    rng = np.random.default_rng(seed)

    # Fields tile a slightly larger disc so that transformed positions remain
    # inside the covered region.
    field_radius = radius * 1.45
    centres = _sunflower_centres(n_cells, field_radius, rng, centre_jitter)
    sigma = field_sigma * np.exp(rng.normal(0.0, sigma_jitter, size=n_cells))
    peak = peak_rate * np.exp(rng.normal(0.0, rate_jitter, size=n_cells))

    if trajectory:
        u = _random_walk_disc(n_samples, radius, rng)
    else:
        ang = rng.uniform(0, 2 * np.pi, n_samples)
        rad = radius * np.sqrt(rng.uniform(0, 1, n_samples))
        u = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)

    if context_mode == "rotation_family":
        contexts = rotation_family_contexts(n_rotations, context_span, context_feature_mode)
    elif context_mode == "discrete":
        contexts = default_contexts(radius, morph=morph_context)
    else:
        raise ValueError(f"unknown context_mode {context_mode!r}")
    acts = []
    for ci, spec in enumerate(contexts):
        u_c = spec.apply(u)
        pk = peak
        if rate_remap > 0 and ci > 0:
            pk = peak * np.exp(rng.normal(0.0, rate_remap, size=n_cells))
        acts.append(place_field_rates(u_c, centres, sigma, pk, baseline))
    activity = np.stack(acts, axis=1)  # (N, C, cells)
    activity = add_observation_noise(activity, noise, noise_kind, rng)

    ctx_feats = np.stack([s.features for s in contexts], axis=0)
    ref = int(np.argmin([abs(s.group_params.get("theta", 0.0)) for s in contexts]))
    return ContextualDataset(
        activity=activity.astype(np.float32),
        latent=u.astype(np.float32),
        context_features=ctx_feats.astype(np.float32),
        contexts=contexts,
        reference=ref,
        meta={
            "simulation": "hippocampus",
            "centres": centres,
            "sigma": sigma,
            "peak": peak,
            "radius": radius,
            "field_radius": field_radius,
            "latent_topology": "disc",
            "true_latent_dim": 2,
        },
    )


def _random_walk_disc(n: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    """Momentum random walk on a disc with specular reflection at the wall."""
    pos = np.zeros((n, 2))
    p = rng.normal(0, 0.2 * radius, size=2)
    v = rng.normal(0, 1, size=2)
    v /= np.linalg.norm(v) + 1e-9
    speed = 0.045 * radius
    for t in range(n):
        v = v + rng.normal(0, 0.55, size=2)
        v /= np.linalg.norm(v) + 1e-9
        cand = p + speed * v
        r = np.linalg.norm(cand)
        if r > radius:
            nrm = cand / (r + 1e-9)
            v = v - 2 * (v @ nrm) * nrm          # reflect
            v /= np.linalg.norm(v) + 1e-9
            cand = p + speed * v
            if np.linalg.norm(cand) > radius:    # fallback: project inside
                cand = 0.98 * radius * cand / (np.linalg.norm(cand) + 1e-9)
        p = cand
        pos[t] = p
    return pos


def rate_map(
    dataset: ContextualDataset,
    cell: int,
    context: int,
    bins: int = 36,
    smooth: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Occupancy-normalised rate map of one cell in one context.

    Returns ``(map, extent)`` where ``map`` is ``(bins, bins)`` with NaN in
    unvisited bins.  This is the standard way experimentalists visualise place
    fields, and is what Figure 1 shows.
    """
    from scipy.ndimage import gaussian_filter

    radius = dataset.meta["radius"]
    u = dataset.latent
    r = dataset.activity[:, context, cell]
    edges = np.linspace(-radius, radius, bins + 1)
    occ, _, _ = np.histogram2d(u[:, 0], u[:, 1], bins=[edges, edges])
    tot, _, _ = np.histogram2d(u[:, 0], u[:, 1], bins=[edges, edges], weights=r)
    if smooth > 0:
        occ = gaussian_filter(occ, smooth)
        tot = gaussian_filter(tot, smooth)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(occ > 1e-6, tot / occ, np.nan)
    return m.T, np.array([-radius, radius, -radius, radius])

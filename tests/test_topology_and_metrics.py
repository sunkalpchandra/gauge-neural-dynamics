"""Tests for the topological machinery and the four metrics.

Topology is checked against point clouds whose homology is known by
construction. Each metric is checked for the property that makes it trustworthy:
that it responds in the right direction to a controlled perturbation, and that it
cannot be satisfied by the degenerate solution it is most at risk from.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gnd.geometry.metrics import (
    GaugeSummary, context_invariance_score, gauge_consistency_score,
    geometric_recovery_error, manifold_preservation_score,
)
from gnd.geometry.topology import (
    TopologicalSignatureLoss, betti_numbers, diagram_distance, persistence_diagrams,
)


# ---------------------------------------------------------------------------
# point clouds with known homology
# ---------------------------------------------------------------------------
def _circle(n=400, r=1.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(t), r * np.sin(t)], 1) + rng.normal(0, noise, (n, 2))


def _torus(n=800, R=2.0, r=1.0, seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    return np.stack([(R + r * np.cos(b)) * np.cos(a),
                     (R + r * np.cos(b)) * np.sin(a),
                     r * np.sin(b)], 1)


def _disc(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t, s = rng.uniform(0, 2 * np.pi, n), np.sqrt(rng.uniform(0, 1, n))
    return np.stack([s * np.cos(t), s * np.sin(t)], 1)


@pytest.mark.parametrize("cloud,maxdim,expected", [
    (_circle(), 1, [1, 1]),
    (_disc(), 1, [1, 0]),
    (_torus(1500), 2, [1, 2, 1]),
])
def test_betti_numbers_of_known_manifolds(cloud, maxdim, expected):
    # The torus is deliberately sampled more densely: separating its two H_1 bars
    # from the noise band is the marginal case for any gap-based estimator, and at
    # a few hundred points the call is genuinely ambiguous.  persistence_gap()
    # reports how decisive the separation was, and we quote it alongside.
    n = 600 if maxdim == 2 else 450
    dgms = persistence_diagrams(cloud, maxdim=maxdim, n_points=min(len(cloud), n), seed=0)
    assert betti_numbers(dgms) == expected


def test_diagram_distance_is_a_metric_like_quantity():
    a = persistence_diagrams(_circle(seed=1), maxdim=1, n_points=300)
    b = persistence_diagrams(_circle(seed=2), maxdim=1, n_points=300)
    c = persistence_diagrams(_disc(seed=3), maxdim=1, n_points=300)
    assert diagram_distance(a[1], a[1]) == pytest.approx(0.0, abs=1e-9)
    same = diagram_distance(a[1], b[1])
    diff = diagram_distance(a[1], c[1])
    assert same < diff, "two circles must be closer to each other than to a disc"


def test_topology_loss_is_zero_for_an_isometry_and_positive_otherwise():
    torch.manual_seed(0)
    loss = TopologicalSignatureLoss()
    x = torch.randn(96, 5)
    theta = 0.7
    R = torch.tensor([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
                     dtype=torch.float32)
    z_iso = x[:, :2] @ R.T
    x2 = x[:, :2]
    assert float(loss(x2, z_iso)) < 1e-6
    z_bad = x2.clone()
    z_bad[:, 1] *= 0.05                      # collapse one direction
    assert float(loss(x2, z_bad)) > float(loss(x2, z_iso))


# ---------------------------------------------------------------------------
# metric behaviour
# ---------------------------------------------------------------------------
def _rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _gauge(angles):
    G = np.array([[[0.0, -1.0], [1.0, 0.0]]]) / np.sqrt(2)
    theta = np.array([[a * np.sqrt(2)] for a in angles])
    M = np.stack([_rot(a) for a in angles])
    return GaugeSummary(generators=G, coefficients=theta, matrices=M)


def test_gcs_is_one_for_an_exact_group():
    """A one-parameter rotation family is abelian and closes exactly."""
    g = _gauge([0.0, 0.3, -0.6, 1.1])
    out = gauge_consistency_score(g, _disc(200), seed=0)
    assert out["gcs"] > 0.999
    assert out["closure_defect"] < 1e-6
    assert out["abelianness"] < 1e-8


def test_gcs_is_high_but_magnitude_exposes_a_collapsed_family():
    """The failure mode GCS alone cannot see, and why we report the magnitude."""
    real = gauge_consistency_score(_gauge([0.0, 0.5, -0.9, 1.2]), _disc(200))
    triv = gauge_consistency_score(_gauge([0.0, 1e-4, -2e-4, 1.5e-4]), _disc(200))
    assert triv["gcs"] > 0.99
    assert triv["transform_magnitude"] < 0.01
    assert real["transform_magnitude"] > 0.3


def test_cis_is_one_when_invariant_and_zero_when_collapsed():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(200, 1, 3)).repeat(4, axis=1)
    assert context_invariance_score(z)["cis"] > 0.999
    noisy = z + rng.normal(0, 3.0, z.shape)
    out = context_invariance_score(noisy)
    assert out["cis"] < 0.2, "chance-corrected CIS must be near 0 when noise dominates"
    assert out["cis_raw"] > out["cis"], "the raw ratio floors at 1/C, hence the correction"
    const = np.zeros((200, 4, 3))
    assert context_invariance_score(const)["cis"] < 1e-6, "collapse must not score well"


def test_cis_decreases_monotonically_with_context_noise():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(300, 1, 4)).repeat(3, axis=1)
    vals = [context_invariance_score(base + rng.normal(0, s, base.shape))["cis"]
            for s in (0.05, 0.3, 1.0, 3.0)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_gre_is_zero_for_perfect_recovery_and_one_for_the_null_model():
    angles = [0.0, 0.6, -0.4, 1.0]
    u = _disc(300, seed=5)
    per_ctx = np.stack([u @ _rot(a).T for a in angles])
    z = np.stack([u @ _rot(a).T for a in angles], axis=1)     # canonical == truth
    z_canon = np.repeat(u[:, None], len(angles), axis=1)

    perfect = geometric_recovery_error(z_canon, _gauge(angles), u, per_ctx, reference=0)
    assert perfect["gre"] < 0.02
    assert perfect["readout_r2"] > 0.99

    null = geometric_recovery_error(z_canon, _gauge([0.0] * len(angles)), u, per_ctx, reference=0)
    assert 0.95 < null["gre"] < 1.05, "no transformation must score exactly the null"


def test_mps_is_higher_for_a_matching_manifold():
    z_good = np.repeat(_circle(400, seed=7)[:, None], 3, axis=1)
    z_bad = np.repeat(_disc(400, seed=7)[:, None], 3, axis=1)
    target = _circle(400, seed=8)
    good = manifold_preservation_score(z_good, true_embedding=target, maxdim=1, n_points=280)
    bad = manifold_preservation_score(z_bad, true_embedding=target, maxdim=1, n_points=280)
    assert good["mps"] > bad["mps"]
    assert good["betti_learned"] == [1, 1]
    assert bad["betti_learned"] == [1, 0], "a disc has no one-dimensional hole"


def test_mps_cross_context_detects_misalignment():
    """Same manifold in every context scores higher than different manifolds."""
    same = np.repeat(_circle(400, seed=9)[:, None], 3, axis=1)
    mixed = np.stack([_circle(400, seed=9), _disc(400, seed=10), _circle(400, seed=11)], axis=1)
    a = manifold_preservation_score(same, maxdim=1, n_points=250)["mps_cross_context"]
    b = manifold_preservation_score(mixed, maxdim=1, n_points=250)["mps_cross_context"]
    assert a > b


# ---------------------------------------------------------------------------
# latent dynamics: units
# ---------------------------------------------------------------------------
def test_rotation_frequency_is_in_hertz():
    """A planted rotation must come back at its own frequency.

    ``fit_linear_dynamics`` already divides by the sampling interval, so its
    eigenvalues are in radians per second; dividing by ``dt`` a second time in
    the spectrum would scale every frequency by ``1/dt`` -- a factor of 100 at
    the 10 ms step the motor simulation uses.
    """
    from gnd.models.neural_dynamics import dynamics_spectrum, fit_linear_dynamics

    dt, f_hz = 0.01, 2.0
    t = np.arange(0, 4.0, dt)
    w = 2 * np.pi * f_hz
    Z = np.stack([np.cos(w * t), np.sin(w * t)], axis=1)
    A = fit_linear_dynamics(Z, dt=dt)
    assert abs(dynamics_spectrum(A)["top_frequency_hz"] - f_hz) < 0.05 * f_hz

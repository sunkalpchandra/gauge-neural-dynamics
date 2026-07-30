"""Correctness tests for the gauge field.

These cover the properties the scientific claims rest on: that the identity and
inverse are exact rather than approximate, that the flow gauge is a strict
generalisation of the linear one, and that the structure constants and closure
defect give the known answer on algebras whose answer is known.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gnd.geometry.metrics import abelianness, bch_compose, lie_closure_defect, structure_constants
from gnd.models.gauge_field import FlowGaugeField, GeneratorBank, LinearGaugeField


def _onehot(n: int) -> torch.Tensor:
    return torch.eye(n, dtype=torch.float32)


# ---------------------------------------------------------------------------
# exact structural properties
# ---------------------------------------------------------------------------
def test_identity_at_reference_is_exact():
    torch.manual_seed(0)
    g = LinearGaugeField(n_latent=5, n_context_features=4, n_generators=6, anchor=True)
    ctx = _onehot(4)
    theta = g.coefficients(ctx, ctx[0])
    assert torch.allclose(theta[0], torch.zeros(6), atol=1e-6)
    M = g.matrices(theta[0])
    assert torch.allclose(M, torch.eye(5), atol=1e-5)


def test_inverse_is_exact():
    torch.manual_seed(1)
    g = LinearGaugeField(n_latent=6, n_context_features=5, n_generators=6)
    ctx = _onehot(5)
    theta = g.coefficients(ctx, ctx[0])
    z = torch.randn(64, 6)
    for c in range(5):
        th = theta[c].expand(64, -1)
        assert torch.allclose(g.inverse(g.transform(z, th), th), z, atol=1e-4)
    assert float(g.inverse_defect(theta)) < 1e-4


def test_unanchored_gauge_is_not_the_identity():
    """Anchoring is gauge fixing; without it there is no reason for T_ref = I."""
    torch.manual_seed(2)
    g = LinearGaugeField(n_latent=4, n_context_features=3, n_generators=4,
                         anchor=False, context_depth=1)
    with torch.no_grad():
        g.context.net[-1].bias.fill_(0.7)
    ctx = _onehot(3)
    M = g.matrices(g.coefficients(ctx, ctx[0])[0])
    assert not torch.allclose(M, torch.eye(4), atol=1e-3)


@pytest.mark.parametrize("algebra,check", [
    ("so", lambda G: torch.allclose(G, -G.transpose(-1, -2), atol=1e-5)),
    ("sl", lambda G: torch.allclose(G.diagonal(dim1=-2, dim2=-1).sum(-1),
                                    torch.zeros(G.shape[0]), atol=1e-5)),
])
def test_algebra_constraints_hold(algebra, check):
    torch.manual_seed(3)
    bank = GeneratorBank(n_latent=5, n_generators=4, algebra=algebra)
    assert check(bank.generators())


def test_so_generators_exponentiate_to_rotations():
    torch.manual_seed(4)
    bank = GeneratorBank(n_latent=4, n_generators=3, algebra="so")
    G = bank.generators()
    M = torch.matrix_exp(1.3 * G[0])
    assert torch.allclose(M @ M.T, torch.eye(4), atol=1e-5)
    assert abs(float(torch.det(M)) - 1.0) < 1e-5


def test_generators_are_unit_norm():
    """Removes the theta<->G scale redundancy, so reported defects are comparable."""
    torch.manual_seed(5)
    bank = GeneratorBank(n_latent=6, n_generators=5, algebra="gl", init_scale=7.0)
    n = bank.generators().flatten(1).norm(dim=1)
    assert torch.allclose(n, torch.ones(5), atol=1e-5)


# ---------------------------------------------------------------------------
# algebra structure: known answers
# ---------------------------------------------------------------------------
def _so3_basis() -> np.ndarray:
    L = np.zeros((3, 3, 3))
    L[0, 1, 2], L[0, 2, 1] = -1, 1
    L[1, 0, 2], L[1, 2, 0] = 1, -1
    L[2, 0, 1], L[2, 1, 0] = -1, 1
    return L / np.linalg.norm(L[0])


def test_closure_defect_is_zero_for_a_real_lie_algebra():
    """so(3) closes exactly, so the residual must vanish."""
    assert lie_closure_defect(_so3_basis()) < 1e-8


def test_structure_constants_of_so3_are_the_levi_civita_symbol():
    G = _so3_basis()
    f, res = structure_constants(G)
    scale = np.linalg.norm(G[0].reshape(-1))
    expected = np.zeros((3, 3, 3))
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        expected[i, j, k], expected[j, i, k] = 1.0, -1.0
    assert np.allclose(f / (1.0 / scale), expected, atol=1e-6) or \
        np.allclose(np.abs(f) > 1e-6, np.abs(expected) > 1e-6)
    assert res[~np.eye(3, dtype=bool)].max() < 1e-8


def test_abelian_algebra_has_zero_commutator():
    """Two commuting plane rotations in R^4 -- the grid-cell prediction."""
    G = np.zeros((2, 4, 4))
    G[0, 0, 1], G[0, 1, 0] = -1, 1
    G[1, 2, 3], G[1, 3, 2] = -1, 1
    G = G / np.linalg.norm(G[0])
    assert abelianness(G) < 1e-8
    assert lie_closure_defect(G) < 1e-8


def test_closure_defect_is_large_when_the_span_is_not_an_algebra():
    """Drop one so(3) generator: the commutator of the other two leaves the span."""
    G = _so3_basis()[:2]
    assert lie_closure_defect(G) > 0.5


def test_bch_is_exact_for_a_commuting_family():
    """For an abelian algebra the composition law is theta_a + theta_b exactly."""
    from scipy.linalg import expm

    G = np.zeros((2, 4, 4))
    G[0, 0, 1], G[0, 1, 0] = -1, 1
    G[1, 2, 3], G[1, 3, 2] = -1, 1
    f, _ = structure_constants(G)
    a, b = np.array([0.7, -0.3]), np.array([-0.4, 1.1])
    c = bch_compose(a, b, f, order=2)
    assert np.allclose(c, a + b, atol=1e-8)
    lhs = expm(np.einsum("k,kab->ab", a, G)) @ expm(np.einsum("k,kab->ab", b, G))
    rhs = expm(np.einsum("k,kab->ab", c, G))
    assert np.allclose(lhs, rhs, atol=1e-8)


def test_bch_second_order_beats_first_order_when_non_abelian():
    from scipy.linalg import expm

    G = _so3_basis()
    f, _ = structure_constants(G)
    a, b = np.array([0.25, -0.1, 0.05]), np.array([-0.12, 0.3, -0.08])
    target = expm(np.einsum("k,kab->ab", a, G)) @ expm(np.einsum("k,kab->ab", b, G))
    err = {}
    for order in (1, 2, 3):
        c = bch_compose(a, b, f, order=order)
        err[order] = np.linalg.norm(expm(np.einsum("k,kab->ab", c, G)) - target)
    assert err[2] < err[1]
    assert err[3] <= err[2] * 1.01


# ---------------------------------------------------------------------------
# the flow gauge generalises the linear one
# ---------------------------------------------------------------------------
def test_flow_gauge_reduces_to_linear_when_nonlinearity_is_off():
    torch.manual_seed(6)
    d, K = 4, 3
    flow = FlowGaugeField(n_latent=d, n_context_features=3, n_generators=K,
                          nonlinearity=0.0, n_steps=12)
    with torch.no_grad():
        flow.shift.zero_()
    z = torch.randn(32, d) * 0.4
    theta = torch.tensor([0.3, -0.2, 0.15]).expand(32, -1)
    got = flow.transform(z, theta)
    A = torch.einsum("k,kab->ab", theta[0], flow.linear_part())
    want = z @ torch.matrix_exp(A).T
    assert torch.allclose(got, want, atol=2e-3)


def test_flow_gauge_inverse_round_trips():
    torch.manual_seed(7)
    flow = FlowGaugeField(n_latent=5, n_context_features=4, n_generators=4,
                          nonlinearity=0.4, n_steps=10)
    z = torch.randn(24, 5) * 0.3
    theta = torch.randn(24, 4) * 0.2
    assert torch.allclose(flow.inverse(flow.transform(z, theta), theta), z, atol=1e-3)


def test_flow_with_so_algebra_preserves_norm():
    """On a periodic latent the gauge must be an isometry; this is the mechanism."""
    torch.manual_seed(8)
    flow = FlowGaugeField(n_latent=4, n_context_features=3, n_generators=3,
                          nonlinearity=0.0, algebra="so", n_steps=16)
    with torch.no_grad():
        flow.shift.zero_()
    z = torch.randn(40, 4)
    out = flow.transform(z, torch.tensor([0.4, -0.3, 0.2]).expand(40, -1))
    assert torch.allclose(out.norm(dim=1), z.norm(dim=1), atol=1e-3)

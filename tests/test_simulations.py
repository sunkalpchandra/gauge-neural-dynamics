"""Tests that the simulations generate what the paper says they generate.

A silent error here would invalidate every downstream claim, because the whole
evaluation rests on knowing the ground-truth transformation. These check the
generative assumption itself: that a single shared tuning map is used in every
context, that the context action is the one we report, and that the latent has
the topology we attribute to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from gnd.simulations.grid_cells import (
    _integer_rotation_matrix, grid_rates, simulate_grid_cells, torus_embedding,
    wave_vectors,
)
from gnd.simulations.hippocampus import (
    place_field_rates, rotation_family_contexts, simulate_place_cells,
)
from gnd.simulations.motor_cortex import ReachTaskConfig, equivariance_defect, simulate_motor_cortex


# ---------------------------------------------------------------------------
# the generative assumption
# ---------------------------------------------------------------------------
def test_place_cells_use_one_shared_tuning_map():
    """x_c = f(T_c u) with f shared: applying T_c by hand must reproduce the data."""
    ds = simulate_place_cells(n_cells=40, n_samples=200, noise=0.0, seed=0)
    m = ds.meta
    for c, spec in enumerate(ds.contexts):
        expected = place_field_rates(spec.apply(ds.latent), m["centres"], m["sigma"],
                                     m["peak"], baseline=0.2)
        assert np.allclose(ds.activity[:, c], expected, atol=1e-5), spec.name


def test_place_cell_contexts_are_the_transformations_we_claim():
    ds = simulate_place_cells(n_cells=20, n_samples=50, noise=0.0, seed=0)
    names = [s.name for s in ds.contexts]
    assert "A: familiar" in names[0]
    assert np.allclose(ds.contexts[0].matrix, np.eye(2))          # reference is identity
    rot = ds.contexts[1]
    assert abs(np.linalg.det(rot.matrix) - 1.0) < 1e-9            # a rotation
    assert np.allclose(rot.matrix @ rot.matrix.T, np.eye(2), atol=1e-9)
    assert abs(np.degrees(np.arctan2(rot.matrix[1, 0], rot.matrix[0, 0])) - 60) < 1e-6
    assert not ds.contexts[-1].is_affine, "the morph context must be non-affine"


def test_morph_context_is_a_diffeomorphism_but_not_affine():
    ds = simulate_place_cells(n_cells=10, n_samples=400, noise=0.0, seed=0)
    morph = ds.contexts[-1]
    u = ds.latent
    v = morph.apply(u)
    # norm preserving (it is a radially graded rotation) but not linear
    assert np.allclose(np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1), atol=1e-9)
    U = np.concatenate([u, np.ones((len(u), 1))], 1)
    resid = np.linalg.lstsq(U, v, rcond=None)[1]
    assert resid.sum() > 1e-3, "a purely affine warp would be fitted exactly"


def test_grid_activity_depends_only_on_phase():
    """This is what makes the population manifold a torus."""
    ds = simulate_grid_cells(n_cells=30, n_samples=600, noise=0.0, context_kind="translation", seed=0)
    th = ds.latent
    # two samples with (nearly) the same wrapped phase must have the same rates
    d = np.abs(th[:, None, :] - th[None, :, :])
    d = np.minimum(d, 2 * np.pi - d).max(-1)
    np.fill_diagonal(d, np.inf)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    assert d[i, j] < 0.05, "need a close phase pair for this test to mean anything"
    scale = ds.activity[:, 0].std()
    assert np.abs(ds.activity[i, 0] - ds.activity[j, 0]).max() < 0.25 * scale


def test_grid_rates_are_periodic_in_phase():
    rng = np.random.default_rng(0)
    phases = rng.uniform(-np.pi, np.pi, size=(12, 2))
    peak = np.ones(12)
    th = rng.uniform(-np.pi, np.pi, size=(40, 2))
    a = grid_rates(th, phases, peak)
    b = grid_rates(th + 2 * np.pi * np.array([[1.0, -2.0]]), phases, peak)
    assert np.allclose(a, b, atol=1e-9)


def test_lattice_rotation_is_an_integer_matrix():
    """Only integer matrices descend to maps of the torus."""
    M = _integer_rotation_matrix(0.42, 0.21)
    assert np.allclose(M, np.round(M))
    assert abs(abs(np.linalg.det(M)) - 1.0) < 1e-9
    assert np.allclose(np.linalg.matrix_power(M, 6), np.eye(2), atol=1e-6), "order 6"


def test_hexagonal_wave_vectors_close():
    """k3 = k2 - k1 is why one module has a two-dimensional phase."""
    k = wave_vectors(0.42, 0.21)
    assert np.allclose(k[2], k[1] - k[0], atol=1e-9)
    assert np.allclose(np.linalg.norm(k, axis=1), np.linalg.norm(k[0]), atol=1e-9)


def test_torus_embedding_turns_translation_into_block_rotation():
    """The prediction tested in Experiment 2."""
    from gnd.simulations.grid_cells import block_rotation

    rng = np.random.default_rng(0)
    th = rng.uniform(-np.pi, np.pi, size=(60, 2))
    delta = np.array([0.7, -1.1])
    lhs = torus_embedding(th + delta)
    rhs = torus_embedding(th) @ block_rotation(delta).T
    assert np.allclose(lhs, rhs, atol=1e-9)


def test_rotation_family_covers_the_circle_and_anchors_at_zero():
    specs = rotation_family_contexts(12, span=2 * np.pi, feature_mode="circular")
    ang = np.array([s.group_params["theta"] for s in specs])
    assert len(specs) == 12
    assert abs(ang[0]) < 1e-12
    assert np.allclose(np.diff(ang), 2 * np.pi / 12)
    lifted = rotation_family_contexts(12, span=2 * np.pi, feature_mode="lifted")
    assert lifted[0].features.shape == (1,), "the cover is one dimensional"
    assert specs[0].features.shape == (2,), "the circle is presented as (cos, sin)"


# ---------------------------------------------------------------------------
# the reaching network
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_reach_rnn_learns_the_task():
    cfg = ReachTaskConfig(n_units=64, train_steps=400)
    ds = simulate_motor_cortex(n_recorded=40, n_trials=2, cfg=cfg, noise=0.0, seed=0)
    curve = np.asarray(ds.meta["rnn_loss_curve"])
    assert curve[-1] < 0.35 * curve[0], "BPTT should reduce the task loss substantially"
    defect = equivariance_defect(ds)
    assert 0.0 < defect["equivariance_residual_mean"] < 1.0, (
        "the circuit should be partly but not perfectly equivariant; that is the "
        "point of training it rather than constructing it"
    )


def test_reaches_are_curved_so_the_reach_plane_is_two_dimensional():
    """Straight reaches would make the recovery metric ill-posed by construction."""
    from gnd.simulations.motor_cortex import _task_tensors

    _, Yt, _, _ = _task_tensors(ReachTaskConfig())
    ref = Yt[0].numpy()
    s = np.linalg.svd(ref - ref.mean(0), compute_uv=False)
    assert s[1] / s[0] > 0.05, "second singular value must be non-negligible"


# ---------------------------------------------------------------------------
# dataset bookkeeping
# ---------------------------------------------------------------------------
def test_split_does_not_leak_trials():
    ds = simulate_motor_cortex(n_recorded=20, n_trials=6,
                               cfg=ReachTaskConfig(n_units=32, train_steps=20), seed=0)
    tr, te = ds.split(0.75, seed=0)
    assert set(np.unique(tr.trial_index)).isdisjoint(np.unique(te.trial_index))
    assert tr.n_samples + te.n_samples == ds.n_samples


def test_standardise_pools_over_contexts():
    """Per-context scaling would remove part of the effect being measured."""
    ds = simulate_place_cells(n_cells=25, n_samples=300, noise=0.1, seed=0)
    z = ds.standardise()
    flat = z.activity.reshape(-1, z.n_neurons)
    assert np.allclose(flat.mean(0), 0, atol=1e-5)
    assert np.allclose(flat.std(0), 1, atol=1e-3)
    per_ctx_std = z.activity[:, 1].std(0)
    assert per_ctx_std.std() > 1e-3, "contexts should not each be separately unit variance"


def test_contexts_actually_differ():
    ds = simulate_place_cells(n_cells=40, n_samples=300, noise=0.0, seed=0)
    ref = ds.activity[:, 0]
    for c in range(1, ds.n_contexts):
        rel = np.linalg.norm(ds.activity[:, c] - ref) / np.linalg.norm(ref)
        assert rel > 0.05, f"context {c} is indistinguishable from the reference"

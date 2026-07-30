"""End-to-end checks on a small, fast problem with a known answer.

The experiments in the paper are the real evidence; these are regression tests.
They fit the model to a deliberately easy place-cell dataset and check that it
recovers the planted transformation, beats the no-transformation null, and that
the ablation which removes the gauge does measurably worse. If the pipeline ever
silently stops learning the geometry, these fail.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from gnd.experiments.common import run_gnd
from gnd.experiments.exp1_hippocampus import build_ground_truth
from gnd.models.gnd import GNDConfig
from gnd.simulations.hippocampus import simulate_place_cells
from gnd.utils.common import set_seed

pytestmark = pytest.mark.slow


def _fit(**overrides):
    set_seed(0)
    ds = simulate_place_cells(n_cells=60, n_samples=900, noise=0.05,
                              morph_context=False, seed=0).standardise()
    train, test = ds.split(0.8, seed=0)
    gt = build_ground_truth(test)
    cfg = replace(
        GNDConfig(n_latent=4, hidden=96, depth=2, n_generators=4, epochs=120,
                  batch_size=128, lr=3e-3, w_topology=0.2, seed=0),
        **overrides,
    )
    res, art = run_gnd(train, test, gt, cfg, "GND", "cpu", False, topology_points=180)
    return res, art


@pytest.fixture(scope="module")
def fitted():
    return _fit()


def test_recovers_the_planted_transformation(fitted):
    res, _ = fitted
    assert res["gre"] < 0.6, (
        f"GRE {res['gre']:.3f} -- 1.0 is the score of assuming no transformation, "
        "so anything near or above it means the geometry was not recovered"
    )
    assert res["readout_r2"] > 0.9, "the canonical latent should be an affine chart of position"


def test_canonical_latent_is_context_invariant(fitted):
    res, _ = fitted
    assert res["cis"] > 0.8
    assert res["context_leakage"] < 0.4, "context should be hard to decode from z"


def test_transformations_move_the_latent_and_still_compose(fitted):
    res, _ = fitted
    assert res["transform_magnitude"] > 0.1, "a collapsed family would be trivially consistent"
    assert res["gcs"] > 0.6


def test_predicts_held_out_context_activity(fitted):
    res, _ = fitted
    assert res["transport_r2"] > 0.3, "must beat each context's own mean by a clear margin"


def test_gauge_is_exactly_identity_and_invertible_after_training(fitted):
    _, art = fitted
    theta, M = art["theta"], art["matrices"]
    assert np.allclose(theta[0], 0, atol=1e-5), "anchoring should hold after training"
    assert np.allclose(M[0], np.eye(M.shape[-1]), atol=1e-5)
    for Mi in M:
        assert abs(np.linalg.det(Mi)) > 1e-6, "group elements must stay invertible"


def test_removing_the_gauge_hurts():
    """Ablation A1: the no-context latent model, on the same data and budget."""
    full, _ = _fit()
    none, _ = _fit(gauge="none", w_group=0.0, w_closure=0.0)
    assert none["transport_r2"] < full["transport_r2"] - 0.1, (
        f"gauge {full['transport_r2']:.3f} vs no gauge {none['transport_r2']:.3f}"
    )
    assert none["gre"] > full["gre"]


def test_results_are_reproducible_given_a_seed():
    a, _ = _fit()
    b, _ = _fit()
    assert a["transport_r2"] == pytest.approx(b["transport_r2"], abs=1e-6)
    assert a["gre"] == pytest.approx(b["gre"], abs=1e-6)

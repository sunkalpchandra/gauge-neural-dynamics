"""Container types shared by the three biological simulations.

Every simulation produces a :class:`ContextualDataset`, which realises the
generative assumption of the paper

    x_c = f( T_c(z) ),                                              (GND)

for a *shared* observation map ``f`` (the tuning curves / readout) and a
*context-dependent* latent transformation ``T_c``.  The dataset therefore
stores, for each of ``n_samples`` latent states, the observed population
activity in every one of ``n_contexts`` contexts, together with the ground
truth transformation used to generate it.

Storing the same latent index across contexts gives us *paired* data.  Pairing
is used for (a) the transport loss of the paired GND variant and (b) *all*
evaluation metrics.  The unpaired variant of the model never touches it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch


@dataclass
class ContextSpec:
    """Ground-truth description of one context.

    Attributes
    ----------
    name:
        Human readable label, used in figures.
    features:
        The observable context variable handed to the model's context encoder
        (e.g. a one-hot environment identity, or a low-dimensional cue vector).
        The model never sees ``matrix``/``offset``.
    matrix, offset:
        Ground-truth affine action on the *generative* latent coordinate,
        ``u -> matrix @ u + offset``.  ``None`` when the ground-truth action is
        not affine (e.g. the non-linear morph context).
    warp:
        Optional callable implementing a non-affine ground-truth action.
    group_params:
        Dictionary of the interpretable parameters of the transformation
        (rotation angle, scale, ...) used by the geometric-recovery metric.
    """

    name: str
    features: np.ndarray
    matrix: np.ndarray | None = None
    offset: np.ndarray | None = None
    warp: object | None = None
    group_params: dict = field(default_factory=dict)

    def apply(self, u: np.ndarray) -> np.ndarray:
        """Apply the ground-truth action to generative coordinates ``u`` (N, d)."""
        if self.warp is not None:
            return np.asarray(self.warp(u))
        out = u
        if self.matrix is not None:
            out = out @ np.asarray(self.matrix).T
        if self.offset is not None:
            out = out + np.asarray(self.offset)[None, :]
        return out

    @property
    def is_affine(self) -> bool:
        return self.warp is None


@dataclass
class ContextualDataset:
    """Paired multi-context neural recordings.

    Attributes
    ----------
    activity:
        ``(n_samples, n_contexts, n_neurons)`` observed population activity.
    latent:
        ``(n_samples, d_gen)`` ground-truth generative latent (position, grid
        phase, ...).  Shared across contexts by construction.
    context_features:
        ``(n_contexts, d_ctx)`` observable context variables.
    contexts:
        List of :class:`ContextSpec`.
    reference:
        Index of the reference context (the one at which ``T_c`` is the
        identity).  Used for gauge fixing.
    time_index:
        Optional ``(n_samples,)`` integer trial/time bookkeeping, used by the
        motor-cortex simulation where samples form trajectories.
    trial_index:
        Optional ``(n_samples,)`` trial id, same purpose.
    """

    activity: np.ndarray
    latent: np.ndarray
    context_features: np.ndarray
    contexts: list[ContextSpec]
    reference: int = 0
    time_index: np.ndarray | None = None
    trial_index: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    # -- shape helpers -----------------------------------------------------
    @property
    def n_samples(self) -> int:
        return self.activity.shape[0]

    @property
    def n_contexts(self) -> int:
        return self.activity.shape[1]

    @property
    def n_neurons(self) -> int:
        return self.activity.shape[2]

    @property
    def d_context(self) -> int:
        return self.context_features.shape[1]

    def __post_init__(self) -> None:
        assert self.activity.ndim == 3, "activity must be (samples, contexts, neurons)"
        assert self.activity.shape[1] == len(self.contexts)
        assert self.activity.shape[0] == self.latent.shape[0]
        assert self.context_features.shape[0] == len(self.contexts)

    # -- transforms --------------------------------------------------------
    def standardise(self) -> "ContextualDataset":
        """Z-score each neuron using statistics pooled over samples *and*
        contexts.

        Pooling matters: per-context standardisation would silently remove part
        of the transformation we are trying to recover.
        """
        flat = self.activity.reshape(-1, self.n_neurons)
        mu = flat.mean(axis=0, keepdims=True)
        sd = flat.std(axis=0, keepdims=True) + 1e-6
        act = (self.activity - mu[None]) / sd[None]
        self.meta = {**self.meta, "standardise_mean": mu, "standardise_std": sd}
        return ContextualDataset(
            activity=act,
            latent=self.latent,
            context_features=self.context_features,
            contexts=self.contexts,
            reference=self.reference,
            time_index=self.time_index,
            trial_index=self.trial_index,
            meta=self.meta,
        )

    def split(self, frac: float = 0.8, seed: int = 0) -> tuple["ContextualDataset", "ContextualDataset"]:
        """Split along the *sample* axis, keeping all contexts on both sides.

        When ``trial_index`` is present the split is performed at the level of
        whole trials so that no trajectory straddles the boundary.
        """
        rng = np.random.default_rng(seed)
        if self.trial_index is not None:
            trials = np.unique(self.trial_index)
            perm = rng.permutation(trials)
            n_tr = int(round(frac * len(trials)))
            train_tr, test_tr = set(perm[:n_tr].tolist()), set(perm[n_tr:].tolist())
            idx_tr = np.array([i for i, t in enumerate(self.trial_index) if t in train_tr])
            idx_te = np.array([i for i, t in enumerate(self.trial_index) if t in test_tr])
        else:
            perm = rng.permutation(self.n_samples)
            n_tr = int(round(frac * self.n_samples))
            idx_tr, idx_te = perm[:n_tr], perm[n_tr:]
        return self._subset(idx_tr), self._subset(idx_te)

    def _subset(self, idx: np.ndarray) -> "ContextualDataset":
        return ContextualDataset(
            activity=self.activity[idx],
            latent=self.latent[idx],
            context_features=self.context_features,
            contexts=self.contexts,
            reference=self.reference,
            time_index=None if self.time_index is None else self.time_index[idx],
            trial_index=None if self.trial_index is None else self.trial_index[idx],
            meta=self.meta,
        )

    def select_contexts(self, keep: Sequence[int]) -> "ContextualDataset":
        """Restrict to a subset of contexts (used for held-out-context tests)."""
        keep = list(keep)
        ref = keep.index(self.reference) if self.reference in keep else 0
        return ContextualDataset(
            activity=self.activity[:, keep],
            latent=self.latent,
            context_features=self.context_features[keep],
            contexts=[self.contexts[i] for i in keep],
            reference=ref,
            time_index=self.time_index,
            trial_index=self.trial_index,
            meta=self.meta,
        )

    # -- torch bridge ------------------------------------------------------
    def tensors(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        return {
            "activity": torch.as_tensor(self.activity, dtype=torch.float32, device=device),
            "latent": torch.as_tensor(self.latent, dtype=torch.float32, device=device),
            "context_features": torch.as_tensor(
                self.context_features, dtype=torch.float32, device=device
            ),
        }

    def flat(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten to ``(X, context_id, latent)`` with ``N = samples*contexts``."""
        n, c, p = self.activity.shape
        X = self.activity.transpose(0, 1, 2).reshape(n * c, p)
        cid = np.tile(np.arange(c), (n, 1)).reshape(-1)
        z = np.repeat(self.latent, c, axis=0)
        return X, cid, z


def add_observation_noise(
    activity: np.ndarray,
    noise: float,
    kind: str = "gaussian",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Corrupt clean rates with observation noise.

    ``gaussian`` adds i.i.d. noise scaled by the global activity s.d.;
    ``poisson`` draws spike counts with mean ``rate / noise`` and rescales,
    giving a Fano factor of 1 with the noise level controlling the count scale.
    """
    rng = rng or np.random.default_rng(0)
    if noise <= 0:
        return activity
    if kind == "gaussian":
        scale = activity.std()
        return activity + rng.normal(0.0, noise * scale, size=activity.shape)
    if kind == "poisson":
        gain = 1.0 / max(noise, 1e-6)
        counts = rng.poisson(np.clip(activity, 0, None) * gain)
        return counts / gain
    raise ValueError(f"unknown noise kind {kind!r}")

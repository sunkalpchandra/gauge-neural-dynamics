r"""Latent dynamics in the canonical (gauge-fixed) frame.

If the latent computation is shared across contexts, then so is its *flow*.
Writing the canonical dynamics as ``dz/dt = F(z)``, the dynamics an
experimenter would measure in context ``c`` are the push-forward

    F_c(w) = D T_c( T_c^{-1} w ) F( T_c^{-1} w ),                        (4)

which for a linear gauge ``T_c = M_c`` reduces to matrix conjugation

    F_c = M_c F M_c^{-1} .                                              (5)

Equation (5) has an immediate, testable consequence: **the eigenvalues of the
linearised dynamics are context invariant, while their eigenvectors -- the
rotation planes -- are not**.  Rotation frequencies should therefore be
preserved across reach conditions even when the population geometry rotates,
which is what Experiment 3 checks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .encoder import _mlp


class LatentDynamics(nn.Module):
    """Vector field ``F`` on the canonical latent manifold.

    ``kind='linear'`` gives ``F(z) = A z + a`` (the case where (5) is exact);
    ``kind='mlp'`` gives a general smooth field, for which (4) still holds and
    (5) holds for the Jacobian at matched points.
    """

    def __init__(
        self,
        n_latent: int,
        kind: str = "linear",
        hidden: int = 128,
        depth: int = 2,
        dt: float = 1.0,
    ):
        super().__init__()
        self.kind = kind
        self.n_latent = n_latent
        self.dt = dt
        if kind == "linear":
            self.A = nn.Parameter(torch.randn(n_latent, n_latent) * 0.1 / np.sqrt(n_latent))
            self.a = nn.Parameter(torch.zeros(n_latent))
        elif kind == "mlp":
            self.net = _mlp(n_latent, hidden, depth, n_latent, "tanh", layer_norm=False)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            raise ValueError(kind)

    def field(self, z: torch.Tensor) -> torch.Tensor:
        if self.kind == "linear":
            return z @ self.A.T + self.a
        return self.net(z)

    forward = field

    def step(self, z: torch.Tensor, dt: float | None = None) -> torch.Tensor:
        """One RK4 step of the canonical flow."""
        h = self.dt if dt is None else dt
        k1 = self.field(z)
        k2 = self.field(z + 0.5 * h * k1)
        k3 = self.field(z + 0.5 * h * k2)
        k4 = self.field(z + h * k3)
        return z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rollout(self, z0: torch.Tensor, n_steps: int, dt: float | None = None) -> torch.Tensor:
        zs = [z0]
        z = z0
        for _ in range(n_steps):
            z = self.step(z, dt)
            zs.append(z)
        return torch.stack(zs, dim=1)

    def jacobian(self, z: torch.Tensor) -> torch.Tensor:
        """``D F(z)``; analytic for the linear field, autograd otherwise."""
        if self.kind == "linear":
            return self.A.expand(z.shape[0], -1, -1)
        z = z.detach().requires_grad_(True)
        return torch.stack([torch.autograd.functional.jacobian(self.field, zi[None]).squeeze() for zi in z])

    # -- push-forward ------------------------------------------------------
    def pushforward(self, w: torch.Tensor, gauge, theta: torch.Tensor) -> torch.Tensor:
        """Observed vector field in the context with coefficients ``theta``.

        Implements (4) by differentiating through the gauge map, so it is valid
        for both the linear and the flow gauge.
        """
        z = gauge.inverse(w, theta)
        v = self.field(z)
        z = z.detach().requires_grad_(True)
        out = gauge.transform(z, theta)
        (jvp,) = torch.autograd.grad(out, z, grad_outputs=v, create_graph=self.training)
        return jvp


def dynamics_spectrum(A: np.ndarray, dt: float = 1.0) -> dict:
    """Continuous-time eigen-summary of a linear latent flow.

    Returns decay rates, rotation frequencies (Hz if ``dt`` is in seconds) and
    the raw eigenvalues.  These are the conjugation-invariant quantities that
    (5) predicts to be shared across contexts.
    """
    ev = np.linalg.eigvals(np.asarray(A, float))
    order = np.argsort(-np.abs(ev.imag))
    ev = ev[order]
    return {
        "eigenvalues": ev,
        "rates": ev.real,
        "frequencies_hz": ev.imag / (2 * np.pi * dt),
        "top_frequency_hz": float(np.abs(ev.imag).max() / (2 * np.pi * dt)),
    }


def fit_linear_dynamics(Z: np.ndarray, dt: float = 1.0, ridge: float = 1e-4) -> np.ndarray:
    """Least-squares continuous-time generator from a trajectory ``Z`` (T, d).

    Fits ``(z_{t+1} - z_t)/dt = A z_t + a`` and returns ``A``.
    """
    Z = np.asarray(Z, float)
    X, Y = Z[:-1], (Z[1:] - Z[:-1]) / dt
    Xa = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    G = Xa.T @ Xa + ridge * np.eye(Xa.shape[1])
    W = np.linalg.solve(G, Xa.T @ Y)
    return W[:-1].T


def rotational_plane_angle(A: np.ndarray, B: np.ndarray) -> float:
    """Principal angle between the dominant rotation planes of two generators.

    The dominant plane is spanned by the real and imaginary parts of the
    eigenvector with the largest ``|Im lambda|``.  Under (5) this plane rotates
    with the context while the frequency stays fixed.
    """
    def plane(M: np.ndarray) -> np.ndarray:
        ev, V = np.linalg.eig(np.asarray(M, float))
        i = int(np.argmax(np.abs(ev.imag)))
        P = np.stack([V[:, i].real, V[:, i].imag], axis=1)
        Q, _ = np.linalg.qr(P)
        return Q

    Pa, Pb = plane(A), plane(B)
    s = np.linalg.svd(Pa.T @ Pb, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s, -1, 1))).mean())

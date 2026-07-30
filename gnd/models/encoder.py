"""Encoders mapping population activity to the observed-frame latent.

The encoder is *shared across contexts*: it sees only the neural activity
vector, never the context label.  This is deliberate.  In the generative model
``x_c = f(T_c(z))`` the encoder is the approximate inverse of ``f``, so it
returns the *observed-frame* latent ``w_c = T_c(z)``; all context dependence is
then carried by the gauge field, which pulls ``w_c`` back to the canonical
frame via ``z = T_c^{-1}(w_c)``.  Giving the encoder access to the context
would let it absorb the transformation and make the gauge field unidentifiable.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(d_in: int, hidden: int, depth: int, d_out: int, activation: str = "gelu",
         layer_norm: bool = True) -> nn.Sequential:
    act = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}[activation]
    layers: list[nn.Module] = []
    d = d_in
    for _ in range(depth):
        layers.append(nn.Linear(d, hidden))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(act())
        d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class MLPEncoder(nn.Module):
    """Deterministic encoder ``x -> w``."""

    def __init__(
        self,
        n_neurons: int,
        n_latent: int,
        hidden: int = 256,
        depth: int = 3,
        activation: str = "gelu",
        layer_norm: bool = True,
        normalise_latent: bool = False,
    ):
        super().__init__()
        self.net = _mlp(n_neurons, hidden, depth, n_latent, activation, layer_norm)
        self.n_latent = n_latent
        self.normalise_latent = normalise_latent

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        w = self.net(x)
        if self.normalise_latent:
            w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return w, {}


class VariationalEncoder(nn.Module):
    """Gaussian encoder ``q(w | x) = N(mu(x), diag(sigma(x)^2))``.

    Returns a reparameterised sample during training and the mean at eval time,
    together with the per-sample KL to a standard normal prior.
    """

    def __init__(
        self,
        n_neurons: int,
        n_latent: int,
        hidden: int = 256,
        depth: int = 3,
        activation: str = "gelu",
        layer_norm: bool = True,
        logvar_min: float = -8.0,
        logvar_max: float = 4.0,
    ):
        super().__init__()
        self.trunk = _mlp(n_neurons, hidden, depth, 2 * n_latent, activation, layer_norm)
        self.n_latent = n_latent
        self.logvar_min, self.logvar_max = logvar_min, logvar_max

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        h = self.trunk(x)
        mu, logvar = h.chunk(2, dim=-1)
        logvar = logvar.clamp(self.logvar_min, self.logvar_max)
        if self.training:
            w = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            w = mu
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(-1)
        return w, {"mu": mu, "logvar": logvar, "kl": kl}


def build_encoder(kind: str, **kwargs) -> nn.Module:
    """``kind`` in ``{"mlp", "vae"}``."""
    if kind == "mlp":
        return MLPEncoder(**kwargs)
    if kind == "vae":
        kwargs.pop("normalise_latent", None)
        return VariationalEncoder(**kwargs)
    raise ValueError(f"unknown encoder kind {kind!r}")

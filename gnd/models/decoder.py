"""Decoders realising the shared observation map ``f``.

Like the encoder, the decoder never sees the context: it *is* the fixed tuning
of the population.  Whatever differs between contexts must therefore be
expressed by the gauge field, which is what makes the framework falsifiable --
a context effect that no latent transformation can produce (e.g. rate
remapping, or the non-integer grid rescaling of Experiment 2) shows up as
irreducible reconstruction error rather than being silently absorbed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import _mlp


class MLPDecoder(nn.Module):
    """``z -> x``.  ``output`` selects the link function.

    ``linear`` suits z-scored activity; ``softplus`` produces non-negative
    rates and pairs with the Poisson likelihood.
    """

    def __init__(
        self,
        n_latent: int,
        n_neurons: int,
        hidden: int = 256,
        depth: int = 3,
        activation: str = "gelu",
        layer_norm: bool = True,
        output: str = "linear",
    ):
        super().__init__()
        self.net = _mlp(n_latent, hidden, depth, n_neurons, activation, layer_norm)
        self.output = output

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.net(z)
        if self.output == "softplus":
            return F.softplus(y) + 1e-4
        return y


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "mse") -> torch.Tensor:
    """Per-sample-mean reconstruction loss."""
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "huber":
        return F.huber_loss(pred, target, delta=1.0)
    if kind == "poisson":
        return (pred - target * torch.log(pred.clamp_min(1e-6))).mean()
    raise ValueError(f"unknown reconstruction loss {kind!r}")


def build_decoder(**kwargs) -> nn.Module:
    return MLPDecoder(**kwargs)

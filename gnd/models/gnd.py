r"""The Gauge Neural Dynamics model.

Architecture
------------
Three shared, context-blind components and one context-dependent one:

    w_c = E(x_c)                 encoder      (context blind)
    z   = T_c^{-1}(w_c)          gauge        (context dependent)
    x_c = D(T_c(z))              decoder      (context blind)

Because ``E`` and ``D`` never see the context, the *only* way the model can
explain a context difference is through ``T_c``.  This is what makes the
framework falsifiable rather than merely flexible.

Objective
---------
    L = w_rec  L_rec                     autoencoding, per context
      + w_tra  L_transport               cross-context prediction
      + w_inv  L_invariance              canonical latent is context invariant
      + w_grp  L_group                   BCH composition / identity / inverse
      + w_clo  L_closure                 the algebra should close
      + w_top  L_topology                H_0 persistence signature preserved
      + w_kl   KL  +  w_reg ||z||^2      regularisation

``L_transport`` is the term that carries the scientific content: it asks the
model to predict the population response in context ``b`` from a response
recorded in context ``a``, using only the shared decoder and the learned
transformation.  A model with no genuine gauge structure cannot do this.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..geometry.topology import TopologicalSignatureLoss
from .decoder import build_decoder, reconstruction_loss
from .encoder import build_encoder
from .gauge_field import FlowGaugeField, IdentityGauge, LinearGaugeField, build_gauge_field
from .neural_dynamics import LatentDynamics


@dataclass
class GNDConfig:
    """All model and optimisation hyper-parameters."""

    # -- architecture
    n_latent: int = 6
    encoder: str = "mlp"                 # mlp | vae
    hidden: int = 256
    depth: int = 3
    activation: str = "gelu"
    layer_norm: bool = True
    decoder_output: str = "linear"       # linear | softplus
    recon_kind: str = "mse"              # mse | huber | poisson

    # -- gauge
    gauge: str = "linear"                # linear | flow | none
    algebra: str = "gl"                  # gl | so | sl | aff | se
    n_generators: int = 6
    context_hidden: int = 64
    context_depth: int = 2
    anchor: bool = True
    bch_order: int = 2
    flow_steps: int = 6
    flow_nonlinearity: float = 0.5

    # -- latent dynamics (Experiment 3)
    dynamics: str = "none"               # none | linear | mlp
    dynamics_hidden: int = 128

    # -- loss weights
    w_recon: float = 1.0
    w_transport: float = 1.0
    w_invariance: float = 1.0
    w_group: float = 0.2
    w_closure: float = 0.05
    w_topology: float = 0.5
    w_kl: float = 1e-3
    w_latent_norm: float = 1e-3
    w_dynamics: float = 0.0
    alignment_mode: str = "paired"       # paired | mmd  (mmd = correspondence free)

    # -- loss bookkeeping
    n_transport_pairs: int = 8
    n_group_pairs: int = 12
    group_probe_points: int = 64
    topology_points: int = 96
    topology_every: int = 2
    closure_every: int = 5
    struct_refresh_every: int = 25       # flow gauge: cost of the vector-field bracket

    # -- optimisation
    lr: float = 2e-3
    weight_decay: float = 1e-5
    epochs: int = 400
    batch_size: int = 256
    grad_clip: float = 1.0
    warmup_frac: float = 0.15            # ramp gauge-dependent losses in slowly
    seed: int = 0


# ---------------------------------------------------------------------------
# distributional alignment (correspondence-free variant)
# ---------------------------------------------------------------------------
def gaussian_mmd(x: torch.Tensor, y: torch.Tensor, scales: tuple[float, ...] = (0.5, 1.0, 2.0)) -> torch.Tensor:
    """Multi-scale Gaussian-kernel MMD^2 with a median-heuristic bandwidth."""
    xy = torch.cat([x, y], dim=0)
    d2 = torch.cdist(xy, xy).pow(2)
    n = x.shape[0]
    with torch.no_grad():
        med = d2[d2 > 0].median().clamp_min(1e-8)
    out = x.new_zeros(())
    for s in scales:
        K = torch.exp(-d2 / (s * med))
        Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
        m = y.shape[0]
        out = out + (
            (Kxx.sum() - Kxx.diagonal().sum()) / max(n * (n - 1), 1)
            + (Kyy.sum() - Kyy.diagonal().sum()) / max(m * (m - 1), 1)
            - 2.0 * Kxy.mean()
        )
    return out / len(scales)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class GaugeNeuralDynamics(nn.Module):
    def __init__(self, n_neurons: int, n_context_features: int, cfg: GNDConfig):
        super().__init__()
        self.cfg = cfg
        self.n_neurons = n_neurons
        self.encoder = build_encoder(
            cfg.encoder,
            n_neurons=n_neurons,
            n_latent=cfg.n_latent,
            hidden=cfg.hidden,
            depth=cfg.depth,
            activation=cfg.activation,
            layer_norm=cfg.layer_norm,
        )
        self.decoder = build_decoder(
            n_latent=cfg.n_latent,
            n_neurons=n_neurons,
            hidden=cfg.hidden,
            depth=cfg.depth,
            activation=cfg.activation,
            layer_norm=cfg.layer_norm,
            output=cfg.decoder_output,
        )
        self.gauge = build_gauge_field(
            cfg.gauge,
            n_latent=cfg.n_latent,
            n_context_features=n_context_features,
            n_generators=cfg.n_generators,
            algebra=cfg.algebra,
            context_hidden=cfg.context_hidden,
            context_depth=cfg.context_depth,
            anchor=cfg.anchor,
            bch_order=cfg.bch_order,
            hidden=cfg.hidden // 2,
            depth=2,
            n_steps=cfg.flow_steps,
            nonlinearity=cfg.flow_nonlinearity,
        )
        self.dynamics = (
            LatentDynamics(cfg.n_latent, cfg.dynamics, cfg.dynamics_hidden)
            if cfg.dynamics != "none"
            else None
        )
        self.topo_loss = TopologicalSignatureLoss()
        self._step = 0

    # -- basic maps --------------------------------------------------------
    def coefficients(self, ctx: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return self.gauge.coefficients(ctx, ref)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Observed-frame latent ``w_c = E(x_c)``."""
        return self.encoder(x)

    def canonicalise(self, w: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """``z = T_c^{-1}(w)``."""
        return self.gauge.inverse(w, theta)

    def express(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """``w = T_c(z)``."""
        return self.gauge.transform(z, theta)

    def decode(self, w: torch.Tensor) -> torch.Tensor:
        return self.decoder(w)

    @torch.no_grad()
    def latents(self, x: torch.Tensor, ctx: torch.Tensor, ref: torch.Tensor):
        """Return ``(w, z, theta)`` for ``x`` of shape ``(N, C, P)``."""
        self.eval()
        N, C, P = x.shape
        theta = self.coefficients(ctx, ref)
        w, _ = self.encode(x.reshape(N * C, P))
        w = w.reshape(N, C, -1)
        th = theta[None].expand(N, C, -1).reshape(N * C, -1)
        z = self.canonicalise(w.reshape(N * C, -1), th).reshape(N, C, -1)
        return w, z, theta

    @torch.no_grad()
    def transport(self, x_a: torch.Tensor, theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
        """Predict activity in context ``b`` from activity recorded in ``a``."""
        self.eval()
        w, _ = self.encode(x_a)
        z = self.canonicalise(w, theta_a.expand(x_a.shape[0], -1))
        return self.decode(self.express(z, theta_b.expand(x_a.shape[0], -1)))

    # -- objective ---------------------------------------------------------
    def loss(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        ref: torch.Tensor,
        step: int = 0,
        total_steps: int = 1,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict]:
        cfg = self.cfg
        self._step = step
        N, C, P = x.shape
        dev = x.device
        theta = self.coefficients(ctx, ref)                      # (C, K)
        linear = isinstance(self.gauge, LinearGaugeField)

        # Group elements depend only on the context, so exponentiate once.
        if linear:
            Mf = self.gauge.matrices(theta)                      # (C, m, m)
            Mi = self.gauge.matrices(-theta)
        else:
            Mf = Mi = None

        xf = x.reshape(N * C, P)
        w, aux = self.encode(xf)
        if linear:
            Mi_full = Mi[None].expand(N, C, -1, -1).reshape(N * C, *Mi.shape[1:])
            z = self.gauge.act(Mi_full, w).reshape(N, C, -1)
        else:
            th = theta[None].expand(N, C, -1).reshape(N * C, -1)
            z = self.canonicalise(w, th).reshape(N, C, -1)

        parts: dict[str, torch.Tensor] = {}

        # (a) autoencoding in the observed frame
        parts["recon"] = reconstruction_loss(self.decode(w), xf, cfg.recon_kind)

        # (b) context invariance of the canonical latent
        if cfg.alignment_mode == "paired":
            zbar = z.mean(dim=1, keepdim=True)
            scale = z.reshape(-1, z.shape[-1]).var(0).sum().clamp_min(1e-6)
            parts["invariance"] = ((z - zbar) ** 2).sum(-1).mean() / scale
        else:  # correspondence-free: match distributions across contexts
            vals = [gaussian_mmd(z[:, 0], z[:, c]) for c in range(C) if c != 0]
            parts["invariance"] = torch.stack(vals).mean() if vals else z.new_zeros(())

        # (c) cross-context transport -- the load-bearing term.  All pairs are
        #     evaluated in a single batched decoder pass.
        pairs = self._sample_pairs(C, cfg.n_transport_pairs, dev, generator)
        if pairs:
            ai = torch.tensor([p[0] for p in pairs], device=dev)
            bi = torch.tensor([p[1] for p in pairs], device=dev)
            za = z[:, ai].reshape(-1, z.shape[-1])               # (N*P, d)
            if linear:
                Mb = Mf[bi][None].expand(N, -1, -1, -1).reshape(-1, *Mf.shape[1:])
                wb = self.gauge.act(Mb, za)
            else:
                tb = theta[bi][None].expand(N, -1, -1).reshape(-1, theta.shape[-1])
                wb = self.express(za, tb)
            tgt = x[:, bi].reshape(-1, P)
            parts["transport"] = reconstruction_loss(self.decode(wb), tgt, cfg.recon_kind)
        else:
            parts["transport"] = x.new_zeros(())

        # (d) approximate-group structure
        parts["group"] = self._group_loss(z, theta, Mf, C, dev, generator)

        # (e) Lie-algebra closure
        if cfg.w_closure > 0 and (step % cfg.closure_every == 0):
            parts["closure"] = (
                self.gauge.closure_defect(z[: min(48, N), 0].detach())
                if isinstance(self.gauge, FlowGaugeField)
                else self.gauge.closure_defect()
            )
        else:
            parts["closure"] = x.new_zeros(())

        # (f) topology preservation between activity space and canonical latent
        if cfg.w_topology > 0 and (step % cfg.topology_every == 0):
            m = min(cfg.topology_points, N)
            idx = torch.randperm(N, device=dev, generator=generator)[:m]
            parts["topology"] = self.topo_loss(x[idx, 0], z[idx, 0])
        else:
            parts["topology"] = x.new_zeros(())

        # (g) regularisation
        parts["kl"] = aux["kl"].mean() if "kl" in aux else x.new_zeros(())
        parts["latent_norm"] = (z ** 2).sum(-1).mean()

        # (h) optional shared latent dynamics
        parts["dynamics"] = self._dynamics_loss(z) if self.dynamics is not None else x.new_zeros(())

        ramp = min(1.0, (step + 1) / max(cfg.warmup_frac * total_steps, 1.0))
        total = (
            cfg.w_recon * parts["recon"]
            + ramp * cfg.w_transport * parts["transport"]
            + ramp * cfg.w_invariance * parts["invariance"]
            + ramp * cfg.w_group * parts["group"]
            + ramp * cfg.w_closure * parts["closure"]
            + cfg.w_topology * parts["topology"]
            + cfg.w_kl * parts["kl"]
            + cfg.w_latent_norm * parts["latent_norm"]
            + cfg.w_dynamics * parts["dynamics"]
        )
        return total, {k: float(v.detach()) for k, v in parts.items()}

    # -- pieces ------------------------------------------------------------
    @staticmethod
    def _sample_pairs(C: int, n: int, device, generator) -> list[tuple[int, int]]:
        if C < 2:
            return []
        all_pairs = [(a, b) for a in range(C) for b in range(C) if a != b]
        if n >= len(all_pairs):
            return all_pairs
        idx = torch.randperm(len(all_pairs), device=device, generator=generator)[:n].tolist()
        return [all_pairs[i] for i in idx]

    def _group_loss(self, z, theta, Mf, C, device, generator) -> torch.Tensor:
        """Composition + invertibility, measured in latent space.

        Composition: ``T_a(T_b(z))`` should equal ``T_{a.b}(z)`` where the
        composed coefficients come from the truncated BCH series.  Normalising
        by how far the composed map actually moves points prevents the trivial
        solution ``T = id``.  All context pairs are evaluated in one batch on a
        fixed-size probe set.
        """
        cfg = self.cfg
        if isinstance(self.gauge, IdentityGauge) or C < 2:
            return z.new_zeros(())
        pairs = self._sample_pairs(C, cfg.n_group_pairs, device, generator)
        if not pairs:
            return z.new_zeros(())
        z0 = z[: min(cfg.group_probe_points, z.shape[0]), 0]
        n = z0.shape[0]
        ai = torch.tensor([p[0] for p in pairs], device=device)
        bi = torch.tensor([p[1] for p in pairs], device=device)
        P = len(pairs)
        zr = z0[:, None].expand(n, P, z0.shape[-1]).reshape(-1, z0.shape[-1])

        if isinstance(self.gauge, FlowGaugeField):
            ta = theta[ai][None].expand(n, -1, -1).reshape(-1, theta.shape[-1])
            tb = theta[bi][None].expand(n, -1, -1).reshape(-1, theta.shape[-1])
            zab = self.express(self.express(zr, tb), ta)
            tc = self.gauge.compose(theta[ai], theta[bi], z0.detach(),
                                    refresh=(self._step % cfg.struct_refresh_every == 0))
        else:
            Ma = Mf[ai][None].expand(n, -1, -1, -1).reshape(-1, *Mf.shape[1:])
            Mb = Mf[bi][None].expand(n, -1, -1, -1).reshape(-1, *Mf.shape[1:])
            zab = self.gauge.act(Ma, self.gauge.act(Mb, zr))
            tc = self.gauge.compose(theta[ai], theta[bi])
        tc_full = tc[None].expand(n, -1, -1).reshape(-1, tc.shape[-1])
        zc = self.express(zr, tc_full)

        num = ((zab - zc) ** 2).sum(-1).mean()
        den = ((zab - zr) ** 2).sum(-1).mean().clamp_min(1e-6)
        comp = num / den
        return comp + 0.1 * self.gauge.inverse_defect(theta)

    def _dynamics_loss(self, z: torch.Tensor) -> torch.Tensor:
        """One-step prediction error of the shared canonical flow.

        Samples must be ordered in time; the experiment driver guarantees this
        by passing contiguous trajectory chunks.
        """
        zc = z[:, 0]
        if zc.shape[0] < 3:
            return z.new_zeros(())
        pred = self.dynamics.step(zc[:-1])
        return ((pred - zc[1:]) ** 2).sum(-1).mean() / zc.var(0).sum().clamp_min(1e-6)

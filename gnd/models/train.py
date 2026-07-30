"""Training loop for :class:`~gnd.models.gnd.GaugeNeuralDynamics`."""

from __future__ import annotations

import time

import numpy as np
import torch

from ..simulations.base import ContextualDataset
from ..utils.common import count_parameters, set_seed
from .gnd import GaugeNeuralDynamics, GNDConfig


def _batches(n: int, bs: int, generator: torch.Generator, contiguous: bool = False, device="cpu"):
    """Yield index batches.

    ``contiguous=True`` returns runs of consecutive samples, which the latent
    dynamics loss needs (it is a one-step prediction error in time).
    """
    if contiguous:
        n_chunks = max(n // bs, 1)
        starts = torch.randint(0, max(n - bs, 1), (n_chunks,), generator=generator, device=device)
        for s in starts.tolist():
            yield torch.arange(s, min(s + bs, n), device=device)
    else:
        perm = torch.randperm(n, generator=generator, device=device)
        for i in range(0, n, bs):
            yield perm[i: i + bs]


def fit_gnd(
    dataset: ContextualDataset,
    cfg: GNDConfig,
    device: torch.device | str = "cpu",
    verbose: bool = False,
    log_every: int = 100,
) -> tuple[GaugeNeuralDynamics, dict]:
    """Fit the model; returns ``(model, history)``."""
    set_seed(cfg.seed)
    device = torch.device(device)
    t = dataset.tensors(device)
    X, ctx = t["activity"], t["context_features"]
    ref = ctx[dataset.reference]

    model = GaugeNeuralDynamics(dataset.n_neurons, dataset.d_context, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.05)
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 17)

    contiguous = cfg.w_dynamics > 0
    n = X.shape[0]
    steps_per_epoch = max(n // cfg.batch_size, 1)
    total_steps = cfg.epochs * steps_per_epoch
    history: dict[str, list] = {"total": [], "epoch": []}
    step = 0
    t0 = time.time()

    model.train()
    for epoch in range(cfg.epochs):
        agg: dict[str, float] = {}
        nb = 0
        for idx in _batches(n, cfg.batch_size, gen, contiguous, device):
            if len(idx) < 8:
                continue
            opt.zero_grad(set_to_none=True)
            total, parts = model.loss(X[idx], ctx, ref, step, total_steps, gen)
            total.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            step += 1
            nb += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            agg["total"] = agg.get("total", 0.0) + float(total.detach())
        sched.step()
        if nb:
            for k, v in agg.items():
                history.setdefault(k, []).append(v / nb)
            history["epoch"].append(epoch)
        if verbose and (epoch % log_every == 0 or epoch == cfg.epochs - 1):
            msg = "  ".join(f"{k}={history[k][-1]:.4f}" for k in ("total", "recon", "transport", "invariance", "group") if k in history)
            print(f"[epoch {epoch:4d}] {msg}")

    model.eval()
    history["wall_time_s"] = time.time() - t0
    history["n_parameters"] = count_parameters(model)
    return model, history


@torch.no_grad()
def held_out_context_score(
    model: GaugeNeuralDynamics,
    dataset: ContextualDataset,
    device: torch.device | str = "cpu",
) -> dict:
    """R^2 of predicting each context's activity from the reference context.

    This is the model's core empirical claim in its most direct form: given a
    population response recorded in the reference context and *only* the label
    of another context, reproduce the response there.
    """
    t = dataset.tensors(device)
    X, ctx = t["activity"], t["context_features"]
    ref = ctx[dataset.reference]
    theta = model.coefficients(ctx, ref)
    out = {}
    r2s = []
    for c in range(dataset.n_contexts):
        if c == dataset.reference:
            continue
        pred = model.transport(X[:, dataset.reference], theta[dataset.reference], theta[c])
        tgt = X[:, c]
        ss_res = ((pred - tgt) ** 2).sum()
        ss_tot = ((tgt - tgt.mean(0, keepdim=True)) ** 2).sum()
        r2 = float(1 - ss_res / ss_tot.clamp_min(1e-9))
        out[f"transport_r2_ctx{c}"] = r2
        r2s.append(r2)
    out["transport_r2"] = float(np.mean(r2s)) if r2s else float("nan")
    return out

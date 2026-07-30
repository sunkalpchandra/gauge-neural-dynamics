"""Recurrent-network model of motor cortex during centre-out reaching.

Biological setting
------------------
Motor cortical population activity during reaching is well described as the
state of an autonomous dynamical system whose condition-dependent initial state
is set during preparation (Churchland et al., 2012; Shenoy et al., 2013;
Sussillo et al., 2015).  Because the centre-out task is rotationally symmetric,
different reach directions are natural candidates for being *the same latent
computation expressed in a rotated coordinate frame* -- precisely the
hypothesis this paper formalises.

Network
-------
A continuous-time rate RNN

    tau dx/dt = -x + W tanh(x) + B u(t) + b + sqrt(2 tau) sigma xi(t),

read out as ``y = C tanh(x)``, is trained by BPTT to emit a bell-shaped hand
velocity profile in the cued direction after a go cue.  We do *not* impose any
symmetry on ``W``: whatever equivariance the solution has is learned from the
task.  This matters, because it lets us measure an empirical ceiling on
recoverability -- the circuit's own equivariance defect (see
:func:`equivariance_defect`) -- rather than assuming the ground truth is exact.

Contexts
--------
Contexts are the ``n_directions x n_extents`` reach conditions.  The
ground-truth latent action on the reach plane is ``e_c R(phi_c)``: a rotation
by the reach angle composed with a scaling by the reach extent, i.e. a
two-parameter abelian group ``SO(2) x R_+``.  Samples are indexed by time
within the trial and are paired across contexts by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .base import ContextSpec, ContextualDataset, add_observation_noise


@dataclass
class ReachTaskConfig:
    n_units: int = 128
    n_directions: int = 8
    extents: tuple[float, ...] = (1.0, 0.6)
    dt: float = 0.01          # s
    tau: float = 0.05         # s
    t_prep: float = 0.30      # s  cue on, hold
    t_move: float = 0.50      # s  after go cue
    peak_speed_time: float = 0.20
    speed_width: float = 0.07
    curvature: float = 0.30   # lateral velocity component (fraction of peak speed)
    noise_std: float = 0.02
    train_steps: int = 1500
    lr: float = 4e-3
    reg_rate: float = 1e-4
    reg_weight: float = 1e-4


class ReachRNN(nn.Module):
    """Continuous-time tanh rate network, Euler-discretised."""

    def __init__(self, cfg: ReachTaskConfig, n_inputs: int = 3, n_outputs: int = 2):
        super().__init__()
        n = cfg.n_units
        self.cfg = cfg
        g = 1.2                                     # slightly chaotic init
        self.W = nn.Parameter(torch.randn(n, n) * g / np.sqrt(n))
        self.B = nn.Parameter(torch.randn(n, n_inputs) / np.sqrt(n_inputs))
        self.b = nn.Parameter(torch.zeros(n))
        self.C = nn.Parameter(torch.randn(n_outputs, n) / np.sqrt(n))
        self.x0 = nn.Parameter(torch.zeros(n))

    def forward(self, u: torch.Tensor, noise: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """``u`` is (batch, time, n_inputs); returns hidden ``x`` and output ``y``."""
        cfg = self.cfg
        bsz, T, _ = u.shape
        alpha = cfg.dt / cfg.tau
        x = self.x0.expand(bsz, -1).clone()
        xs = []
        for t in range(T):
            r = torch.tanh(x)
            dx = -x + r @ self.W.T + u[:, t] @ self.B.T + self.b
            x = x + alpha * dx
            if noise and cfg.noise_std > 0:
                x = x + np.sqrt(2 * alpha) * cfg.noise_std * torch.randn_like(x)
            xs.append(x)
        X = torch.stack(xs, dim=1)
        Y = torch.tanh(X) @ self.C.T
        return X, Y


def _task_tensors(cfg: ReachTaskConfig) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Build inputs and targets for every (direction, extent) condition."""
    n_prep = int(round(cfg.t_prep / cfg.dt))
    n_move = int(round(cfg.t_move / cfg.dt))
    T = n_prep + n_move
    dirs = np.arange(cfg.n_directions) * 2 * np.pi / cfg.n_directions
    conds = [(d, e) for e in cfg.extents for d in dirs]

    t = np.arange(T) * cfg.dt
    t_go = n_prep * cfg.dt
    arg = (t - t_go - cfg.peak_speed_time) / cfg.speed_width
    speed = np.exp(-0.5 * arg ** 2)
    # Reaches are slightly curved, so the lateral velocity component is the
    # derivative of the speed profile.  This matters for identifiability: with
    # perfectly straight reaches the canonical reach-plane coordinate would be
    # confined to a line, the second row of any readout from latent to reach
    # plane would be unconstrained, and no method could be scored on recovering
    # a rotation of that plane.
    lateral = cfg.curvature * (-arg) * np.exp(-0.5 * arg ** 2)
    speed[:n_prep] = 0.0
    lateral[:n_prep] = 0.0

    U = np.zeros((len(conds), T, 3), dtype=np.float32)
    Yt = np.zeros((len(conds), T, 2), dtype=np.float32)
    for i, (phi, ext) in enumerate(conds):
        c, sn = np.cos(phi), np.sin(phi)
        U[i, :, 0] = ext * c                # target cue, on throughout
        U[i, :, 1] = ext * sn
        U[i, n_prep:, 2] = 1.0              # go signal
        Yt[i, :, 0] = ext * (speed * c - lateral * sn)
        Yt[i, :, 1] = ext * (speed * sn + lateral * c)
    return (
        torch.as_tensor(U),
        torch.as_tensor(Yt),
        np.array([c[0] for c in conds]),
        np.array([c[1] for c in conds]),
    )


def train_reach_rnn(cfg: ReachTaskConfig, seed: int = 0, verbose: bool = False):
    """Train the RNN on centre-out reaching; returns ``(model, history, task)``."""
    torch.manual_seed(seed)
    U, Yt, phis, exts = _task_tensors(cfg)
    model = ReachRNN(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.train_steps)
    hist = []
    for step in range(cfg.train_steps):
        opt.zero_grad()
        X, Y = model(U, noise=True)
        loss_task = ((Y - Yt) ** 2).mean()
        loss = (
            loss_task
            + cfg.reg_rate * (torch.tanh(X) ** 2).mean()
            + cfg.reg_weight * (model.W ** 2).mean()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        hist.append(float(loss_task.detach()))
        if verbose and step % 250 == 0:
            print(f"  rnn step {step:4d}  task loss {hist[-1]:.5f}")
    return model, hist, (U, Yt, phis, exts)


def simulate_motor_cortex(
    n_recorded: int = 100,
    n_trials: int = 12,
    cfg: ReachTaskConfig | None = None,
    noise: float = 0.15,
    noise_kind: str = "gaussian",
    readout_mixing: bool = True,
    seed: int = 0,
    verbose: bool = False,
) -> ContextualDataset:
    """Train the reach RNN and package its activity as a contextual dataset.

    ``n_trials`` repeats of every condition are simulated with independent
    intrinsic noise; ``readout_mixing`` passes the rates through a random dense
    projection so that the recorded population is a mixture of network units,
    as with real electrode arrays.

    Samples are ``(trial, time)`` pairs and are paired across contexts by
    construction: sample ``s`` corresponds to the same trial repeat and the same
    time bin in every reach condition.
    """
    cfg = cfg or ReachTaskConfig()
    model, hist, (U, Yt, phis, exts) = train_reach_rnn(cfg, seed=seed, verbose=verbose)

    n_cond, T, _ = U.shape
    torch.manual_seed(seed + 991)
    rng = np.random.default_rng(seed + 991)

    rates = np.zeros((n_trials, T, n_cond, cfg.n_units), dtype=np.float32)
    with torch.no_grad():
        for tr in range(n_trials):
            X, _ = model(U, noise=True)
            rates[tr] = torch.tanh(X).permute(1, 0, 2).numpy()

    if readout_mixing:
        P = rng.normal(0, 1.0 / np.sqrt(cfg.n_units), size=(cfg.n_units, n_recorded))
        obs = rates @ P
    else:
        keep = rng.choice(cfg.n_units, size=min(n_recorded, cfg.n_units), replace=False)
        obs = rates[..., keep]

    n_samples = n_trials * T
    activity = obs.reshape(n_samples, n_cond, -1)
    activity = add_observation_noise(activity, noise, noise_kind, rng)

    # Latent bookkeeping: normalised time within trial (the shared "phase" of
    # the movement) plus a movement-epoch indicator.
    tt = np.tile(np.arange(T) / (T - 1.0), n_trials)
    trial_idx = np.repeat(np.arange(n_trials), T)
    n_prep = int(round(cfg.t_prep / cfg.dt))
    epoch = (np.tile(np.arange(T), n_trials) >= n_prep).astype(np.float32)
    latent = np.stack([tt, epoch], axis=1).astype(np.float32)

    contexts = []
    for i in range(n_cond):
        phi, ext = float(phis[i]), float(exts[i])
        c, s = np.cos(phi), np.sin(phi)
        contexts.append(
            ContextSpec(
                name=f"reach {np.degrees(phi):.0f}deg, extent {ext:g}",
                features=np.zeros(0),
                matrix=ext * np.array([[c, -s], [s, c]]),
                offset=np.zeros(2),
                group_params={"angle": phi, "extent": ext},
            )
        )
    # Reference = 0 deg at full extent; identity by construction.
    k = len(contexts)
    for i, spec in enumerate(contexts):
        f = np.zeros(k)
        f[i] = 1.0
        spec.features = f

    return ContextualDataset(
        activity=activity.astype(np.float32),
        latent=latent,
        context_features=np.stack([s.features for s in contexts]).astype(np.float32),
        contexts=contexts,
        reference=0,
        time_index=np.tile(np.arange(T), n_trials),
        trial_index=trial_idx,
        meta={
            "simulation": "motor_cortex",
            "n_time": T,
            "n_prep": n_prep,
            "dt": cfg.dt,
            "angles": phis,
            "extents": exts,
            "rnn_loss": hist[-1],
            "rnn_loss_curve": hist,
            "target_output": Yt.numpy(),
            "unit_rates": rates.mean(0),      # (T, cond, units) trial-averaged
            "latent_topology": "trajectory",
            "true_latent_dim": 2,
        },
    )


def equivariance_defect(dataset: ContextualDataset, n_comp: int = 8) -> dict:
    """How equivariant is the *simulated circuit itself*?

    For every pair of reach directions at equal extent we fit the optimal
    orthogonal map between the trial-averaged, PCA-reduced trajectories
    (orthogonal Procrustes) and report the normalised residual.  This is an
    empirical ceiling: no method that models context as a latent isometry can
    do better than the circuit's own departure from exact equivariance.
    """
    from scipy.linalg import orthogonal_procrustes
    from sklearn.decomposition import PCA

    n_time = dataset.meta["n_time"]
    A = dataset.activity.reshape(-1, n_time, dataset.n_contexts, dataset.n_neurons).mean(0)
    flat = A.reshape(-1, dataset.n_neurons)
    pca = PCA(n_components=min(n_comp, flat.shape[1])).fit(flat)
    Z = pca.transform(flat).reshape(n_time, dataset.n_contexts, -1)

    exts = dataset.meta["extents"]
    res = []
    for a in range(dataset.n_contexts):
        for b in range(dataset.n_contexts):
            if a == b or exts[a] != exts[b]:
                continue
            Za, Zb = Z[:, a], Z[:, b]
            R, _ = orthogonal_procrustes(Za, Zb)
            res.append(np.linalg.norm(Za @ R - Zb) / (np.linalg.norm(Zb) + 1e-9))
    return {
        "equivariance_residual_mean": float(np.mean(res)),
        "equivariance_residual_sem": float(np.std(res, ddof=1) / np.sqrt(len(res))),
        "n_pairs": len(res),
    }

r"""Learned gauge transformations on a neural latent manifold.

The paper models observed population activity as

    x_c = f( T_c(z) ),                                                    (1)

with a *shared* observation map ``f`` and a context-dependent latent
transformation ``T_c``.  This module implements ``T_c`` as the exponential of a
context-dependent element of a **learned finite-dimensional Lie algebra**.

Linear gauge
------------
Let ``G_1, ..., G_K`` be learned matrices spanning a subspace
``g = span{G_k} \subset gl(d)`` and let ``theta: C -> R^K`` be a learned
context embedding.  Then

    A(c) = sum_k theta_k(c) G_k,        T_c(z) = exp( A(c) ) z .          (2)

This construction gives three of the desired group properties *for free*:

* **Identity.**  ``theta(c_ref) = 0`` (enforced exactly by anchoring, see
  ``anchor``) gives ``T_{c_ref} = I``.
* **Invertibility.**  ``T_c^{-1} = exp(-A(c))`` exactly, for every ``c``.
* **Smoothness.**  ``c -> T_c`` is smooth because ``exp`` and ``theta`` are.

**Composition** is the non-trivial one, and it is exactly where the geometry
lives.  ``exp(A) exp(B) = exp(A + B + 1/2 [A,B] + ...)`` by Baker--Campbell--
Hausdorff, so the family ``{T_c}`` is closed under composition iff ``g`` is a
Lie subalgebra, i.e. iff ``[G_i, G_j] \in g`` for all ``i, j``.  We do not
impose this; we *measure* it.  Projecting the commutators onto ``g`` by least
squares yields the structure constants

    [G_i, G_j] = sum_k f^k_{ij} G_k + R_{ij},                             (3)

and the normalised residual ``||R_{ij}|| / ||[G_i,G_j]||`` is the **Lie closure
defect** reported throughout the paper.  Given ``f``, the composition law on
the algebra is the truncated BCH series, which is what
:meth:`LinearGaugeField.compose` implements and what the gauge consistency loss
and score are built on.

Flow gauge
----------
Replacing matrices by vector fields generalises (2) to diffeomorphisms.  With
learned fields ``V_k: R^d -> R^d`` the transformation is the time-one flow of
``sum_k theta_k V_k``.  We parameterise ``V_k(z) = G_k z + v_k + eta m_k(z)``
so that ``eta = 0`` recovers the linear gauge exactly; the flow is integrated
with RK4 and inverted by integrating the negated field, so invertibility again
holds up to integrator error only.

Gauge fixing
------------
Equation (1) is invariant under ``z -> S z``, ``T_c -> S T_c S^{-1}``,
``f -> f o S^{-1}`` for any invertible ``S``: the latent frame is only defined
up to conjugation.  Every recovery metric in :mod:`gnd.geometry.metrics` is
therefore computed either modulo conjugation or from conjugation-invariant
quantities (eigenvalue spectra).  Anchoring ``theta(c_ref) = 0`` fixes the
remaining global left-multiplication freedom.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# context embedding
# ---------------------------------------------------------------------------
class ContextEncoder(nn.Module):
    """Maps observable context variables to Lie-algebra coefficients.

    Works both for categorical contexts (one-hot ``features``) and for
    continuous cue vectors.  When ``anchor`` is set the output at the reference
    context is subtracted, which makes ``T_{c_ref} = I`` hold exactly.  This is
    a *gauge fixing* choice: it removes the global left-multiplication
    redundancy of (1) rather than adding modelling power.
    """

    def __init__(
        self,
        n_features: int,
        n_generators: int,
        hidden: int = 64,
        depth: int = 2,
        anchor: bool = True,
        init_scale: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = n_features
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(d_in, hidden), nn.Tanh()]
            d_in = hidden
        head = nn.Linear(d_in, n_generators)
        nn.init.normal_(head.weight, std=init_scale)
        nn.init.zeros_(head.bias)
        layers.append(head)
        self.net = nn.Sequential(*layers)
        self.anchor = anchor
        self.n_generators = n_generators

    def forward(self, features: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
        theta = self.net(features)
        if self.anchor and reference is not None:
            theta = theta - self.net(reference.unsqueeze(0) if reference.dim() == 1 else reference)
        return theta


# ---------------------------------------------------------------------------
# generator bank
# ---------------------------------------------------------------------------
_ALGEBRAS = ("gl", "so", "sl", "aff", "se")


class GeneratorBank(nn.Module):
    """A learned basis of a candidate Lie algebra.

    ``algebra`` selects the ambient algebra the generators are projected onto:

    ``gl``   unconstrained ``gl(d)``;
    ``so``   antisymmetric -> ``exp`` is a rotation (isometric gauge);
    ``sl``   traceless -> ``exp`` is volume preserving;
    ``aff``  affine ``gl(d) x| R^d`` acting on homogeneous coordinates;
    ``se``   Euclidean ``so(d) x| R^d`` (rigid motions).

    Generators are normalised to unit Frobenius norm on every forward pass,
    which removes the ``theta <-> G`` scale redundancy and makes the reported
    closure defects and structure constants comparable across runs.
    """

    def __init__(self, n_latent: int, n_generators: int, algebra: str = "gl", init_scale: float = 0.5):
        super().__init__()
        if algebra not in _ALGEBRAS:
            raise ValueError(f"algebra must be one of {_ALGEBRAS}, got {algebra!r}")
        self.algebra = algebra
        self.n_latent = n_latent
        self.n_generators = n_generators
        self.affine = algebra in ("aff", "se")
        self.dim = n_latent + 1 if self.affine else n_latent
        self.raw = nn.Parameter(torch.randn(n_generators, n_latent, n_latent) * init_scale / math.sqrt(n_latent))
        if self.affine:
            self.raw_shift = nn.Parameter(torch.randn(n_generators, n_latent) * init_scale / math.sqrt(n_latent))
        else:
            self.register_parameter("raw_shift", None)

    # -- construction ------------------------------------------------------
    def generators(self, normalise: bool = True) -> torch.Tensor:
        """Return the generator bank, shape ``(K, dim, dim)``."""
        A = self.raw
        if self.algebra in ("so", "se"):
            A = 0.5 * (A - A.transpose(-1, -2))
        elif self.algebra == "sl":
            tr = A.diagonal(dim1=-2, dim2=-1).sum(-1) / self.n_latent
            A = A - tr[:, None, None] * torch.eye(self.n_latent, device=A.device, dtype=A.dtype)
        if self.affine:
            K = A.shape[0]
            G = A.new_zeros(K, self.dim, self.dim)
            G[:, : self.n_latent, : self.n_latent] = A
            G[:, : self.n_latent, self.n_latent] = self.raw_shift
        else:
            G = A
        if normalise:
            nrm = G.flatten(1).norm(dim=1).clamp_min(1e-8)
            G = G / nrm[:, None, None]
        return G

    # -- algebra structure -------------------------------------------------
    def structure_constants(self, generators: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Least-squares structure constants and commutator residuals.

        Solves ``[G_i, G_j] \approx sum_k f^k_{ij} G_k`` in Frobenius norm and
        returns ``(f, residual)`` with ``f`` of shape ``(K, K, K)`` indexed
        ``f[i, j, k]`` and ``residual`` of shape ``(K, K)`` holding
        ``||R_{ij}||_F / ||[G_i, G_j]||_F``.
        """
        G = self.generators() if generators is None else generators
        K = G.shape[0]
        Gf = G.reshape(K, -1)                                   # (K, m^2)
        comm = torch.einsum("iab,jbc->ijac", G, G) - torch.einsum("jab,ibc->ijac", G, G)
        Cf = comm.reshape(K * K, -1)                            # (K^2, m^2)
        sol = torch.linalg.lstsq(Gf.T, Cf.T).solution           # (K, K^2)
        f = sol.T.reshape(K, K, K)                              # f[i, j, k]
        recon = torch.einsum("ijk,kab->ijab", f, G)
        num = (comm - recon).flatten(2).norm(dim=2)
        den = comm.flatten(2).norm(dim=2).clamp_min(1e-8)
        return f, num / den

    def closure_defect(self) -> torch.Tensor:
        """Mean normalised commutator residual over distinct generator pairs.

        Zero means ``span{G_k}`` is a genuine Lie subalgebra, so the learned
        transformations form a group; one means the commutators leave the span
        entirely.  Pairs whose commutator is numerically zero (already-closed
        abelian directions) are excluded, since their residual is undefined.
        """
        G = self.generators()
        f, res = self.structure_constants(G)
        K = G.shape[0]
        comm = torch.einsum("iab,jbc->ijac", G, G) - torch.einsum("jab,ibc->ijac", G, G)
        mag = comm.flatten(2).norm(dim=2)
        mask = (~torch.eye(K, dtype=torch.bool, device=G.device)) & (mag > 1e-4)
        if mask.sum() == 0:
            return torch.zeros((), device=G.device)
        return res[mask].mean()

    def commutator_norm(self) -> torch.Tensor:
        """Mean ``||[G_i, G_j]||_F`` over distinct pairs -- an abelianness index.

        Because generators are unit-normalised this is directly comparable
        across runs; values near zero indicate a commuting (abelian) algebra,
        which is the prediction for pure grid-cell phase translations.
        """
        G = self.generators()
        K = G.shape[0]
        comm = torch.einsum("iab,jbc->ijac", G, G) - torch.einsum("jab,ibc->ijac", G, G)
        mag = comm.flatten(2).norm(dim=2)
        off = ~torch.eye(K, dtype=torch.bool, device=G.device)
        return mag[off].mean()


# ---------------------------------------------------------------------------
# linear (matrix-exponential) gauge
# ---------------------------------------------------------------------------
class LinearGaugeField(nn.Module):
    """``T_c(z) = exp(sum_k theta_k(c) G_k) z``.

    Exactly invertible, exactly the identity at the reference context, and with
    a closed-form composition law on the algebra given by BCH.
    """

    def __init__(
        self,
        n_latent: int,
        n_context_features: int,
        n_generators: int = 6,
        algebra: str = "gl",
        context_hidden: int = 64,
        context_depth: int = 2,
        anchor: bool = True,
        bch_order: int = 2,
    ):
        super().__init__()
        self.bank = GeneratorBank(n_latent, n_generators, algebra)
        self.context = ContextEncoder(
            n_context_features, n_generators, context_hidden, context_depth, anchor
        )
        self.n_latent = n_latent
        self.affine = self.bank.affine
        self.bch_order = bch_order

    # -- coefficients ------------------------------------------------------
    def coefficients(self, features: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
        return self.context(features, reference)

    def algebra_element(self, theta: torch.Tensor) -> torch.Tensor:
        """``A(theta) = sum_k theta_k G_k``; ``theta`` is ``(..., K)``."""
        return torch.einsum("...k,kab->...ab", theta, self.bank.generators())

    def matrices(self, theta: torch.Tensor) -> torch.Tensor:
        """Group elements ``exp(A(theta))``, shape ``(..., dim, dim)``."""
        return torch.matrix_exp(self.algebra_element(theta))

    # -- action ------------------------------------------------------------
    def _act(self, M: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.affine:
            z1 = torch.cat([z, torch.ones_like(z[..., :1])], dim=-1)
            out = torch.einsum("...ab,...b->...a", M, z1)
            return out[..., : self.n_latent] / out[..., self.n_latent:].clamp_min(1e-6)
        return torch.einsum("...ab,...b->...a", M, z)

    def transform(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """``T_c(z)``.  Broadcasting: ``z`` is ``(N, d)`` and ``theta`` ``(N, K)``
        or ``(K,)``."""
        return self._act(self.matrices(theta), z)

    def inverse(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """``T_c^{-1}(z) = exp(-A(theta)) z``; exact, not an approximation."""
        return self._act(self.matrices(-theta), z)

    def act(self, M: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Apply pre-computed group elements.

        ``theta`` depends only on the context, so ``exp`` needs to be evaluated
        once per context rather than once per sample; the training loop caches
        the ``(C, dim, dim)`` matrices and calls this instead of
        :meth:`transform`.
        """
        return self._act(M, z)

    forward = transform

    # -- group law ---------------------------------------------------------
    def bracket(self, a: torch.Tensor, b: torch.Tensor, f: torch.Tensor | None = None) -> torch.Tensor:
        """Lie bracket in coefficient space, ``[a, b]^k = f^k_{ij} a^i b^j``."""
        if f is None:
            f, _ = self.bank.structure_constants()
        return torch.einsum("ijk,...i,...j->...k", f, a, b)

    def compose(self, a: torch.Tensor, b: torch.Tensor, order: int | None = None) -> torch.Tensor:
        """Truncated BCH composition ``theta(a . b)``.

        Order 1 is the abelian law ``a + b``; order 2 adds ``1/2 [a, b]``;
        order 3 adds the two cubic terms.  Exact whenever the learned algebra
        closes and the series converges.
        """
        order = self.bch_order if order is None else order
        f, _ = self.bank.structure_constants()
        out = a + b
        if order >= 2:
            ab = self.bracket(a, b, f)
            out = out + 0.5 * ab
            if order >= 3:
                out = out + (1.0 / 12.0) * (
                    self.bracket(a, ab, f) - self.bracket(b, ab, f)
                )
        return out

    # -- diagnostics -------------------------------------------------------
    def closure_defect(self) -> torch.Tensor:
        return self.bank.closure_defect()

    def commutator_norm(self) -> torch.Tensor:
        return self.bank.commutator_norm()

    def identity_defect(self, reference_features: torch.Tensor) -> torch.Tensor:
        """``||T_{c_ref} - I||_F``; exactly zero when ``anchor=True``."""
        theta = self.coefficients(reference_features.unsqueeze(0), reference_features)
        M = self.matrices(theta)[0]
        return (M - torch.eye(M.shape[-1], device=M.device)).norm()

    def inverse_defect(self, theta: torch.Tensor) -> torch.Tensor:
        M = self.matrices(theta)
        Mi = self.matrices(-theta)
        eye = torch.eye(M.shape[-1], device=M.device).expand_as(M)
        return (M @ Mi - eye).flatten(1).norm(dim=1).mean()


# ---------------------------------------------------------------------------
# flow (Neural-ODE) gauge
# ---------------------------------------------------------------------------
class FlowGaugeField(nn.Module):
    r"""``T_c`` = time-one flow of ``sum_k theta_k(c) V_k``.

    The learned vector fields are

        V_k(z) = G_k z + v_k + eta * m_k(z),

    with ``m_k`` a small MLP.  Setting ``eta = 0`` reduces the flow exactly to
    the linear gauge (up to RK4 truncation error), so this class is a strict
    generalisation used for the non-affine morph contexts.  The Lie bracket of
    vector fields, ``[V_i, V_j] = DV_j V_i - DV_i V_j``, is evaluated with
    autograd and projected pointwise onto ``span{V_k}`` to give the closure
    defect, exactly mirroring the matrix case.
    """

    def __init__(
        self,
        n_latent: int,
        n_context_features: int,
        n_generators: int = 6,
        hidden: int = 64,
        depth: int = 2,
        context_hidden: int = 64,
        context_depth: int = 2,
        anchor: bool = True,
        n_steps: int = 6,
        nonlinearity: float = 0.5,
        bch_order: int = 2,
        algebra: str = "gl",
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_generators = n_generators
        self.n_steps = n_steps
        self.eta = nonlinearity
        self.bch_order = bch_order
        self.algebra = algebra
        self.affine = False
        self.linear = nn.Parameter(torch.randn(n_generators, n_latent, n_latent) * 0.5 / math.sqrt(n_latent))
        self.shift = nn.Parameter(torch.zeros(n_generators, n_latent))
        layers: list[nn.Module] = []
        d_in = n_latent
        for _ in range(depth):
            layers += [nn.Linear(d_in, hidden), nn.Tanh()]
            d_in = hidden
        layers.append(nn.Linear(d_in, n_generators * n_latent))
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.mlp = nn.Sequential(*layers)
        self.context = ContextEncoder(
            n_context_features, n_generators, context_hidden, context_depth, anchor
        )
        self.register_buffer("_f_cache", torch.zeros(n_generators, n_generators, n_generators))
        self.register_buffer("_f_valid", torch.zeros((), dtype=torch.bool))

    # -- fields ------------------------------------------------------------
    def linear_part(self) -> torch.Tensor:
        """The linear coefficient of each field, after the algebra projection.

        With ``algebra="so"`` the linear part is antisymmetric, so with
        ``nonlinearity=0`` the flow is an exact rotation.  On a periodic latent
        manifold this matters: an unconstrained linear part lets the flow
        expand or contract and destroys the torus.
        """
        A = self.linear
        if self.algebra in ("so", "se"):
            A = 0.5 * (A - A.transpose(-1, -2))
        elif self.algebra == "sl":
            tr = A.diagonal(dim1=-2, dim2=-1).sum(-1) / self.n_latent
            A = A - tr[:, None, None] * torch.eye(self.n_latent, device=A.device, dtype=A.dtype)
        return A

    def basis_fields(self, z: torch.Tensor) -> torch.Tensor:
        """``V_k(z)`` for all ``k``; shape ``(N, K, d)``."""
        lin = torch.einsum("kab,nb->nka", self.linear_part(), z) + self.shift[None]
        if self.eta == 0:
            return lin
        nl = self.mlp(z).reshape(z.shape[0], self.n_generators, self.n_latent)
        return lin + self.eta * nl

    def velocity(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        V = self.basis_fields(z)
        if theta.dim() == 1:
            theta = theta.expand(z.shape[0], -1)
        return torch.einsum("nk,nka->na", theta, V)

    def coefficients(self, features: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
        return self.context(features, reference)

    # -- flow --------------------------------------------------------------
    def _integrate(self, z: torch.Tensor, theta: torch.Tensor, sign: float) -> torch.Tensor:
        h = sign / self.n_steps
        for _ in range(self.n_steps):
            k1 = self.velocity(z, theta)
            k2 = self.velocity(z + 0.5 * h * k1, theta)
            k3 = self.velocity(z + 0.5 * h * k2, theta)
            k4 = self.velocity(z + h * k3, theta)
            z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return z

    def transform(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self._integrate(z, theta, +1.0)

    def inverse(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self._integrate(z, theta, -1.0)

    forward = transform

    # -- algebra structure -------------------------------------------------
    def _jacobian_action(self, z: torch.Tensor, k: int, w: torch.Tensor) -> torch.Tensor:
        """``D V_k(z) w``, the Jacobian-vector product.

        A single ``autograd.grad`` of ``V_k`` with ``grad_outputs=w`` would give
        the *vector*-Jacobian product ``J^T w``, which is a different quantity
        whenever ``J`` is not symmetric -- and the linear part of a gauge
        generator is never symmetric.  The double-backward below is the standard
        way to get ``J w`` from reverse mode: differentiating ``J^T v`` with
        respect to the dummy ``v`` contracts the transpose away.

        ``torch.enable_grad`` is forced on so that the bracket can also be
        evaluated inside ``no_grad`` blocks at analysis time.  The result is
        always detached: structure constants enter the objective as *derived
        constants*, in the same spirit as a target network, which avoids a
        double-backward through the vector-field MLP on every step.
        """
        with torch.enable_grad():
            zz = z.detach().requires_grad_(True)
            v = torch.zeros_like(zz, requires_grad=True)
            Vk = self.basis_fields(zz)[:, k]
            (vjp,) = torch.autograd.grad(Vk, zz, grad_outputs=v, create_graph=True)
            (jvp,) = torch.autograd.grad(vjp, v, grad_outputs=w.detach(), retain_graph=False)
        return jvp.detach()

    def structure_constants(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pointwise least-squares structure constants of the vector fields.

        Uses ``[V_i, V_j] = D V_j V_i - D V_i V_j`` and projects onto
        ``span{V_k}`` over the probe batch, exactly mirroring the matrix case.
        """
        K = self.n_generators
        with torch.no_grad():
            V = self.basis_fields(z).detach()                     # (N, K, d)
        comm = []
        for i in range(K):
            row = []
            for j in range(K):
                if i == j:
                    row.append(torch.zeros_like(V[:, 0]))
                    continue
                row.append(self._jacobian_action(z, j, V[:, i]) - self._jacobian_action(z, i, V[:, j]))
            comm.append(torch.stack(row, dim=1))
        comm = torch.stack(comm, dim=1)                           # (N, K, K, d)
        N = z.shape[0]
        A = V.permute(0, 2, 1).reshape(N * self.n_latent, K)
        Y = comm.permute(0, 3, 1, 2).reshape(N * self.n_latent, K * K)
        sol = torch.linalg.lstsq(A, Y).solution                   # (K, K^2)
        f = sol.T.reshape(K, K, K)
        recon = torch.einsum("ijk,nka->nija", f, V)
        num = (comm - recon).norm(dim=-1).mean(0)
        den = comm.norm(dim=-1).mean(0).clamp_min(1e-8)
        return f, num / den

    def cached_structure_constants(self, z: torch.Tensor, refresh: bool = False) -> torch.Tensor:
        """Structure constants, recomputed only when asked.

        The bracket needs ``K^2`` Jacobian-vector products, so refreshing it on
        every optimisation step dominates the cost of the flow gauge.  The
        training loop refreshes periodically instead.
        """
        if refresh or not bool(self._f_valid):
            f, _ = self.structure_constants(z)
            self._f_cache.copy_(f)
            self._f_valid.fill_(True)
        return self._f_cache

    def closure_defect(self, z: torch.Tensor | None = None) -> torch.Tensor:
        if z is None:
            z = torch.randn(64, self.n_latent, device=self.linear.device)
        _, res = self.structure_constants(z)
        K = self.n_generators
        off = ~torch.eye(K, dtype=torch.bool, device=res.device)
        return res[off].mean()

    def commutator_norm(self, z: torch.Tensor | None = None) -> torch.Tensor:
        if z is None:
            z = torch.randn(64, self.n_latent, device=self.linear.device)
        K = self.n_generators
        with torch.no_grad():
            V = self.basis_fields(z).detach()
        scale = V.norm(dim=-1).mean().clamp_min(1e-8)
        vals = []
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                c = self._jacobian_action(z, j, V[:, i]) - self._jacobian_action(z, i, V[:, j])
                vals.append(c.norm(dim=-1).mean() / scale)
        return torch.stack(vals).mean()

    def bracket(self, a: torch.Tensor, b: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ijk,...i,...j->...k", f, a, b)

    def compose(
        self, a: torch.Tensor, b: torch.Tensor, z: torch.Tensor,
        order: int | None = None, refresh: bool = False,
    ) -> torch.Tensor:
        order = self.bch_order if order is None else order
        f = self.cached_structure_constants(z, refresh)
        out = a + b
        if order >= 2:
            ab = self.bracket(a, b, f)
            # Minus, not plus.  ``f`` are the structure constants of the *vector
            # field* bracket, which for a linear field carries the opposite sign
            # to the matrix commutator: [V_A, V_B] = V_{-[A,B]}.  BCH in matrix
            # coordinates is A + B + (1/2)[A,B], so in these coordinates the
            # second-order term is -(1/2)[a,b].  The third-order term is
            # unaffected: both of its brackets flip sign, and the two flips
            # cancel.
            out = out - 0.5 * ab
            if order >= 3:
                out = out + (1.0 / 12.0) * (self.bracket(a, ab, f) - self.bracket(b, ab, f))
        return out

    def identity_defect(self, reference_features: torch.Tensor) -> torch.Tensor:
        z = torch.randn(64, self.n_latent, device=self.linear.device)
        theta = self.coefficients(reference_features.unsqueeze(0), reference_features)
        return (self.transform(z, theta[0]) - z).norm(dim=-1).mean()

    def inverse_defect(self, theta: torch.Tensor) -> torch.Tensor:
        z = torch.randn(64, self.n_latent, device=self.linear.device)
        errs = []
        for t in theta:
            errs.append((self.inverse(self.transform(z, t), t) - z).norm(dim=-1).mean())
        return torch.stack(errs).mean()


def build_gauge_field(kind: str, **kwargs) -> nn.Module:
    """Factory: ``kind`` in ``{"linear", "flow", "none"}``."""
    if kind == "linear":
        kwargs.pop("hidden", None)
        kwargs.pop("depth", None)
        kwargs.pop("n_steps", None)
        kwargs.pop("nonlinearity", None)
        return LinearGaugeField(**kwargs)
    if kind == "flow":
        return FlowGaugeField(**kwargs)
    if kind == "none":
        return IdentityGauge(kwargs["n_latent"], kwargs["n_context_features"], kwargs.get("n_generators", 1))
    raise ValueError(f"unknown gauge kind {kind!r}")


class IdentityGauge(nn.Module):
    """Ablation stub: ``T_c = id`` for every context (Ablation 1)."""

    def __init__(self, n_latent: int, n_context_features: int, n_generators: int = 1):
        super().__init__()
        self.n_latent = n_latent
        self.n_generators = n_generators
        self.affine = False
        self._dummy = nn.Parameter(torch.zeros(1))

    def coefficients(self, features: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
        return features.new_zeros(features.shape[0], self.n_generators) + 0.0 * self._dummy

    def transform(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return z + 0.0 * self._dummy

    def inverse(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return z + 0.0 * self._dummy

    forward = transform

    def compose(self, a: torch.Tensor, b: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return a + b

    def closure_defect(self, *args, **kwargs) -> torch.Tensor:
        return self._dummy.new_zeros(())

    def commutator_norm(self, *args, **kwargs) -> torch.Tensor:
        return self._dummy.new_zeros(())

    def identity_defect(self, *args, **kwargs) -> torch.Tensor:
        return self._dummy.new_zeros(())

    def inverse_defect(self, *args, **kwargs) -> torch.Tensor:
        return self._dummy.new_zeros(())

    def matrices(self, theta: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(self.n_latent, device=theta.device)
        return eye.expand(*theta.shape[:-1], self.n_latent, self.n_latent)

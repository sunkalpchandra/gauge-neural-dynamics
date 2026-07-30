"""Baseline latent-variable and alignment methods, on a common footing with GND.

Every baseline is asked to produce exactly what GND produces:

* ``w`` -- an *observed-frame* latent for each sample in each context;
* ``matrices`` -- an estimated linear map ``T_c`` taking the canonical frame to
  context ``c``'s frame;
* a generative pathway able to turn a latent back into population activity.

The canonical latent is then ``z_c = T_c^{-1} w_c`` and every metric in
:mod:`gnd.geometry.metrics` applies unchanged.  Methods that do not estimate a
transformation themselves (PCA, UMAP, AE, VAE) get one fitted post hoc by least
squares *on the training split*, which is a deliberately generous choice: it
hands them the paired correspondence that GND's correspondence-free variant
does not use.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..simulations.base import ContextualDataset
from ..utils.common import set_seed


# ---------------------------------------------------------------------------
# common machinery
# ---------------------------------------------------------------------------
@dataclass
class BaselineResult:
    """Uniform container returned by every baseline."""

    name: str
    w_train: np.ndarray                       # (N_tr, C, d) observed-frame latents
    w_test: np.ndarray                        # (N_te, C, d)
    matrices: np.ndarray                      # (C, d, d) canonical -> context c
    recon_test: np.ndarray | None = None      # (N_te, C, P) native reconstruction
    meta: dict = field(default_factory=dict)

    @property
    def n_latent(self) -> int:
        return self.w_train.shape[-1]

    def canonical(self, which: str = "test") -> np.ndarray:
        w = self.w_test if which == "test" else self.w_train
        inv = np.stack([np.linalg.pinv(M) for M in self.matrices])
        return np.einsum("cab,ncb->nca", inv, w)


def estimate_linear_transforms(w: np.ndarray, reference: int = 0, ridge: float = 1e-6) -> np.ndarray:
    """Least-squares maps ``T_c`` with ``w_c ~ T_c w_ref``, fitted on paired data.

    Returns ``(C, d, d)``.  ``T_ref`` is the identity by construction.
    """
    w = np.asarray(w, float)
    N, C, d = w.shape
    X = w[:, reference]
    G = X.T @ X + ridge * np.eye(d)
    out = np.zeros((C, d, d))
    for c in range(C):
        out[c] = np.linalg.solve(G, X.T @ w[:, c]).T
    out[reference] = np.eye(d)
    return out


class RandomFeatureReadout:
    """Closed-form non-linear readout ``latent -> activity`` (RFF ridge).

    Used as the *common* generative pathway when comparing transport accuracy
    across methods, so that differences reflect the latent and the
    transformation rather than decoder capacity.
    """

    def __init__(self, n_features: int = 512, gamma: float | None = None, ridge: float = 1e-3, seed: int = 0):
        self.n_features, self.gamma, self.ridge, self.seed = n_features, gamma, ridge, seed

    def fit(self, Z: np.ndarray, X: np.ndarray) -> "RandomFeatureReadout":
        Z, X = np.asarray(Z, float), np.asarray(X, float)
        self.mu_, self.sd_ = Z.mean(0), Z.std(0) + 1e-8
        Zs = (Z - self.mu_) / self.sd_
        d = Zs.shape[1]
        g = self.gamma if self.gamma is not None else 1.0 / d
        rng = np.random.default_rng(self.seed)
        self.W_ = rng.normal(0, np.sqrt(2 * g), size=(d, self.n_features))
        self.b_ = rng.uniform(0, 2 * np.pi, size=self.n_features)
        F = self._features(Zs)
        A = F.T @ F + self.ridge * len(F) * np.eye(F.shape[1])
        self.C_ = np.linalg.solve(A, F.T @ X)
        return self

    def _features(self, Zs: np.ndarray) -> np.ndarray:
        P = np.sqrt(2.0 / self.n_features) * np.cos(Zs @ self.W_ + self.b_)
        return np.concatenate([P, np.ones((len(P), 1))], axis=1)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        Zs = (np.asarray(Z, float) - self.mu_) / self.sd_
        return self._features(Zs) @ self.C_


# ---------------------------------------------------------------------------
# linear / classical embeddings
# ---------------------------------------------------------------------------
def baseline_pca(train: ContextualDataset, test: ContextualDataset, n_latent: int, seed: int = 0, **_) -> BaselineResult:
    """PCA on activity pooled over samples and contexts."""
    from sklearn.decomposition import PCA

    Xtr = train.activity.reshape(-1, train.n_neurons)
    pca = PCA(n_components=n_latent, random_state=seed).fit(Xtr)
    wt = pca.transform(train.activity.reshape(-1, train.n_neurons)).reshape(train.n_samples, train.n_contexts, -1)
    we = pca.transform(test.activity.reshape(-1, test.n_neurons)).reshape(test.n_samples, test.n_contexts, -1)
    rec = pca.inverse_transform(we.reshape(-1, n_latent)).reshape(test.n_samples, test.n_contexts, -1)
    return BaselineResult("PCA", wt, we, estimate_linear_transforms(wt, train.reference), rec)


def baseline_umap(train: ContextualDataset, test: ContextualDataset, n_latent: int, seed: int = 0,
                  n_neighbors: int = 20, min_dist: float = 0.1, max_fit: int = 6000, **_) -> BaselineResult:
    """UMAP embedding; a non-linear, non-parametric manifold baseline."""
    import umap

    Xtr = train.activity.reshape(-1, train.n_neurons)
    # UMAP dominates baseline runtime at >10k points; the embedding is fitted on a
    # capped subsample and then applied to everything, which is how UMAP is used
    # on large recordings anyway.
    rng = np.random.default_rng(seed)
    fit_idx = (rng.choice(len(Xtr), max_fit, replace=False)
               if len(Xtr) > max_fit else np.arange(len(Xtr)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_components=n_latent, n_neighbors=n_neighbors, min_dist=min_dist,
            random_state=seed, init="spectral", verbose=False,
        ).fit(Xtr[fit_idx])
        wt = reducer.transform(Xtr)
        we = reducer.transform(test.activity.reshape(-1, test.n_neurons))
    wt = wt.reshape(train.n_samples, train.n_contexts, -1)
    we = we.reshape(test.n_samples, test.n_contexts, -1)
    return BaselineResult("UMAP", wt, we, estimate_linear_transforms(wt, train.reference), None)


def baseline_procrustes(train: ContextualDataset, test: ContextualDataset, n_latent: int, seed: int = 0, **_) -> BaselineResult:
    """Per-context PCA followed by orthogonal Procrustes alignment to the reference.

    This is the standard way population geometry is compared across sessions or
    conditions; it can only express a rigid motion of the latent.
    """
    from scipy.linalg import orthogonal_procrustes
    from sklearn.decomposition import PCA

    C = train.n_contexts
    models = [PCA(n_components=n_latent, random_state=seed).fit(train.activity[:, c]) for c in range(C)]
    wt = np.stack([models[c].transform(train.activity[:, c]) for c in range(C)], axis=1)
    we = np.stack([models[c].transform(test.activity[:, c]) for c in range(C)], axis=1)
    ref = train.reference
    Ms = np.zeros((C, n_latent, n_latent))
    for c in range(C):
        R, _ = orthogonal_procrustes(wt[:, ref], wt[:, c])
        Ms[c] = R.T                                   # canonical -> context c
    return BaselineResult("Procrustes", wt, we, Ms, None)


def baseline_cca(train: ContextualDataset, test: ContextualDataset, n_latent: int, seed: int = 0, **_) -> BaselineResult:
    """CCA alignment of each context onto the reference context.

    Uses the paired correspondence explicitly, as in canonical-correlation
    alignment of neural populations across days.
    """
    from sklearn.cross_decomposition import CCA

    C, ref = train.n_contexts, train.reference
    Xref_tr = train.activity[:, ref]
    wt = np.zeros((train.n_samples, C, n_latent))
    we = np.zeros((test.n_samples, C, n_latent))
    for c in range(C):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cca = CCA(n_components=n_latent, max_iter=800)
            cca.fit(train.activity[:, c], Xref_tr)
        wt[:, c] = cca.transform(train.activity[:, c])
        we[:, c] = cca.transform(test.activity[:, c])
    return BaselineResult("CCA", wt, we, estimate_linear_transforms(wt, ref), None)


def baseline_manifold_alignment(train: ContextualDataset, test: ContextualDataset, n_latent: int,
                                seed: int = 0, k: int = 12, mu: float = 1.0, **_) -> BaselineResult:
    """Semi-supervised manifold alignment (Ham, Lee & Saul, 2005).

    Builds a joint graph whose within-context edges come from k-nearest
    neighbours and whose across-context edges come from the known
    correspondences, then embeds with the graph Laplacian.  Out-of-sample
    points are placed by Nystrom-style kernel regression.
    """
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh
    from sklearn.neighbors import kneighbors_graph

    rng = np.random.default_rng(seed)
    C = train.n_contexts
    n_sub = min(train.n_samples, 600)                 # dense eigenproblem: keep it small
    idx = np.sort(rng.choice(train.n_samples, n_sub, replace=False))
    blocks = [train.activity[idx, c] for c in range(C)]

    rows, cols, vals = [], [], []
    for c in range(C):
        G = kneighbors_graph(blocks[c], n_neighbors=min(k, n_sub - 1), mode="connectivity")
        G = G.maximum(G.T).tocoo()
        rows.append(G.row + c * n_sub)
        cols.append(G.col + c * n_sub)
        vals.append(np.ones_like(G.row, dtype=float))
    for a in range(C):
        for b in range(a + 1, C):
            i = np.arange(n_sub)
            rows += [i + a * n_sub, i + b * n_sub]
            cols += [i + b * n_sub, i + a * n_sub]
            vals += [np.full(n_sub, mu), np.full(n_sub, mu)]
    W = csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(C * n_sub, C * n_sub),
    )
    d = np.asarray(W.sum(1)).ravel()
    L = diags(d) - W
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals_, vecs = eigsh(L.astype(float), k=n_latent + 1, M=diags(np.clip(d, 1e-8, None)), sigma=-1e-5, which="LM")
    emb = vecs[:, 1: n_latent + 1]                     # drop the constant eigenvector
    emb = emb / (emb.std(0) + 1e-9)
    Y = emb.reshape(C, n_sub, n_latent).transpose(1, 0, 2)

    # Nystrom-style out-of-sample extension: one RFF ridge per context.
    wt = np.zeros((train.n_samples, C, n_latent))
    we = np.zeros((test.n_samples, C, n_latent))
    for c in range(C):
        rr = RandomFeatureReadout(n_features=256, ridge=1e-2, seed=seed).fit(blocks[c], Y[:, c])
        wt[:, c] = rr.predict(train.activity[:, c])
        we[:, c] = rr.predict(test.activity[:, c])
    return BaselineResult("ManifoldAlign", wt, we, estimate_linear_transforms(wt, train.reference), None)


# ---------------------------------------------------------------------------
# neural autoencoders
# ---------------------------------------------------------------------------
class _AE(nn.Module):
    def __init__(self, p: int, d: int, hidden: int, depth: int, variational: bool):
        super().__init__()
        from .encoder import _mlp

        self.variational = variational
        self.enc = _mlp(p, hidden, depth, 2 * d if variational else d, "gelu", True)
        self.dec = _mlp(d, hidden, depth, p, "gelu", True)
        self.d = d

    def encode(self, x):
        h = self.enc(x)
        if not self.variational:
            return h, None, None
        mu, logvar = h.chunk(2, -1)
        logvar = logvar.clamp(-8, 4)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        return z, mu, logvar


def _fit_ae(train: ContextualDataset, test: ContextualDataset, n_latent: int, variational: bool,
            seed: int, hidden: int = 256, depth: int = 3, epochs: int = 400, lr: float = 2e-3,
            batch: int = 256, beta: float = 1e-3, device: str = "cpu") -> BaselineResult:
    set_seed(seed)
    Xtr = torch.as_tensor(train.activity.reshape(-1, train.n_neurons), dtype=torch.float32, device=device)
    model = _AE(train.n_neurons, n_latent, hidden, depth, variational).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator(device=device).manual_seed(seed)
    n = Xtr.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g, device=device)
        for i in range(0, n, batch):
            xb = Xtr[perm[i: i + batch]]
            opt.zero_grad(set_to_none=True)
            z, mu, logvar = model.encode(xb)
            loss = ((model.dec(z) - xb) ** 2).mean()
            if variational:
                loss = loss + beta * (0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(-1)).mean()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()

    def embed(ds):
        X = torch.as_tensor(ds.activity.reshape(-1, ds.n_neurons), dtype=torch.float32, device=device)
        with torch.no_grad():
            z, _, _ = model.encode(X)
            rec = model.dec(z)
        return (z.cpu().numpy().reshape(ds.n_samples, ds.n_contexts, -1),
                rec.cpu().numpy().reshape(ds.n_samples, ds.n_contexts, -1))

    wt, _ = embed(train)
    we, rec = embed(test)
    name = "VAE" if variational else "Autoencoder"
    return BaselineResult(name, wt, we, estimate_linear_transforms(wt, train.reference), rec)


def baseline_autoencoder(train, test, n_latent, seed=0, device="cpu", epochs=400, **_):
    return _fit_ae(train, test, n_latent, False, seed, epochs=epochs, device=device)


def baseline_vae(train, test, n_latent, seed=0, device="cpu", epochs=400, **_):
    return _fit_ae(train, test, n_latent, True, seed, epochs=epochs, device=device)


BASELINES = {
    "pca": baseline_pca,
    "umap": baseline_umap,
    "autoencoder": baseline_autoencoder,
    "vae": baseline_vae,
    "cca": baseline_cca,
    "procrustes": baseline_procrustes,
    "manifold_align": baseline_manifold_alignment,
}

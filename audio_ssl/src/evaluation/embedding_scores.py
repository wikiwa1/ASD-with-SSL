from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import torch
from torch import nn


class EmbeddingScorer(Protocol):
    """One-class anomaly scorer fit on a target's normal embeddings (higher = anomalous)."""

    def fit(self, normal: np.ndarray) -> "EmbeddingScorer": ...
    def score(self, embeddings: np.ndarray) -> np.ndarray: ...


class MahalanobisScorer:
    """Squared Mahalanobis distance to the normal cluster, with Ledoit-Wolf shrinkage so
    the covariance is well-conditioned even when D is close to the number of clips."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.precision: np.ndarray | None = None

    def fit(self, normal: np.ndarray) -> "MahalanobisScorer":
        from sklearn.covariance import LedoitWolf

        self.mean = normal.mean(axis=0)
        self.precision = LedoitWolf().fit(normal).precision_.astype(np.float32)
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.precision is not None
        centered = embeddings - self.mean
        return np.einsum("nd,dk,nk->n", centered, self.precision, centered).astype(np.float64)


class KNNScorer:
    """Mean cosine distance to the k nearest normal embeddings (L2-normalized)."""

    def __init__(self, k: int = 4) -> None:
        self.k = k
        self.normal: np.ndarray | None = None

    @staticmethod
    def _l2(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    def fit(self, normal: np.ndarray) -> "KNNScorer":
        self.normal = self._l2(normal.astype(np.float32))
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.normal is not None
        sims = self._l2(embeddings.astype(np.float32)) @ self.normal.T  # cosine similarity
        k = min(self.k, sims.shape[1])
        topk = np.partition(sims, -k, axis=1)[:, -k:]
        return (1.0 - topk.mean(axis=1)).astype(np.float64)


class GMMScorer:
    """Negative log-likelihood under a full-covariance Gaussian mixture — captures the
    multi-modality that single-Gaussian Mahalanobis misses, and is robust at small N."""

    def __init__(self, n_components: int = 4, reg_covar: float = 1e-4, seed: int = 42) -> None:
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.seed = seed
        self.gmm = None

    def fit(self, normal: np.ndarray) -> "GMMScorer":
        from sklearn.mixture import GaussianMixture

        k = max(1, min(self.n_components, len(normal) // 50))  # keep enough points per mode
        self.gmm = GaussianMixture(n_components=k, covariance_type="full",
                                   reg_covar=self.reg_covar, random_state=self.seed).fit(normal)
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.gmm is not None
        return (-self.gmm.score_samples(embeddings)).astype(np.float64)


class _MaskedLinear(nn.Linear):
    """Linear layer with a fixed binary connectivity mask on the weights (MADE)."""

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor) -> None:
        super().__init__(in_features, out_features)
        self.register_buffer("conn", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight * self.conn, self.bias)


class _MADE(nn.Module):
    """Masked MLP (Germain et al. 2015): outputs `out_per_dim` parameters per input dim,
    where the parameters for dim i depend only on inputs with index < i. One forward pass
    yields all autoregressive-conditioner parameters (MAF density evaluation is parallel)."""

    def __init__(self, dim: int, hidden: int, out_per_dim: int) -> None:
        super().__init__()
        in_deg = torch.arange(1, dim + 1)
        h_deg = (torch.arange(hidden) % max(1, dim - 1)) + 1  # degrees cycle over 1..D-1
        out_deg = in_deg.repeat_interleave(out_per_dim)       # grouped per dim -> reshape below
        self.dim, self.out_per_dim = dim, out_per_dim
        self.l1 = _MaskedLinear(dim, hidden, (h_deg[:, None] >= in_deg[None, :]).float())
        self.l2 = _MaskedLinear(hidden, hidden, (h_deg[:, None] >= h_deg[None, :]).float())
        self.l3 = _MaskedLinear(hidden, dim * out_per_dim, (out_deg[:, None] > h_deg[None, :]).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.l2(torch.relu(self.l1(x))))
        return self.l3(h).view(-1, self.dim, self.out_per_dim)  # (B, D, P)


def _rq_spline(x: torch.Tensor, w: torch.Tensor, h: torch.Tensor, d: torch.Tensor,
               bound: float = 3.0, min_bin: float = 1e-3, min_deriv: float = 1e-3):
    """Monotonic rational-quadratic spline (Durkan et al. 2019) with identity tails.

    x: (B, D); w, h: (B, D, K) unnormalized bin widths/heights; d: (B, D, K-1) unnormalized
    interior derivatives. Returns (z, log|dz/dx|) elementwise; outside [-bound, bound] the
    transform is the identity (logdet 0), so standardized data can't fall off the spline.
    """
    K = w.shape[-1]
    inside = (x > -bound) & (x < bound)

    widths = min_bin + (1 - min_bin * K) * torch.softmax(w, dim=-1)
    heights = min_bin + (1 - min_bin * K) * torch.softmax(h, dim=-1)
    derivs = min_deriv + torch.nn.functional.softplus(d)
    derivs = torch.nn.functional.pad(derivs, (1, 1), value=1.0)  # boundary derivative 1 -> C1 with tails

    cum_w = torch.nn.functional.pad(torch.cumsum(widths, -1), (1, 0)) * 2 * bound - bound
    cum_h = torch.nn.functional.pad(torch.cumsum(heights, -1), (1, 0)) * 2 * bound - bound
    widths, heights = widths * 2 * bound, heights * 2 * bound

    xc = x.clamp(-bound, bound - 1e-6)
    bin_idx = (torch.searchsorted(cum_w, xc.unsqueeze(-1), right=True) - 1).clamp(0, K - 1)

    take = lambda t: t.gather(-1, bin_idx).squeeze(-1)
    x_lo, y_lo = take(cum_w), take(cum_h)
    bw, bh = take(widths), take(heights)
    d_lo, d_hi = take(derivs[..., :-1]), take(derivs[..., 1:])
    s = bh / bw  # bin slope

    theta = ((xc - x_lo) / bw).clamp(0, 1)
    om = 1 - theta
    denom = s + (d_hi + d_lo - 2 * s) * theta * om
    z_in = y_lo + bh * (s * theta**2 + d_lo * theta * om) / denom
    logdet_in = (torch.log(s**2 * (d_hi * theta**2 + 2 * s * theta * om + d_lo * om**2))
                 - 2 * torch.log(denom))

    z = torch.where(inside, z_in, x)
    logdet = torch.where(inside, logdet_in, torch.zeros_like(x))
    return z, logdet


class _MAFRQS(nn.Module):
    """Masked autoregressive flow with rational-quadratic-spline transforms; feature order
    is reversed between layers so every dim gets conditioned both ways."""

    def __init__(self, dim: int, hidden: int, n_layers: int, bins: int = 8, bound: float = 3.0) -> None:
        super().__init__()
        self.dim, self.bins, self.bound = dim, bins, bound
        self.mades = nn.ModuleList([_MADE(dim, hidden, 3 * bins - 1) for _ in range(n_layers)])
        self.register_buffer("flip", torch.arange(dim - 1, -1, -1))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = x, torch.zeros(len(x), device=x.device)
        for i, made in enumerate(self.mades):
            if i > 0:
                z = z[:, self.flip]
            params = made(z)
            w, h = params[..., : self.bins], params[..., self.bins : 2 * self.bins]
            d = params[..., 2 * self.bins :]
            z, ld = _rq_spline(z, w, h, d, bound=self.bound)
            log_det = log_det + ld.sum(-1)
        log_base = -0.5 * (z**2).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi)
        return log_base + log_det


class _Coupling(nn.Module):
    """RealNVP affine coupling: transform the unmasked dims conditioned on the masked ones."""

    def __init__(self, dim: int, hidden: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim * 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        scale, shift = self.net(x_masked).chunk(2, dim=-1)
        scale = torch.tanh(scale) * (1 - self.mask)  # bounded -> exp(s) in [0.37, 2.7]
        shift = shift * (1 - self.mask)
        z = x_masked + (1 - self.mask) * (x * torch.exp(scale) + shift)
        return z, scale.sum(-1)  # log|det| = sum of scales on transformed dims


class _RealNVP(nn.Module):
    def __init__(self, dim: int, hidden: int, n_layers: int) -> None:
        super().__init__()
        self.dim = dim
        layers = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % 2::2] = 1.0  # alternate which half is conditioning
            layers.append(_Coupling(dim, hidden, mask))
        self.layers = nn.ModuleList(layers)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = x, torch.zeros(len(x), device=x.device)
        for layer in self.layers:
            z, ld = layer(z)
            log_det = log_det + ld
        log_base = -0.5 * (z ** 2).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi)
        return log_base + log_det


class FlowScorer:
    """Negative log-likelihood under a max-likelihood normalizing flow — an
    arbitrarily-shaped normal density vs Mahalanobis' single Gaussian.

    flow_type "rqs" (default): masked autoregressive flow with rational-quadratic-spline
    transforms (MAF + RQS, Durkan et al. 2019) — far more expressive than the legacy
    "realnvp" (bounded affine couplings), which is kept for comparison.

    Training hygiene (the legacy scorer had none): inputs are ALWAYS standardized with
    fit-set mean/std (spline tails assume ~unit scale); a seeded val split is held out;
    early stopping on val NLL with best-weights restore; per-epoch history kept in
    `self.history` and a one-line summary printed per fit. Still best paired with
    pca_dim reduction — a 256-d density on ~600 clips is data-starved regardless."""

    def __init__(self, flow_type: str = "rqs", n_layers: int = 4, hidden: int = 128,
                 bins: int = 8, epochs: int = 400, lr: float = 1e-3, batch_size: int = 256,
                 val_frac: float = 0.15, patience: int = 30, weight_decay: float = 1e-5,
                 seed: int = 42) -> None:
        self.flow_type = flow_type
        self.n_layers = n_layers
        self.hidden = hidden
        self.bins = bins
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.patience = patience
        self.weight_decay = weight_decay
        self.seed = seed
        self.flow: nn.Module | None = None
        self.mu: torch.Tensor | None = None
        self.sigma: torch.Tensor | None = None
        self.history: dict[str, list[float]] = {}

    def _build(self, dim: int) -> nn.Module:
        if self.flow_type == "rqs":
            return _MAFRQS(dim, self.hidden, self.n_layers, bins=self.bins)
        if self.flow_type == "realnvp":
            return _RealNVP(dim, self.hidden, self.n_layers)
        raise ValueError(f"unknown flow_type '{self.flow_type}' (rqs|realnvp)")

    @torch.no_grad()
    def _nll(self, x: torch.Tensor) -> float:
        self.flow.eval()
        nll = float(-self.flow.log_prob(x).mean())
        self.flow.train()
        return nll if math.isfinite(nll) else float("inf")

    def fit(self, normal: np.ndarray) -> "FlowScorer":
        torch.manual_seed(self.seed)
        x = torch.as_tensor(normal, dtype=torch.float32)
        self.mu = x.mean(dim=0)
        self.sigma = x.std(dim=0).clamp_min(1e-6)
        x = (x - self.mu) / self.sigma

        # Seeded held-out split for early stopping (skipped only when data is tiny).
        n_val = int(len(x) * self.val_frac)
        if n_val >= 8:
            perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(self.seed))
            x_val, x_train = x[perm[:n_val]], x[perm[n_val:]]
        else:
            x_val, x_train = None, x

        self.flow = self._build(x.shape[1])
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        self.flow.train()
        self.history = {"train_nll": [], "val_nll": []}
        best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0

        for epoch in range(self.epochs):
            perm = torch.randperm(len(x_train))
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, len(x_train), self.batch_size):
                batch = x_train[perm[start : start + self.batch_size]]
                loss = -self.flow.log_prob(batch).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 5.0)
                optimizer.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
            self.history["train_nll"].append(epoch_loss / max(1, n_batches))

            if x_val is None:
                continue
            val_nll = self._nll(x_val)
            self.history["val_nll"].append(val_nll)
            if val_nll < best_val:
                best_val, best_epoch, bad = val_nll, epoch, 0
                best_state = {k: v.detach().clone() for k, v in self.flow.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.flow.load_state_dict(best_state)  # restore the val-NLL optimum
        self.flow.eval()
        ran = len(self.history["train_nll"])
        if x_val is not None:
            print(f"[flow] {self.flow_type} d={x.shape[1]} n={len(x_train)}+{n_val}val: "
                  f"best epoch {best_epoch + 1}/{ran} val_nll={best_val:.3f} "
                  f"(train_nll={self.history['train_nll'][best_epoch]:.3f})"
                  f"{' EARLY-STOPPED' if ran < self.epochs else ''}", flush=True)
        else:
            print(f"[flow] {self.flow_type} d={x.shape[1]} n={len(x_train)}: no val split "
                  f"(too few samples), ran all {ran} epochs UNMONITORED", flush=True)
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.flow is not None
        x = (torch.as_tensor(embeddings, dtype=torch.float32) - self.mu) / self.sigma
        with torch.no_grad():
            nll = -self.flow.log_prob(x)
        return nll.numpy().astype(np.float64)


class WhitenedScorer:
    """PCA-whiten the embeddings (fit on the normal set) to `pca_dim`, then delegate to a
    base scorer. Makes density estimation tractable in the small-N / high-D regime."""

    def __init__(self, base: EmbeddingScorer, pca_dim: int) -> None:
        self.base = base
        self.pca_dim = pca_dim
        self.pca = None

    def fit(self, normal: np.ndarray) -> "WhitenedScorer":
        from sklearn.decomposition import PCA

        dim = min(int(self.pca_dim), normal.shape[1], max(1, normal.shape[0] - 1))
        self.pca = PCA(n_components=dim, whiten=True, random_state=42).fit(normal)
        self.base.fit(self.pca.transform(normal).astype(np.float32))
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.pca is not None
        return self.base.score(self.pca.transform(embeddings).astype(np.float32))


def build_scorer(cfg=None, **overrides) -> EmbeddingScorer:
    """Build a one-class embedding scorer from an `embedding` config dict (extra keys such as
    `encoder` are ignored). Accepts a dict, a bare method string, or keyword overrides."""
    if isinstance(cfg, str):
        cfg = {"method": cfg}
    cfg = {**(cfg or {}), **overrides}
    method = str(cfg.get("method", "mahalanobis")).lower()

    if method == "mahalanobis":
        base: EmbeddingScorer = MahalanobisScorer()
    elif method == "knn":
        base = KNNScorer(k=int(cfg.get("knn_k", 4)))
    elif method == "gmm":
        base = GMMScorer(n_components=int(cfg.get("gmm_components", 4)))
    elif method == "flow":
        base = FlowScorer(flow_type=str(cfg.get("flow_type", "rqs")),
                          n_layers=int(cfg.get("flow_layers", 4)),
                          hidden=int(cfg.get("flow_hidden", 128)),
                          bins=int(cfg.get("flow_bins", 8)),
                          epochs=int(cfg.get("flow_epochs", 400)),
                          lr=float(cfg.get("flow_lr", 1e-3)),
                          val_frac=float(cfg.get("flow_val_frac", 0.15)),
                          patience=int(cfg.get("flow_patience", 30)),
                          weight_decay=float(cfg.get("flow_weight_decay", 1e-5)))
    else:
        raise ValueError(f"Unknown embedding scorer '{method}' (mahalanobis|knn|gmm|flow)")

    pca_dim = cfg.get("pca_dim")
    return WhitenedScorer(base, int(pca_dim)) if pca_dim else base

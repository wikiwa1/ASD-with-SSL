from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingScorer(Protocol):
    """One-class anomaly scorer fit on a target's normal embeddings."""

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
        topk = np.partition(sims, -k, axis=1)[:, -k:]  # k highest similarities
        return (1.0 - topk.mean(axis=1)).astype(np.float64)  # distance = 1 - mean cosine


def build_scorer(method: str = "mahalanobis", knn_k: int = 4) -> EmbeddingScorer:
    method = method.lower()
    if method == "mahalanobis":
        return MahalanobisScorer()
    if method == "knn":
        return KNNScorer(k=knn_k)
    raise ValueError(f"Unknown embedding scorer '{method}' (use 'mahalanobis' or 'knn')")

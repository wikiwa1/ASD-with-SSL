from __future__ import annotations

import numpy as np
from sklearn import metrics


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    finite = np.isfinite(scores)
    if not np.all(finite):
        labels = labels[finite]
        scores = scores[finite]
    return float(metrics.roc_auc_score(labels, scores))


def roc_curve_points(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> dict:
    """ROC curve plus AUC and standardized partial AUC.

    pAUC over FPR in [0, max_fpr] is the standard MIMII/DCASE metric (max_fpr=0.1).
    Returns the full fpr/tpr/threshold arrays so the curve can be plotted or saved.
    """
    finite = np.isfinite(scores)
    if not np.all(finite):
        labels = labels[finite]
        scores = scores[finite]
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": float(metrics.roc_auc_score(labels, scores)),
        "pauc": float(metrics.roc_auc_score(labels, scores, max_fpr=max_fpr)),
        "max_fpr": float(max_fpr),
    }


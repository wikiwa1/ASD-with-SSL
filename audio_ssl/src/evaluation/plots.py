from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on compute nodes
import matplotlib.pyplot as plt  # noqa: E402


def save_roc_plot(
    path: str | Path,
    fpr,
    tpr,
    auc: float,
    pauc: float | None = None,
    max_fpr: float | None = None,
    title: str | None = None,
) -> None:
    """Save a single-target ROC curve PNG."""
    fig, ax = plt.subplots(figsize=(5, 5))
    label = f"AUC = {auc:.4f}"
    if pauc is not None:
        label += f"\npAUC = {pauc:.4f}"
    ax.plot(fpr, tpr, lw=2, label=label)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="gray")
    if max_fpr is not None:
        ax.axvspan(0, max_fpr, color="red", alpha=0.06)
        ax.axvline(max_fpr, color="red", ls=":", lw=1, label=f"pAUC FPR ≤ {max_fpr:g}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

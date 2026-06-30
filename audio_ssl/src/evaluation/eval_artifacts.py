from __future__ import annotations

from pathlib import Path

import numpy as np

from audio_ssl.src.data.splits import TargetInfo
from audio_ssl.src.evaluation.auc import roc_curve_points
from audio_ssl.src.utils.io import save_npz
from audio_ssl.src.utils.loggers import end_experiments, resume_comet_experiment


def compute_and_save_roc(
    roc_dir: Path,
    info: TargetInfo,
    scores: np.ndarray,
    labels: np.ndarray,
    checkpoint_path: str | Path,
    eval_files: list,
    max_fpr: float,
) -> dict:
    """Compute AUC/pAUC + ROC, save the curve (png + npz), and return the result entry.
    Shared by every per-target evaluator so the artifact format stays consistent."""
    roc = roc_curve_points(labels, scores, max_fpr=max_fpr)

    npz_path = roc_dir / f"{info.key}.npz"
    save_npz(npz_path, fpr=roc["fpr"], tpr=roc["tpr"], thresholds=roc["thresholds"],
             scores=scores, labels=labels)

    png_path = roc_dir / f"{info.key}.png"
    try:
        from audio_ssl.src.evaluation.plots import save_roc_plot

        save_roc_plot(png_path, roc["fpr"], roc["tpr"], roc["auc"],
                      pauc=roc["pauc"], max_fpr=roc["max_fpr"], title=info.key)
        plot_path: str | None = str(png_path)
    except Exception as exc:  # plotting is optional; never fail eval over a plot
        print(f"  (skipped ROC plot for {info.key}: {exc})", flush=True)
        plot_path = None

    return {
        "AUC": roc["auc"],
        "pAUC": roc["pauc"],
        "pauc_max_fpr": roc["max_fpr"],
        "machine_type": info.machine_type,
        "machine_id": info.machine_id,
        "db": info.db,
        "checkpoint": str(checkpoint_path),
        "num_eval_files": int(len(eval_files)),
        "roc_data": str(npz_path),
        "roc_plot": plot_path,
    }


def log_results_to_comet(
    config: dict,
    experiment_key_file: Path,
    results: dict,
    prefix: str = "test",
) -> None:
    """Reattach to the training Comet experiment and log per-target metrics + ROC images.
    No-op if there is no key / Comet is disabled. `prefix` namespaces the metrics so
    multiple scorers (e.g. prediction-error vs embedding) don't collide."""
    if not results:
        return
    experiment_key = experiment_key_file.read_text().strip() if experiment_key_file.exists() else None
    comet_logger = resume_comet_experiment(config, experiment_key) if experiment_key else None
    if comet_logger is None:
        return
    try:
        aucs = [r["AUC"] for r in results.values()]
        paucs = [r["pAUC"] for r in results.values()]
        metrics = {f"{prefix}_overall_AUC": float(np.mean(aucs)),
                   f"{prefix}_overall_pAUC": float(np.mean(paucs))}
        for key, r in results.items():
            metrics[f"{prefix}_AUC/{key}"] = r["AUC"]
            metrics[f"{prefix}_pAUC/{key}"] = r["pAUC"]
        comet_logger.log_metrics(metrics)
        experiment = comet_logger.experiment
        for key, r in results.items():
            if r["roc_plot"]:
                experiment.log_image(r["roc_plot"], name=f"{prefix}_ROC_{key}")
    except Exception as exc:
        print(f"  (comet eval logging skipped: {exc})", flush=True)
    finally:
        end_experiments([comet_logger])

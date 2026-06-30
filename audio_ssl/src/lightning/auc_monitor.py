from __future__ import annotations

import lightning.pytorch as pl
import numpy as np
import torch

from audio_ssl.src.evaluation.auc import roc_auc
from audio_ssl.src.evaluation.embedding_scores import build_scorer
from audio_ssl.src.evaluation.jepa_embeddings import embed_spectrograms


class PeriodicAUCMonitor(pl.Callback):
    """Every N epochs, compute a single-target embedding-distance AUC and log it (e.g. to
    Comet) so SSL over-optimization is visible *during* training — the pretext val_loss can
    keep dropping while downstream AUC peaks then declines.

    The pre-extracted spectrograms are passed in (no multiprocessing fork happens during
    training — a mid-training fork inherits Lightning's CUDA SIGTERM handler and crashes).
    Add this callback ONLY on global rank 0 (the caller decides), so it neither duplicates
    work across ranks nor calls any collective.

    DIAGNOSTIC ONLY: scores the target's TEST split (the only anomaly-labeled MIMII data),
    so it must not select the final reported checkpoint.
    """

    def __init__(
        self,
        normal_specs: torch.Tensor,
        eval_specs: torch.Tensor,
        labels: np.ndarray,
        every_n_epochs: int = 1,
        batch_size: int = 256,
        encoder: str = "target",
        method: str = "mahalanobis",
        metric_name: str = "monitor_AUC",
    ):
        super().__init__()
        self.normal_specs = normal_specs  # (N, 1, n_mels, T)
        self.eval_specs = eval_specs
        self.labels = labels
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.batch_size = int(batch_size)
        self.encoder = encoder
        self.method = method
        self.metric_name = metric_name

    @torch.inference_mode()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        was_training = pl_module.training
        try:
            device = pl_module.device
            normal = embed_spectrograms(pl_module, self.normal_specs, self.batch_size, device, self.encoder)
            evaluation = embed_spectrograms(pl_module, self.eval_specs, self.batch_size, device, self.encoder)
            scores = build_scorer(self.method).fit(normal).score(evaluation)
            auc = float(roc_auc(self.labels, scores))
        except Exception as exc:  # never let a monitor probe kill training
            print(f"[monitor] skipped at epoch {trainer.current_epoch + 1}: {exc}", flush=True)
            return
        finally:
            if was_training:
                pl_module.train()  # embed_spectrograms put the module in eval()

        for logger in trainer.loggers:
            logger.log_metrics({self.metric_name: auc}, step=trainer.global_step)
        print(f"[monitor] epoch {trainer.current_epoch + 1}: {self.metric_name}={auc:.4f}", flush=True)

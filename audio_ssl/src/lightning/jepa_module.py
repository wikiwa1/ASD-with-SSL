from __future__ import annotations

import copy
import math

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F

from audio_ssl.src.models.jepa.masking import sample_block_mask
from audio_ssl.src.models.jepa.vit import JEPAEncoder, JEPAPredictor


class LitJEPA(pl.LightningModule):
    """Audio I-JEPA: predict EMA-target patch embeddings of masked time-frequency blocks
    from the encoded context. The latent prediction error is later reused as the anomaly
    score (see eval_jepa)."""

    def __init__(
        self,
        n_mels: int = 64,
        target_frames: int = 313,
        patch_mels: int = 16,
        patch_frames: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        heads: int = 4,
        predictor_dim: int = 128,
        predictor_depth: int = 4,
        predictor_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_blocks: int = 4,
        mask_scale: tuple[float, float] = (0.15, 0.3),
        mask_aspect: tuple[float, float] = (0.5, 2.0),
        ema_start: float = 0.996,
        ema_end: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_frac: float = 0.1,
        mel_mean: np.ndarray | None = None,
        mel_std: np.ndarray | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["mel_mean", "mel_std"])
        patch = (patch_mels, patch_frames)
        self.context_encoder = JEPAEncoder(
            n_mels, target_frames, patch, 1, embed_dim, depth, heads, mlp_ratio
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = JEPAPredictor(
            self.context_encoder.num_patches, embed_dim, predictor_dim,
            predictor_depth, predictor_heads, mlp_ratio,
        )
        self.grid = self.context_encoder.grid

        mean = torch.zeros(1, 1, n_mels, 1) if mel_mean is None else torch.as_tensor(
            np.asarray(mel_mean, dtype=np.float32)).view(1, 1, n_mels, 1)
        std = torch.ones(1, 1, n_mels, 1) if mel_std is None else torch.as_tensor(
            np.asarray(mel_std, dtype=np.float32)).view(1, 1, n_mels, 1)
        self.register_buffer("mel_mean", mean)
        self.register_buffer("mel_std", std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mel_mean) / self.mel_std

    def _loss(self, x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        x = self.normalize(x)
        target_idx, context_idx = sample_block_mask(
            self.grid[0], self.grid[1], self.hparams.num_blocks,
            self.hparams.mask_scale, self.hparams.mask_aspect, generator=generator,
        )
        target_idx = target_idx.to(x.device)
        context_idx = context_idx.to(x.device)
        with torch.no_grad():
            targets = self.target_encoder(x)[:, target_idx, :]
        context_enc = self.context_encoder(x, keep_idx=context_idx)
        preds = self.predictor(context_enc, context_idx, target_idx)
        return F.smooth_l1_loss(preds, targets)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self._loss(batch[0])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        if batch[0].numel() == 0:
            return None
        generator = torch.Generator().manual_seed(1234 + batch_idx)  # stable val masks
        loss = self._loss(batch[0], generator=generator)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def _ema_update(self, momentum: float) -> None:
        for tp, cp in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            tp.mul_(momentum).add_(cp.detach(), alpha=1.0 - momentum)

    def on_train_batch_end(self, *args) -> None:
        total = max(1, int(self.trainer.estimated_stepping_batches))
        progress = min(1.0, self.global_step / total)
        momentum = self.hparams.ema_start + (self.hparams.ema_end - self.hparams.ema_start) * progress
        self._ema_update(momentum)

    def configure_optimizers(self):
        params = list(self.context_encoder.parameters()) + list(self.predictor.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        total = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = max(1, int(self.hparams.warmup_frac * total))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

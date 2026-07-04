from __future__ import annotations

import math

import lightning.pytorch as pl
import numpy as np
import torch

from audio_ssl.src.models.jepa.vit import JEPAEncoder
from audio_ssl.src.models.lejepa.augment import make_views
from audio_ssl.src.models.lejepa.sigreg import SIGReg


class LitLeJEPA(pl.LightningModule):
    """LeJEPA (Balestriero & LeCun 2025): a single ViT encoder trained by view-invariance +
    SIGReg (isotropic-Gaussian regularization). No EMA teacher, no predictor, no stop-grad —
    SIGReg prevents collapse, and the embeddings are Gaussian by construction (so Mahalanobis
    scoring is well-specified). Exposes target/context_encoder so the existing embedding eval
    and AUC monitor work unchanged.
    """

    def __init__(
        self,
        n_mels: int = 64,
        target_frames: int = 313,
        patch_mels: int = 16,
        patch_frames: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        num_views: int = 4,
        lam: float = 0.05,
        sigreg_slices: int = 512,
        sigreg_points: int = 17,
        augment: dict | None = None,
        lr: float = 5e-4,
        weight_decay: float = 0.05,
        warmup_frac: float = 0.1,
        lr_schedule: str = "cosine",
        mel_mean: np.ndarray | None = None,
        mel_std: np.ndarray | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["mel_mean", "mel_std"])
        self.encoder = JEPAEncoder(
            n_mels, target_frames, (patch_mels, patch_frames), 1, embed_dim, depth, heads, mlp_ratio)
        self.sigreg = SIGReg(num_slices=sigreg_slices, num_points=sigreg_points)
        self.augment_cfg = augment or {}

        mean = torch.zeros(1, 1, n_mels, 1) if mel_mean is None else torch.as_tensor(
            np.asarray(mel_mean, dtype=np.float32)).view(1, 1, n_mels, 1)
        std = torch.ones(1, 1, n_mels, 1) if mel_std is None else torch.as_tensor(
            np.asarray(mel_std, dtype=np.float32)).view(1, 1, n_mels, 1)
        self.register_buffer("mel_mean", mean)
        self.register_buffer("mel_std", std)

    # --- eval/monitor compatibility (embed_spectrograms accesses these) ---
    @property
    def target_encoder(self):
        return self.encoder

    @property
    def context_encoder(self):
        return self.encoder

    @property
    def grid(self):
        return self.encoder.grid

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mel_mean) / self.mel_std

    def _view_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        # One batched forward over all V views (V*B, 1, M, T) -> (B, V, D). Encoder batch is
        # V*B, so the config batch_size is per-clip; scale it down by num_views to match a
        # plain encoder's memory footprint.
        v = self.hparams.num_views
        views = make_views(x, v, self.augment_cfg)
        stacked = torch.cat([self.normalize(view) for view in views], dim=0)  # (V*B, 1, M, T)
        emb = self.encoder(stacked).mean(dim=1)                               # (V*B, D)
        return emb.view(v, x.shape[0], -1).transpose(0, 1)                    # (B, V, D)

    def _losses(self, x: torch.Tensor):
        z = self._view_embeddings(x)
        mu = z.mean(dim=1, keepdim=True)
        pred = ((z - mu) ** 2).mean()                       # view invariance (mean over dim, matches repo)
        reg = self.sigreg(z.reshape(-1, z.shape[-1]))       # isotropic-Gaussian on embeddings
        return (1 - self.hparams.lam) * pred + self.hparams.lam * reg, pred, reg

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, pred, reg = self._losses(batch[0])
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("pred_loss", pred, on_step=False, on_epoch=True, sync_dist=True)
        self.log("sigreg", reg, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        if batch[0].numel() == 0:
            return None
        loss, _, _ = self._losses(batch[0])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        total = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = max(1, int(self.hparams.warmup_frac * total))
        constant = self.hparams.lr_schedule == "constant"

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / warmup
            if constant:
                return 1.0
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

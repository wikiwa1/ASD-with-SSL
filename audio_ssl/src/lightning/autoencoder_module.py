from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch import nn

from audio_ssl.src.models.autoencoder import DenseAutoEncoder


class LitAutoEncoder(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (64, 64, 8, 64, 64),
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = DenseAutoEncoder(input_dim=input_dim, hidden_dims=hidden_dims)
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, target = batch
        loss = self.criterion(self(x), target)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, target = batch
        loss = self.criterion(self(x), target)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


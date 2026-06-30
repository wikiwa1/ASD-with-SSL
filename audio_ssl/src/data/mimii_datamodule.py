from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from audio_ssl.src.features.logmel import list_to_vector_array


class MIMIIAutoEncoderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_files: list[Path],
        feature_kwargs: dict,
        batch_size: int = 512,
        validation_split: float = 0.1,
        shuffle: bool = True,
        num_workers: int = 4,
        seed: int = 42,
        train_features: np.ndarray | None = None,
    ):
        super().__init__()
        self.train_files = train_files
        self.feature_kwargs = feature_kwargs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.seed = seed
        self.train_features = train_features
        self.train_dataset: TensorDataset | None = None
        self.val_dataset: TensorDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_features is None:
            self.train_features = list_to_vector_array(
                self.train_files,
                msg="generate train features",
                **self.feature_kwargs,
            )
        if len(self.train_features) == 0:
            raise ValueError("No training feature vectors were generated")

        features = torch.from_numpy(self.train_features.astype(np.float32, copy=False))
        n_total = len(features)
        n_val = int(n_total * self.validation_split)
        if n_val <= 0 and self.validation_split > 0.0 and n_total > 1:
            n_val = 1

        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(n_total, generator=generator)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        if len(train_indices) == 0:
            raise ValueError("Validation split consumed all training vectors")

        train_x = features[train_indices]
        self.train_dataset = TensorDataset(train_x, train_x)
        if len(val_indices) > 0:
            val_x = features[val_indices]
            self.val_dataset = TensorDataset(val_x, val_x)
        else:
            self.val_dataset = TensorDataset(train_x[:0], train_x[:0])

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


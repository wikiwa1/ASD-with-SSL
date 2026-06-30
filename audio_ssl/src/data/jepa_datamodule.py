from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from audio_ssl.src.features.spectrogram import stack_logmels


class JEPASpectrogramDataModule(pl.LightningDataModule):
    """Serves full-clip log-mel spectrograms (N, 1, n_mels, T) for global JEPA pretraining.

    `train_files` should be the pooled normal-train clips across all targets (the eval
    held-out normals + abnormals are excluded upstream, so pretraining never sees them).
    Raw log-mels are served; normalization lives in the model (checkpointed buffers).
    """

    def __init__(
        self,
        train_files: list[str | Path],
        feature_kwargs: dict,
        batch_size: int = 256,
        num_workers: int = 8,
        val_split: float = 0.02,
        seed: int = 42,
        cache_path: str | Path | None = None,
    ):
        super().__init__()
        self.train_files = train_files
        self.feature_kwargs = feature_kwargs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.cache_path = Path(cache_path) if cache_path else None
        self.specs: np.ndarray | None = None
        self.mel_mean: np.ndarray | None = None
        self.mel_std: np.ndarray | None = None
        self.train_dataset: TensorDataset | None = None
        self.val_dataset: TensorDataset | None = None

    def _load_or_extract(self) -> np.ndarray:
        if self.cache_path and self.cache_path.exists():
            specs = np.load(self.cache_path, mmap_mode="r")
            if specs.shape[0] == len(self.train_files):
                return np.asarray(specs)
        specs = stack_logmels(self.train_files, msg="extract JEPA spectrograms", **self.feature_kwargs)
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp.npy")
            np.save(tmp, specs)
            tmp.replace(self.cache_path)  # atomic: waiters only see the complete file
        return specs

    def setup(self, stage: str | None = None) -> None:
        if self.specs is not None:
            return
        specs = self._load_or_extract()  # (N, n_mels, T)
        if len(specs) == 0:
            raise ValueError("No spectrograms extracted for JEPA pretraining")
        # Per-mel-bin standardization stats over (clips, time). Floor the std so
        # near-silent bins (std ~ 0) don't blow up to huge normalized values.
        self.mel_mean = specs.mean(axis=(0, 2)).astype(np.float32)
        std = specs.std(axis=(0, 2))
        std_floor = 0.1 * float(specs.std())  # 10% of the global std
        self.mel_std = np.maximum(std, std_floor).astype(np.float32)
        self.specs = specs

        tensor = torch.from_numpy(specs.astype(np.float32, copy=False)).unsqueeze(1)  # (N,1,M,T)
        n_total = len(tensor)
        n_val = max(1, int(n_total * self.val_split)) if self.val_split > 0 else 0
        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n_total, generator=generator)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        self.train_dataset = TensorDataset(tensor[train_idx])
        self.val_dataset = TensorDataset(tensor[val_idx]) if n_val > 0 else TensorDataset(tensor[:0])

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
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

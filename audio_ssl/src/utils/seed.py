from __future__ import annotations

import lightning.pytorch as pl


def seed_everything(seed: int) -> None:
    pl.seed_everything(seed, workers=True)


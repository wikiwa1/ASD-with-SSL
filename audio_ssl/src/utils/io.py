from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)


def save_npz(path: str | Path, **arrays) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    np.savez_compressed(target, **arrays)


def load_npz(path: str | Path):
    return np.load(Path(path), allow_pickle=False)


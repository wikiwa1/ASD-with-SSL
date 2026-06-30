from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TargetInfo:
    target_dir: Path
    db: str
    machine_type: str
    machine_id: str

    @property
    def key(self) -> str:
        return f"{self.machine_type}_{self.machine_id}_{self.db}"


@dataclass(frozen=True)
class TargetSplit:
    train_files: list[Path]
    train_labels: np.ndarray
    eval_files: list[Path]
    eval_labels: np.ndarray


def find_target_dirs(base_directory: str | Path, machines: list[str] | None = None) -> list[Path]:
    """Return MIMII target dirs with normal/abnormal children, optionally filtered to a
    set of machine types.

    Supports both common layouts:
    - base/db/machine_type/id_xx/{normal,abnormal}
    - base/machine_type/id_xx/{normal,abnormal}
    """
    base = Path(base_directory)
    dirs = sorted(
        path
        for path in base.rglob("*")
        if path.is_dir() and (path / "normal").is_dir() and (path / "abnormal").is_dir()
    )
    if machines:
        wanted = set(machines)
        dirs = [d for d in dirs if parse_target_info(d, base_directory).machine_type in wanted]
    return dirs


def discover_targets(config: dict) -> list[Path]:
    """Config-aware target discovery: applies the optional `data.machines` filter so an
    experiment (e.g. fan-only JEPA) is defined entirely by its config."""
    data_cfg = config["data"]
    return find_target_dirs(data_cfg["base_directory"], machines=data_cfg.get("machines"))


def parse_target_info(target_dir: str | Path, base_directory: str | Path | None = None) -> TargetInfo:
    target = Path(target_dir)
    if base_directory is not None:
        try:
            parts = target.relative_to(Path(base_directory)).parts
        except ValueError:
            parts = target.parts
    else:
        parts = target.parts

    if len(parts) >= 3:
        db, machine_type, machine_id = parts[-3], parts[-2], parts[-1]
    elif len(parts) == 2:
        db, machine_type, machine_id = "default", parts[-2], parts[-1]
    else:
        db, machine_type, machine_id = "default", target.parent.name, target.name

    return TargetInfo(
        target_dir=target,
        db=db,
        machine_type=machine_type,
        machine_id=machine_id,
    )


def make_baseline_split(
    target_dir: str | Path,
    normal_dir_name: str = "normal",
    abnormal_dir_name: str = "abnormal",
    ext: str = "wav",
) -> TargetSplit:
    """Match the original MIMII baseline split.

    Training uses normal files after the first N normal files, where N is the
    number of abnormal files. Evaluation uses the held-out first N normal files
    plus all abnormal files.
    """
    target = Path(target_dir)
    normal_files = sorted((target / normal_dir_name).glob(f"*.{ext}"))
    abnormal_files = sorted((target / abnormal_dir_name).glob(f"*.{ext}"))

    if not normal_files:
        raise FileNotFoundError(f"No normal '*.{ext}' files under {target / normal_dir_name}")
    if not abnormal_files:
        raise FileNotFoundError(f"No abnormal '*.{ext}' files under {target / abnormal_dir_name}")
    if len(normal_files) <= len(abnormal_files):
        raise ValueError(
            "Need more normal files than abnormal files for the baseline split: "
            f"{len(normal_files)} normal, {len(abnormal_files)} abnormal"
        )

    n_eval_normal = len(abnormal_files)
    train_files = normal_files[n_eval_normal:]
    eval_files = normal_files[:n_eval_normal] + abnormal_files

    return TargetSplit(
        train_files=train_files,
        train_labels=np.zeros(len(train_files), dtype=np.int64),
        eval_files=eval_files,
        eval_labels=np.concatenate(
            [
                np.zeros(n_eval_normal, dtype=np.int64),
                np.ones(len(abnormal_files), dtype=np.int64),
            ]
        ),
    )

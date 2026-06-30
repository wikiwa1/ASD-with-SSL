from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _coerce_cli_value(value):
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r") as handle:
        return yaml.safe_load(handle)


def merge_cli_overrides(config: dict[str, Any], args) -> dict[str, Any]:
    trainer = config.setdefault("trainer", {})
    for field in ("accelerator", "devices", "num_nodes", "strategy", "precision"):
        value = getattr(args, field, None)
        if value is not None:
            trainer[field] = _coerce_cli_value(value)
    if getattr(args, "max_epochs", None) is not None:
        config.setdefault("fit", {})["epochs"] = args.max_epochs
    if getattr(args, "base_directory", None) is not None:
        config.setdefault("data", {})["base_directory"] = args.base_directory
    return config

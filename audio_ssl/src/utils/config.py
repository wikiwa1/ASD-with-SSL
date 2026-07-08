from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _coerce_cli_value(value):
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config. A top-level `extends: <relative-path>` key inherits from a base
    config and deep-merges the overrides — so an experiment config can be a few lines."""
    path = Path(path)
    with path.open("r") as handle:
        config = yaml.safe_load(handle)
    parent = config.pop("extends", None)
    if parent is not None:
        config = _deep_merge(load_config(path.parent / parent), config)
    return config


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

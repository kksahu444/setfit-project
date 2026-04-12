"""Utilities to load project settings from YAML config files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default.yaml"


ConfigPathLike = Union[str, Path, None]


def resolve_config_path(config_path: ConfigPathLike) -> Path:
    """Resolve config path relative to the project root when needed."""
    if config_path is None:
        return DEFAULT_CONFIG_PATH

    raw = str(config_path).strip()
    if not raw:
        return DEFAULT_CONFIG_PATH

    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_config(config_path: ConfigPathLike = None) -> Dict[str, Any]:
    """Load YAML configuration into a dictionary."""
    resolved = resolve_config_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping.")

    return data

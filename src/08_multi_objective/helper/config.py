"""Config loading helpers for the v2 optimizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .paths import AVAILABILITY_CONFIG, ENDPOINTS_CONFIG, INGREDIENTS_CONFIG, OPTIMIZATION_CONFIG


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML file and return an empty dict for an empty document."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_ingredients_config(path: str | Path = INGREDIENTS_CONFIG) -> dict[str, Any]:
    return load_yaml(path)


def load_endpoints_config(path: str | Path = ENDPOINTS_CONFIG) -> dict[str, Any]:
    return load_yaml(path)


def load_optimization_config(path: str | Path = OPTIMIZATION_CONFIG) -> dict[str, Any]:
    return load_yaml(path)


def load_availability_config(path: str | Path = AVAILABILITY_CONFIG) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return load_yaml(path)


def nested_get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Return a dotted-path value from nested mappings."""
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current

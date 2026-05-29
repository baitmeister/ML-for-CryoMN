"""Endpoint normalization and intact-patch gate logic."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


TRUE_VALUES = {"1", "true", "t", "yes", "y", "pass", "passed"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "fail", "failed"}


def parse_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def intact_patch_formation_pass(
    row: Mapping[str, Any],
    min_intact_tip_count: int = 90,
    total_tip_count_default: int = 100,
    min_intact_tip_fraction: float = 0.90,
) -> bool:
    """Evaluate the required intact-patch formation screening gate."""
    explicit = parse_bool(row.get("intact_patch_formation_pass"))
    if explicit is not None:
        return explicit

    no_slurry = parse_bool(row.get("no_slurry"))
    no_collapse = parse_bool(row.get("no_collapse"))
    if no_slurry is False or no_collapse is False:
        return False

    intact_tip_count = row.get("intact_tip_count")
    if intact_tip_count is None or pd.isna(intact_tip_count):
        return False
    total_tip_count = row.get("total_tip_count", total_tip_count_default)
    if total_tip_count is None or pd.isna(total_tip_count):
        total_tip_count = total_tip_count_default

    threshold = max(int(min_intact_tip_count), float(total_tip_count) * min_intact_tip_fraction)
    return float(intact_tip_count) >= threshold


def canonical_endpoint_name(name: str) -> str:
    return str(name).strip().lower()

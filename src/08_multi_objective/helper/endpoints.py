"""Endpoint normalization and intact-patch gate logic."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping

import pandas as pd


TRUE_VALUES = {"1", "true", "t", "yes", "y", "pass", "passed"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "fail", "failed"}
INTACT_PATCH_ENDPOINT = "intact_patch_formation_pass"
INTACT_PATCH_REPLICATE_POLICY = "all_pass"
PREPARATION_VISCOSITY_MAX_MPA_S = 3000.0
PREPARATION_MIN_FILLED_CAVITIES = 90
PREPARATION_REFERENCE_TOTAL_CAVITIES = 100


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


def preparation_gate_pass(
    row: Mapping[str, Any],
    apparent_viscosity_max_mPa_s: float = PREPARATION_VISCOSITY_MAX_MPA_S,
    minimum_filled_cavities: int = PREPARATION_MIN_FILLED_CAVITIES,
    reference_total_cavities: int = PREPARATION_REFERENCE_TOTAL_CAVITIES,
) -> bool | None:
    """Evaluate the provisional Round-5 cell-free preparation gate.

    ``None`` means the detailed gate is incomplete. The caller must not turn an
    incomplete record into a pass label.
    """
    required_booleans = [
        parse_bool(row.get("homogeneous_after_preparation_pass")),
        parse_bool(row.get("homogeneous_after_4C_30min_pass")),
        parse_bool(row.get("no_sediment_or_crystallization_2h_pass")),
    ]
    viscosity = pd.to_numeric(
        row.get("apparent_viscosity_mPa_s_25C_10s"), errors="coerce"
    )
    filled = pd.to_numeric(row.get("filled_cavity_count"), errors="coerce")
    total = pd.to_numeric(row.get("total_cavity_count"), errors="coerce")
    if (
        any(value is None for value in required_booleans)
        or pd.isna(viscosity)
        or pd.isna(filled)
        or pd.isna(total)
    ):
        return None
    if float(total) <= 0:
        return False
    fill_fraction = float(filled) / float(total)
    reference_fraction = float(minimum_filled_cavities) / float(
        reference_total_cavities
    )
    return bool(
        all(required_booleans)
        and float(viscosity) <= float(apparent_viscosity_max_mPa_s)
        and float(filled) >= float(minimum_filled_cavities)
        and fill_fraction >= reference_fraction
    )


def aggregate_intact_patch_replicates(values: Iterable[Any]) -> float | None:
    """Collapse measured patch replicates with the conservative all-pass rule.

    Missing values are ignored. A formulation passes only when every measured
    replicate passes; one failed replicate therefore makes the formulation-level
    gate fail. ``None`` is returned when no replicate has a measured gate result.
    """
    measured: list[bool] = []
    for value in values:
        if value is None or pd.isna(value):
            continue
        parsed = parse_bool(value)
        if parsed is None:
            raise ValueError(
                "Intact-patch replicate values must be boolean-like (0/1, "
                "true/false, or pass/fail)."
            )
        measured.append(parsed)
    if not measured:
        return None
    return 1.0 if all(measured) else 0.0


def canonical_endpoint_name(name: str) -> str:
    return str(name).strip().lower()

"""Soft constraint and acquisition penalty helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import nested_get
from .registry import IngredientRegistry, presence_threshold


def _row_value(row: Mapping[str, Any] | pd.Series, feature_name: str) -> float:
    value = row.get(feature_name, 0.0)
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def count_active_ingredients(
    row: Mapping[str, Any] | pd.Series,
    registry: IngredientRegistry,
) -> int:
    """Count active formulation ingredients above practical presence floors."""
    count = 0
    for feature_name in registry.feature_names:
        value = abs(_row_value(row, feature_name))
        if value >= presence_threshold(feature_name):
            count += 1
    return count


def active_ingredient_excess(
    row: Mapping[str, Any] | pd.Series,
    registry: IngredientRegistry,
    soft_limit: int = 8,
) -> int:
    """Return ingredient count above the soft limit; this is not a hard cutoff."""
    return max(0, count_active_ingredients(row, registry) - int(soft_limit))


def single_molar_excesses(
    row: Mapping[str, Any] | pd.Series,
    registry: IngredientRegistry,
    limit_M: float = 0.5,
) -> dict[str, float]:
    """Return per-ingredient molar excess above 500 mM.

    This intentionally does not sum total molar solute concentration before
    checking the rule. Each molar ingredient is evaluated independently.
    """
    excesses: dict[str, float] = {}
    for ingredient in registry.active_ingredients():
        if ingredient.unit != "M" or not ingredient.penalize_single_over_500mM:
            continue
        value = _row_value(row, ingredient.feature_name)
        excess = max(0.0, value - float(limit_M))
        if excess > 0.0:
            excesses[ingredient.feature_name] = excess
    return excesses


def acquisition_penalty(
    row: Mapping[str, Any] | pd.Series,
    registry: IngredientRegistry,
    optimization_config: Mapping[str, Any],
    intact_failure_probability: float = 0.0,
) -> float:
    """Compute additive acquisition penalty for soft feasibility pressure."""
    soft_limit = int(nested_get(optimization_config, "penalties.active_ingredient_soft_limit", 8))
    count_weight = float(
        nested_get(optimization_config, "penalties.active_ingredient_excess_weight", 0.05)
    )
    molar_limit = float(nested_get(optimization_config, "penalties.single_molar_ingredient_limit_M", 0.5))
    molar_weight = float(nested_get(optimization_config, "penalties.single_molar_excess_weight", 0.20))
    intact_weight = float(nested_get(optimization_config, "penalties.intact_failure_weight", 0.80))

    ingredient_count_penalty = count_weight * active_ingredient_excess(row, registry, soft_limit)
    molar_penalty = molar_weight * sum(single_molar_excesses(row, registry, molar_limit).values())
    intact_penalty = intact_weight * float(np.clip(intact_failure_probability, 0.0, 1.0))
    return float(ingredient_count_penalty + molar_penalty + intact_penalty)


def constraint_report(
    row: Mapping[str, Any] | pd.Series,
    registry: IngredientRegistry,
    optimization_config: Mapping[str, Any],
    intact_failure_probability: float = 0.0,
) -> dict[str, Any]:
    soft_limit = int(nested_get(optimization_config, "penalties.active_ingredient_soft_limit", 8))
    molar_limit = float(nested_get(optimization_config, "penalties.single_molar_ingredient_limit_M", 0.5))
    count = count_active_ingredients(row, registry)
    molar_excess = single_molar_excesses(row, registry, molar_limit)
    return {
        "active_ingredient_count": count,
        "active_ingredient_excess_above_8": max(0, count - soft_limit),
        "single_molar_excess_features": ";".join(sorted(molar_excess)),
        "single_molar_excess_total_M": float(sum(molar_excess.values())),
        "intact_failure_probability": float(np.clip(intact_failure_probability, 0.0, 1.0)),
        "acquisition_penalty": acquisition_penalty(
            row,
            registry,
            optimization_config,
            intact_failure_probability=intact_failure_probability,
        ),
    }

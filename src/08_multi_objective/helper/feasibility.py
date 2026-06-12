"""Forward-only formulation feasibility and support diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import nested_get
from .registry import IngredientRegistry, presence_threshold


POLICY_VERSION_DEFAULT = "round2_candidate_feasibility_v1"


@dataclass(frozen=True)
class SupportContext:
    observed_scaled: np.ndarray
    lower_bounds: np.ndarray
    ranges: np.ndarray
    radius: float


def policy_activation(
    optimization_config: Mapping[str, Any],
    target_round_number: int | None,
) -> tuple[bool, str, int]:
    version = str(
        nested_get(
            optimization_config,
            "formulation_feasibility.policy_version",
            POLICY_VERSION_DEFAULT,
        )
    )
    start_round = int(
        nested_get(optimization_config, "formulation_feasibility.start_round", 2)
    )
    active = target_round_number is not None and int(target_round_number) >= start_round
    return active, version, start_round


def _numeric(row: Mapping[str, Any] | pd.Series, feature_name: str) -> float:
    value = pd.to_numeric(row.get(feature_name, 0.0), errors="coerce")
    return 0.0 if pd.isna(value) else float(value)


def formulation_totals(
    row: Mapping[str, Any] | pd.Series,
    optimization_config: Mapping[str, Any],
) -> dict[str, float | int]:
    cfg = nested_get(optimization_config, "formulation_feasibility", {}) or {}
    polymer_features = list(
        cfg.get(
            "polymer_features",
            ["pvp_pct", "dextran_pct", "hyaluronic_acid_pct", "methylcellulose_pct"],
        )
    )
    serum_features = list(
        cfg.get("serum_protein_features", ["fbs_pct", "hsa_pct", "human_serum_pct"])
    )
    sugar_features = list(
        cfg.get(
            "sugar_features",
            ["trehalose_M", "sucrose_M", "glucose_M", "raffinose_M"],
        )
    )
    nonpermeating_features = list(
        cfg.get(
            "nonpermeating_osmolyte_features",
            [
                "ectoin_M",
                "trehalose_M",
                "sucrose_M",
                "raffinose_M",
                "taurine_M",
                "myo_inositol_M",
                "methoxyphenyl_beta_d_glucopyranoside_M",
                "betaine_M",
                "proline_M",
                "glucose_M",
                "creatine_M",
                "acetamide_M",
            ],
        )
    )
    active_polymers = sum(
        _numeric(row, feature) >= presence_threshold(feature)
        for feature in polymer_features
    )
    total_polymer = sum(_numeric(row, feature) for feature in polymer_features)
    total_serum = sum(_numeric(row, feature) for feature in serum_features)
    total_sugar = sum(_numeric(row, feature) for feature in sugar_features)
    total_nonpermeating = sum(
        _numeric(row, feature) for feature in nonpermeating_features
    )
    return {
        "active_polymer_count": int(active_polymers),
        "total_polymer_pct": float(total_polymer),
        "total_serum_protein_pct": float(total_serum),
        "total_polymer_serum_pct": float(total_polymer + total_serum),
        "total_sugar_M": float(total_sugar),
        "total_nonpermeating_solute_M": float(total_nonpermeating),
    }


def estimated_small_solute_g_L(
    row: Mapping[str, Any] | pd.Series,
    registry: IngredientRegistry,
) -> float:
    total = 0.0
    for ingredient in registry.active_ingredients():
        if ingredient.unit != "M" or ingredient.molecular_weight_g_mol is None:
            continue
        total += _numeric(row, ingredient.feature_name) * ingredient.molecular_weight_g_mol
    return float(total)


def feasibility_report(
    row: Mapping[str, Any] | pd.Series,
    registry: IngredientRegistry,
    optimization_config: Mapping[str, Any],
    policy_active: bool,
) -> dict[str, Any]:
    totals = formulation_totals(row, optimization_config)
    reasons: list[str] = []
    cfg = nested_get(optimization_config, "formulation_feasibility", {}) or {}

    for ingredient in registry.active_ingredients():
        value = _numeric(row, ingredient.feature_name)
        if value < ingredient.lower_bound - 1e-12:
            reasons.append(f"{ingredient.feature_name}_below_lower_bound")
        if value > ingredient.upper_bound + 1e-12:
            reasons.append(f"{ingredient.feature_name}_above_upper_bound")

    if policy_active:
        for feature_name, cap in (cfg.get("ingredient_caps", {}) or {}).items():
            if _numeric(row, str(feature_name)) > float(cap) + 1e-12:
                reasons.append(f"{feature_name}_above_campaign_cap")
        if totals["active_polymer_count"] > int(cfg.get("max_active_polymers", 1)):
            reasons.append("multiple_viscosity_active_polymers")
        if totals["total_polymer_pct"] > float(cfg.get("max_total_polymer_pct", 10.0)) + 1e-12:
            reasons.append("total_polymer_pct_exceeds_limit")
        if totals["total_serum_protein_pct"] > float(
            cfg.get("max_total_serum_protein_pct", 10.0)
        ) + 1e-12:
            reasons.append("total_serum_protein_pct_exceeds_limit")
        if totals["total_polymer_serum_pct"] > float(
            cfg.get("max_total_polymer_serum_pct", 15.0)
        ) + 1e-12:
            reasons.append("combined_polymer_serum_pct_exceeds_limit")
        if totals["active_polymer_count"] > 0:
            if totals["total_sugar_M"] > float(
                cfg.get("max_sugar_M_with_polymer", 0.50)
            ) + 1e-12:
                reasons.append("sugar_M_with_polymer_exceeds_limit")
            if totals["total_nonpermeating_solute_M"] > float(
                cfg.get("max_nonpermeating_M_with_polymer", 0.75)
            ) + 1e-12:
                reasons.append("nonpermeating_M_with_polymer_exceeds_limit")

    return {
        **totals,
        "estimated_small_solute_g_L": estimated_small_solute_g_L(row, registry),
        "feasibility_pass": len(reasons) == 0,
        "feasibility_reasons": ";".join(reasons),
    }


def annotate_feasibility(
    candidates: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping[str, Any],
    policy_active: bool,
) -> pd.DataFrame:
    annotated = candidates.copy()
    reports = [
        feasibility_report(row, registry, optimization_config, policy_active)
        for _, row in annotated.iterrows()
    ]
    report_frame = pd.DataFrame(reports, index=annotated.index)
    for column in report_frame.columns:
        annotated[column] = report_frame[column]
    return annotated


def build_support_context(
    formulations: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping[str, Any],
) -> SupportContext:
    ingredients = registry.active_ingredients()
    lower = np.array([ingredient.lower_bound for ingredient in ingredients], dtype=float)
    upper = np.array([ingredient.upper_bound for ingredient in ingredients], dtype=float)
    ranges = np.maximum(upper - lower, 1e-12)
    if formulations.empty:
        return SupportContext(np.empty((0, len(ingredients))), lower, ranges, np.inf)
    matrix = (
        formulations[registry.feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    scaled = (matrix - lower) / ranges
    if len(scaled) < 2:
        radius = np.inf
    else:
        diffs = scaled[:, None, :] - scaled[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        percentile = float(
            nested_get(optimization_config, "support_policy.radius_percentile", 95.0)
        )
        multiplier = float(
            nested_get(optimization_config, "support_policy.radius_multiplier", 1.25)
        )
        radius = float(np.percentile(nearest, percentile) * multiplier)
    return SupportContext(scaled, lower, ranges, radius)


def annotate_support(
    candidates: pd.DataFrame,
    registry: IngredientRegistry,
    support: SupportContext,
) -> pd.DataFrame:
    annotated = candidates.copy()
    if annotated.empty:
        annotated["nearest_support_distance"] = pd.Series(dtype=float)
        annotated["support_status"] = pd.Series(dtype=str)
        return annotated
    matrix = (
        annotated[registry.feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    scaled = (matrix - support.lower_bounds) / support.ranges
    if len(support.observed_scaled) == 0:
        distances = np.full(len(annotated), np.inf)
    else:
        distances = np.min(
            np.linalg.norm(
                scaled[:, None, :] - support.observed_scaled[None, :, :],
                axis=2,
            ),
            axis=1,
        )
    annotated["nearest_support_distance"] = distances
    annotated["support_status"] = np.where(
        distances <= support.radius,
        "in_support",
        "boundary",
    )
    return annotated

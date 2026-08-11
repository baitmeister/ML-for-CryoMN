"""Forward-only formulation feasibility and support diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import nested_get
from .registry import IngredientRegistry, presence_threshold


POLICY_VERSION_DEFAULT = "round2_candidate_feasibility_v1"
ROUND5_POLICY_VERSION = "round5_solubility_viscosity_v2"


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
    cfg = nested_get(optimization_config, "formulation_feasibility", {}) or {}
    configured_versions = cfg.get("policy_versions") or []
    schedule: list[tuple[int, str]] = []
    for item in configured_versions:
        if not isinstance(item, Mapping):
            continue
        schedule.append(
            (
                int(item.get("start_round", 2)),
                str(item.get("policy_version", POLICY_VERSION_DEFAULT)),
            )
        )
    if not schedule:
        schedule = [
            (
                int(cfg.get("start_round", 2)),
                str(cfg.get("policy_version", POLICY_VERSION_DEFAULT)),
            )
        ]
    schedule.sort(key=lambda item: item[0])
    if target_round_number is None:
        start_round, version = schedule[-1]
        return False, version, start_round
    eligible = [item for item in schedule if int(target_round_number) >= item[0]]
    if not eligible:
        start_round, version = schedule[0]
        return False, version, start_round
    start_round, version = eligible[-1]
    return True, version, start_round


def latest_feasibility_policy_version(
    optimization_config: Mapping[str, Any],
) -> str:
    """Return the newest configured policy for direct validation APIs."""
    return policy_activation(optimization_config, None)[1]


def ingredient_upper_bound_for_policy(
    ingredient: Any,
    optimization_config: Mapping[str, Any],
    policy_version: str | None,
) -> float:
    """Use one source of truth for all discrete and continuous generators."""
    version = policy_version or latest_feasibility_policy_version(optimization_config)
    cfg = nested_get(optimization_config, "formulation_feasibility", {}) or {}
    upper = float(ingredient.upper_bound)
    if version == ROUND5_POLICY_VERSION:
        if ingredient.practical_upper_bound is not None:
            upper = min(upper, float(ingredient.practical_upper_bound))
        return upper
    legacy_cap = (cfg.get("ingredient_caps") or {}).get(ingredient.feature_name)
    if legacy_cap is not None:
        upper = min(upper, float(legacy_cap))
    return upper


def _numeric(row: Mapping[str, Any] | pd.Series, feature_name: str) -> float:
    value = pd.to_numeric(row.get(feature_name, 0.0), errors="coerce")
    return 0.0 if pd.isna(value) else float(value)


def formulation_totals(
    row: Mapping[str, Any] | pd.Series,
    optimization_config: Mapping[str, Any],
    registry: IngredientRegistry | None = None,
) -> dict[str, float | int]:
    cfg = nested_get(optimization_config, "formulation_feasibility", {}) or {}
    if registry is not None:
        def grouped(group: str) -> list[str]:
            return [
                ingredient.feature_name
                for ingredient in registry.active_ingredients()
                if group in ingredient.aggregate_groups
            ]

        polymer_features = grouped("viscosity_active_polymer")
        serum_features = grouped("serum_protein")
        sugar_features = grouped("sugar")
        nonpermeating_features = grouped("nonpermeating_osmolyte")
        permeating_features = grouped("permeating_cpa")
    else:
        polymer_features = list(
            cfg.get(
                "polymer_features",
                ["pvp_pct", "dextran_pct", "hyaluronic_acid_pct", "methylcellulose_pct"],
            )
        )
        serum_features = list(
            cfg.get(
                "serum_protein_features",
                ["fbs_pct", "hsa_pct", "human_serum_pct", "sericin_pct"],
            )
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
        permeating_features = [
            "dmso_M",
            "ethylene_glycol_M",
            "glycerol_M",
            "propylene_glycol_M",
        ]
    active_polymers = sum(
        _numeric(row, feature) >= presence_threshold(feature)
        for feature in polymer_features
    )
    strict_active_polymers = sum(
        _numeric(row, feature) > 1e-12 for feature in polymer_features
    )
    total_polymer = sum(_numeric(row, feature) for feature in polymer_features)
    total_serum = sum(_numeric(row, feature) for feature in serum_features)
    total_sugar = sum(_numeric(row, feature) for feature in sugar_features)
    total_nonpermeating = sum(
        _numeric(row, feature) for feature in nonpermeating_features
    )
    total_permeating = sum(_numeric(row, feature) for feature in permeating_features)
    return {
        "active_polymer_count": int(active_polymers),
        "strict_active_polymer_count": int(strict_active_polymers),
        "total_polymer_pct": float(total_polymer),
        "total_serum_protein_pct": float(total_serum),
        "total_polymer_serum_pct": float(total_polymer + total_serum),
        "total_viscosity_active_macromolecule_pct": float(total_polymer + total_serum),
        "total_permeating_cpa_M": float(total_permeating),
        "total_sugar_M": float(total_sugar),
        "total_nonpermeating_solute_M": float(total_nonpermeating),
    }


def crystalline_solute_saturation_burden(
    row: Mapping[str, Any] | pd.Series,
    registry: IngredientRegistry,
) -> float:
    burden = 0.0
    for ingredient in registry.active_ingredients():
        if "crystalline_solute" not in ingredient.aggregate_groups:
            continue
        solubility = ingredient.aqueous_solubility_M
        if solubility is None or solubility <= 0:
            raise ValueError(
                f"Missing positive solubility for {ingredient.feature_name}"
            )
        burden += _numeric(row, ingredient.feature_name) / float(solubility)
    return float(burden)


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
    policy_version: str | None = None,
) -> dict[str, Any]:
    selected_version = policy_version or latest_feasibility_policy_version(
        optimization_config
    )
    totals = formulation_totals(row, optimization_config, registry)
    if selected_version == ROUND5_POLICY_VERSION:
        totals["active_polymer_count"] = totals["strict_active_polymer_count"]
    saturation_burden = crystalline_solute_saturation_burden(row, registry)
    reasons: list[str] = []
    cfg = nested_get(optimization_config, "formulation_feasibility", {}) or {}

    for ingredient in registry.active_ingredients():
        value = _numeric(row, ingredient.feature_name)
        if value < ingredient.lower_bound - 1e-12:
            reasons.append(f"{ingredient.feature_name}_below_lower_bound")
        if value > ingredient.upper_bound + 1e-12:
            reasons.append(f"{ingredient.feature_name}_above_upper_bound")

    if policy_active and selected_version == ROUND5_POLICY_VERSION:
        rules = cfg.get("round5_rules") or {}
        for ingredient in registry.active_ingredients():
            value = _numeric(row, ingredient.feature_name)
            practical_upper = ingredient_upper_bound_for_policy(
                ingredient, optimization_config, selected_version
            )
            if value > practical_upper + 1e-12:
                reasons.append(
                    f"{ingredient.feature_name}_above_practical_upper_bound"
                )
        if totals["active_polymer_count"] > int(rules.get("max_active_polymers", 1)):
            reasons.append("multiple_viscosity_active_polymers")
        if totals["total_polymer_pct"] > float(
            rules.get("max_total_polymer_pct", 10.0)
        ) + 1e-12:
            reasons.append("total_polymer_pct_exceeds_limit")
        if totals["total_serum_protein_pct"] > float(
            rules.get("max_total_serum_protein_pct", 10.0)
        ) + 1e-12:
            reasons.append("total_serum_protein_pct_exceeds_limit")
        if totals["total_polymer_serum_pct"] > float(
            rules.get("max_total_polymer_serum_pct", 15.0)
        ) + 1e-12:
            reasons.append("combined_polymer_serum_pct_exceeds_limit")
        if totals["total_permeating_cpa_M"] > float(
            rules.get("max_total_permeating_cpa_M", 2.50)
        ) + 1e-12:
            reasons.append("total_permeating_cpa_M_exceeds_limit")
        if totals["total_sugar_M"] > float(
            rules.get("max_total_sugar_M", 0.50)
        ) + 1e-12:
            reasons.append("total_sugar_M_exceeds_limit")
        if totals["total_nonpermeating_solute_M"] > float(
            rules.get("max_total_nonpermeating_solute_M", 0.50)
        ) + 1e-12:
            reasons.append("total_nonpermeating_solute_M_exceeds_limit")
        if totals["active_polymer_count"] > 0:
            if totals["total_sugar_M"] > float(
                rules.get("max_sugar_M_with_polymer", 0.30)
            ) + 1e-12:
                reasons.append("sugar_M_with_polymer_exceeds_limit")
            if totals["total_nonpermeating_solute_M"] > float(
                rules.get("max_nonpermeating_M_with_polymer", 0.30)
            ) + 1e-12:
                reasons.append("nonpermeating_M_with_polymer_exceeds_limit")
        if saturation_burden > float(
            rules.get("max_crystalline_solute_saturation_burden", 0.80)
        ) + 1e-12:
            reasons.append("crystalline_solute_saturation_burden_exceeds_limit")
    elif policy_active:
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

    report = {
        key: totals[key]
        for key in (
            "active_polymer_count",
            "total_polymer_pct",
            "total_serum_protein_pct",
            "total_polymer_serum_pct",
            "total_sugar_M",
            "total_nonpermeating_solute_M",
        )
    }
    if policy_active and selected_version == ROUND5_POLICY_VERSION:
        report.update(
            {
                "total_viscosity_active_macromolecule_pct": totals[
                    "total_viscosity_active_macromolecule_pct"
                ],
                "total_permeating_cpa_M": totals["total_permeating_cpa_M"],
                "crystalline_solute_saturation_burden": saturation_burden,
                "feasibility_policy_version": selected_version,
            }
        )
    return {
        **report,
        "estimated_small_solute_g_L": estimated_small_solute_g_L(row, registry),
        "feasibility_pass": len(reasons) == 0,
        "feasibility_reasons": ";".join(reasons),
    }


def annotate_feasibility(
    candidates: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping[str, Any],
    policy_active: bool,
    policy_version: str | None = None,
) -> pd.DataFrame:
    annotated = candidates.copy()
    reports = [
        feasibility_report(
            row,
            registry,
            optimization_config,
            policy_active,
            policy_version=policy_version,
        )
        for _, row in annotated.iterrows()
    ]
    report_frame = pd.DataFrame(reports, index=annotated.index)
    for column in report_frame.columns:
        annotated[column] = report_frame[column]
    return annotated


def _support_source_formulations(
    formulations: pd.DataFrame,
    observations: pd.DataFrame | None,
) -> pd.DataFrame:
    if formulations.empty or observations is None or observations.empty:
        return formulations
    if "formulation_id" not in formulations.columns or "formulation_id" not in observations.columns:
        return formulations
    observed_ids = (
        observations["formulation_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    observed_ids = set(observed_ids[observed_ids != ""].unique().tolist())
    if not observed_ids:
        return formulations

    support_formulations = formulations[
        formulations["formulation_id"].astype(str).isin(observed_ids)
    ].copy()
    if support_formulations.empty:
        return formulations
    return support_formulations.drop_duplicates("formulation_id", keep="last")


def build_support_context(
    formulations: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping[str, Any],
    observations: pd.DataFrame | None = None,
) -> SupportContext:
    ingredients = registry.active_ingredients()
    lower = np.array([ingredient.lower_bound for ingredient in ingredients], dtype=float)
    upper = np.array([ingredient.upper_bound for ingredient in ingredients], dtype=float)
    ranges = np.maximum(upper - lower, 1e-12)
    support_formulations = _support_source_formulations(formulations, observations)
    if support_formulations.empty:
        return SupportContext(np.empty((0, len(ingredients))), lower, ranges, np.inf)
    matrix = (
        support_formulations[registry.feature_names]
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

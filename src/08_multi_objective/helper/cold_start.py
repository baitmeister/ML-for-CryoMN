"""Campaign-evidence classification for cold-start formulation ingredients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from .config import nested_get
from .registry import IngredientRegistry, presence_threshold
from .status import parse_round_number


@dataclass(frozen=True)
class ColdStartPolicy:
    policy_version: str
    start_round: int
    active: bool
    evidence_source_types: tuple[str, ...]
    evidence_endpoint: str
    minimum_distinct_formulations: int
    max_ordinary_rows_per_ingredient: int
    graduation_slots_per_round: int
    exempt_candidate_origins: tuple[str, ...]
    graduation_max_cold_ingredients_per_row: int
    graduation_require_distinct_active_sets: bool
    same_origin_replacement_preferred: bool
    preserve_origin_allocation: bool


@dataclass(frozen=True)
class ColdStartContext:
    policy: ColdStartPolicy
    evidence_counts: dict[str, int]
    last_campaign_round: dict[str, int | None]
    cold_ingredients: tuple[str, ...]

    def is_exempt_origin(self, origin: object) -> bool:
        return str(origin or "").strip() in self.policy.exempt_candidate_origins


def resolve_cold_start_policy(
    optimization_config: Mapping[str, Any],
    target_round_number: int | None,
) -> ColdStartPolicy:
    cfg = nested_get(optimization_config, "cold_start_policy", {}) or {}
    start_round = int(cfg.get("start_round", 6))
    minimum = int(cfg.get("minimum_distinct_formulations", 3))
    cap = int(cfg.get("max_ordinary_rows_per_ingredient", 2))
    slots = int(cfg.get("graduation_slots_per_round", 4))
    max_cold = int(cfg.get("graduation_max_cold_ingredients_per_row", 1))
    if minimum < 1:
        raise ValueError(
            "cold_start_policy.minimum_distinct_formulations must be at least 1."
        )
    if cap < 1:
        raise ValueError(
            "cold_start_policy.max_ordinary_rows_per_ingredient must be at least 1."
        )
    if slots < 0:
        raise ValueError(
            "cold_start_policy.graduation_slots_per_round must be non-negative."
        )
    if max_cold < 1:
        raise ValueError(
            "cold_start_policy.graduation_max_cold_ingredients_per_row must be at least 1."
        )
    return ColdStartPolicy(
        policy_version=str(
            cfg.get("policy_version", "round6_cold_start_graduation_v1")
        ),
        start_round=start_round,
        active=bool(
            target_round_number is not None
            and int(target_round_number) >= start_round
        ),
        evidence_source_types=tuple(
            str(value).strip()
            for value in cfg.get("evidence_source_types", ["wetlab_feedback"])
            if str(value).strip()
        ),
        evidence_endpoint=str(cfg.get("evidence_endpoint", "viability_percent")),
        minimum_distinct_formulations=minimum,
        max_ordinary_rows_per_ingredient=cap,
        graduation_slots_per_round=slots,
        exempt_candidate_origins=tuple(
            str(value).strip()
            for value in cfg.get(
                "exempt_candidate_origins", ["retest", "rescue_dilution"]
            )
            if str(value).strip()
        ),
        graduation_max_cold_ingredients_per_row=max_cold,
        graduation_require_distinct_active_sets=bool(
            cfg.get("graduation_require_distinct_active_sets", True)
        ),
        same_origin_replacement_preferred=bool(
            cfg.get("same_origin_replacement_preferred", True)
        ),
        preserve_origin_allocation=bool(
            cfg.get("preserve_origin_allocation", True)
        ),
    )


def build_cold_start_context(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
    policy: ColdStartPolicy,
    target_round_number: int | None,
    unavailable_feature_names: list[str] | tuple[str, ...] = (),
) -> ColdStartContext:
    counts = {feature_name: 0 for feature_name in registry.feature_names}
    last_round: dict[str, int | None] = {
        feature_name: None for feature_name in registry.feature_names
    }
    if policy.active and not formulations.empty and not observations.empty:
        required = {
            "formulation_id",
            "batch_id",
            "endpoint",
            "value",
            "source_type",
        }
        if required.issubset(observations.columns):
            obs = observations.loc[
                observations["endpoint"].astype(str).eq(policy.evidence_endpoint)
                & observations["source_type"].astype(str).isin(
                    policy.evidence_source_types
                )
                & pd.to_numeric(observations["value"], errors="coerce").notna()
            ].copy()
            obs["round_number"] = obs["batch_id"].map(parse_round_number)
            obs = obs.loc[obs["round_number"].notna()].copy()
            if target_round_number is not None:
                obs = obs.loc[
                    obs["round_number"].astype(int) < int(target_round_number)
                ]
            tested = obs[["formulation_id", "round_number"]].drop_duplicates()
            tested = tested.merge(
                formulations[["formulation_id", *registry.feature_names]],
                on="formulation_id",
                how="inner",
            )
            for feature_name in registry.feature_names:
                present = pd.to_numeric(
                    tested[feature_name], errors="coerce"
                ).fillna(0.0).abs() >= presence_threshold(feature_name)
                evidence = tested.loc[present, ["formulation_id", "round_number"]]
                counts[feature_name] = int(evidence["formulation_id"].nunique())
                if not evidence.empty:
                    last_round[feature_name] = int(evidence["round_number"].max())

    unavailable = set(str(value) for value in unavailable_feature_names)
    cold = (
        tuple(
            feature_name
            for feature_name in registry.feature_names
            if feature_name not in unavailable
            and counts[feature_name] < policy.minimum_distinct_formulations
        )
        if policy.active
        else tuple()
    )
    return ColdStartContext(
        policy=policy,
        evidence_counts=counts,
        last_campaign_round=last_round,
        cold_ingredients=cold,
    )


def cold_ingredients_in_row(
    row: Mapping[str, Any] | pd.Series,
    registry: IngredientRegistry,
    context: ColdStartContext,
) -> tuple[str, ...]:
    present: list[str] = []
    for feature_name in context.cold_ingredients:
        numeric = pd.to_numeric(row.get(feature_name, 0.0), errors="coerce")
        value = float(numeric) if pd.notna(numeric) else 0.0
        if abs(value) >= presence_threshold(feature_name):
            present.append(feature_name)
    return tuple(present)


def annotate_cold_start_candidates(
    candidates: pd.DataFrame,
    registry: IngredientRegistry,
    context: ColdStartContext,
) -> pd.DataFrame:
    annotated = candidates.copy()
    cold_sets = [
        cold_ingredients_in_row(row, registry, context)
        for _, row in annotated.iterrows()
    ]
    annotated["cold_start_ingredients"] = [";".join(values) for values in cold_sets]
    annotated["cold_start_ingredient_count"] = [len(values) for values in cold_sets]
    annotated["cold_start_prior_evidence_counts"] = [
        ";".join(
            f"{feature_name}={context.evidence_counts.get(feature_name, 0)}"
            for feature_name in values
        )
        for values in cold_sets
    ]
    annotated["cold_start_ordinary_exempt"] = [
        context.is_exempt_origin(row.get("candidate_origin", ""))
        for _, row in annotated.iterrows()
    ]
    annotated["cold_start_graduation_eligible"] = [
        bool(
            context.policy.active
            and not context.is_exempt_origin(row.get("candidate_origin", ""))
            and len(values)
            <= context.policy.graduation_max_cold_ingredients_per_row
            and len(values) == 1
        )
        for values, (_, row) in zip(cold_sets, annotated.iterrows())
    ]
    ingredient_priority = {
        feature_name: position
        for position, feature_name in enumerate(
            graduation_ingredient_order(context, registry), start=1
        )
    }
    annotated["cold_start_graduation_priority"] = [
        (
            ingredient_priority.get(values[0], "")
            if eligible and len(values) == 1
            else ""
        )
        for values, eligible in zip(
            cold_sets,
            annotated["cold_start_graduation_eligible"].tolist(),
        )
    ]
    annotated["cold_start_graduation_reason"] = [
        (
            f"eligible:{values[0]};"
            f"prior_distinct_formulations={context.evidence_counts.get(values[0], 0)};"
            f"graduation_threshold={context.policy.minimum_distinct_formulations}"
            if eligible and len(values) == 1
            else (
                "ineligible:special_origin_exemption"
                if context.is_exempt_origin(row.get("candidate_origin", ""))
                else (
                    "ineligible:must_contain_exactly_one_cold_start_ingredient"
                    if context.policy.active
                    else "inactive_before_policy_start_round"
                )
            )
        )
        for values, eligible, (_, row) in zip(
            cold_sets,
            annotated["cold_start_graduation_eligible"].tolist(),
            annotated.iterrows(),
        )
    ]
    return annotated


def graduation_ingredient_order(
    context: ColdStartContext,
    registry: IngredientRegistry,
) -> list[str]:
    """Order cold ingredients by closeness to graduation and recency."""
    registry_order = {
        feature_name: index
        for index, feature_name in enumerate(registry.feature_names)
    }
    return sorted(
        context.cold_ingredients,
        key=lambda feature_name: (
            -context.evidence_counts.get(feature_name, 0),
            context.last_campaign_round.get(feature_name)
            if context.last_campaign_round.get(feature_name) is not None
            else -1,
            registry_order[feature_name],
        ),
    )


def planned_graduation_allocations(
    context: ColdStartContext,
    registry: IngredientRegistry,
) -> list[str]:
    """Allocate one breadth-first slot, then bounded second slots."""
    if not context.policy.active or context.policy.graduation_slots_per_round <= 0:
        return []
    ordered = graduation_ingredient_order(context, registry)
    allocations: list[str] = []
    allocated_counts = {feature_name: 0 for feature_name in ordered}

    for feature_name in ordered:
        if len(allocations) >= context.policy.graduation_slots_per_round:
            break
        deficit = max(
            context.policy.minimum_distinct_formulations
            - context.evidence_counts.get(feature_name, 0),
            0,
        )
        if deficit > 0:
            allocations.append(feature_name)
            allocated_counts[feature_name] += 1

    while len(allocations) < context.policy.graduation_slots_per_round:
        eligible = [
            feature_name
            for feature_name in ordered
            if allocated_counts[feature_name]
            < min(
                context.policy.max_ordinary_rows_per_ingredient,
                max(
                    context.policy.minimum_distinct_formulations
                    - context.evidence_counts.get(feature_name, 0),
                    0,
                ),
            )
        ]
        if not eligible:
            break
        eligible.sort(
            key=lambda feature_name: (
                -(
                    context.evidence_counts.get(feature_name, 0)
                    + allocated_counts[feature_name]
                ),
                ordered.index(feature_name),
            )
        )
        feature_name = eligible[0]
        allocations.append(feature_name)
        allocated_counts[feature_name] += 1
    return allocations


def graduation_allocation_attempts(
    context: ColdStartContext,
    registry: IngredientRegistry,
) -> list[str]:
    """Return the ordered allocation queue, including reassignment backups."""
    if not context.policy.active or context.policy.graduation_slots_per_round <= 0:
        return []
    ordered = graduation_ingredient_order(context, registry)
    attempts = list(ordered)
    allocated_counts = {feature_name: 1 for feature_name in ordered}
    while True:
        eligible = [
            feature_name
            for feature_name in ordered
            if allocated_counts[feature_name]
            < min(
                context.policy.max_ordinary_rows_per_ingredient,
                max(
                    context.policy.minimum_distinct_formulations
                    - context.evidence_counts.get(feature_name, 0),
                    0,
                ),
            )
        ]
        if not eligible:
            break
        eligible.sort(
            key=lambda feature_name: (
                -(
                    context.evidence_counts.get(feature_name, 0)
                    + allocated_counts[feature_name]
                ),
                ordered.index(feature_name),
            )
        )
        feature_name = eligible[0]
        attempts.append(feature_name)
        allocated_counts[feature_name] += 1
    return attempts


def cold_start_policy_metadata(context: ColdStartContext) -> dict[str, Any]:
    policy = context.policy
    return {
        "policy_version": policy.policy_version,
        "start_round": policy.start_round,
        "active": policy.active,
        "evidence_source_types": list(policy.evidence_source_types),
        "evidence_endpoint": policy.evidence_endpoint,
        "minimum_distinct_formulations": policy.minimum_distinct_formulations,
        "max_ordinary_rows_per_ingredient": policy.max_ordinary_rows_per_ingredient,
        "graduation_slots_per_round": policy.graduation_slots_per_round,
        "exempt_candidate_origins": list(policy.exempt_candidate_origins),
        "special_rows_count_toward_cold_cap": False,
        "special_rows_may_contribute_future_evidence": True,
        "same_origin_replacement_preferred": policy.same_origin_replacement_preferred,
        "preserve_origin_allocation": policy.preserve_origin_allocation,
        "cold_ingredients": list(context.cold_ingredients),
        "prior_distinct_formulation_counts": {
            feature_name: int(context.evidence_counts.get(feature_name, 0))
            for feature_name in context.cold_ingredients
        },
        "last_campaign_round": {
            feature_name: context.last_campaign_round.get(feature_name)
            for feature_name in context.cold_ingredients
        },
    }

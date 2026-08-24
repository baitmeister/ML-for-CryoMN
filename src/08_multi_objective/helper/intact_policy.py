"""Empirical exact-combination intact-patch feasibility policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import nested_get
from .endpoints import INTACT_PATCH_ENDPOINT, aggregate_intact_patch_replicates
from .registry import IngredientRegistry, presence_threshold
from .status import parse_round_number


@dataclass(frozen=True)
class IntactCombinationPolicy:
    policy_version: str
    start_round: int
    active: bool
    evidence_source_types: tuple[str, ...]
    evidence_radius: float
    beta_prior_pass: float
    beta_prior_fail: float
    screening_neutral_probability: float
    screening_max_penalty: float
    mechanics_mode: str
    compatibility_mode: str
    numerical_probability_floor: float
    mechanics_primary_test_count: int
    mechanics_backup_behavior: str


def resolve_intact_combination_policy(
    optimization_config: Mapping[str, Any],
    target_round_number: int | None,
) -> IntactCombinationPolicy:
    cfg = nested_get(optimization_config, "intact_combination_policy", {}) or {}
    start_round = int(cfg.get("start_round", 6))
    radius = float(cfg.get("evidence_radius", 0.35))
    prior_pass = float(cfg.get("beta_prior_pass", 1.0))
    prior_fail = float(cfg.get("beta_prior_fail", 1.0))
    neutral = float(cfg.get("screening_neutral_probability", 0.50))
    max_penalty = float(cfg.get("screening_max_penalty", 0.20))
    probability_floor = float(cfg.get("numerical_probability_floor", 1e-9))
    primary_test_count = int(cfg.get("mechanics_primary_test_count", 4))
    source_types = tuple(
        str(value).strip()
        for value in cfg.get("evidence_source_types", ["wetlab_feedback"])
        if str(value).strip()
    )
    mechanics_mode = str(
        cfg.get("mechanics_mode", "empirical_feasibility_weighted")
    ).strip()
    compatibility_mode = str(
        cfg.get("compatibility_mode", "classifier_threshold_penalty")
    ).strip()

    if radius <= 0.0:
        raise ValueError("intact_combination_policy.evidence_radius must be > 0.")
    if prior_pass <= 0.0 or prior_fail <= 0.0:
        raise ValueError("Intact Beta-prior parameters must both be > 0.")
    if not 0.0 < neutral < 1.0:
        raise ValueError(
            "intact_combination_policy.screening_neutral_probability must be in (0, 1)."
        )
    if max_penalty < 0.0:
        raise ValueError(
            "intact_combination_policy.screening_max_penalty must be non-negative."
        )
    if not 0.0 < probability_floor < 1.0:
        raise ValueError(
            "intact_combination_policy.numerical_probability_floor must be in (0, 1)."
        )
    if primary_test_count < 1:
        raise ValueError(
            "intact_combination_policy.mechanics_primary_test_count must be at least 1."
        )
    if mechanics_mode not in {
        "empirical_feasibility_weighted",
        "classifier_threshold_penalty",
    }:
        raise ValueError(
            "intact_combination_policy.mechanics_mode must be "
            "'empirical_feasibility_weighted' or 'classifier_threshold_penalty'."
        )

    return IntactCombinationPolicy(
        policy_version=str(
            cfg.get("policy_version", "round6_empirical_combination_intact_v1")
        ),
        start_round=start_round,
        active=bool(
            target_round_number is not None
            and int(target_round_number) >= start_round
        ),
        evidence_source_types=source_types,
        evidence_radius=radius,
        beta_prior_pass=prior_pass,
        beta_prior_fail=prior_fail,
        screening_neutral_probability=neutral,
        screening_max_penalty=max_penalty,
        mechanics_mode=mechanics_mode,
        compatibility_mode=compatibility_mode,
        numerical_probability_floor=probability_floor,
        mechanics_primary_test_count=primary_test_count,
        mechanics_backup_behavior=str(
            cfg.get(
                "mechanics_backup_behavior",
                "actual_intact_priority_promotion",
            )
        ).strip(),
    )


def active_ingredient_set(
    row: Mapping[str, Any] | pd.Series,
    registry: IngredientRegistry,
) -> tuple[str, ...]:
    """Return the exact ordered set of ingredients above presence floors."""
    active: list[str] = []
    for feature_name in registry.feature_names:
        value = pd.to_numeric(row.get(feature_name, 0.0), errors="coerce")
        if pd.notna(value) and abs(float(value)) >= presence_threshold(feature_name):
            active.append(feature_name)
    return tuple(active)


def build_intact_evidence(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
    policy: IntactCombinationPolicy,
    target_round_number: int | None,
) -> pd.DataFrame:
    """Build one prior-round intact label per formulation and campaign batch."""
    columns = [
        "formulation_id",
        "batch_id",
        "intact_patch_formation_pass",
        "active_ingredient_set",
        *registry.feature_names,
    ]
    if formulations.empty or observations.empty or not policy.active:
        return pd.DataFrame(columns=columns)

    required = {"formulation_id", "batch_id", "endpoint", "value", "source_type"}
    if not required.issubset(observations.columns):
        return pd.DataFrame(columns=columns)

    obs = observations.loc[
        observations["endpoint"].astype(str).eq(INTACT_PATCH_ENDPOINT)
        & observations["source_type"].astype(str).isin(policy.evidence_source_types)
    ].copy()
    if obs.empty:
        return pd.DataFrame(columns=columns)
    obs["round_number"] = obs["batch_id"].map(parse_round_number)
    obs = obs.loc[obs["round_number"].notna()].copy()
    if target_round_number is not None:
        obs = obs.loc[obs["round_number"].astype(int) < int(target_round_number)]
    if obs.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        obs.groupby(["formulation_id", "batch_id"], as_index=False, dropna=False)
        .agg(value=("value", aggregate_intact_patch_replicates))
        .rename(columns={"value": "intact_patch_formation_pass"})
    )
    feature_frame = formulations[["formulation_id", *registry.feature_names]].copy()
    evidence = grouped.merge(feature_frame, on="formulation_id", how="inner")
    for feature_name in registry.feature_names:
        evidence[feature_name] = pd.to_numeric(
            evidence[feature_name], errors="coerce"
        ).fillna(0.0)
    evidence["active_ingredient_set"] = evidence.apply(
        lambda row: active_ingredient_set(row, registry), axis=1
    )
    return evidence[columns].reset_index(drop=True)


def _combination_evidence_for_row(
    row: Mapping[str, Any] | pd.Series,
    evidence: pd.DataFrame,
    registry: IngredientRegistry,
    policy: IntactCombinationPolicy,
) -> dict[str, Any]:
    combo = active_ingredient_set(row, registry)
    prior_pass = policy.beta_prior_pass
    prior_fail = policy.beta_prior_fail
    prior_probability = prior_pass / (prior_pass + prior_fail)
    if not combo or evidence.empty:
        return {
            "empirical_combination_pass_probability": prior_probability,
            "empirical_combination_weighted_passes": 0.0,
            "empirical_combination_weighted_failures": 0.0,
            "nearest_matching_intact_pass_distance": np.nan,
            "nearest_matching_intact_failure_distance": np.nan,
            "intact_combination_screening_penalty": 0.0,
            "intact_combination_policy_version": policy.policy_version,
        }

    same = evidence.loc[
        evidence["active_ingredient_set"].map(
            lambda value: tuple(value) == combo
        )
    ]
    if same.empty:
        return {
            "empirical_combination_pass_probability": prior_probability,
            "empirical_combination_weighted_passes": 0.0,
            "empirical_combination_weighted_failures": 0.0,
            "nearest_matching_intact_pass_distance": np.nan,
            "nearest_matching_intact_failure_distance": np.nan,
            "intact_combination_screening_penalty": 0.0,
            "intact_combination_policy_version": policy.policy_version,
        }

    ranges = {
        feature_name: max(
            registry.get_by_feature(feature_name).upper_bound
            - registry.get_by_feature(feature_name).lower_bound,
            1e-12,
        )
        for feature_name in combo
    }
    candidate_values = {}
    for feature_name in combo:
        numeric = pd.to_numeric(row.get(feature_name, 0.0), errors="coerce")
        candidate_values[feature_name] = (
            float(numeric) if pd.notna(numeric) else 0.0
        )
    weighted_passes = 0.0
    weighted_failures = 0.0
    pass_distances: list[float] = []
    failure_distances: list[float] = []
    for _, anchor in same.iterrows():
        distance = float(
            np.sqrt(
                np.mean(
                    [
                        (
                            (candidate_values[feature_name] - float(anchor[feature_name]))
                            / ranges[feature_name]
                        )
                        ** 2
                        for feature_name in combo
                    ]
                )
            )
        )
        passed = float(anchor["intact_patch_formation_pass"]) >= 0.5
        if passed:
            pass_distances.append(distance)
        else:
            failure_distances.append(distance)
        weight = max(0.0, 1.0 - distance / policy.evidence_radius)
        if passed:
            weighted_passes += weight
        else:
            weighted_failures += weight

    probability = (
        prior_pass + weighted_passes
    ) / (
        prior_pass + prior_fail + weighted_passes + weighted_failures
    )
    neutral = policy.screening_neutral_probability
    screening_penalty = policy.screening_max_penalty * max(
        0.0,
        (neutral - probability) / neutral,
    )
    return {
        "empirical_combination_pass_probability": float(probability),
        "empirical_combination_weighted_passes": float(weighted_passes),
        "empirical_combination_weighted_failures": float(weighted_failures),
        "nearest_matching_intact_pass_distance": (
            float(min(pass_distances)) if pass_distances else np.nan
        ),
        "nearest_matching_intact_failure_distance": (
            float(min(failure_distances)) if failure_distances else np.nan
        ),
        "intact_combination_screening_penalty": float(screening_penalty),
        "intact_combination_policy_version": policy.policy_version,
    }


def annotate_intact_combination_evidence(
    candidates: pd.DataFrame,
    evidence: pd.DataFrame,
    registry: IngredientRegistry,
    policy: IntactCombinationPolicy,
) -> pd.DataFrame:
    """Attach empirical combination feasibility diagnostics to candidates."""
    annotated = candidates.copy()
    reports = [
        _combination_evidence_for_row(row, evidence, registry, policy)
        for _, row in annotated.iterrows()
    ]
    report_frame = pd.DataFrame(reports, index=annotated.index)
    for column in report_frame.columns:
        annotated[column] = report_frame[column]
    return annotated


def intact_policy_metadata(
    policy: IntactCombinationPolicy,
    evidence: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "policy_version": policy.policy_version,
        "start_round": policy.start_round,
        "active": policy.active,
        "evidence_source_types": list(policy.evidence_source_types),
        "evidence_radius": policy.evidence_radius,
        "distance_metric": "registry_bounds_normalized_rms_exact_active_set",
        "beta_prior_pass": policy.beta_prior_pass,
        "beta_prior_fail": policy.beta_prior_fail,
        "unseen_combination_pass_probability": (
            policy.beta_prior_pass
            / (policy.beta_prior_pass + policy.beta_prior_fail)
        ),
        "screening_neutral_probability": policy.screening_neutral_probability,
        "screening_max_penalty": policy.screening_max_penalty,
        "mechanics_mode": policy.mechanics_mode,
        "compatibility_mode": policy.compatibility_mode,
        "mechanics_primary_test_count": policy.mechanics_primary_test_count,
        "mechanics_backup_behavior": policy.mechanics_backup_behavior,
        "classifier_probability_selection_role": (
            "compatibility_only"
            if policy.active
            and policy.mechanics_mode == "empirical_feasibility_weighted"
            else "active"
        ),
        "prior_formulation_batch_evidence_count": int(len(evidence)),
    }

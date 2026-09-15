"""Candidate scoring and next-batch selection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .acquisition import (
    botorch_available,
    minmax,
    qlognehvi_proxy_scores,
    try_botorch_optimize_qlognehvi,
    try_botorch_qlognehvi_scores,
)
from .artifacts import EDITABLE_WETLAB_COLUMNS
from .candidates import stable_formulation_id
from .cold_start import (
    ColdStartContext,
    annotate_cold_start_candidates,
    build_cold_start_context,
    cold_ingredients_in_row,
    cold_start_policy_metadata,
    graduation_allocation_attempts,
    planned_graduation_allocations,
    resolve_cold_start_policy,
)
from .config import nested_get
from .feasibility import (
    ROUND5_POLICY_VERSION,
    annotate_feasibility,
    annotate_support,
    build_support_context,
    feasibility_report,
    ingredient_upper_bound_for_policy,
    policy_activation,
)
from .models import EndpointModels, train_endpoint_models, paired_objective_frame
from .intact_policy import (
    IntactCombinationPolicy,
    annotate_intact_combination_evidence,
    build_intact_evidence,
    intact_policy_metadata,
    resolve_intact_combination_policy,
)
from .phase import (
    PHASE_BOOTSTRAP,
    PHASE_HYBRID,
    PHASE_MECHANICS,
    PHASE_SCREENING,
    PhaseResolution,
    resolve_phase_mode,
)
from .penalties import constraint_report, count_active_ingredients
from .prediction_labels import (
    annotate_viability_prediction_labels,
    is_unknown_viability_status,
)
from .registry import IngredientRegistry, presence_threshold
from .retest import build_retest_candidates
from .similarity import (
    SimilarityAudit,
    build_history_similarity_index,
    filter_frame_by_similarity,
    is_retest_row,
    resolve_similarity_policy,
    similarity_priority_order,
    validate_selected_similarity,
)

def _feature_matrix(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    return frame[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _scaled_matrix(matrix: np.ndarray) -> np.ndarray:
    low = np.nanmin(matrix, axis=0)
    high = np.nanmax(matrix, axis=0)
    spread = np.where((high - low) < 1e-12, 1.0, high - low)
    return (matrix - low) / spread


def _registry_scaled_feature_matrix(
    frame: pd.DataFrame,
    registry: IngredientRegistry,
) -> np.ndarray:
    if frame.empty:
        return np.empty((0, len(registry.feature_names)), dtype=float)
    matrix = _feature_matrix(frame, registry.feature_names)
    lower = np.asarray(
        [registry.get_by_feature(name).lower_bound for name in registry.feature_names],
        dtype=float,
    )
    upper = np.asarray(
        [registry.get_by_feature(name).upper_bound for name in registry.feature_names],
        dtype=float,
    )
    spread = np.where((upper - lower) < 1e-12, 1.0, upper - lower)
    return (matrix - lower) / spread


def _mechanical_history_counts(observations: pd.DataFrame) -> pd.Series:
    if observations.empty:
        return pd.Series(dtype=int)
    required = {"formulation_id", "batch_id", "endpoint", "value"}
    if not required.issubset(observations.columns):
        return pd.Series(dtype=int)
    mechanical = observations.loc[
        observations["endpoint"].astype(str).eq(
            "critical_axial_load_N_per_needle"
        )
        & pd.to_numeric(observations["value"], errors="coerce").notna(),
        ["formulation_id", "batch_id"],
    ].drop_duplicates()
    if mechanical.empty:
        return pd.Series(dtype=int)
    return mechanical.groupby("formulation_id")["batch_id"].nunique().astype(int)


def _annotate_mechanical_history(
    frame: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    annotated = frame.copy()
    counts = _mechanical_history_counts(observations)
    annotated["prior_mechanical_observation_count"] = (
        annotated.get("formulation_id", pd.Series("", index=annotated.index))
        .astype(str)
        .map(counts)
        .fillna(0)
        .astype(int)
    )
    is_anchor = annotated.get(
        "recommendation_type", pd.Series("", index=annotated.index)
    ).astype(str).eq("mechanics_anchor")
    annotated["mechanical_repeat_allowed"] = (
        annotated["prior_mechanical_observation_count"].eq(0) | is_anchor
    )
    annotated["mechanical_repeat_status"] = np.select(
        [
            is_anchor,
            annotated["prior_mechanical_observation_count"].gt(0),
        ],
        ["anchor_allowed", "previously_measured"],
        default="unmeasured",
    )
    if "mechanical_transition_role" not in annotated.columns:
        annotated["mechanical_transition_role"] = ""
    return annotated


def annotate_candidates(
    candidates: pd.DataFrame,
    models: EndpointModels,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    policy_active: bool = False,
) -> pd.DataFrame:
    annotated = candidates.copy()
    x = _feature_matrix(annotated, registry.feature_names)

    viability = models.viability.predict(x)
    critical_load = models.critical_load.predict(x)
    stiffness = models.initial_stiffness.predict(x)
    intact_probability = models.intact.predict_proba(x)
    if not policy_active and not models.intact.fitted:
        # Preserve the already-executed ROUND_001 scoring path exactly.
        intact_probability = np.ones(x.shape[0], dtype=float)

    kappa_v = float(nested_get(optimization_config, "selection.viability_ucb_kappa", 0.35))
    kappa_m = float(nested_get(optimization_config, "selection.mechanical_ucb_kappa", 0.50))

    annotated["predicted_viability_percent"] = viability.mean
    viability_std = np.asarray(viability.std, dtype=float)
    in_support = np.ones(len(annotated), dtype=bool)
    cap_percentile = float(
        nested_get(
            optimization_config,
            "support_policy.uncertainty_cap_percentile",
            90.0,
        )
    )
    if policy_active and "support_status" in annotated.columns:
        in_support = annotated["support_status"].astype(str).eq("in_support").to_numpy()
        if np.any(in_support):
            viability_cap = float(
                np.percentile(viability_std[in_support], cap_percentile)
            )
            viability_std = np.where(
                in_support,
                viability_std,
                np.minimum(viability_std, viability_cap),
            )
    annotated["viability_std"] = viability_std
    annotated["viability_ucb"] = viability.mean + kappa_v * viability_std
    annotated["predicted_critical_axial_load_N_per_needle"] = critical_load.mean
    critical_load_std = np.asarray(critical_load.std, dtype=float)
    if policy_active and np.any(in_support):
        critical_load_cap = float(
            np.percentile(critical_load_std[in_support], cap_percentile)
        )
        critical_load_std = np.where(
            in_support,
            critical_load_std,
            np.minimum(critical_load_std, critical_load_cap),
        )
    annotated["critical_axial_load_std"] = critical_load_std
    annotated["critical_axial_load_ucb"] = (
        critical_load.mean + kappa_m * critical_load_std
    )
    annotated["predicted_initial_stiffness_N_per_mm_per_needle"] = stiffness.mean
    annotated["initial_stiffness_std"] = stiffness.std
    annotated["intact_patch_pass_probability"] = np.clip(intact_probability, 0.0, 1.0)
    if policy_active:
        preparation_probability = models.preparation.predict_proba(x)
        annotated["preparation_feasibility_probability"] = np.clip(
            preparation_probability,
            0.0,
            1.0,
        )

    reports = [
        constraint_report(
            row,
            registry,
            optimization_config,
            intact_failure_probability=1.0 - float(row["intact_patch_pass_probability"]),
        )
        for _, row in annotated.iterrows()
    ]
    report_frame = pd.DataFrame(reports)
    for column in report_frame.columns:
        annotated[column] = report_frame[column].to_numpy()

    # Screening is driven by normalized viability UCB. Chemistry/support
    # deductions and the conservative exact-combination intact deduction can
    # change candidate ranking without changing the viability GP or the
    # diagnostic additive intact classifier.
    support_penalty = np.zeros(len(annotated), dtype=float)
    if policy_active and "support_status" in annotated.columns:
        penalty_value = float(
            nested_get(
                optimization_config,
                "support_policy.out_of_support_score_penalty",
                0.20,
            )
        )
        support_penalty = np.where(
            annotated["support_status"].astype(str).eq("boundary"),
            penalty_value,
            0.0,
        )
    annotated["screening_phase_score"] = (
        minmax(annotated["viability_ucb"].to_numpy(dtype=float))
        - annotated["screening_acquisition_penalty"].to_numpy(dtype=float)
        - support_penalty
        - pd.to_numeric(
            annotated.get(
                "intact_combination_screening_penalty",
                pd.Series(0.0, index=annotated.index),
            ),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
    )
    if "recommendation_type" not in annotated.columns:
        annotated["recommendation_type"] = ""
    if "selection_explanation" not in annotated.columns:
        annotated["selection_explanation"] = ""
    return annotated


def _mechanics_phase_scores(
    annotated: pd.DataFrame,
    models: EndpointModels,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    intact_policy: IntactCombinationPolicy | None = None,
) -> tuple[np.ndarray, dict]:
    train_frame = paired_objective_frame(models.training_frame)
    for objective_column in ["viability_percent", "critical_axial_load_N_per_needle"]:
        if objective_column not in train_frame.columns:
            train_frame[objective_column] = np.nan
    paired = train_frame[
        train_frame[["viability_percent", "critical_axial_load_N_per_needle"]].notna().all(axis=1)
    ].copy()
    train_x = (
        _feature_matrix(paired, registry.feature_names)
        if not paired.empty
        else np.empty((0, len(registry.feature_names)))
    )
    train_y = (
        paired[["viability_percent", "critical_axial_load_N_per_needle"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
        if not paired.empty
        else np.empty((0, 2))
    )
    candidate_x = _feature_matrix(annotated, registry.feature_names)
    ref_cfg = nested_get(optimization_config, "selection.reference_point", {})
    reference_point = (
        float(ref_cfg.get("viability_percent", 0.0)),
        float(ref_cfg.get("critical_axial_load_N_per_needle", 0.0)),
    )
    acquisition, botorch_metadata = try_botorch_qlognehvi_scores(
        train_x=train_x,
        train_y=train_y,
        candidate_x=candidate_x,
        reference_point=reference_point,
    )
    empirical_mode = bool(
        intact_policy is not None
        and intact_policy.active
        and intact_policy.mechanics_mode == "empirical_feasibility_weighted"
    )
    empirical_probability = np.clip(
        pd.to_numeric(
            annotated.get(
                "empirical_combination_pass_probability",
                pd.Series(0.5, index=annotated.index),
            ),
            errors="coerce",
        ).fillna(0.5).to_numpy(dtype=float),
        0.0,
        1.0,
    )
    if acquisition is None:
        mode = "qlognehvi_proxy"
        acquisition = qlognehvi_proxy_scores(
            annotated,
            annotated["viability_ucb"].to_numpy(dtype=float),
            annotated["critical_axial_load_ucb"].to_numpy(dtype=float),
            reference_point=(0.0, 0.0),
            feasibility_probability=(
                empirical_probability if empirical_mode else None
            ),
        )
    else:
        mode = "qlognehvi_botorch"
        if empirical_mode:
            probability_floor = intact_policy.numerical_probability_floor
            acquisition = acquisition + np.log(
                np.maximum(empirical_probability, probability_floor)
            )
    if empirical_mode:
        # The objective acquisition is feasibility-weighted above. Retain the
        # pre-existing non-intact chemistry pressure, but never subtract the
        # additive classifier-derived intact term in this mode.
        score = acquisition - annotated[
            "screening_acquisition_penalty"
        ].to_numpy(dtype=float)
        classifier_role = "diagnostic_only"
    else:
        score = acquisition - annotated["acquisition_penalty"].to_numpy(dtype=float)
        classifier_role = "active_compatibility_mode"
    metadata = {
        "pool_selection_mode": mode,
        "botorch_available": bool(botorch_available()),
        "botorch_metadata": botorch_metadata,
        "intact_feasibility_mode": (
            "empirical_combination_probability_weighting"
            if empirical_mode
            else "classifier_threshold_penalty"
        ),
        "classifier_probability_selection_role": classifier_role,
        "empirical_probability_minimum": float(
            np.min(empirical_probability) if len(empirical_probability) else np.nan
        ),
        "empirical_probability_maximum": float(
            np.max(empirical_probability) if len(empirical_probability) else np.nan
        ),
    }
    return np.asarray(score, dtype=float), metadata


def _candidate_masks(
    candidate_pool: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
) -> list[tuple[int, ...]]:
    max_size = int(
        nested_get(
            optimization_config,
            "continuous_qlognehvi.max_sparse_mask_size",
            4,
        )
    )
    max_masks = int(
        nested_get(optimization_config, "continuous_qlognehvi.max_masks", 16)
    )
    masks: set[tuple[int, ...]] = set()
    for _, row in candidate_pool.iterrows():
        active = tuple(
            index
            for index, feature in enumerate(registry.feature_names)
            if abs(float(pd.to_numeric(row.get(feature, 0.0), errors="coerce") or 0.0))
            >= presence_threshold(feature)
        )
        if 0 < len(active) <= max_size:
            masks.add(active)
    return sorted(masks, key=lambda mask: (len(mask), mask))[:max_masks]


def _continuous_mechanics_candidates(
    candidate_pool: pd.DataFrame,
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    models: EndpointModels,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    policy_active: bool,
    policy_version: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    metadata: dict = {
        "continuous_optimizer_enabled": False,
        "continuous_optimizer_used": False,
        "continuous_optimizer_fallback": True,
    }
    if not policy_active or not bool(
        nested_get(optimization_config, "continuous_qlognehvi.enabled", True)
    ):
        metadata["continuous_optimizer_reason"] = "policy inactive or optimizer disabled"
        return candidate_pool.head(0).copy(), metadata

    train_frame = paired_objective_frame(models.training_frame)
    paired = train_frame[
        train_frame.get("viability_percent", pd.Series(index=train_frame.index, dtype=float)).notna()
        & train_frame.get(
            "critical_axial_load_N_per_needle",
            pd.Series(index=train_frame.index, dtype=float),
        ).notna()
    ].copy()
    if len(paired) < 2:
        metadata["continuous_optimizer_reason"] = "insufficient paired objective rows"
        return candidate_pool.head(0).copy(), metadata

    ingredients = registry.active_ingredients()
    lower = np.array([ingredient.lower_bound for ingredient in ingredients], dtype=float)
    upper = np.array(
        [
            ingredient_upper_bound_for_policy(
                ingredient,
                optimization_config,
                policy_version,
            )
            for ingredient in ingredients
        ],
        dtype=float,
    )
    masks = _candidate_masks(candidate_pool, registry, optimization_config)
    support = build_support_context(
        formulations,
        registry,
        optimization_config,
        observations,
    )
    preparation_threshold = float(
        nested_get(optimization_config, "preparation_model.probability_threshold", 0.50)
    )

    def feasible(vector: np.ndarray) -> bool:
        row = dict(zip(registry.feature_names, vector))
        report = feasibility_report(
            row,
            registry,
            optimization_config,
            policy_active=True,
            policy_version=policy_version,
        )
        if not bool(report["feasibility_pass"]):
            return False
        if models.preparation.fitted:
            probability = float(models.preparation.predict_proba(vector.reshape(1, -1))[0])
            if probability < preparation_threshold:
                return False
        return True

    ref_cfg = nested_get(optimization_config, "selection.reference_point", {})
    reference_point = (
        float(ref_cfg.get("viability_percent", 0.0)),
        float(ref_cfg.get("critical_axial_load_N_per_needle", 0.0)),
    )
    target = int(
        nested_get(
            optimization_config,
            "continuous_qlognehvi.generated_candidate_target",
            24,
        )
    )
    optimized, botorch_metadata = try_botorch_optimize_qlognehvi(
        train_x=_feature_matrix(paired, registry.feature_names),
        train_y=paired[
            ["viability_percent", "critical_axial_load_N_per_needle"]
        ].to_numpy(dtype=float),
        lower_bounds=lower,
        upper_bounds=upper,
        active_masks=masks,
        reference_point=reference_point,
        n_candidates=target,
        feasibility_callback=feasible,
        random_seed=int(optimization_config.get("random_seed", 42)),
    )
    metadata.update(botorch_metadata)
    metadata["continuous_optimizer_enabled"] = True
    if optimized is None or len(optimized) == 0:
        metadata["continuous_optimizer_reason"] = botorch_metadata.get(
            "botorch_error",
            "continuous optimization failed",
        )
        return candidate_pool.head(0).copy(), metadata

    rows = []
    for index, vector in enumerate(optimized):
        row = dict(zip(registry.feature_names, vector))
        row.update(
            {
                "candidate_id": f"qlognehvi_{index + 1:04d}",
                "formulation_id": stable_formulation_id(row, registry),
                "active_ingredient_count": count_active_ingredients(row, registry),
                "candidate_origin": "continuous_qlognehvi",
            }
        )
        rows.append(row)
    generated = pd.DataFrame(rows).drop_duplicates("formulation_id", keep="first")
    generated = annotate_feasibility(
        generated,
        registry,
        optimization_config,
        policy_active=True,
        policy_version=policy_version,
    )
    generated = annotate_support(generated, registry, support)
    generated = generated.loc[generated["feasibility_pass"].astype(bool)].reset_index(drop=True)
    metadata["continuous_optimizer_used"] = not generated.empty
    metadata["continuous_optimizer_fallback"] = generated.empty
    return generated, metadata


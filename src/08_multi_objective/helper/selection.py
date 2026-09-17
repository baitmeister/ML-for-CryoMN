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
from .models import EndpointModels, train_endpoint_models
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

from .selection_constraints import (
    _active_ingredient_set,
    _cold_start_ordinary_counts,
    _cold_start_trial_passes_shared_constraints,
    _combination_cap_for_size,
    _drop_zero_active_candidates,
    _enforce_cold_start_policy,
    _enforce_ingredient_combination_cap,
    _enforce_ingredient_frequency_cap,
    _enforce_shared_ingredient_pair_cap,
    _exact_combination_caps_pass,
    _ingredient_appearance_counts,
    _is_protected_diversity_row,
    _shared_pair_counts,
    _support_boundary_count,
)
from . import selection_scoring as _selection_scoring
from .selection_scoring import (
    _annotate_mechanical_history,
    _candidate_masks,
    _continuous_mechanics_candidates,
    _feature_matrix,
    _mechanical_history_counts,
    _mechanics_phase_scores as _mechanics_phase_scores_impl,
    _registry_scaled_feature_matrix,
    _scaled_matrix,
    annotate_candidates,
)


def _mechanics_phase_scores(*args, **kwargs):
    """Compatibility bridge for callers that patch the former module global."""

    original = _selection_scoring.try_botorch_qlognehvi_scores
    _selection_scoring.try_botorch_qlognehvi_scores = try_botorch_qlognehvi_scores
    try:
        return _mechanics_phase_scores_impl(*args, **kwargs)
    finally:
        _selection_scoring.try_botorch_qlognehvi_scores = original



from . import selection_policy as _selection_policy
from .selection_policy import (
    _allocate_screening_origin_quota,
    _batch_sort_key,
    _bootstrap_mechanical_order,
    _greedy_diverse_pick,
    _hybrid_mechanical_order,
    _kcenter_pick,
    _mechanical_eligibility_mask,
    _select_bootstrap_anchor,
    _select_round_slate,
    select_mechanical_tests as _select_mechanical_tests_impl,
)
from .selection_reporting import (
    _format_candidate_line,
    _write_summary,
    write_selection_result as _write_selection_result_impl,
)

@dataclass(frozen=True)
class SelectionResult:
    viability_screen: pd.DataFrame
    mechanical_tests: pd.DataFrame
    candidate_pool: pd.DataFrame
    metadata: dict


def select_mechanical_tests(*args, **kwargs):
    """Compatibility bridge for callers patching the former module internals."""

    original = _selection_policy._mechanics_phase_scores
    _selection_policy._mechanics_phase_scores = _mechanics_phase_scores
    try:
        return _select_mechanical_tests_impl(*args, **kwargs)
    finally:
        _selection_policy._mechanics_phase_scores = original


def _prepare_and_score_candidate_pool(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    models: EndpointModels,
    phase_resolution: PhaseResolution,
    similarity_policy,
    similarity_audit: SimilarityAudit,
    intact_combination_policy: IntactCombinationPolicy,
    intact_evidence,
    cold_start_context: ColdStartContext,
    target_round_number: int | None,
    policy_active: bool,
    policy_version: str,
    unavailable_feature_names: list[str] | tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    """Prepare evidence, annotate predictions, and apply phase-appropriate scores."""

    continuous_metadata = {
        "continuous_optimizer_enabled": False,
        "continuous_optimizer_used": False,
        "continuous_optimizer_fallback": True,
        "continuous_optimizer_reason": (
            f"continuous optimization is disabled during "
            f"{phase_resolution.active_phase}"
        ),
    }
    if phase_resolution.active_phase in {PHASE_HYBRID, PHASE_MECHANICS}:
        if (
            phase_resolution.active_phase == PHASE_MECHANICS
            and "candidate_origin" in candidate_pool.columns
        ):
            preserved_origin_mask = candidate_pool["candidate_origin"].astype(str).isin(
                ["boundary_probe", "rescue_dilution"]
            )
            candidate_pool.loc[~preserved_origin_mask, "candidate_origin"] = (
                "finite_pool_fallback"
            )
        continuous_candidates, continuous_metadata = _continuous_mechanics_candidates(
            candidate_pool,
            formulations,
            observations,
            models,
            registry,
            optimization_config,
            policy_active=policy_active,
            policy_version=policy_version,
        )
        if not continuous_candidates.empty:
            candidate_pool = pd.concat(
                [continuous_candidates, candidate_pool],
                ignore_index=True,
                sort=False,
            ).drop_duplicates("formulation_id", keep="first")
    if similarity_policy.active:
        similarity_index = build_history_similarity_index(
            formulations,
            observations,
            registry,
            similarity_policy,
        )
        if similarity_audit.history_reference_count == 0:
            similarity_audit.history_reference_count = len(similarity_index)
        candidate_pool, _ = filter_frame_by_similarity(
            similarity_priority_order(candidate_pool),
            similarity_index,
            similarity_audit,
            accepted_reference_kind="generated_pool",
        )
        continuous_survivors = int(
            (
                candidate_pool.get(
                    "candidate_origin",
                    pd.Series("", index=candidate_pool.index),
                ).astype(str)
                == "continuous_qlognehvi"
            ).sum()
        )
        if bool(continuous_metadata.get("continuous_optimizer_used", False)):
            continuous_metadata["similarity_survivor_count"] = continuous_survivors
            if continuous_survivors == 0:
                continuous_metadata["continuous_optimizer_used"] = False
                continuous_metadata["continuous_optimizer_fallback"] = True
                continuous_metadata["continuous_optimizer_reason"] = (
                    "all continuous candidates were rejected by the active "
                    "formulation-similarity policy"
                )
    retest_candidates = build_retest_candidates(
        formulations,
        observations,
        models,
        registry,
        optimization_config,
    )
    retest_policy_metadata = {
        "policy_version": str(
            nested_get(
                optimization_config,
                "retest.policy_version",
                "campaign_observed_instability_v2",
            )
        ),
        "eligible_source_types": ["wetlab_feedback"],
        "max_candidates_per_round": int(
            nested_get(optimization_config, "retest.max_candidates_per_round", 2)
        ),
        "formulation_disagreement_threshold_percent": float(
            nested_get(
                optimization_config,
                "retest.formulation_disagreement_threshold_percent",
                15.0,
            )
        ),
        "within_batch_std_threshold_percent": float(
            nested_get(
                optimization_config,
                "retest.within_batch_std_threshold_percent",
                8.0,
            )
        ),
        "within_batch_min_replicates": int(
            nested_get(
                optimization_config,
                "retest.within_batch_min_replicates",
                3,
            )
        ),
        "local_residual_threshold_percent": float(
            nested_get(
                optimization_config,
                "retest.local_residual_threshold_percent",
                20.0,
            )
        ),
        "one_time_anomaly_confirmation": bool(
            nested_get(
                optimization_config,
                "retest.one_time_anomaly_confirmation",
                True,
            )
        ),
        "eligibility_uses_model_uncertainty": False,
        "eligible_before_feasibility": int(len(retest_candidates)),
    }
    retest_audit_rows: list[dict] = []
    retest_candidates_rejected_by_feasibility = 0
    if policy_active and not retest_candidates.empty:
        support = build_support_context(
            formulations,
            registry,
            optimization_config,
            observations,
        )
        retest_candidates = annotate_feasibility(
            retest_candidates,
            registry,
            optimization_config,
            policy_active=True,
            policy_version=policy_version,
        )
        retest_candidates = annotate_support(retest_candidates, registry, support)
        retest_candidates_rejected_by_feasibility = int(
            (~retest_candidates["feasibility_pass"].astype(bool)).sum()
        )
        for _, row in retest_candidates.iterrows():
            retest_audit_rows.append(
                {
                    "formulation_id": str(row.get("formulation_id", "")),
                    "feedback_batch_count": int(
                        pd.to_numeric(
                            row.get("feedback_batch_count", 0),
                            errors="coerce",
                        )
                        or 0
                    ),
                    "batch_mean_range_percent": float(
                        pd.to_numeric(
                            row.get("same_formulation_range", 0.0),
                            errors="coerce",
                        )
                        or 0.0
                    ),
                    "latest_batch_replicate_count": int(
                        pd.to_numeric(
                            row.get("viability_replicate_count", 0),
                            errors="coerce",
                        )
                        or 0
                    ),
                    "latest_batch_replicate_sd_percent": float(
                        pd.to_numeric(
                            row.get("viability_replicate_sd", 0.0),
                            errors="coerce",
                        )
                        or 0.0
                    ),
                    "nearest_neighbor_residual_percent": float(
                        pd.to_numeric(
                            row.get("local_neighbor_residual", 0.0),
                            errors="coerce",
                        )
                        or 0.0
                    ),
                    "nearest_neighbor_bounds_normalized_distance": (
                        None
                        if pd.isna(row.get("_retest_nearest_neighbor_distance"))
                        else float(row["_retest_nearest_neighbor_distance"])
                    ),
                    "eligibility_reason": str(
                        row.get("_retest_eligibility_reason", "")
                    ),
                    "observed_evidence_severity": float(
                        pd.to_numeric(
                            row.get("retest_priority_score", 0.0),
                            errors="coerce",
                        )
                        or 0.0
                    ),
                    "model_uncertainty_tiebreak": float(
                        pd.to_numeric(
                            row.get("viability_std", 0.0),
                            errors="coerce",
                        )
                        or 0.0
                    ),
                    "feasibility_pass": bool(row.get("feasibility_pass", False)),
                }
            )
        retest_candidates = retest_candidates.loc[
            retest_candidates["feasibility_pass"].astype(bool)
        ].reset_index(drop=True)
        retest_candidates["candidate_origin"] = "retest"
    elif not retest_candidates.empty:
        for _, row in retest_candidates.iterrows():
            retest_audit_rows.append(
                {
                    "formulation_id": str(row.get("formulation_id", "")),
                    "feedback_batch_count": int(row.get("feedback_batch_count", 0)),
                    "batch_mean_range_percent": float(
                        row.get("same_formulation_range", 0.0)
                    ),
                    "latest_batch_replicate_count": int(
                        row.get("viability_replicate_count", 0)
                    ),
                    "latest_batch_replicate_sd_percent": float(
                        row.get("viability_replicate_sd", 0.0)
                    ),
                    "nearest_neighbor_residual_percent": float(
                        row.get("local_neighbor_residual", 0.0)
                    ),
                    "eligibility_reason": str(
                        row.get("_retest_eligibility_reason", "")
                    ),
                    "feasibility_pass": None,
                }
            )
    retest_policy_metadata["rejected_by_feasibility"] = (
        retest_candidates_rejected_by_feasibility
    )
    retest_policy_metadata["eligible_after_feasibility"] = int(
        len(retest_candidates)
    )
    retest_policy_metadata["eligible_candidates"] = retest_audit_rows

    anchor_candidates, anchor_metadata = _select_bootstrap_anchor(
        formulations,
        observations,
        registry,
        optimization_config,
        phase_resolution,
        policy_active=policy_active,
        policy_version=policy_version,
        unavailable_feature_names=unavailable_feature_names,
    )
    if not anchor_candidates.empty and policy_active:
        anchor_candidates = annotate_support(
            anchor_candidates,
            registry,
            build_support_context(
                formulations,
                registry,
                optimization_config,
                observations,
            ),
        )

    # Retest-only audit fields belong in metadata and explanations, not in
    # the stable operator worksheet schema.
    retained_retest_columns = set(candidate_pool.columns) | {
        "candidate_id",
        "formulation_id",
        "recommendation_type",
        "selection_explanation",
        "active_ingredient_count",
        "candidate_origin",
        "same_formulation_range",
        "local_neighbor_residual",
        "retest_priority_score",
        *registry.feature_names,
    }
    retest_candidates = retest_candidates[
        [
            column
            for column in retest_candidates.columns
            if column in retained_retest_columns
        ]
    ].copy()
    combined_pool = candidate_pool.copy()
    if not retest_candidates.empty:
        combined_pool = pd.concat([combined_pool, retest_candidates], ignore_index=True, sort=False)
        combined_pool = combined_pool.drop_duplicates("candidate_id", keep="first")
    if not anchor_candidates.empty:
        combined_pool = pd.concat(
            [anchor_candidates, combined_pool], ignore_index=True, sort=False
        ).drop_duplicates("candidate_id", keep="first")
    combined_pool, zero_active_filtered_count = _drop_zero_active_candidates(combined_pool, registry)
    combined_pool = annotate_intact_combination_evidence(
        combined_pool,
        intact_evidence,
        registry,
        intact_combination_policy,
    )
    combined_pool = annotate_cold_start_candidates(
        combined_pool,
        registry,
        cold_start_context,
    )
    combined_pool["mechanical_feasibility_weight"] = pd.to_numeric(
        combined_pool["empirical_combination_pass_probability"],
        errors="coerce",
    ).fillna(0.50)
    annotated = annotate_candidates(
        combined_pool,
        models,
        registry,
        optimization_config,
        policy_active=policy_active,
    )
    annotated = _annotate_mechanical_history(annotated, observations)
    annotated = annotate_viability_prediction_labels(
        annotated,
        observations,
        models,
        cold_start_context,
        optimization_config,
        target_round_number=target_round_number,
    )
    if policy_active and models.preparation.fitted:
        preparation_threshold = float(
            nested_get(
                optimization_config,
                "preparation_model.probability_threshold",
                0.50,
            )
        )
        annotated = annotated.loc[
            annotated["preparation_feasibility_probability"] >= preparation_threshold
        ].reset_index(drop=True)
    pool_selection_metadata = {"pool_selection_mode": "screening_phase"}
    if phase_resolution.active_phase in {PHASE_HYBRID, PHASE_MECHANICS}:
        mechanics_scores, pool_selection_metadata = _mechanics_phase_scores(
            annotated,
            models,
            registry,
            optimization_config,
            intact_policy=intact_combination_policy,
        )
        annotated["mechanics_phase_score"] = mechanics_scores
        if phase_resolution.active_phase == PHASE_HYBRID:
            screening_weight = float(
                nested_get(
                    optimization_config,
                    "mechanics_transition.hybrid.screening_slate_weight",
                    0.50,
                )
            )
            mechanics_weight = float(
                nested_get(
                    optimization_config,
                    "mechanics_transition.hybrid.mechanics_slate_weight",
                    0.50,
                )
            )
            annotated["hybrid_phase_score"] = (
                screening_weight
                * minmax(annotated["screening_phase_score"].to_numpy(dtype=float))
                + mechanics_weight
                * minmax(annotated["mechanics_phase_score"].to_numpy(dtype=float))
            )
            pool_selection_metadata = {
                **pool_selection_metadata,
                "pool_selection_mode": "hybrid_screening_mechanics",
                "screening_slate_weight": screening_weight,
                "mechanics_slate_weight": mechanics_weight,
            }
        else:
            annotated["hybrid_phase_score"] = np.nan
    else:
        annotated["mechanics_phase_score"] = np.nan
        annotated["hybrid_phase_score"] = np.nan
    return annotated, {
        "continuous_metadata": continuous_metadata,
        "retest_candidates_rejected_by_feasibility": retest_candidates_rejected_by_feasibility,
        "retest_policy_metadata": retest_policy_metadata,
        "retest_audit_rows": retest_audit_rows,
        "anchor_metadata": anchor_metadata,
        "zero_active_filtered_count": zero_active_filtered_count,
        "pool_selection_metadata": pool_selection_metadata,
    }


def _assemble_selection_metadata(
    annotated: pd.DataFrame,
    viability_screen: pd.DataFrame,
    mechanical_tests: pd.DataFrame,
    mechanical_metadata: dict,
    phase_resolution: PhaseResolution,
    anchor_metadata: dict,
    pool_selection_metadata: dict,
    continuous_metadata: dict,
    retest_candidates_rejected_by_feasibility: int,
    retest_policy_metadata: dict,
    models: EndpointModels,
    optimization_config: Mapping,
    target_round_number: int | None,
    similarity_audit: SimilarityAudit,
    similarity_validation: dict,
    intact_combination_policy: IntactCombinationPolicy,
    intact_evidence,
    cold_start_context: ColdStartContext,
    selected_pair_counts: dict,
    shared_pair_cap: int,
    ingredient_frequency_cap: int,
    ingredient_frequency_start_round: int,
    ingredient_frequency_active: bool,
    ingredient_frequency_metadata: dict,
    final_ingredient_counts: dict,
    zero_active_filtered_count: int,
) -> dict:
    """Assemble the auditable protocol record without changing the selection."""

    surrogate_config = (
        nested_get(optimization_config, "surrogate_model", {}) or {}
    )
    prediction_labeling_config = (
        nested_get(optimization_config, "prediction_labeling", {}) or {}
    )

    def uncertainty_summary(column: str) -> dict[str, float | None]:
        values = pd.to_numeric(
            annotated.get(column, pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        if values.empty:
            return {"minimum": None, "median": None, "maximum": None}
        return {
            "minimum": float(values.min()),
            "median": float(values.median()),
            "maximum": float(values.max()),
        }

    if phase_resolution.active_phase == PHASE_SCREENING:
        optimizer_mode = (
            "support_aware_finite_pool_screening"
            if policy_active
            else "legacy_uniform_finite_pool_screening"
        )
        optimizer_fallback_status = "not_applicable"
    elif phase_resolution.active_phase == PHASE_BOOTSTRAP:
        optimizer_mode = "bootstrap_utility_diversity"
        optimizer_fallback_status = "not_applicable"
    elif continuous_metadata.get("continuous_optimizer_used", False):
        optimizer_mode = (
            "hybrid_continuous_qlognehvi"
            if phase_resolution.active_phase == PHASE_HYBRID
            else "continuous_qlognehvi"
        )
        optimizer_fallback_status = "not_used"
    else:
        optimizer_mode = (
            "hybrid_finite_pool_fallback"
            if phase_resolution.active_phase == PHASE_HYBRID
            else "finite_pool_fallback"
        )
        optimizer_fallback_status = "used"
    metadata = {
        "viability_screen_count": int(len(viability_screen)),
        "mechanical_test_count": int(
            mechanical_tests.get(
                "mechanical_primary_recommended",
                pd.Series(False, index=mechanical_tests.index),
            ).astype(bool).sum()
        ),
        "mechanical_policy": mechanical_metadata,
        "objective_endpoints": ["viability_percent", "critical_axial_load_N_per_needle"],
        "secondary_endpoint": "initial_stiffness_N_per_mm_per_needle",
        "screening_gate": "intact_patch_formation_pass",
        "active_phase": phase_resolution.active_phase,
        "phase_resolution": phase_resolution.to_metadata(),
        "mechanics_transition": {
            "policy_version": phase_resolution.transition_policy_version,
            "anchor_selection": anchor_metadata,
        },
        "pool_selection_policy": pool_selection_metadata,
        "continuous_qlognehvi": continuous_metadata,
        "optimizer_mode": optimizer_mode,
        "optimizer_fallback_status": optimizer_fallback_status,
        "retest_candidate_count": int((annotated["recommendation_type"] == "retest_priority").sum()),
        "retest_candidate_count_rejected_by_feasibility": retest_candidates_rejected_by_feasibility,
        "retest_policy": retest_policy_metadata,
        "surrogate_uncertainty": {
            "policy_version": str(
                surrogate_config.get(
                    "policy_version",
                    "round3_surrogate_uncertainty_v2",
                )
            ),
            "start_round": int(surrogate_config.get("start_round", 3)),
            "active": bool(
                target_round_number is not None
                and int(target_round_number)
                >= int(surrogate_config.get("start_round", 3))
            ),
            "regression_kernel": str(
                surrogate_config.get(
                    "regression_kernel",
                    "matern_2p5_explicit_observation_noise",
                )
            ),
            "observation_noise_mechanism": "per_observation_alpha",
            "candidate_pool_viability_std": uncertainty_summary(
                "raw_surrogate_viability_std"
            ),
            "candidate_pool_critical_load_std": uncertainty_summary(
                "critical_axial_load_std"
            ),
            "candidate_pool_initial_stiffness_std": uncertainty_summary(
                "initial_stiffness_std"
            ),
        },
        "viability_prediction_labeling": {
            "policy_version": str(
                prediction_labeling_config.get(
                    "policy_version",
                    "round7_viability_prediction_labeling_v1",
                )
            ),
            "start_round": int(prediction_labeling_config.get("start_round", 7)),
            "active": bool(
                prediction_labeling_config.get("enabled", True)
                and target_round_number is not None
                and int(target_round_number)
                >= int(prediction_labeling_config.get("start_round", 7))
            ),
            "public_unknown_rule": (
                "unobserved cold-start, support boundary, or surrogate "
                "prior reversion"
            ),
            "raw_surrogate_retained_for_acquisition_and_evaluation": True,
            "selected_status_counts": {
                str(status): int(count)
                for status, count in viability_screen[
                    "viability_prediction_status"
                ].value_counts(dropna=False).items()
            },
            "candidate_pool_status_counts": {
                str(status): int(count)
                for status, count in annotated[
                    "viability_prediction_status"
                ].value_counts(dropna=False).items()
            },
        },
        "shared_ingredient_pair_diversity": {
            "definition": (
                "unordered active registry pair above practical presence "
                "thresholds; additional ingredients do not change membership"
            ),
            "max_non_retest_rows_per_pair": shared_pair_cap,
            "retests_exempt": True,
            "rescue_rows_included": True,
            "replacement_count": int(
                viability_screen.attrs.get(
                    "shared_pair_replacement_count",
                    0,
                )
            ),
            "selected_pair_counts": {
                " + ".join(pair): int(count)
                for pair, count in sorted(selected_pair_counts.items())
            },
            "maximum_pair_multiplicity": int(
                max(selected_pair_counts.values(), default=0)
            ),
        },
        "ingredient_frequency_diversity": {
            "policy_version": str(
                nested_get(
                    optimization_config,
                    "selection.ingredient_frequency_policy_version",
                    "round3_marginal_ingredient_diversity_v1",
                )
            ),
            "start_round": ingredient_frequency_start_round,
            "active": ingredient_frequency_active,
            "max_rows_per_ingredient": ingredient_frequency_cap,
            "presence_rule": "registry practical presence thresholds",
            "special_rows_count_toward_cap": True,
            "retests_protected_from_removal": True,
            "rescue_rows_protected_from_removal": True,
            "same_origin_replacement_required": True,
            **ingredient_frequency_metadata,
            "counts_after": {
                feature_name: int(count)
                for feature_name, count in final_ingredient_counts.items()
            },
            "maximum_ingredient_frequency": int(
                max(final_ingredient_counts.values(), default=0)
            ),
        },
        "boundary_slate_cap": {
            "max_support_boundary_rows": int(
                nested_get(
                    optimization_config,
                    "support_policy.max_boundary_candidates_per_slate",
                    1,
                )
            ),
            "selected_support_boundary_rows": int(
                (
                    viability_screen.get(
                        "support_status",
                        pd.Series("", index=viability_screen.index),
                    ).astype(str)
                    == "boundary"
                ).sum()
            ),
            "selected_boundary_probe_origin_rows": int(
                (
                    viability_screen.get(
                        "candidate_origin",
                        pd.Series("", index=viability_screen.index),
                    ).astype(str)
                    == "boundary_probe"
                ).sum()
            ),
            "clarification": (
                "max_boundary_candidates_per_slate caps support_status=boundary "
                "rows, not candidate_origin=boundary_probe rows"
            ),
        },
        "zero_active_candidate_count_filtered": zero_active_filtered_count,
        "target_round_number": target_round_number,
        "preparation_model_fitted": bool(models.preparation.fitted),
        "preparation_observation_count": models.preparation_observation_count,
        "formulation_similarity": {
            **similarity_audit.to_metadata(),
            "final_validation": similarity_validation,
        },
        "intact_combination_policy": intact_policy_metadata(
            intact_combination_policy,
            intact_evidence,
        ),
        "cold_start_policy": {
            **cold_start_policy_metadata(cold_start_context),
            **dict(
                viability_screen.attrs.get("cold_start_policy", {})
            ),
        },
    }
    return metadata


def select_next_round(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    requested_phase_mode: str | None = None,
    target_round_number: int | None = None,
    policy_active: bool = False,
    policy_version: str | None = None,
    similarity_audit: SimilarityAudit | None = None,
    unavailable_feature_names: list[str] | tuple[str, ...] = (),
) -> SelectionResult:
    if policy_version is None:
        policy_version = policy_activation(
            optimization_config,
            target_round_number,
        )[1]
    models = train_endpoint_models(
        formulations,
        observations,
        registry,
        optimization_config=dict(optimization_config),
    )
    intact_combination_policy = resolve_intact_combination_policy(
        optimization_config,
        target_round_number,
    )
    intact_evidence = build_intact_evidence(
        formulations,
        observations,
        registry,
        intact_combination_policy,
        target_round_number,
    )
    cold_start_policy = resolve_cold_start_policy(
        optimization_config,
        target_round_number,
    )
    cold_start_context = build_cold_start_context(
        formulations,
        observations,
        registry,
        cold_start_policy,
        target_round_number,
        unavailable_feature_names=unavailable_feature_names,
    )
    phase_resolution = resolve_phase_mode(
        formulations,
        observations,
        registry,
        optimization_config,
        requested_phase_mode=requested_phase_mode,
        target_round_number=target_round_number,
    )
    similarity_policy = resolve_similarity_policy(
        optimization_config,
        target_round_number,
    )
    if similarity_audit is None:
        similarity_audit = SimilarityAudit(similarity_policy)
    annotated, preparation = _prepare_and_score_candidate_pool(
        formulations=formulations,
        observations=observations,
        candidate_pool=candidate_pool,
        registry=registry,
        optimization_config=optimization_config,
        models=models,
        phase_resolution=phase_resolution,
        similarity_policy=similarity_policy,
        similarity_audit=similarity_audit,
        intact_combination_policy=intact_combination_policy,
        intact_evidence=intact_evidence,
        cold_start_context=cold_start_context,
        target_round_number=target_round_number,
        policy_active=policy_active,
        policy_version=policy_version,
        unavailable_feature_names=unavailable_feature_names,
    )
    continuous_metadata = preparation["continuous_metadata"]
    retest_candidates_rejected_by_feasibility = preparation[
        "retest_candidates_rejected_by_feasibility"
    ]
    retest_policy_metadata = preparation["retest_policy_metadata"]
    retest_audit_rows = preparation["retest_audit_rows"]
    anchor_metadata = preparation["anchor_metadata"]
    zero_active_filtered_count = preparation["zero_active_filtered_count"]
    pool_selection_metadata = preparation["pool_selection_metadata"]

    n_viability = int(nested_get(optimization_config, "round_policy.viability_screens_per_round", 12))
    n_mechanical = int(
        nested_get(
            optimization_config,
            "mechanics_transition.bootstrap.mechanical_capacity",
            4,
        )
    )
    if len(annotated) < n_viability:
        raise ValueError(
            "Candidate pool contains fewer rows than the required viability slate "
            f"after active filters: {len(annotated)}/{n_viability}."
        )

    viability_screen = _select_round_slate(
        annotated,
        registry,
        optimization_config,
        phase_resolution,
        n=n_viability,
        policy_active=policy_active,
        target_round_number=target_round_number,
        cold_start_context=cold_start_context,
    )
    mechanical_tests, mechanical_metadata = select_mechanical_tests(
        viability_screen,
        models,
        registry,
        optimization_config,
        phase_resolution,
        n=n_mechanical,
        intact_policy=intact_combination_policy,
    )
    similarity_validation = validate_selected_similarity(
        viability_screen,
        formulations,
        observations,
        registry,
        similarity_policy,
    )
    selected_pair_counts = _shared_pair_counts(viability_screen, registry)
    shared_pair_cap = int(
        nested_get(
            optimization_config,
            "selection.max_candidates_per_shared_ingredient_pair",
            5,
        )
    )
    ingredient_frequency_cap = int(
        nested_get(
            optimization_config,
            "selection.max_candidates_per_ingredient",
            5,
        )
    )
    ingredient_frequency_start_round = int(
        nested_get(
            optimization_config,
            "selection.ingredient_frequency_start_round",
            3,
        )
    )
    ingredient_frequency_active = bool(
        target_round_number is not None
        and int(target_round_number) >= ingredient_frequency_start_round
    )
    ingredient_frequency_metadata = dict(
        viability_screen.attrs.get(
            "ingredient_frequency_diversity",
            {},
        )
    )
    final_ingredient_counts = _ingredient_appearance_counts(
        viability_screen,
        registry,
    )
    if ingredient_frequency_active:
        frequency_violations = {
            feature_name: count
            for feature_name, count in final_ingredient_counts.items()
            if count > ingredient_frequency_cap
        }
        if frequency_violations:
            raise ValueError(
                "Final selected slate violates the active marginal ingredient "
                f"frequency policy: {frequency_violations}."
            )
    selected_retest_ids = set(
        viability_screen.loc[
            viability_screen["recommendation_type"].astype(str)
            == "retest_priority",
            "formulation_id",
        ].astype(str)
    )
    retest_policy_metadata["selected_candidates"] = [
        row
        for row in retest_audit_rows
        if str(row.get("formulation_id", "")) in selected_retest_ids
    ]
    metadata = _assemble_selection_metadata(
        annotated=annotated,
        viability_screen=viability_screen,
        mechanical_tests=mechanical_tests,
        mechanical_metadata=mechanical_metadata,
        phase_resolution=phase_resolution,
        anchor_metadata=anchor_metadata,
        pool_selection_metadata=pool_selection_metadata,
        continuous_metadata=continuous_metadata,
        retest_candidates_rejected_by_feasibility=retest_candidates_rejected_by_feasibility,
        retest_policy_metadata=retest_policy_metadata,
        models=models,
        optimization_config=optimization_config,
        target_round_number=target_round_number,
        similarity_audit=similarity_audit,
        similarity_validation=similarity_validation,
        intact_combination_policy=intact_combination_policy,
        intact_evidence=intact_evidence,
        cold_start_context=cold_start_context,
        selected_pair_counts=selected_pair_counts,
        shared_pair_cap=shared_pair_cap,
        ingredient_frequency_cap=ingredient_frequency_cap,
        ingredient_frequency_start_round=ingredient_frequency_start_round,
        ingredient_frequency_active=ingredient_frequency_active,
        ingredient_frequency_metadata=ingredient_frequency_metadata,
        final_ingredient_counts=final_ingredient_counts,
        zero_active_filtered_count=zero_active_filtered_count,
    )
    return SelectionResult(
        viability_screen=viability_screen,
        mechanical_tests=mechanical_tests,
        candidate_pool=annotated,
        metadata=metadata,
    )


def write_selection_result(
    result: SelectionResult,
    output_dir: str | Path,
    batch_id: str = "",
    total_candidate_pool_path: str | Path | None = None,
    registry: IngredientRegistry | None = None,
) -> None:
    """Write selection artifacts through the stable V2 facade."""

    return _write_selection_result_impl(
        result=result,
        output_dir=output_dir,
        batch_id=batch_id,
        total_candidate_pool_path=total_candidate_pool_path,
        registry=registry,
    )

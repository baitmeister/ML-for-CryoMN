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
from .selection_scoring import (
    _annotate_mechanical_history,
    _candidate_masks,
    _continuous_mechanics_candidates,
    _feature_matrix,
    _mechanical_history_counts,
    _mechanics_phase_scores,
    _registry_scaled_feature_matrix,
    _scaled_matrix,
    annotate_candidates,
)

def _batch_sort_key(batch_id: str) -> tuple[int, int | str]:
    value = str(batch_id).strip()
    if value.startswith("ROUND_") and value.removeprefix("ROUND_").isdigit():
        return (1, int(value.removeprefix("ROUND_")))
    return (0, value)


def _select_bootstrap_anchor(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    phase_resolution: PhaseResolution,
    policy_active: bool,
    policy_version: str,
    unavailable_feature_names: list[str] | tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict]:
    metadata: dict = {
        "enabled": False,
        "selected": False,
        "source_batch_id": "",
        "formulation_id": "",
        "selection_score": None,
        "reason": "anchor is not scheduled for this bootstrap batch",
    }
    if phase_resolution.active_phase != PHASE_BOOTSTRAP:
        return formulations.head(0).copy(), metadata
    enabled_index = int(
        nested_get(
            optimization_config,
            "mechanics_transition.anchor.enabled_for_bootstrap_batch_index",
            2,
        )
    )
    allowed = bool(
        nested_get(
            optimization_config,
            "mechanics_transition.repeat_policy.allow_bootstrap_anchor",
            True,
        )
    )
    if not allowed or phase_resolution.bootstrap_batch_index != enabled_index:
        return formulations.head(0).copy(), metadata
    metadata["enabled"] = True
    if observations.empty:
        metadata["reason"] = "no observations are available for anchor selection"
        return formulations.head(0).copy(), metadata

    campaign = observations.copy()
    if "source_type" in campaign.columns:
        campaign = campaign[campaign["source_type"].astype(str).eq("wetlab_feedback")]
    campaign["value"] = pd.to_numeric(campaign.get("value"), errors="coerce")
    campaign = campaign[campaign["value"].notna()].copy()
    if campaign.empty:
        metadata["reason"] = "no measured campaign observations are available"
        return formulations.head(0).copy(), metadata

    grouping = ["formulation_id", "batch_id"]
    viability = (
        campaign[campaign["endpoint"].astype(str).eq("viability_percent")]
        .groupby(grouping, as_index=False)
        .agg(
            viability_percent=("value", "mean"),
            viability_replicate_sd=("value", "std"),
        )
    )
    critical = (
        campaign[
            campaign["endpoint"].astype(str).eq(
                "critical_axial_load_N_per_needle"
            )
        ]
        .groupby(grouping, as_index=False)["value"]
        .mean()
        .rename(columns={"value": "critical_axial_load_N_per_needle"})
    )
    intact = (
        campaign[
            campaign["endpoint"].astype(str).eq(
                "intact_patch_formation_pass"
            )
        ]
        .groupby(grouping, as_index=False)["value"]
        .min()
        .rename(columns={"value": "intact_patch_formation_pass"})
    )
    paired = viability.merge(critical, on=grouping, how="inner").merge(
        intact, on=grouping, how="inner"
    )
    if paired.empty:
        metadata["reason"] = "no same-batch viability/load/intact record is available"
        return formulations.head(0).copy(), metadata
    source_batch = sorted(
        critical["batch_id"].astype(str).unique(), key=_batch_sort_key
    )[-1]
    metadata["source_batch_id"] = source_batch
    paired = paired[paired["batch_id"].astype(str).eq(source_batch)].copy()
    paired = paired[paired["intact_patch_formation_pass"].eq(1.0)].copy()
    if paired.empty:
        metadata["reason"] = (
            "the immediately preceding mechanical batch has no paired "
            "actual-intact formulation"
        )
        return formulations.head(0).copy(), metadata
    maximum_exact_repeats = int(
        nested_get(
            optimization_config,
            "mechanics_transition.anchor.maximum_exact_repeats",
            1,
        )
    )
    prior_batch_counts = (
        critical[["formulation_id", "batch_id"]]
        .drop_duplicates()
        .groupby("formulation_id")["batch_id"]
        .nunique()
    )
    paired["_prior_mechanical_batch_count"] = (
        paired["formulation_id"].astype(str).map(prior_batch_counts).fillna(0)
    )
    paired = paired.loc[
        (paired["_prior_mechanical_batch_count"] - 1)
        < maximum_exact_repeats
    ].reset_index(drop=True)
    if paired.empty:
        metadata["reason"] = "all source-batch anchors reached the exact-repeat limit"
        return formulations.head(0).copy(), metadata
    paired = paired.merge(formulations, on="formulation_id", how="inner")
    if paired.empty:
        metadata["reason"] = "source-batch formulations are absent from the database"
        return formulations.head(0).copy(), metadata

    unavailable = set(unavailable_feature_names)
    if unavailable:
        available_mask = np.ones(len(paired), dtype=bool)
        for feature_name in unavailable:
            if feature_name not in paired.columns:
                continue
            values = pd.to_numeric(paired[feature_name], errors="coerce").fillna(0.0)
            available_mask &= values.abs().lt(presence_threshold(feature_name)).to_numpy()
        paired = paired.loc[available_mask].reset_index(drop=True)
    if paired.empty:
        metadata["reason"] = "all source-batch anchors use unavailable ingredients"
        return formulations.head(0).copy(), metadata

    preparation_failures = set(
        campaign.loc[
            campaign["endpoint"].astype(str).eq("preparation_feasibility_pass")
            & campaign["value"].eq(0.0),
            "formulation_id",
        ].astype(str)
    )
    if preparation_failures:
        paired = paired.loc[
            ~paired["formulation_id"].astype(str).isin(preparation_failures)
        ].reset_index(drop=True)
    if "preparation_feasibility_pass" in paired.columns:
        preparation = pd.to_numeric(
            paired["preparation_feasibility_pass"], errors="coerce"
        )
        paired = paired.loc[~preparation.eq(0.0)].reset_index(drop=True)
    if paired.empty:
        metadata["reason"] = "all source-batch anchors have preparation failures"
        return formulations.head(0).copy(), metadata

    if policy_active:
        paired = annotate_feasibility(
            paired,
            registry,
            optimization_config,
            policy_active=True,
            policy_version=policy_version,
        )
        paired = paired[paired["feasibility_pass"].astype(bool)].reset_index(drop=True)
    if paired.empty:
        metadata["reason"] = "all source-batch anchors fail active feasibility rules"
        return formulations.head(0).copy(), metadata

    viability_weight = float(
        nested_get(
            optimization_config,
            "mechanics_transition.anchor.viability_weight",
            0.50,
        )
    )
    load_weight = float(
        nested_get(
            optimization_config,
            "mechanics_transition.anchor.critical_load_weight",
            0.50,
        )
    )
    paired["_anchor_score"] = (
        viability_weight
        * minmax(paired["viability_percent"].to_numpy(dtype=float))
        + load_weight
        * minmax(
            paired["critical_axial_load_N_per_needle"].to_numpy(dtype=float)
        )
    )
    paired["viability_replicate_sd"] = pd.to_numeric(
        paired["viability_replicate_sd"], errors="coerce"
    ).fillna(float("inf"))
    winner = paired.sort_values(
        ["_anchor_score", "viability_replicate_sd", "formulation_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0].copy()
    winner["candidate_id"] = f"mechanics_anchor_{winner['formulation_id']}"
    winner["candidate_origin"] = "mechanics_anchor"
    winner["recommendation_type"] = "mechanics_anchor"
    winner["selection_explanation"] = (
        "mechanics_anchor: paired actual-intact formulation selected from the "
        "preceding mechanical batch by balanced observed viability and critical load"
    )
    winner["mechanics_anchor_source_batch"] = source_batch
    winner["mechanics_anchor_selection_score"] = float(winner["_anchor_score"])
    metadata.update(
        {
            "selected": True,
            "formulation_id": str(winner["formulation_id"]),
            "selection_score": float(winner["_anchor_score"]),
            "reason": "selected balanced paired actual-intact anchor",
        }
    )
    return pd.DataFrame([winner]).drop(columns=["_anchor_score"], errors="ignore"), metadata


def _greedy_diverse_pick(
    frame: pd.DataFrame,
    score: np.ndarray,
    feature_names: list[str],
    n: int,
    diversity_weight: float = 0.10,
    competitive_utility_band: float | None = None,
    max_boundary_candidates: int | None = None,
) -> list[int]:
    if frame.empty or n <= 0:
        return []
    n = min(n, len(frame))
    matrix = _scaled_matrix(_feature_matrix(frame, feature_names))
    selected = [int(np.nanargmax(score))]
    while len(selected) < n:
        remaining = [index for index in range(len(frame)) if index not in selected]
        if max_boundary_candidates is not None and "support_status" in frame.columns:
            selected_boundary = sum(
                str(frame.iloc[index].get("support_status", "")) == "boundary"
                for index in selected
            )
            if selected_boundary >= max_boundary_candidates:
                remaining = [
                    index
                    for index in remaining
                    if str(frame.iloc[index].get("support_status", "")) != "boundary"
                ]
        if not remaining:
            break
        if competitive_utility_band is not None:
            best_remaining = float(np.nanmax(score[remaining]))
            competitive = [
                index
                for index in remaining
                if float(score[index]) >= best_remaining - competitive_utility_band
            ]
            if competitive:
                remaining = competitive
        distances = np.linalg.norm(matrix[remaining, None, :] - matrix[selected][None, :, :], axis=2)
        min_distances = np.min(distances, axis=1)
        combined = minmax(score[remaining]) + diversity_weight * minmax(min_distances)
        next_index = remaining[int(np.nanargmax(combined))]
        selected.append(int(next_index))
    return selected


def _kcenter_pick(frame: pd.DataFrame, seed_score: np.ndarray, feature_names: list[str], n: int) -> list[int]:
    if frame.empty or n <= 0:
        return []
    n = min(n, len(frame))
    matrix = _scaled_matrix(_feature_matrix(frame, feature_names))
    selected = [int(np.nanargmax(seed_score))]
    while len(selected) < n:
        remaining = [index for index in range(len(frame)) if index not in selected]
        distances = np.linalg.norm(matrix[remaining, None, :] - matrix[selected][None, :, :], axis=2)
        min_distances = np.min(distances, axis=1)
        selected.append(remaining[int(np.nanargmax(min_distances))])
    return selected


def _allocate_screening_origin_quota(
    remaining: pd.DataFrame,
    score: np.ndarray,
    registry: IngredientRegistry,
    n: int,
    local_quota: int,
    explore_probe_quota: int,
    explore_probe_per_category_cap: int,
    diversity_weight: float,
    competitive_utility_band: float | None,
) -> list[int]:
    """Pick `n` screening-pool indices using a fixed local/explore/probe mix.

    Pure top-score selection collapses onto whichever origin currently has
    the best-scoring cluster (e.g. local_perturbation candidates seeded near
    a legacy high-viability formulation), starving sparse_exploration and
    boundary_probe even though the pool was generated with a deliberate
    40/35/25 local/sparse/boundary mix. This reproduces that intent at the
    selection stage: `local_quota` slots go to the best local_perturbation
    candidates, and `explore_probe_quota` slots are split between
    sparse_exploration and boundary_probe by score, with each category
    capped at `explore_probe_per_category_cap` so neither one can take all
    of the explore/probe slots.

    Within each bucket the existing greedy diversity pick is reused so
    candidates are still spaced out, not just top-K by raw score.
    """
    if remaining.empty or n <= 0:
        return []

    origin = (
        remaining["candidate_origin"].astype(str)
        if "candidate_origin" in remaining.columns
        else pd.Series("", index=remaining.index)
    )

    def _bucket_pick(mask: pd.Series, count: int, exclude: set[int]) -> list[int]:
        if count <= 0:
            return []
        positions = [
            position
            for position, keep in enumerate(mask.to_numpy())
            if keep and position not in exclude
        ]
        if not positions:
            return []
        bucket_frame = remaining.iloc[positions].reset_index(drop=True)
        bucket_score = score[positions]
        local_indices = _greedy_diverse_pick(
            bucket_frame,
            bucket_score,
            registry.feature_names,
            n=min(count, len(positions)),
            diversity_weight=diversity_weight,
            competitive_utility_band=competitive_utility_band,
        )
        return [positions[index] for index in local_indices]

    selected: list[int] = []
    selected_set: set[int] = set()

    local_mask = origin.eq("local_perturbation")
    local_picks = _bucket_pick(local_mask, local_quota, selected_set)
    selected.extend(local_picks)
    selected_set.update(local_picks)

    sparse_mask = origin.eq("sparse_exploration")
    boundary_mask = origin.eq("boundary_probe")

    sparse_cap = min(explore_probe_per_category_cap, explore_probe_quota)
    boundary_cap = min(explore_probe_per_category_cap, explore_probe_quota)

    sparse_positions = [
        position for position, keep in enumerate(sparse_mask.to_numpy()) if keep
    ]
    boundary_positions = [
        position for position, keep in enumerate(boundary_mask.to_numpy()) if keep
    ]

    # Score-weighted split: rank each category's best available score, then
    # fill greedily by score across both categories together (so a category
    # with no competitive candidates yields its slots to the other), while
    # respecting the per-category cap.
    explore_probe_filled = 0
    sparse_taken = 0
    boundary_taken = 0
    sparse_remaining = [position for position in sparse_positions if position not in selected_set]
    boundary_remaining = [position for position in boundary_positions if position not in selected_set]

    while explore_probe_filled < explore_probe_quota and (sparse_remaining or boundary_remaining):
        candidates: list[tuple[float, str, int]] = []
        if sparse_remaining and sparse_taken < sparse_cap:
            best = max(sparse_remaining, key=lambda position: score[position])
            candidates.append((float(score[best]), "sparse", best))
        if boundary_remaining and boundary_taken < boundary_cap:
            best = max(boundary_remaining, key=lambda position: score[position])
            candidates.append((float(score[best]), "boundary", best))
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, category, position = candidates[0]
        selected.append(position)
        selected_set.add(position)
        explore_probe_filled += 1
        if category == "sparse":
            sparse_taken += 1
            sparse_remaining.remove(position)
        else:
            boundary_taken += 1
            boundary_remaining.remove(position)

    # Backfill: if local/sparse/boundary buckets together couldn't fill `n`
    # (e.g. a thin pool), fall back to best-remaining-score across all
    # origins so the slate still reaches its target size.
    if len(selected) < n:
        fallback_positions = [
            position for position in range(len(remaining)) if position not in selected_set
        ]
        fallback_positions.sort(key=lambda position: score[position], reverse=True)
        for position in fallback_positions:
            if len(selected) >= n:
                break
            selected.append(position)
            selected_set.add(position)

    return selected[:n]


def _select_round_slate(
    annotated: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    phase_resolution: PhaseResolution,
    n: int,
    policy_active: bool = False,
    target_round_number: int | None = None,
    cold_start_context: ColdStartContext | None = None,
) -> pd.DataFrame:
    if phase_resolution.active_phase == PHASE_MECHANICS:
        score_column = "mechanics_phase_score"
        default_recommendation_type = "joint_candidate"
    elif phase_resolution.active_phase == PHASE_HYBRID:
        score_column = "hybrid_phase_score"
        default_recommendation_type = "joint_candidate"
    else:
        score_column = "screening_phase_score"
        default_recommendation_type = "screening_candidate"

    # Retest eligibility is based only on observed campaign evidence. Model
    # uncertainty is the second ranking key after evidence severity and
    # cannot create a retest by itself. Feasibility was applied before this
    # point, so an infeasible diagnostic row cannot consume either slot.
    anchors = annotated[
        annotated.get(
            "recommendation_type", pd.Series("", index=annotated.index)
        ).astype(str).eq("mechanics_anchor")
    ].copy()
    retest_limit = int(nested_get(optimization_config, "retest.max_candidates_per_round", 2))
    retests = annotated[annotated["recommendation_type"] == "retest_priority"].copy()
    selected_parts: list[pd.DataFrame] = []
    selected_ids: set[str] = set()
    if not anchors.empty:
        selected_anchor = anchors.sort_values(
            ["candidate_id"], ascending=[True], kind="mergesort"
        ).head(1)
        selected_parts.append(selected_anchor)
        selected_ids.update(selected_anchor["candidate_id"].astype(str))
    if not retests.empty and retest_limit > 0:
        selected_retests = retests.sort_values(
            ["retest_priority_score", "viability_std", "formulation_id"],
            ascending=[False, False, True],
            kind="mergesort",
        ).head(min(retest_limit, n)).copy()
        selected_parts.append(selected_retests)
        selected_ids = set(selected_retests["candidate_id"].astype(str))

    rescue_limit = (
        int(nested_get(optimization_config, "candidate_generation.rescue_candidates_per_round", 2))
        if policy_active
        and phase_resolution.active_phase
        in {PHASE_SCREENING, PHASE_BOOTSTRAP, PHASE_HYBRID}
        else 0
    )
    if rescue_limit > 0:
        rescue_candidates = annotated.loc[
            annotated.get("candidate_origin", pd.Series("", index=annotated.index)).astype(str).eq("rescue_dilution")
            & ~annotated["candidate_id"].astype(str).isin(selected_ids)
        ].copy()
        if not rescue_candidates.empty:
            remaining_capacity = max(n - sum(len(part) for part in selected_parts), 0)
            if "rescue_scale_factor" not in rescue_candidates.columns:
                rescue_candidates["rescue_scale_factor"] = 1.0
            selected_rescue = rescue_candidates.sort_values(
                ["rescue_scale_factor", "viability_ucb"],
                ascending=[True, False],
            ).head(min(rescue_limit, remaining_capacity)).copy()
            if not selected_rescue.empty:
                selected_parts.append(selected_rescue)
                selected_ids.update(selected_rescue["candidate_id"].astype(str))

    # The retest + rescue mechanisms are reserved a combined budget of
    # (retest_limit + rescue_limit) slots. Any of that reserved budget left
    # unused this round (e.g. no retest-eligible formulation, or fewer
    # rescue candidates than the cap) is backfilled with the best-scoring
    # local_perturbation candidates rather than silently shrinking the
    # slate, so the round always gets a full n candidates.
    rescue_retest_reserve = retest_limit + rescue_limit
    rescue_retest_filled = sum(len(part) for part in selected_parts)
    rescue_retest_unused = max(rescue_retest_reserve - rescue_retest_filled, 0)

    # The additive classifier probability does not narrow this screening
    # pool. Exact-combination evidence has already supplied a bounded
    # screening deduction, while rescue rows keep their reserved priority.
    remaining = annotated.loc[~annotated["candidate_id"].astype(str).isin(selected_ids)].reset_index(drop=True)
    remaining_n = max(n - rescue_retest_filled, 0)
    if remaining_n > 0 and not remaining.empty:
        diversity_weight = (
            float(nested_get(optimization_config, "support_policy.diversity_weight", 0.05))
            if policy_active
            else 0.10
        )
        competitive_band = (
            float(
                nested_get(
                    optimization_config,
                    "support_policy.competitive_utility_band",
                    0.15,
                )
            )
            if policy_active
            else None
        )
        max_boundary = (
            int(
                nested_get(
                    optimization_config,
                    "support_policy.max_boundary_candidates_per_slate",
                    1,
                )
            )
            if policy_active
            else None
        )
        score = remaining[score_column].to_numpy(dtype=float)
        if policy_active and phase_resolution.active_phase == PHASE_SCREENING:
            # Fixed origin mix instead of pure top-score selection: otherwise
            # a tight high-viability local_perturbation cluster (often an
            # echo of legacy-transfer formulations) crowds out
            # sparse_exploration/boundary_probe entirely, even though the
            # pool was deliberately generated with a 40/35/25 mix. The
            # backfill slots from an unused retest/rescue reserve are
            # treated as additional local_perturbation budget.
            base_local_quota = int(
                nested_get(optimization_config, "round_policy.screening_local_quota", 3)
            )
            explore_probe_quota = max(remaining_n - base_local_quota - rescue_retest_unused, 0)
            local_quota = remaining_n - explore_probe_quota
            explore_probe_cap = int(
                nested_get(
                    optimization_config,
                    "round_policy.screening_explore_probe_category_cap",
                    3,
                )
            )
            selected_indices = _allocate_screening_origin_quota(
                remaining,
                score,
                registry,
                n=remaining_n,
                local_quota=local_quota,
                explore_probe_quota=explore_probe_quota,
                explore_probe_per_category_cap=explore_probe_cap,
                diversity_weight=diversity_weight,
                competitive_utility_band=competitive_band,
            )
        else:
            selected_indices = _greedy_diverse_pick(
                remaining,
                score,
                registry.feature_names,
                n=remaining_n,
                diversity_weight=diversity_weight,
                competitive_utility_band=competitive_band,
                max_boundary_candidates=max_boundary,
            )
        if selected_indices:
            selected_parts.append(remaining.iloc[selected_indices].copy())

    if selected_parts:
        selected = pd.concat(selected_parts, ignore_index=True)
    else:
        selected = annotated.head(0).copy()

    selected = _enforce_ingredient_combination_cap(
        selected,
        annotated,
        registry,
        optimization_config,
        score_column=score_column,
    )
    selected = _enforce_shared_ingredient_pair_cap(
        selected,
        annotated,
        registry,
        optimization_config,
        score_column=score_column,
    )
    shared_pair_replacement_count = int(
        selected.attrs.get("shared_pair_replacement_count", 0)
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
    if ingredient_frequency_active:
        selected = _enforce_ingredient_frequency_cap(
            selected,
            annotated,
            registry,
            optimization_config,
            score_column=score_column,
        )
    ingredient_frequency_metadata = selected.attrs.get(
        "ingredient_frequency_diversity",
        {
            "counts_before": _ingredient_appearance_counts(
                selected,
                registry,
            ),
            "counts_after": _ingredient_appearance_counts(
                selected,
                registry,
            ),
            "protected_retest_count": int(
                selected.apply(is_retest_row, axis=1).sum()
            ),
            "protected_rescue_count": int(
                selected.get(
                    "candidate_origin",
                    pd.Series("", index=selected.index),
                )
                .astype(str)
                .eq("rescue_dilution")
                .sum()
            ),
            "replacement_count": 0,
            "replacement_audit": [],
            "maximum_ingredient_frequency": int(
                max(
                    _ingredient_appearance_counts(
                        selected,
                        registry,
                    ).values(),
                    default=0,
                )
            ),
        },
    )
    if cold_start_context is not None and cold_start_context.policy.active:
        selected = _enforce_cold_start_policy(
            selected,
            annotated,
            registry,
            optimization_config,
            score_column=score_column,
            context=cold_start_context,
        )
    selected["recommendation_type"] = selected["recommendation_type"].replace("", pd.NA).fillna(default_recommendation_type)
    selected.insert(0, "selection_rank", range(1, len(selected) + 1))
    selected["selection_role"] = "round_candidate"
    selected.attrs["shared_pair_replacement_count"] = shared_pair_replacement_count
    selected.attrs["ingredient_frequency_diversity"] = (
        ingredient_frequency_metadata
    )
    return selected


def _mechanical_eligibility_mask(
    frame: pd.DataFrame,
    optimization_config: Mapping,
) -> tuple[pd.Series, dict]:
    prior_count = pd.to_numeric(
        frame.get(
            "prior_mechanical_observation_count",
            pd.Series(0, index=frame.index),
        ),
        errors="coerce",
    ).fillna(0)
    recommendation = frame.get(
        "recommendation_type", pd.Series("", index=frame.index)
    ).astype(str)
    is_anchor = recommendation.eq("mechanics_anchor")
    is_retest = recommendation.eq("retest_priority")
    allow_prior = bool(
        nested_get(
            optimization_config,
            "mechanics_transition.repeat_policy.allow_other_prior_mechanical_formulations",
            False,
        )
    )
    allow_retests = bool(
        nested_get(
            optimization_config,
            "mechanics_transition.repeat_policy.allow_retest_priority_for_mechanics",
            False,
        )
    )
    is_viability_control = recommendation.eq("campaign_control") | frame.get(
        "experimental_role", pd.Series("", index=frame.index)
    ).eq("campaign_control")
    historical_mechanical_reference = is_viability_control & frame.get(
        "mechanical_reference_eligible", pd.Series(False, index=frame.index)
    ).fillna(False).astype(bool)
    is_viability_control = is_viability_control & ~historical_mechanical_reference
    repeat_eligible = prior_count.eq(0) | is_anchor | allow_prior | historical_mechanical_reference
    retest_eligible = ~is_retest | allow_retests
    mask = repeat_eligible & retest_eligible & ~is_viability_control
    return mask, {
        "viability_control_excluded_count": int(is_viability_control.sum()),
        "prior_mechanics_excluded_count": int((~repeat_eligible).sum()),
        "retest_excluded_count": int((~retest_eligible).sum()),
        "allow_other_prior_mechanical_formulations": allow_prior,
        "allow_retest_priority_for_mechanics": allow_retests,
    }


def _bootstrap_mechanical_order(
    pool: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
) -> tuple[pd.DataFrame, dict]:
    if pool.empty:
        return pool.copy(), {"anchor_selected": False}
    screening_weight = float(
        nested_get(
            optimization_config,
            "mechanics_transition.bootstrap.screening_utility_weight",
            0.70,
        )
    )
    intact_weight = float(
        nested_get(
            optimization_config,
            "mechanics_transition.bootstrap.empirical_intact_weight",
            0.30,
        )
    )
    diversity_weight = float(
        nested_get(
            optimization_config,
            "mechanics_transition.bootstrap.diversity_weight",
            0.40,
        )
    )
    ordered_pool = pool.reset_index(drop=True).copy()
    screening_score = pd.to_numeric(
        ordered_pool.get("screening_phase_score", 0.0), errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    intact_probability = pd.to_numeric(
        ordered_pool.get("empirical_combination_pass_probability", 0.50),
        errors="coerce",
    ).fillna(0.50).to_numpy(dtype=float)
    utility = screening_weight * minmax(screening_score) + intact_weight * np.clip(
        intact_probability, 0.0, 1.0
    )
    ordered_pool["bootstrap_utility"] = utility
    matrix = _registry_scaled_feature_matrix(ordered_pool, registry)
    anchor_positions = np.flatnonzero(
        ordered_pool.get(
            "recommendation_type", pd.Series("", index=ordered_pool.index)
        ).astype(str).eq("mechanics_anchor").to_numpy()
    ).tolist()
    selected_positions: list[int] = anchor_positions[:1]
    first_new_position: int | None = None
    remaining = [
        position
        for position in range(len(ordered_pool))
        if position not in selected_positions
    ]
    while remaining:
        if first_new_position is None:
            best_utility = max(utility[position] for position in remaining)
            competitive = [
                position
                for position in remaining
                if abs(utility[position] - best_utility) < 1e-12
            ]
            chosen = min(
                competitive,
                key=lambda position: str(
                    ordered_pool.iloc[position].get("candidate_id", "")
                ),
            )
        else:
            distances = np.linalg.norm(
                matrix[remaining, None, :] - matrix[selected_positions][None, :, :],
                axis=2,
            )
            minimum_distance = np.min(distances, axis=1)
            combined = (
                (1.0 - diversity_weight) * minmax(utility[remaining])
                + diversity_weight * minmax(minimum_distance)
            )
            ranking = sorted(
                range(len(remaining)),
                key=lambda offset: (
                    -float(combined[offset]),
                    str(
                        ordered_pool.iloc[remaining[offset]].get(
                            "candidate_id", ""
                        )
                    ),
                ),
            )
            chosen = remaining[ranking[0]]
        if first_new_position is None:
            first_new_position = chosen
        selected_positions.append(chosen)
        remaining.remove(chosen)

    ranked = ordered_pool.iloc[selected_positions].copy().reset_index(drop=True)
    ranked["mechanical_transition_role"] = "bootstrap_coverage"
    if anchor_positions:
        ranked.loc[
            ranked["recommendation_type"].astype(str).eq("mechanics_anchor"),
            "mechanical_transition_role",
        ] = "anchor"
    if first_new_position is not None:
        first_new_candidate = str(
            ordered_pool.iloc[first_new_position].get("candidate_id", "")
        )
        ranked.loc[
            ranked["candidate_id"].astype(str).eq(first_new_candidate),
            "mechanical_transition_role",
        ] = "bootstrap_utility"
    return ranked, {
        "anchor_selected": bool(anchor_positions),
        "screening_utility_weight": screening_weight,
        "empirical_intact_weight": intact_weight,
        "diversity_weight": diversity_weight,
    }


def _hybrid_mechanical_order(
    pool: pd.DataFrame,
    models: EndpointModels,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    primary_capacity: int,
) -> tuple[pd.DataFrame, dict]:
    if pool.empty:
        return pool.copy(), {"role_fallbacks": []}
    working = pool.reset_index(drop=True).copy()
    mechanics = pd.to_numeric(
        working.get("mechanics_phase_score", 0.0), errors="coerce"
    ).fillna(float("-inf")).to_numpy(dtype=float)
    hybrid = pd.to_numeric(
        working.get("hybrid_phase_score", mechanics), errors="coerce"
    ).fillna(float("-inf")).to_numpy(dtype=float)
    matrix = _registry_scaled_feature_matrix(working, registry)
    selected: list[int] = []
    role_by_position: dict[int, str] = {}
    fallbacks: list[dict] = []

    qlog_slots = int(
        nested_get(
            optimization_config,
            "mechanics_transition.hybrid.qlognehvi_slots",
            2,
        )
    )
    local_slots = int(
        nested_get(
            optimization_config,
            "mechanics_transition.hybrid.local_slots",
            1,
        )
    )
    coverage_slots = int(
        nested_get(
            optimization_config,
            "mechanics_transition.hybrid.coverage_slots",
            1,
        )
    )

    def best_position(candidates: list[int], score: np.ndarray) -> int | None:
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda position: (
                -float(score[position]),
                str(working.iloc[position].get("candidate_id", "")),
            ),
        )[0]

    for slot in range(qlog_slots):
        candidates = [position for position in range(len(working)) if position not in selected]
        if not candidates:
            break
        if slot == 0 or not selected:
            chosen = best_position(candidates, mechanics)
        else:
            distance = np.min(
                np.linalg.norm(
                    matrix[candidates, None, :] - matrix[selected][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            combined = minmax(mechanics[candidates]) + float(
                nested_get(optimization_config, "selection.diversity_weight", 0.10)
            ) * minmax(distance)
            chosen = candidates[
                sorted(
                    range(len(candidates)),
                    key=lambda offset: (
                        -float(combined[offset]),
                        str(
                            working.iloc[candidates[offset]].get("candidate_id", "")
                        ),
                    ),
                )[0]
            ]
        if chosen is not None:
            selected.append(chosen)
            role_by_position[chosen] = "hybrid_qlognehvi"

    for _ in range(local_slots):
        candidates = [
            position
            for position in range(len(working))
            if position not in selected
            and str(working.iloc[position].get("candidate_origin", ""))
            == "local_perturbation"
        ]
        chosen = best_position(candidates, mechanics)
        if chosen is None:
            fallbacks.append({"role": "hybrid_local", "reason": "no eligible local_perturbation candidate"})
            continue
        selected.append(chosen)
        role_by_position[chosen] = "hybrid_local"

    for _ in range(coverage_slots):
        screening = pd.to_numeric(
            working.get("screening_phase_score", 0.0), errors="coerce"
        ).fillna(float("-inf"))
        quantile = float(
            nested_get(
                optimization_config,
                "mechanics_transition.hybrid.coverage_min_screening_quantile",
                0.50,
            )
        )
        score_floor = float(screening.quantile(quantile))
        probability_floor = float(
            nested_get(
                optimization_config,
                "mechanics_transition.hybrid.coverage_min_empirical_intact_probability",
                0.50,
            )
        )
        probability = pd.to_numeric(
            working.get("empirical_combination_pass_probability", 0.50),
            errors="coerce",
        ).fillna(0.50)
        candidates = [
            position
            for position in range(len(working))
            if position not in selected
            and float(screening.iloc[position]) >= score_floor
            and float(probability.iloc[position]) >= probability_floor
        ]
        if not candidates:
            fallbacks.append({"role": "hybrid_coverage", "reason": "no candidate meets coverage floors"})
            continue
        historical = models.training_frame.copy()
        if "critical_axial_load_N_per_needle" in historical.columns:
            historical = historical.loc[
                pd.to_numeric(
                    historical["critical_axial_load_N_per_needle"], errors="coerce"
                ).notna()
            ]
        else:
            historical = historical.head(0)
        reference_matrix = _registry_scaled_feature_matrix(historical, registry)
        if selected:
            reference_matrix = np.vstack([reference_matrix, matrix[selected]])
        if reference_matrix.size == 0:
            coverage_distance = np.ones(len(candidates), dtype=float)
        else:
            coverage_distance = np.min(
                np.linalg.norm(
                    matrix[candidates, None, :] - reference_matrix[None, :, :],
                    axis=2,
                ),
                axis=1,
            )
        chosen = candidates[
            sorted(
                range(len(candidates)),
                key=lambda offset: (
                    -float(coverage_distance[offset]),
                    str(working.iloc[candidates[offset]].get("candidate_id", "")),
                ),
            )[0]
        ]
        selected.append(chosen)
        role_by_position[chosen] = "hybrid_coverage"

    while len(selected) < min(primary_capacity, len(working)):
        candidates = [position for position in range(len(working)) if position not in selected]
        chosen = best_position(candidates, hybrid)
        if chosen is None:
            break
        selected.append(chosen)
        role_by_position[chosen] = "hybrid_qlognehvi"
        fallbacks.append(
            {
                "role": "hybrid_fallback",
                "candidate_id": str(working.iloc[chosen].get("candidate_id", "")),
                "reason": "unfilled transition role backfilled by hybrid score",
            }
        )

    remaining = [position for position in range(len(working)) if position not in selected]
    remaining.sort(
        key=lambda position: (
            -float(mechanics[position]),
            str(working.iloc[position].get("candidate_id", "")),
        )
    )
    ordered_positions = [*selected, *remaining]
    ranked = working.iloc[ordered_positions].copy().reset_index(drop=True)
    ranked["mechanical_transition_role"] = [
        role_by_position.get(position, "ordered_backup")
        for position in ordered_positions
    ]
    return ranked, {
        "qlognehvi_slots": qlog_slots,
        "local_slots": local_slots,
        "coverage_slots": coverage_slots,
        "role_fallbacks": fallbacks,
    }


def select_mechanical_tests(
    annotated: pd.DataFrame,
    models: EndpointModels,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    phase_resolution: PhaseResolution,
    n: int,
    intact_policy: IntactCombinationPolicy | None = None,
) -> tuple[pd.DataFrame, dict]:
    threshold = float(nested_get(optimization_config, "round_policy.intact_probability_threshold", 0.50))
    mechanical_count = models.mechanical_observation_count
    if phase_resolution.active_phase == PHASE_SCREENING:
        selected = annotated.head(0).copy()
        selected.insert(0, "mechanical_selection_rank", pd.Series(dtype=int))
        selected["mechanical_primary_recommended"] = pd.Series(dtype=bool)
        selected["mechanical_backup_status"] = pd.Series(dtype=str)
        selected["selection_role"] = "mechanical_test_disabled"
        selected["mechanical_selection_mode"] = "disabled_screening_only"
        return selected, {
            "mechanical_selection_mode": "disabled_screening_only",
            "mechanical_selection_reason": "mechanical recommendations remain off until the screening-count gate is satisfied",
            "mechanical_observation_count": mechanical_count,
            "intact_probability_threshold": threshold,
            "intact_probability_threshold_role": "unused_screening_only",
            "pass_pool_size": 0,
            "primary_mechanical_capacity": int(n),
            "primary_recommendation_count": 0,
            "backup_count": 0,
            "botorch_available": bool(botorch_available()),
            "active_phase": phase_resolution.active_phase,
        }

    eligibility, eligibility_metadata = _mechanical_eligibility_mask(
        annotated, optimization_config
    )
    pool = annotated.loc[eligibility].reset_index(drop=True).copy()
    primary_count = min(int(n), len(pool))
    mechanics_metadata: dict = {"botorch_metadata": {}}
    transition_metadata: dict = {}

    if phase_resolution.active_phase == PHASE_BOOTSTRAP:
        selected, transition_metadata = _bootstrap_mechanical_order(
            pool, registry, optimization_config
        )
        mode = "bootstrap_utility_diversity_with_actual_intact_backups"
        threshold_role = "empirical_probability_in_bootstrap_utility"
    else:
        if "mechanics_phase_score" in pool.columns and pd.to_numeric(
            pool["mechanics_phase_score"], errors="coerce"
        ).notna().all():
            score = pd.to_numeric(
                pool["mechanics_phase_score"], errors="coerce"
            ).to_numpy(dtype=float)
            mechanics_metadata = {
                "pool_selection_mode": "precomputed_feasibility_weighted",
                "botorch_metadata": {},
            }
        else:
            score, mechanics_metadata = _mechanics_phase_scores(
                pool,
                models,
                registry,
                optimization_config,
                intact_policy=intact_policy,
            )
            pool["mechanics_phase_score"] = score
        if phase_resolution.active_phase == PHASE_HYBRID:
            selected, transition_metadata = _hybrid_mechanical_order(
                pool,
                models,
                registry,
                optimization_config,
                primary_capacity=primary_count,
            )
            mode = "hybrid_2qlognehvi_1local_1coverage_with_actual_intact_backups"
        else:
            ranking = pd.DataFrame(
                {
                    "position": np.arange(len(pool), dtype=int),
                    "score": score,
                    "candidate_id": pool["candidate_id"].astype(str).to_numpy(),
                }
            ).sort_values(
                ["score", "candidate_id"],
                ascending=[False, True],
                kind="mergesort",
            )
            selected = pool.iloc[ranking["position"].to_numpy(dtype=int)].copy()
            selected["mechanical_transition_role"] = "ordered_backup"
            if primary_count:
                selected.iloc[
                    :primary_count,
                    selected.columns.get_loc("mechanical_transition_role"),
                ] = ""
            mode = "empirical_feasibility_weighted_with_actual_intact_backups"
        threshold_role = "compatibility_only_not_applied"

    selected = selected.reset_index(drop=True)
    selected.insert(0, "mechanical_selection_rank", range(1, len(selected) + 1))
    selected["mechanical_primary_recommended"] = (
        selected["mechanical_selection_rank"] <= primary_count
    )
    selected["mechanical_backup_status"] = np.where(
        selected["mechanical_primary_recommended"], "primary", "ordered_backup"
    )
    selected["selection_role"] = np.where(
        selected["mechanical_primary_recommended"],
        "mechanical_test_primary",
        "mechanical_test_backup",
    )
    selected["mechanical_selection_mode"] = mode
    anchor_rows = selected.loc[
        selected.get(
            "recommendation_type", pd.Series("", index=selected.index)
        ).astype(str).eq("mechanics_anchor")
    ]
    metadata = {
        "mechanical_selection_mode": mode,
        "mechanical_observation_count": mechanical_count,
        "intact_probability_threshold": threshold,
        "intact_probability_threshold_role": threshold_role,
        "pass_pool_size": int(len(pool)),
        "mechanically_ineligible_count": int((~eligibility).sum()),
        "primary_mechanical_capacity": int(n),
        "primary_recommendation_count": int(
            selected["mechanical_primary_recommended"].sum()
        ),
        "backup_count": int((~selected["mechanical_primary_recommended"]).sum()),
        "actual_intact_backup_rule": (
            "test the first ranked rows that actually pass intact until "
            "primary_mechanical_capacity is reached; unranked rows are ineligible"
        ),
        "botorch_available": bool(botorch_available()),
        "active_phase": phase_resolution.active_phase,
        "classifier_probability_selection_role": "diagnostic_only",
        "eligibility": eligibility_metadata,
        "transition_allocation": transition_metadata,
        "anchor": {
            "selected": not anchor_rows.empty,
            "formulation_id": (
                str(anchor_rows.iloc[0].get("formulation_id", ""))
                if not anchor_rows.empty
                else ""
            ),
            "source_batch_id": (
                str(anchor_rows.iloc[0].get("mechanics_anchor_source_batch", ""))
                if not anchor_rows.empty
                else ""
            ),
            "selection_score": (
                None
                if anchor_rows.empty
                or pd.isna(anchor_rows.iloc[0].get("mechanics_anchor_selection_score"))
                else float(anchor_rows.iloc[0]["mechanics_anchor_selection_score"])
            ),
        },
        "botorch_metadata": mechanics_metadata.get("botorch_metadata", {}),
    }
    return selected, metadata


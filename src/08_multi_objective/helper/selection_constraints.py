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

def _drop_zero_active_candidates(frame: pd.DataFrame, registry: IngredientRegistry) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame.copy(), 0
    filtered = frame.copy()
    filtered["active_ingredient_count"] = filtered.apply(
        lambda row: count_active_ingredients(row, registry),
        axis=1,
    )
    mask = pd.to_numeric(filtered["active_ingredient_count"], errors="coerce").fillna(0).astype(int) > 0
    removed = int((~mask).sum())
    return filtered.loc[mask].reset_index(drop=True).copy(), removed


def _active_ingredient_set(row: pd.Series, registry: IngredientRegistry) -> frozenset[str]:
    """Return the exact set of registry-recognized ingredients active in `row`.

    Uses `registry.feature_names` (the authoritative ingredient list, same
    one `count_active_ingredients` uses) rather than
    any `_M`/`_pct`-suffix heuristic, so derived/aggregate columns like
    `total_polymer_pct` or `total_nonpermeating_solute_M` are never mistaken
    for selectable ingredients.
    """
    active: list[str] = []
    for feature_name in registry.feature_names:
        value = pd.to_numeric(row.get(feature_name, 0.0), errors="coerce")
        if pd.isna(value):
            continue
        if abs(float(value)) >= presence_threshold(feature_name):
            active.append(feature_name)
    return frozenset(active)


def _combination_cap_for_size(optimization_config: Mapping, combo_size: int) -> int:
    """Combination occurrence cap, with pairs allowed more repeats than
    larger combinations.

    Pairs (size 2) use `selection.max_candidates_per_ingredient_combination`
    (default 2) -- the original, looser cap. Any exact combination of size 3
    or larger (trio, four-a-kind, etc.) is far more specific and far less
    likely to be a coincidence, so it defaults to a much tighter cap of 1 via
    `selection.max_candidates_per_larger_ingredient_combination`: at most
    one candidate per round may carry any *exact* size-3+ active-ingredient
    set.
    """
    if combo_size <= 2:
        return int(
            nested_get(
                optimization_config,
                "selection.max_candidates_per_ingredient_combination",
                2,
            )
        )
    return int(
        nested_get(
            optimization_config,
            "selection.max_candidates_per_larger_ingredient_combination",
            1,
        )
    )


def _enforce_ingredient_combination_cap(
    selected: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    score_column: str,
) -> pd.DataFrame:
    """Cap how many selected candidates may share the exact same active-
    ingredient set, regardless of how many ingredients are in that set.

    A pure-viability score can collapse the slate onto repeats of one
    high-scoring combination (e.g. ectoin + ethylene_glycol), even after
    origin-bucket diversity is enforced, because every bucket independently
    re-discovers the same favored combination. This swaps out the lowest-
    scoring offender past the cap for the best-scoring pool candidate whose
    own combination is not already at the cap. A same-origin replacement is
    preferred so rescue and exploration allocation is retained where the pool
    permits, while still mirroring
    the candidate's full active-ingredient set (size 2+).

    The cap is size-dependent (see `_combination_cap_for_size`): pairs get a
    looser cap, exact trios/quadruples/etc. get a much tighter one (1 by
    default), since an exact match on 3+ ingredients simultaneously is a much
    stronger signal of redundant exploration than a repeated pair.
    Combinations of size 0-1 are left uncapped here: an empty or
    single-ingredient formulation isn't the "ectoin+EG cluster" failure mode
    this guards against. Size-1 spacing is enforced by the unified formulation
    similarity policy before selection.
    """
    if selected.empty:
        return selected.copy()

    adjusted = selected.copy().reset_index(drop=True)
    ranked_pool = candidate_pool.sort_values(
        [score_column, "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    def combo_of(row: pd.Series) -> frozenset[str]:
        return _active_ingredient_set(row, registry)

    while True:
        combos = [
            frozenset() if is_retest_row(row) else combo_of(row)
            for _, row in adjusted.iterrows()
        ]
        counts: dict[frozenset[str], int] = {}
        for combo in combos:
            if len(combo) < 2:
                continue
            counts[combo] = counts.get(combo, 0) + 1
        over_cap = {
            combo: count
            for combo, count in counts.items()
            if count > _combination_cap_for_size(optimization_config, len(combo))
        }
        if not over_cap:
            break

        # Pick the most-over-cap combination (relative to its own size's
        # cap), then within it the lowest-scoring candidate as the swap-out
        # target.
        worst_combo = max(
            over_cap,
            key=lambda combo: over_cap[combo] - _combination_cap_for_size(optimization_config, len(combo)),
        )
        worst_cap = _combination_cap_for_size(optimization_config, len(worst_combo))
        offender_positions = [
            position
            for position, combo in enumerate(combos)
            if combo == worst_combo
        ]
        offender_positions.sort(
            key=lambda position: float(
                pd.to_numeric(adjusted.iloc[position].get(score_column, 0.0), errors="coerce") or 0.0
            )
        )
        loser_position = offender_positions[0]
        loser_id = str(adjusted.iloc[loser_position]["candidate_id"])
        loser_origin = str(
            adjusted.iloc[loser_position].get("candidate_origin", "")
        )

        selected_ids = set(adjusted["candidate_id"].astype(str))
        replacement_row: pd.DataFrame | None = None
        for require_same_origin in (True, False):
            for _, candidate in ranked_pool.iterrows():
                candidate_id = str(candidate.get("candidate_id", ""))
                if (
                    not candidate_id
                    or candidate_id == loser_id
                    or candidate_id in selected_ids
                    or is_retest_row(candidate)
                ):
                    continue
                candidate_origin = str(candidate.get("candidate_origin", ""))
                if require_same_origin and candidate_origin != loser_origin:
                    continue
                if not require_same_origin and candidate_origin == loser_origin:
                    continue
                candidate_combo = combo_of(candidate)
                if candidate_combo == worst_combo:
                    continue
                if len(candidate_combo) >= 2:
                    candidate_cap = _combination_cap_for_size(
                        optimization_config,
                        len(candidate_combo),
                    )
                    if counts.get(candidate_combo, 0) >= candidate_cap:
                        continue
                replacement_row = pd.DataFrame([candidate])
                break
            if replacement_row is not None:
                break

        if replacement_row is None:
            # No eligible replacement exists in the pool; leave this
            # over-cap combination as-is rather than shrinking the slate.
            break

        loser_row = adjusted.iloc[[loser_position]]
        adjusted = pd.concat(
            [
                adjusted.iloc[:loser_position],
                replacement_row,
                adjusted.iloc[loser_position + 1 :],
            ],
            ignore_index=True,
        )
        del loser_row, worst_cap

    return adjusted


def _shared_pair_counts(
    frame: pd.DataFrame,
    registry: IngredientRegistry,
) -> dict[tuple[str, str], int]:
    """Count every active ingredient pair among non-retest rows.

    A row with three active ingredients contributes three pairs, so adding an
    ingredient does not hide membership in a repeatedly selected core pair.
    """
    counts: dict[tuple[str, str], int] = {}
    for _, row in frame.iterrows():
        if is_retest_row(row):
            continue
        active = sorted(_active_ingredient_set(row, registry))
        for pair in combinations(active, 2):
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def _exact_combination_caps_pass(
    frame: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
) -> bool:
    counts: dict[frozenset[str], int] = {}
    for _, row in frame.iterrows():
        if is_retest_row(row):
            continue
        active = _active_ingredient_set(row, registry)
        if len(active) < 2:
            continue
        counts[active] = counts.get(active, 0) + 1
        if counts[active] > _combination_cap_for_size(
            optimization_config,
            len(active),
        ):
            return False
    return True


def _enforce_shared_ingredient_pair_cap(
    selected: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    score_column: str,
) -> pd.DataFrame:
    """Replace ordinary rows until every shared core pair is within its cap."""
    if selected.empty:
        adjusted = selected.copy()
        adjusted.attrs["shared_pair_replacement_count"] = 0
        return adjusted

    cap = int(
        nested_get(
            optimization_config,
            "selection.max_candidates_per_shared_ingredient_pair",
            5,
        )
    )
    if cap < 1:
        raise ValueError(
            "selection.max_candidates_per_shared_ingredient_pair must be at least 1."
        )

    adjusted = selected.copy().reset_index(drop=True)
    ranked_pool = candidate_pool.copy()
    ranked_pool["_pair_score"] = pd.to_numeric(
        ranked_pool.get(score_column, 0.0),
        errors="coerce",
    ).fillna(float("-inf"))
    ranked_pool = ranked_pool.sort_values(
        ["_pair_score", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).drop(columns=["_pair_score"]).reset_index(drop=True)
    replacement_count = 0

    while True:
        pair_counts = _shared_pair_counts(adjusted, registry)
        over_cap = {
            pair: count for pair, count in pair_counts.items() if count > cap
        }
        if not over_cap:
            break
        offending_pair = sorted(
            over_cap,
            key=lambda pair: (-(over_cap[pair] - cap), pair),
        )[0]

        offender_positions: list[int] = []
        for position, (_, row) in enumerate(adjusted.iterrows()):
            if is_retest_row(row):
                continue
            if str(row.get("candidate_origin", "")) == "rescue_dilution":
                continue
            active = _active_ingredient_set(row, registry)
            if set(offending_pair).issubset(active):
                offender_positions.append(position)
        offender_positions.sort(
            key=lambda position: (
                float(
                    pd.to_numeric(
                        adjusted.iloc[position].get(score_column, float("-inf")),
                        errors="coerce",
                    )
                ),
                str(adjusted.iloc[position].get("candidate_id", "")),
            )
        )
        if not offender_positions:
            raise ValueError(
                "Shared ingredient-pair cap cannot be satisfied without "
                f"removing a protected retest/rescue row: {offending_pair}."
            )

        replacement_made = False
        selected_ids = set(adjusted["candidate_id"].astype(str))
        for loser_position in offender_positions:
            loser = adjusted.iloc[loser_position]
            loser_origin = str(loser.get("candidate_origin", ""))
            for _, candidate in ranked_pool.iterrows():
                candidate_id = str(candidate.get("candidate_id", ""))
                if not candidate_id or candidate_id in selected_ids:
                    continue
                if is_retest_row(candidate):
                    continue
                if str(candidate.get("candidate_origin", "")) != loser_origin:
                    continue
                trial = pd.concat(
                    [
                        adjusted.iloc[:loser_position],
                        pd.DataFrame([candidate]),
                        adjusted.iloc[loser_position + 1 :],
                    ],
                    ignore_index=True,
                )
                if max(_shared_pair_counts(trial, registry).values(), default=0) > cap:
                    continue
                if not _exact_combination_caps_pass(
                    trial,
                    registry,
                    optimization_config,
                ):
                    continue
                adjusted = trial
                replacement_count += 1
                replacement_made = True
                break
            if replacement_made:
                break
        if not replacement_made:
            raise ValueError(
                "Shared ingredient-pair cap cannot be satisfied from the "
                "eligible candidate pool while preserving origin allocation; "
                f"unresolved pair={offending_pair}, count={over_cap[offending_pair]}, cap={cap}."
            )

    final_counts = _shared_pair_counts(adjusted, registry)
    violations = {
        pair: count for pair, count in final_counts.items() if count > cap
    }
    if violations:
        raise ValueError(
            "Refusing to freeze a slate that violates the shared ingredient-pair "
            f"cap: {violations}."
        )
    adjusted.attrs["shared_pair_replacement_count"] = replacement_count
    return adjusted


def _ingredient_appearance_counts(
    frame: pd.DataFrame,
    registry: IngredientRegistry,
) -> dict[str, int]:
    """Count marginal ingredient presence across every selected row."""
    counts = {feature_name: 0 for feature_name in registry.feature_names}
    for _, row in frame.iterrows():
        for feature_name in _active_ingredient_set(row, registry):
            counts[feature_name] += 1
    return {feature_name: count for feature_name, count in counts.items() if count}


def _is_protected_diversity_row(row: pd.Series) -> bool:
    """Retests and rescue hypotheses count toward caps but cannot be removed."""
    return is_retest_row(row) or str(
        row.get("candidate_origin", "")
    ).strip() == "rescue_dilution"


def _support_boundary_count(frame: pd.DataFrame) -> int:
    if "support_status" not in frame.columns:
        return 0
    return int(frame["support_status"].astype(str).eq("boundary").sum())


def _enforce_ingredient_frequency_cap(
    selected: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    score_column: str,
) -> pd.DataFrame:
    """Hard-cap marginal ingredient appearances while preserving origins."""
    cap = int(
        nested_get(
            optimization_config,
            "selection.max_candidates_per_ingredient",
            5,
        )
    )
    if cap < 1:
        raise ValueError(
            "selection.max_candidates_per_ingredient must be at least 1."
        )
    if selected.empty:
        adjusted = selected.copy()
        adjusted.attrs["ingredient_frequency_diversity"] = {
            "counts_before": {},
            "counts_after": {},
            "protected_retest_count": 0,
            "protected_rescue_count": 0,
            "replacement_count": 0,
            "replacement_audit": [],
            "maximum_ingredient_frequency": 0,
        }
        return adjusted

    adjusted = selected.copy().reset_index(drop=True)
    selected_size = len(adjusted)
    origin_counts_before = (
        adjusted.get(
            "candidate_origin",
            pd.Series("", index=adjusted.index),
        )
        .fillna("")
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    counts_before = _ingredient_appearance_counts(adjusted, registry)
    protected_retest_count = int(
        adjusted.apply(is_retest_row, axis=1).sum()
    )
    protected_rescue_count = int(
        adjusted.get(
            "candidate_origin",
            pd.Series("", index=adjusted.index),
        )
        .astype(str)
        .eq("rescue_dilution")
        .sum()
    )

    ranked_pool = candidate_pool.copy()
    ranked_pool["_frequency_score"] = pd.to_numeric(
        ranked_pool.get(score_column, 0.0),
        errors="coerce",
    ).fillna(float("-inf"))
    ranked_pool = (
        ranked_pool.sort_values(
            ["_frequency_score", "candidate_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        .drop(columns=["_frequency_score"])
        .reset_index(drop=True)
    )
    feature_order = {
        feature_name: index
        for index, feature_name in enumerate(registry.feature_names)
    }
    shared_pair_cap = int(
        nested_get(
            optimization_config,
            "selection.max_candidates_per_shared_ingredient_pair",
            5,
        )
    )
    boundary_cap = int(
        nested_get(
            optimization_config,
            "support_policy.max_boundary_candidates_per_slate",
            1,
        )
    )
    replacement_audit: list[dict[str, object]] = []

    def score_value(row: pd.Series) -> float:
        value = pd.to_numeric(
            row.get(score_column, float("-inf")),
            errors="coerce",
        )
        return float(value) if pd.notna(value) else float("-inf")

    while True:
        current_counts = _ingredient_appearance_counts(adjusted, registry)
        over_cap = {
            feature_name: count
            for feature_name, count in current_counts.items()
            if count > cap
        }
        if not over_cap:
            break
        trigger_feature = sorted(
            over_cap,
            key=lambda feature_name: (
                -(over_cap[feature_name] - cap),
                feature_order[feature_name],
            ),
        )[0]

        offender_positions = [
            position
            for position, (_, row) in enumerate(adjusted.iterrows())
            if trigger_feature in _active_ingredient_set(row, registry)
            and not _is_protected_diversity_row(row)
        ]
        offender_positions.sort(
            key=lambda position: (
                score_value(adjusted.iloc[position]),
                str(adjusted.iloc[position].get("candidate_id", "")),
            )
        )
        if not offender_positions:
            raise ValueError(
                "Ingredient-frequency cap cannot be satisfied without "
                "removing protected retest/rescue rows: "
                f"{trigger_feature} appears {over_cap[trigger_feature]} "
                f"times, cap={cap}."
            )

        replacement_made = False
        selected_ids = set(adjusted["candidate_id"].astype(str))
        for loser_position in offender_positions:
            loser = adjusted.iloc[loser_position]
            loser_origin = str(loser.get("candidate_origin", ""))
            for _, candidate in ranked_pool.iterrows():
                candidate_id = str(candidate.get("candidate_id", ""))
                if not candidate_id or candidate_id in selected_ids:
                    continue
                if _is_protected_diversity_row(candidate):
                    continue
                if str(candidate.get("candidate_origin", "")) != loser_origin:
                    continue
                feasibility_value = candidate.get("feasibility_pass", True)
                if pd.notna(feasibility_value) and not bool(feasibility_value):
                    continue

                trial = pd.concat(
                    [
                        adjusted.iloc[:loser_position],
                        pd.DataFrame([candidate]),
                        adjusted.iloc[loser_position + 1 :],
                    ],
                    ignore_index=True,
                )
                trial_counts = _ingredient_appearance_counts(trial, registry)
                if trial_counts.get(trigger_feature, 0) >= current_counts.get(
                    trigger_feature,
                    0,
                ):
                    continue
                if any(
                    trial_counts.get(feature_name, 0)
                    > max(cap, current_counts.get(feature_name, 0))
                    for feature_name in registry.feature_names
                ):
                    continue
                if not _exact_combination_caps_pass(
                    trial,
                    registry,
                    optimization_config,
                ):
                    continue
                if max(
                    _shared_pair_counts(trial, registry).values(),
                    default=0,
                ) > shared_pair_cap:
                    continue
                if _support_boundary_count(trial) > boundary_cap:
                    continue

                replacement_audit.append(
                    {
                        "trigger_ingredient": trigger_feature,
                        "count_before": int(
                            current_counts.get(trigger_feature, 0)
                        ),
                        "count_after": int(
                            trial_counts.get(trigger_feature, 0)
                        ),
                        "removed_candidate_id": str(
                            loser.get("candidate_id", "")
                        ),
                        "added_candidate_id": candidate_id,
                        "candidate_origin": loser_origin,
                        "removed_score": score_value(loser),
                        "added_score": score_value(candidate),
                        "score_change": score_value(candidate)
                        - score_value(loser),
                    }
                )
                adjusted = trial
                replacement_made = True
                break
            if replacement_made:
                break
        if not replacement_made:
            raise ValueError(
                "Ingredient-frequency cap cannot be satisfied from the "
                "eligible candidate pool while preserving origin allocation "
                "and existing diversity constraints: "
                f"{trigger_feature} appears {over_cap[trigger_feature]} "
                f"times, cap={cap}."
            )

    counts_after = _ingredient_appearance_counts(adjusted, registry)
    violations = {
        feature_name: count
        for feature_name, count in counts_after.items()
        if count > cap
    }
    origin_counts_after = (
        adjusted.get(
            "candidate_origin",
            pd.Series("", index=adjusted.index),
        )
        .fillna("")
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    if violations:
        raise ValueError(
            "Refusing to freeze a slate that violates the marginal ingredient "
            f"frequency cap: {violations}."
        )
    if len(adjusted) != selected_size:
        raise ValueError(
            "Ingredient-frequency enforcement changed the slate size: "
            f"{selected_size} -> {len(adjusted)}."
        )
    if origin_counts_after != origin_counts_before:
        raise ValueError(
            "Ingredient-frequency enforcement changed candidate-origin "
            f"allocation: before={origin_counts_before}, "
            f"after={origin_counts_after}."
        )

    adjusted.attrs["ingredient_frequency_diversity"] = {
        "counts_before": {
            feature_name: int(count)
            for feature_name, count in counts_before.items()
        },
        "counts_after": {
            feature_name: int(count)
            for feature_name, count in counts_after.items()
        },
        "protected_retest_count": protected_retest_count,
        "protected_rescue_count": protected_rescue_count,
        "replacement_count": len(replacement_audit),
        "replacement_audit": replacement_audit,
        "maximum_ingredient_frequency": int(
            max(counts_after.values(), default=0)
        ),
    }
    return adjusted


def _cold_start_ordinary_counts(
    frame: pd.DataFrame,
    registry: IngredientRegistry,
    context: ColdStartContext,
) -> dict[str, int]:
    counts = {feature_name: 0 for feature_name in context.cold_ingredients}
    for _, row in frame.iterrows():
        if context.is_exempt_origin(row.get("candidate_origin", "")):
            continue
        for feature_name in cold_ingredients_in_row(row, registry, context):
            counts[feature_name] += 1
    return counts


def _cold_start_trial_passes_shared_constraints(
    trial: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
) -> bool:
    if not _exact_combination_caps_pass(trial, registry, optimization_config):
        return False
    pair_cap = int(
        nested_get(
            optimization_config,
            "selection.max_candidates_per_shared_ingredient_pair",
            5,
        )
    )
    if max(_shared_pair_counts(trial, registry).values(), default=0) > pair_cap:
        return False
    universal_cap = int(
        nested_get(
            optimization_config,
            "selection.max_candidates_per_ingredient",
            5,
        )
    )
    if max(
        _ingredient_appearance_counts(trial, registry).values(), default=0
    ) > universal_cap:
        return False
    boundary_cap = int(
        nested_get(
            optimization_config,
            "support_policy.max_boundary_candidates_per_slate",
            1,
        )
    )
    return _support_boundary_count(trial) <= boundary_cap


def _enforce_cold_start_policy(
    selected: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    score_column: str,
    context: ColdStartContext,
) -> pd.DataFrame:
    """Reserve graduation rows, then enforce each cold ingredient's cap."""
    adjusted = selected.copy().reset_index(drop=True)
    policy = context.policy
    base_metadata: dict[str, object] = {
        "planned_graduation_allocations": [],
        "selected_graduation_allocations": [],
        "graduation_skips": [],
        "graduation_replacements": [],
        "cap_replacements": [],
        "ordinary_counts_before": _cold_start_ordinary_counts(
            adjusted, registry, context
        ),
        "ordinary_counts_after": {},
        "cross_origin_replacement_count": 0,
    }
    if not policy.active or not context.cold_ingredients:
        base_metadata["ordinary_counts_after"] = base_metadata[
            "ordinary_counts_before"
        ]
        adjusted.attrs["cold_start_policy"] = base_metadata
        return adjusted

    planned = planned_graduation_allocations(context, registry)
    allocation_attempts = graduation_allocation_attempts(context, registry)
    base_metadata["planned_graduation_allocations"] = list(planned)
    base_metadata["graduation_allocation_attempts"] = list(allocation_attempts)
    base_metadata["graduation_reassignments"] = []
    ranked_pool = candidate_pool.copy()
    ranked_pool["_cold_score"] = pd.to_numeric(
        ranked_pool.get(score_column, 0.0), errors="coerce"
    ).fillna(float("-inf"))
    ranked_pool = ranked_pool.sort_values(
        ["_cold_score", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).drop(columns=["_cold_score"]).reset_index(drop=True)
    assigned_ids: set[str] = set()
    assigned_active_sets: dict[str, set[frozenset[str]]] = {
        feature_name: set() for feature_name in context.cold_ingredients
    }

    def score_value(row: pd.Series) -> float:
        value = pd.to_numeric(row.get(score_column, float("-inf")), errors="coerce")
        return float(value) if pd.notna(value) else float("-inf")

    def eligible_for_ingredient(row: pd.Series, feature_name: str) -> bool:
        cold = cold_ingredients_in_row(row, registry, context)
        return bool(
            not context.is_exempt_origin(row.get("candidate_origin", ""))
            and len(cold) == 1
            and cold[0] == feature_name
            and bool(row.get("feasibility_pass", True))
        )

    def mark_graduation(position: int, feature_name: str) -> None:
        candidate_id = str(adjusted.iloc[position].get("candidate_id", ""))
        active = _active_ingredient_set(adjusted.iloc[position], registry)
        assigned_ids.add(candidate_id)
        assigned_active_sets[feature_name].add(active)
        adjusted.at[position, "recommendation_type"] = "cold_start_graduation"
        prior_count = context.evidence_counts.get(feature_name, 0)
        explanation = (
            "cold_start_graduation: "
            f"ingredient={feature_name}; prior_distinct_formulations={prior_count}; "
            f"graduation_threshold={policy.minimum_distinct_formulations}"
        )
        prior_value = adjusted.iloc[position].get("selection_explanation", "")
        prior_explanation = (
            "" if pd.isna(prior_value) else str(prior_value).strip()
        )
        adjusted.at[position, "selection_explanation"] = (
            f"{prior_explanation}; {explanation}"
            if prior_explanation
            else explanation
        )
        base_metadata["selected_graduation_allocations"].append(
            {
                "ingredient": feature_name,
                "candidate_id": candidate_id,
                "prior_distinct_formulations": int(prior_count),
                "active_ingredient_set": sorted(active),
            }
        )

    for attempt_index, feature_name in enumerate(allocation_attempts):
        if (
            len(base_metadata["selected_graduation_allocations"])
            >= policy.graduation_slots_per_round
        ):
            break
        existing_positions = [
            position
            for position, (_, row) in enumerate(adjusted.iterrows())
            if str(row.get("candidate_id", "")) not in assigned_ids
            and eligible_for_ingredient(row, feature_name)
            and (
                not policy.graduation_require_distinct_active_sets
                or _active_ingredient_set(row, registry)
                not in assigned_active_sets[feature_name]
            )
        ]
        existing_positions.sort(
            key=lambda position: (
                -score_value(adjusted.iloc[position]),
                str(adjusted.iloc[position].get("candidate_id", "")),
            )
        )
        if existing_positions:
            existing_position = existing_positions[0]
            mark_graduation(existing_position, feature_name)
            if attempt_index >= len(planned):
                base_metadata["graduation_reassignments"].append(
                    {
                        "ingredient": feature_name,
                        "candidate_id": str(
                            adjusted.iloc[existing_position].get("candidate_id", "")
                        ),
                        "reason": "earlier planned graduation slot was unfilled",
                    }
                )
            continue

        selected_ids = set(adjusted["candidate_id"].astype(str))
        inserted = False
        for _, candidate in ranked_pool.iterrows():
            candidate_id = str(candidate.get("candidate_id", ""))
            if not candidate_id or candidate_id in selected_ids:
                continue
            if not eligible_for_ingredient(candidate, feature_name):
                continue
            candidate_active = _active_ingredient_set(candidate, registry)
            if (
                policy.graduation_require_distinct_active_sets
                and candidate_active in assigned_active_sets[feature_name]
            ):
                continue
            candidate_origin = str(candidate.get("candidate_origin", ""))
            loser_positions = [
                position
                for position, (_, row) in enumerate(adjusted.iterrows())
                if str(row.get("candidate_id", "")) not in assigned_ids
                and not context.is_exempt_origin(row.get("candidate_origin", ""))
            ]
            loser_positions.sort(
                key=lambda position: (
                    str(adjusted.iloc[position].get("candidate_origin", ""))
                    != candidate_origin,
                    score_value(adjusted.iloc[position]),
                    str(adjusted.iloc[position].get("candidate_id", "")),
                )
            )
            if policy.preserve_origin_allocation:
                loser_positions = [
                    position
                    for position in loser_positions
                    if str(
                        adjusted.iloc[position].get("candidate_origin", "")
                    )
                    == candidate_origin
                ]
            for loser_position in loser_positions:
                loser = adjusted.iloc[loser_position]
                trial = pd.concat(
                    [
                        adjusted.iloc[:loser_position],
                        pd.DataFrame([candidate]),
                        adjusted.iloc[loser_position + 1 :],
                    ],
                    ignore_index=True,
                )
                if not _cold_start_trial_passes_shared_constraints(
                    trial, registry, optimization_config
                ):
                    continue
                before_counts = _cold_start_ordinary_counts(
                    adjusted, registry, context
                )
                trial_counts = _cold_start_ordinary_counts(
                    trial, registry, context
                )
                if any(
                    trial_counts.get(cold_feature, 0)
                    > max(
                        policy.max_ordinary_rows_per_ingredient,
                        before_counts.get(cold_feature, 0),
                    )
                    for cold_feature in context.cold_ingredients
                ):
                    continue
                same_origin = (
                    str(loser.get("candidate_origin", "")) == candidate_origin
                )
                adjusted = trial
                base_metadata["graduation_replacements"].append(
                    {
                        "ingredient": feature_name,
                        "removed_candidate_id": str(
                            loser.get("candidate_id", "")
                        ),
                        "added_candidate_id": candidate_id,
                        "same_origin": same_origin,
                        "score_change": score_value(candidate)
                        - score_value(loser),
                    }
                )
                if not same_origin:
                    base_metadata["cross_origin_replacement_count"] += 1
                mark_graduation(loser_position, feature_name)
                if attempt_index >= len(planned):
                    base_metadata["graduation_reassignments"].append(
                        {
                            "ingredient": feature_name,
                            "candidate_id": candidate_id,
                            "reason": "earlier planned graduation slot was unfilled",
                        }
                    )
                inserted = True
                break
            if inserted:
                break
        if not inserted:
            base_metadata["graduation_skips"].append(
                {
                    "ingredient": feature_name,
                    "reason": (
                        "no eligible distinct candidate/replacement preserved "
                        "the active slate constraints"
                    ),
                }
            )

    while True:
        counts = _cold_start_ordinary_counts(adjusted, registry, context)
        violations = {
            feature_name: count
            for feature_name, count in counts.items()
            if count > policy.max_ordinary_rows_per_ingredient
        }
        if not violations:
            break
        trigger = sorted(
            violations,
            key=lambda feature_name: (
                -(violations[feature_name] - policy.max_ordinary_rows_per_ingredient),
                registry.feature_names.index(feature_name),
            ),
        )[0]
        offenders = [
            position
            for position, (_, row) in enumerate(adjusted.iterrows())
            if trigger in cold_ingredients_in_row(row, registry, context)
            and not context.is_exempt_origin(row.get("candidate_origin", ""))
            and str(row.get("candidate_id", "")) not in assigned_ids
        ]
        offenders.sort(
            key=lambda position: (
                score_value(adjusted.iloc[position]),
                str(adjusted.iloc[position].get("candidate_id", "")),
            )
        )
        if not offenders:
            raise ValueError(
                "Cold-start cap cannot be satisfied without removing a reserved "
                f"graduation/special row: {trigger}={violations[trigger]}."
            )

        replacement_made = False
        selected_ids = set(adjusted["candidate_id"].astype(str))
        for loser_position in offenders:
            loser = adjusted.iloc[loser_position]
            loser_origin = str(loser.get("candidate_origin", ""))
            for _, candidate in ranked_pool.iterrows():
                candidate_id = str(candidate.get("candidate_id", ""))
                if not candidate_id or candidate_id in selected_ids:
                    continue
                if context.is_exempt_origin(candidate.get("candidate_origin", "")):
                    continue
                if not bool(candidate.get("feasibility_pass", True)):
                    continue
                candidate_origin = str(candidate.get("candidate_origin", ""))
                if policy.preserve_origin_allocation and candidate_origin != loser_origin:
                    continue
                trial = pd.concat(
                    [
                        adjusted.iloc[:loser_position],
                        pd.DataFrame([candidate]),
                        adjusted.iloc[loser_position + 1 :],
                    ],
                    ignore_index=True,
                )
                trial_counts = _cold_start_ordinary_counts(
                    trial, registry, context
                )
                if trial_counts.get(trigger, 0) >= counts.get(trigger, 0):
                    continue
                if any(
                    trial_counts.get(feature_name, 0)
                    > max(
                        policy.max_ordinary_rows_per_ingredient,
                        counts.get(feature_name, 0),
                    )
                    for feature_name in context.cold_ingredients
                ):
                    continue
                if not _cold_start_trial_passes_shared_constraints(
                    trial, registry, optimization_config
                ):
                    continue
                adjusted = trial
                base_metadata["cap_replacements"].append(
                    {
                        "trigger_ingredient": trigger,
                        "count_before": int(counts.get(trigger, 0)),
                        "count_after": int(trial_counts.get(trigger, 0)),
                        "removed_candidate_id": str(
                            loser.get("candidate_id", "")
                        ),
                        "added_candidate_id": candidate_id,
                        "same_origin": candidate_origin == loser_origin,
                        "score_change": score_value(candidate)
                        - score_value(loser),
                    }
                )
                replacement_made = True
                break
            if replacement_made:
                break
        if not replacement_made:
            raise ValueError(
                "Cold-start cap cannot be satisfied from the eligible candidate "
                "pool while preserving the selected origin and diversity rules: "
                f"{trigger}={violations[trigger]}, "
                f"cap={policy.max_ordinary_rows_per_ingredient}."
            )

    final_counts = _cold_start_ordinary_counts(adjusted, registry, context)
    final_violations = {
        feature_name: count
        for feature_name, count in final_counts.items()
        if count > policy.max_ordinary_rows_per_ingredient
    }
    if final_violations:
        raise ValueError(
            "Refusing to freeze a slate that violates the cold-start cap: "
            f"{final_violations}."
        )
    base_metadata["ordinary_counts_after"] = {
        feature_name: int(count) for feature_name, count in final_counts.items()
    }
    base_metadata["graduation_selected_count"] = len(
        base_metadata["selected_graduation_allocations"]
    )
    base_metadata["graduation_skip_count"] = len(
        base_metadata["graduation_skips"]
    )
    base_metadata["cap_replacement_count"] = len(
        base_metadata["cap_replacements"]
    )
    adjusted.attrs["cold_start_policy"] = base_metadata
    return adjusted


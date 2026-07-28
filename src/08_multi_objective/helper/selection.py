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
from .config import nested_get
from .feasibility import (
    annotate_feasibility,
    annotate_support,
    build_support_context,
    feasibility_report,
)
from .models import EndpointModels, train_endpoint_models
from .phase import PHASE_MECHANICS, PHASE_SCREENING, PhaseResolution, resolve_phase_mode
from .penalties import constraint_report, count_active_ingredients
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


@dataclass(frozen=True)
class SelectionResult:
    viability_screen: pd.DataFrame
    mechanical_tests: pd.DataFrame
    candidate_pool: pd.DataFrame
    metadata: dict


def _feature_matrix(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    return frame[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _scaled_matrix(matrix: np.ndarray) -> np.ndarray:
    low = np.nanmin(matrix, axis=0)
    high = np.nanmax(matrix, axis=0)
    spread = np.where((high - low) < 1e-12, 1.0, high - low)
    return (matrix - low) / spread


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

    # Screening exists to find viable formulations; intact-needle formation
    # is a mechanics-phase concern (see _mechanics_phase_scores, which uses
    # the full acquisition_penalty including intact-failure risk). Screening
    # therefore scores purely on predicted viability, net of non-intact
    # soft-constraint penalties. The intact gate is preserved for rescue
    # candidate generation (generate_rescue_candidate_pool), not for
    # filtering or scoring the general screening pool.
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
) -> tuple[np.ndarray, dict]:
    train_frame = models.training_frame.copy()
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
    if acquisition is None:
        mode = "qlognehvi_proxy"
        acquisition = qlognehvi_proxy_scores(
            annotated,
            annotated["viability_ucb"].to_numpy(dtype=float),
            annotated["critical_axial_load_ucb"].to_numpy(dtype=float),
            reference_point=(0.0, 0.0),
        )
    else:
        mode = "qlognehvi_botorch"
    score = acquisition - annotated["acquisition_penalty"].to_numpy(dtype=float)
    metadata = {
        "pool_selection_mode": mode,
        "botorch_available": bool(botorch_available()),
        "botorch_metadata": botorch_metadata,
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

    train_frame = models.training_frame.copy()
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
    campaign_caps = (
        nested_get(optimization_config, "formulation_feasibility.ingredient_caps", {})
        or {}
    )
    upper = np.array(
        [
            min(
                ingredient.upper_bound,
                float(campaign_caps.get(ingredient.feature_name, ingredient.upper_bound)),
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
    )
    generated = annotate_support(generated, registry, support)
    generated = generated.loc[generated["feasibility_pass"].astype(bool)].reset_index(drop=True)
    metadata["continuous_optimizer_used"] = not generated.empty
    metadata["continuous_optimizer_fallback"] = generated.empty
    return generated, metadata


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
) -> pd.DataFrame:
    if phase_resolution.active_phase == PHASE_MECHANICS:
        score_column = "mechanics_phase_score"
        default_recommendation_type = "joint_candidate"
    else:
        score_column = "screening_phase_score"
        default_recommendation_type = "screening_candidate"

    # Retest eligibility is based only on observed campaign evidence. Model
    # uncertainty is the second ranking key after evidence severity and
    # cannot create a retest by itself. Feasibility was applied before this
    # point, so an infeasible diagnostic row cannot consume either slot.
    retest_limit = int(nested_get(optimization_config, "retest.max_candidates_per_round", 2))
    retests = annotated[annotated["recommendation_type"] == "retest_priority"].copy()
    selected_parts: list[pd.DataFrame] = []
    selected_ids: set[str] = set()
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
        if policy_active and phase_resolution.active_phase == PHASE_SCREENING
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

    # Screening-phase candidate selection (the `remaining` pool below) is no
    # longer narrowed by predicted intact-formation probability. Screening
    # is solely about predicted viability; intact-needle formation is
    # assessed once the selector enters the mechanics phase (mechanics_phase_score
    # already accounts for intact-failure risk via acquisition_penalty).
    # The intact gate during screening now only acts through rescue
    # candidates (handled above) and select_mechanical_tests, which stays
    # disabled in the screening phase regardless.
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
    selected["recommendation_type"] = selected["recommendation_type"].replace("", pd.NA).fillna(default_recommendation_type)
    selected.insert(0, "selection_rank", range(1, len(selected) + 1))
    selected["selection_role"] = "round_candidate"
    selected.attrs["shared_pair_replacement_count"] = shared_pair_replacement_count
    selected.attrs["ingredient_frequency_diversity"] = (
        ingredient_frequency_metadata
    )
    return selected


def select_mechanical_tests(
    annotated: pd.DataFrame,
    models: EndpointModels,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    phase_resolution: PhaseResolution,
    n: int,
) -> tuple[pd.DataFrame, dict]:
    threshold = float(nested_get(optimization_config, "round_policy.intact_probability_threshold", 0.50))
    pass_pool = annotated[annotated["intact_patch_pass_probability"] >= threshold].reset_index(drop=True)
    fallback_pool = annotated.reset_index(drop=True)
    pool = pass_pool if not pass_pool.empty else fallback_pool

    mechanical_count = models.mechanical_observation_count
    if phase_resolution.active_phase == PHASE_SCREENING:
        selected = pool.head(0).copy()
        selected.insert(0, "mechanical_selection_rank", pd.Series(dtype=int))
        selected["selection_role"] = "mechanical_test_disabled"
        selected["mechanical_selection_mode"] = "disabled_screening_only"
        metadata = {
            "mechanical_selection_mode": "disabled_screening_only",
            "mechanical_selection_reason": "mechanical recommendations stay off until the selector enters mechanics_enabled",
            "mechanical_observation_count": mechanical_count,
            "intact_probability_threshold": threshold,
            "pass_pool_size": int(len(pass_pool)),
            "botorch_available": bool(botorch_available()),
            "active_phase": phase_resolution.active_phase,
        }
        return selected, metadata

    score, mechanics_metadata = _mechanics_phase_scores(pool, models, registry, optimization_config)
    mode = mechanics_metadata["pool_selection_mode"]
    selected_indices = _greedy_diverse_pick(pool, score, registry.feature_names, n)

    selected = pool.iloc[selected_indices].copy() if selected_indices else pool.head(0).copy()
    selected = selected.head(n).copy()
    selected.insert(0, "mechanical_selection_rank", range(1, len(selected) + 1))
    selected["selection_role"] = "mechanical_test"
    selected["mechanical_selection_mode"] = mode
    metadata = {
        "mechanical_selection_mode": mode,
        "mechanical_observation_count": mechanical_count,
        "intact_probability_threshold": threshold,
        "pass_pool_size": int(len(pass_pool)),
        "botorch_available": bool(botorch_available()),
        "active_phase": phase_resolution.active_phase,
    }
    if phase_resolution.active_phase == PHASE_MECHANICS:
        metadata["botorch_metadata"] = mechanics_metadata["botorch_metadata"]
    return selected, metadata


def select_next_round(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    requested_phase_mode: str | None = None,
    target_round_number: int | None = None,
    policy_active: bool = False,
    similarity_audit: SimilarityAudit | None = None,
) -> SelectionResult:
    models = train_endpoint_models(
        formulations,
        observations,
        registry,
        optimization_config=dict(optimization_config),
    )
    phase_resolution = resolve_phase_mode(
        formulations,
        observations,
        registry,
        optimization_config,
        requested_phase_mode=requested_phase_mode,
    )
    similarity_policy = resolve_similarity_policy(
        optimization_config,
        target_round_number,
    )
    if similarity_audit is None:
        similarity_audit = SimilarityAudit(similarity_policy)
    continuous_metadata = {
        "continuous_optimizer_enabled": False,
        "continuous_optimizer_used": False,
        "continuous_optimizer_fallback": True,
        "continuous_optimizer_reason": "screening_only phase",
    }
    if phase_resolution.active_phase == PHASE_MECHANICS:
        if "candidate_origin" in candidate_pool.columns:
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
    combined_pool, zero_active_filtered_count = _drop_zero_active_candidates(combined_pool, registry)
    annotated = annotate_candidates(
        combined_pool,
        models,
        registry,
        optimization_config,
        policy_active=policy_active,
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
    if phase_resolution.active_phase == PHASE_MECHANICS:
        mechanics_scores, pool_selection_metadata = _mechanics_phase_scores(
            annotated,
            models,
            registry,
            optimization_config,
        )
        annotated["mechanics_phase_score"] = mechanics_scores
    else:
        annotated["mechanics_phase_score"] = np.nan

    n_viability = int(nested_get(optimization_config, "round_policy.viability_screens_per_round", 12))
    n_mechanical = int(nested_get(optimization_config, "round_policy.mechanical_tests_per_round", 4))
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
    )
    mechanical_tests, mechanical_metadata = select_mechanical_tests(
        viability_screen,
        models,
        registry,
        optimization_config,
        phase_resolution,
        n=n_mechanical,
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
    surrogate_config = (
        nested_get(optimization_config, "surrogate_model", {}) or {}
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
    elif continuous_metadata.get("continuous_optimizer_used", False):
        optimizer_mode = "continuous_qlognehvi"
        optimizer_fallback_status = "not_used"
    else:
        optimizer_mode = "finite_pool_fallback"
        optimizer_fallback_status = "used"
    metadata = {
        "viability_screen_count": int(len(viability_screen)),
        "mechanical_test_count": int(len(mechanical_tests)),
        "mechanical_policy": mechanical_metadata,
        "objective_endpoints": ["viability_percent", "critical_axial_load_N_per_needle"],
        "secondary_endpoint": "initial_stiffness_N_per_mm_per_needle",
        "screening_gate": "intact_patch_formation_pass",
        "active_phase": phase_resolution.active_phase,
        "phase_resolution": {
            "requested_phase_mode": phase_resolution.requested_phase_mode,
            "active_phase": phase_resolution.active_phase,
            "paired_observation_count": phase_resolution.paired_observation_count,
            "distinct_formulation_count": phase_resolution.distinct_formulation_count,
            "batch_count": phase_resolution.batch_count,
            "reason": phase_resolution.reason,
            "override_used": phase_resolution.override_used,
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
                "viability_std"
            ),
            "candidate_pool_critical_load_std": uncertainty_summary(
                "critical_axial_load_std"
            ),
            "candidate_pool_initial_stiffness_std": uncertainty_summary(
                "initial_stiffness_std"
            ),
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
    }
    return SelectionResult(
        viability_screen=viability_screen,
        mechanical_tests=mechanical_tests,
        candidate_pool=annotated,
        metadata=metadata,
    )


def _format_candidate_line(row: pd.Series, registry: IngredientRegistry) -> str:
    ingredients = []
    for column in registry.feature_names:
        if column not in row.index:
            continue
        value = row.get(column)
        if pd.isna(value) or float(value) <= 0.0:
            continue
        display_name = registry.get_by_feature(column).display_name
        if column.endswith("_pct"):
            ingredients.append(f"{float(value):.3g}% {display_name}")
        elif float(value) >= 1.0:
            ingredients.append(f"{float(value):.3g}M {display_name}")
        else:
            ingredients.append(f"{float(value) * 1000:.3g}mM {display_name}")
    return " + ".join(ingredients) if ingredients else "No active ingredients"


def _write_summary(
    result: SelectionResult,
    selected: pd.DataFrame,
    output_path: Path,
    registry: IngredientRegistry,
) -> None:
    zero_active_filtered = int(result.metadata.get("candidate_pool_rows_filtered_zero_active_at_entry", 0))
    active_phase = result.metadata.get("active_phase", PHASE_SCREENING)
    mechanical_instruction = (
        "3. Do not use this sheet for mechanical-test recommendations yet; the recommender stays off during screening_only."
        if active_phase == PHASE_SCREENING
        else "3. For rows marked mechanical_test_recommended=true and intact_patch_formation_pass=true, run Instron or enter raw critical load."
    )
    lines = [
        "CryoMN v2 Next-Round Candidate Summary",
        "=" * 42,
        "",
        f"Batch ID: {result.metadata.get('batch_id', '')}",
        f"Active phase: {active_phase}",
        f"Phase reason: {result.metadata.get('phase_resolution', {}).get('reason', '')}",
        f"Candidates to make: {len(selected)}",
        f"Mechanical tests requested: {int(selected['mechanical_test_recommended'].sum())}",
        f"Mechanical selection mode: {result.metadata['mechanical_policy']['mechanical_selection_mode']}",
        f"Mechanical observations in database: {result.metadata['mechanical_policy']['mechanical_observation_count']}",
        f"Retest-priority formulations in slate: {int((selected.get('recommendation_type', pd.Series(dtype=str)) == 'retest_priority').sum())}",
        "",
        "Main database used by selector:",
        "- data/processed_v2/formulations.csv",
        "- data/processed_v2/observations.csv",
        "",
        "Temporary selection restrictions:",
        "- "
        + (
            ", ".join(result.metadata.get("temporary_unavailable_features", []))
            if result.metadata.get("temporary_unavailable_features")
            else "none"
        ),
        "",
        "Wet-lab instructions:",
        "1. Make every formulation listed below.",
        "2. Fill viability_percent and intact_patch_formation_pass in next_round_candidates.csv.",
        mechanical_instruction,
        "4. Run 03_run_round/run_round.py after the CSV is filled.",
        "",
        "Candidates:",
    ]
    if bool(result.metadata.get("formulation_feasibility_policy_active", False)):
        policy_lines = [
            "Forward-only formulation policy:",
            f"- Version: {result.metadata.get('formulation_feasibility_policy_version', '')}",
            f"- Activation round: ROUND_{int(result.metadata.get('formulation_feasibility_policy_start_round', 2)):03d}",
            f"- Support radius: {float(result.metadata.get('support_radius', float('nan'))):.4g}",
            f"- Rejected pool rows: {int(result.metadata.get('candidate_pool_rows_rejected_by_feasibility', 0))}",
            f"- Optimizer mode: {result.metadata.get('optimizer_mode', '')}",
            f"- Fallback status: {result.metadata.get('optimizer_fallback_status', '')}",
            f"- Fallback reason: {result.metadata.get('continuous_qlognehvi', {}).get('continuous_optimizer_reason', 'not applicable')}",
            "",
        ]
        insertion_index = lines.index("Wet-lab instructions:")
        lines[insertion_index:insertion_index] = policy_lines
    similarity_metadata = result.metadata.get("formulation_similarity", {})
    if bool(similarity_metadata.get("enabled", False)):
        final_validation = similarity_metadata.get("final_validation", {})
        similarity_lines = [
            "Formulation similarity policy:",
            f"- Version: {similarity_metadata.get('policy_version', '')}",
            f"- Status: {'active' if similarity_metadata.get('active', False) else 'inactive'}",
            f"- Activation round: ROUND_{int(similarity_metadata.get('start_round', 3)):03d}",
            f"- Bounds-normalized distance threshold: {float(similarity_metadata.get('distance_threshold', 0.05)):.4g}",
            "- Same-single-ingredient minimum relative difference: "
            f"{float(similarity_metadata.get('single_ingredient_min_relative_difference', 0.50)):.0%}",
            f"- Historical references: {int(similarity_metadata.get('history_reference_count', 0))}",
            f"- Rejected generation/pool rows: {int(similarity_metadata.get('rejection_count', 0))}",
            "- Final minimum history distance: "
            f"{final_validation.get('minimum_history_distance')}",
            "- Final minimum within-slate distance: "
            f"{final_validation.get('minimum_within_slate_distance')}",
            "",
        ]
        insertion_index = lines.index("Wet-lab instructions:")
        lines[insertion_index:insertion_index] = similarity_lines
    ingredient_frequency_metadata = result.metadata.get(
        "ingredient_frequency_diversity",
        {},
    )
    if bool(ingredient_frequency_metadata.get("active", False)):
        frequency_lines = [
            "Marginal ingredient-frequency policy:",
            f"- Version: {ingredient_frequency_metadata.get('policy_version', '')}",
            f"- Maximum selected rows per ingredient: {int(ingredient_frequency_metadata.get('max_rows_per_ingredient', 5))}",
            "- Retest and rescue rows count toward the limit but are protected from removal.",
            f"- Frequency-driven replacements: {int(ingredient_frequency_metadata.get('replacement_count', 0))}",
            f"- Final maximum ingredient frequency: {int(ingredient_frequency_metadata.get('maximum_ingredient_frequency', 0))}",
            "",
        ]
        insertion_index = lines.index("Wet-lab instructions:")
        lines[insertion_index:insertion_index] = frequency_lines
    if zero_active_filtered:
        warning_lines = [
            "Warnings:",
            "- "
            f"{zero_active_filtered} zero-active candidate-pool rows were removed before scoring.",
            "- Review the supplied candidate pool or upstream candidate-generation logic.",
            "",
        ]
        insertion_index = lines.index("Wet-lab instructions:")
        lines[insertion_index:insertion_index] = warning_lines
    display_columns = [
        "selection_rank",
        "candidate_id",
        "formulation_id",
        "mechanical_test_recommended",
        "predicted_viability_percent",
        "intact_patch_pass_probability",
        "predicted_critical_axial_load_N_per_needle",
        "active_ingredient_count",
    ]
    for _, row in selected.iterrows():
        parts = [
            f"#{int(row['selection_rank'])}",
            f"candidate_id={row['candidate_id']}",
            f"formulation_id={row['formulation_id']}",
            f"recommendation_type={row.get('recommendation_type', '')}",
            f"mechanical_test={bool(row['mechanical_test_recommended'])}",
            f"predicted_viability={float(row['predicted_viability_percent']):.1f}%",
            f"intact_probability={float(row['intact_patch_pass_probability']):.2f}",
        ]
        if bool(result.metadata.get("formulation_feasibility_policy_active", False)):
            parts.extend(
                [
                    f"origin={row.get('candidate_origin', 'finite_pool_fallback')}",
                    f"support={row.get('support_status', 'not_evaluated')}",
                ]
            )
        if result.metadata["mechanical_policy"]["mechanical_observation_count"] > 0 and "predicted_critical_axial_load_N_per_needle" in row and pd.notna(
            row["predicted_critical_axial_load_N_per_needle"]
        ):
            parts.append(
                "predicted_critical_load="
                f"{float(row['predicted_critical_axial_load_N_per_needle']):.3g} N/needle"
            )
        lines.append("- " + "; ".join(parts))
        lines.append(f"  formulation: {_format_candidate_line(row, registry)}")
        if str(row.get("selection_explanation", "")).strip():
            lines.append(f"  note: {row['selection_explanation']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selection_result(
    result: SelectionResult,
    output_dir: str | Path,
    batch_id: str = "",
    total_candidate_pool_path: str | Path | None = None,
    registry: IngredientRegistry | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if registry is None:
        registry = IngredientRegistry.from_config()
    selected = result.viability_screen.copy()
    selected["mechanical_test_recommended"] = selected["candidate_id"].isin(
        set(result.mechanical_tests["candidate_id"]) if not result.mechanical_tests.empty else set()
    )
    selected["mechanical_selection_rank"] = ""
    selected["mechanical_selection_mode"] = ""
    if not result.mechanical_tests.empty:
        rank_map = result.mechanical_tests.set_index("candidate_id")["mechanical_selection_rank"].to_dict()
        mode_map = result.mechanical_tests.set_index("candidate_id")["mechanical_selection_mode"].to_dict()
        selected["mechanical_selection_rank"] = selected["candidate_id"].map(rank_map).fillna("")
        selected["mechanical_selection_mode"] = selected["candidate_id"].map(mode_map).fillna("")

    wetlab_result_columns = [
        "formulation_id",
        "candidate_id",
        "selection_rank",
        "recommendation_type",
        "selection_explanation",
        "mechanical_test_recommended",
        "mechanical_selection_rank",
        "mechanical_selection_mode",
        "batch_id",
        "replicate_id",
        "viability_percent",
        "intact_patch_formation_pass",
        "no_slurry",
        "no_collapse",
        "intact_tip_count",
        "total_tip_count",
        "instron_file",
        "needles_compressed",
        "critical_axial_load_N_per_needle",
        "critical_axial_load_N_total",
        "initial_stiffness_N_per_mm_per_needle",
        "notes",
    ]
    if bool(result.metadata.get("formulation_feasibility_policy_active", False)):
        preparation_columns = [
            "preparation_feasibility_pass",
            "homogeneous_solution_pass",
            "fillability_pass",
            "preparation_failure_reason",
        ]
        notes_index = wetlab_result_columns.index("notes")
        wetlab_result_columns[notes_index:notes_index] = preparation_columns
    selected["batch_id"] = batch_id
    for column in wetlab_result_columns:
        if column not in selected.columns:
            selected[column] = ""
    for column in EDITABLE_WETLAB_COLUMNS:
        if column in selected.columns:
            selected[column] = ""
    result.metadata["batch_id"] = batch_id
    forward_diagnostic_columns = [
        *registry.feature_names,
        "active_ingredient_count",
        "candidate_origin",
        "rescue_scale_factor",
        "rescue_anchor_formulation_id",
        "rescue_anchor_viability_percent",
        "active_polymer_count",
        "total_polymer_pct",
        "total_serum_protein_pct",
        "total_polymer_serum_pct",
        "total_sugar_M",
        "total_nonpermeating_solute_M",
        "estimated_small_solute_g_L",
        "feasibility_pass",
        "feasibility_reasons",
        "nearest_support_distance",
        "support_status",
        "source",
        "source_row_id",
        "formulation_label",
        "source_type",
        "predicted_viability_percent",
        "viability_std",
        "predicted_critical_axial_load_N_per_needle",
        "critical_axial_load_std",
        "intact_patch_pass_probability",
        "same_formulation_range",
        "local_neighbor_residual",
        "retest_priority_score",
        "viability_ucb",
        "critical_axial_load_ucb",
        "predicted_initial_stiffness_N_per_mm_per_needle",
        "initial_stiffness_std",
        "preparation_feasibility_probability",
        "active_ingredient_excess_above_8",
        "single_molar_excess_features",
        "single_molar_excess_total_M",
        "intact_failure_probability",
        "acquisition_penalty",
        "screening_acquisition_penalty",
        "screening_phase_score",
        "mechanics_phase_score",
        "selection_role",
    ]
    if bool(result.metadata.get("formulation_feasibility_policy_active", False)):
        for column in forward_diagnostic_columns:
            if column not in selected.columns:
                selected[column] = ""
        csv_columns = wetlab_result_columns + forward_diagnostic_columns + [
            column
            for column in selected.columns
            if column not in wetlab_result_columns
            and column not in forward_diagnostic_columns
        ]
    else:
        csv_columns = wetlab_result_columns + [
            column
            for column in selected.columns
            if column not in wetlab_result_columns
        ]
    selected[csv_columns].to_csv(output / "next_round_candidates.csv", index=False)

    total_pool = result.candidate_pool.copy()
    total_pool["batch_id"] = batch_id
    total_pool["active_phase"] = result.metadata.get("active_phase", PHASE_SCREENING)
    if bool(result.metadata.get("formulation_feasibility_policy_active", False)):
        total_pool["formulation_feasibility_policy_active"] = True
        total_pool["formulation_feasibility_policy_version"] = result.metadata.get(
            "formulation_feasibility_policy_version",
            "",
        )
        total_pool["formulation_feasibility_policy_start_round"] = result.metadata.get(
            "formulation_feasibility_policy_start_round",
            "",
        )
        total_pool["optimizer_mode"] = result.metadata.get("optimizer_mode", "")
        total_pool["optimizer_fallback_status"] = result.metadata.get(
            "optimizer_fallback_status",
            "",
        )
    total_pool["selected_for_viability_screen"] = total_pool["candidate_id"].isin(
        set(result.viability_screen["candidate_id"])
    )
    total_pool["selected_for_mechanical_test"] = total_pool["candidate_id"].isin(
        set(result.mechanical_tests["candidate_id"]) if not result.mechanical_tests.empty else set()
    )
    rank_map = result.viability_screen.set_index("candidate_id")["selection_rank"].to_dict()
    total_pool["selection_rank"] = total_pool["candidate_id"].map(rank_map).fillna("")
    total_pool["mechanical_selection_rank"] = ""
    total_pool["mechanical_selection_mode"] = ""
    if not result.mechanical_tests.empty:
        mech_rank_map = result.mechanical_tests.set_index("candidate_id")[
            "mechanical_selection_rank"
        ].to_dict()
        mech_mode_map = result.mechanical_tests.set_index("candidate_id")[
            "mechanical_selection_mode"
        ].to_dict()
        total_pool["mechanical_selection_rank"] = (
            total_pool["candidate_id"].map(mech_rank_map).fillna("")
        )
        total_pool["mechanical_selection_mode"] = (
            total_pool["candidate_id"].map(mech_mode_map).fillna("")
        )
    total_pool_output = (
        Path(total_candidate_pool_path)
        if total_candidate_pool_path is not None
        else output.parent / "total_candidate_pool.csv"
    )
    total_pool_output.parent.mkdir(parents=True, exist_ok=True)
    total_pool.to_csv(total_pool_output, index=False)
    _write_summary(result, selected, output / "next_round_summary.txt", registry=registry)
    if bool(result.metadata.get("formulation_feasibility_policy_active", False)):
        metadata_path = output / "next_round_metadata.json"
        metadata_path.write_text(
            json.dumps(
                result.metadata,
                indent=2,
                default=lambda value: value.item() if hasattr(value, "item") else str(value),
            )
            + "\n",
            encoding="utf-8",
        )

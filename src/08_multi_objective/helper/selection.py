"""Candidate scoring and next-batch selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .acquisition import botorch_available, minmax, qlognehvi_proxy_scores, try_botorch_qlognehvi_scores
from .config import nested_get
from .models import EndpointModels, train_endpoint_models
from .phase import PHASE_MECHANICS, PHASE_SCREENING, PhaseResolution, resolve_phase_mode
from .penalties import constraint_report, count_active_ingredients
from .registry import IngredientRegistry
from .retest import build_retest_candidates


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


def _greedy_diverse_pick(frame: pd.DataFrame, score: np.ndarray, feature_names: list[str], n: int) -> list[int]:
    if frame.empty or n <= 0:
        return []
    n = min(n, len(frame))
    matrix = _scaled_matrix(_feature_matrix(frame, feature_names))
    selected = [int(np.nanargmax(score))]
    while len(selected) < n:
        remaining = [index for index in range(len(frame)) if index not in selected]
        distances = np.linalg.norm(matrix[remaining, None, :] - matrix[selected][None, :, :], axis=2)
        min_distances = np.min(distances, axis=1)
        combined = minmax(score[remaining]) + 0.10 * minmax(min_distances)
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
) -> pd.DataFrame:
    annotated = candidates.copy()
    x = _feature_matrix(annotated, registry.feature_names)

    viability = models.viability.predict(x)
    critical_load = models.critical_load.predict(x)
    stiffness = models.initial_stiffness.predict(x)
    intact_probability = models.intact.predict_proba(x)

    kappa_v = float(nested_get(optimization_config, "selection.viability_ucb_kappa", 0.35))
    kappa_m = float(nested_get(optimization_config, "selection.mechanical_ucb_kappa", 0.50))

    annotated["predicted_viability_percent"] = viability.mean
    annotated["viability_std"] = viability.std
    annotated["viability_ucb"] = viability.mean + kappa_v * viability.std
    annotated["predicted_critical_axial_load_N_per_needle"] = critical_load.mean
    annotated["critical_axial_load_std"] = critical_load.std
    annotated["critical_axial_load_ucb"] = critical_load.mean + kappa_m * critical_load.std
    annotated["predicted_initial_stiffness_N_per_mm_per_needle"] = stiffness.mean
    annotated["initial_stiffness_std"] = stiffness.std
    annotated["intact_patch_pass_probability"] = np.clip(intact_probability, 0.0, 1.0)

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

    viability_weight = float(nested_get(optimization_config, "screening_phase.viability_weight", 0.75))
    intact_weight = float(nested_get(optimization_config, "screening_phase.intact_weight", 0.25))
    annotated["screening_phase_score"] = (
        viability_weight * minmax(annotated["viability_ucb"].to_numpy(dtype=float))
        + intact_weight * minmax(annotated["intact_patch_pass_probability"].to_numpy(dtype=float))
        - annotated["acquisition_penalty"].to_numpy(dtype=float)
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


def _select_round_slate(
    annotated: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    phase_resolution: PhaseResolution,
    n: int,
) -> pd.DataFrame:
    if phase_resolution.active_phase == PHASE_MECHANICS:
        score_column = "mechanics_phase_score"
        default_recommendation_type = "joint_candidate"
    else:
        score_column = "screening_phase_score"
        default_recommendation_type = "screening_candidate"

    retest_limit = int(nested_get(optimization_config, "retest.max_candidates_per_round", 2))
    retests = annotated[annotated["recommendation_type"] == "retest_priority"].copy()
    selected_parts: list[pd.DataFrame] = []
    selected_ids: set[str] = set()
    if not retests.empty and retest_limit > 0:
        selected_retests = retests.sort_values("retest_priority_score", ascending=False).head(min(retest_limit, n)).copy()
        selected_parts.append(selected_retests)
        selected_ids = set(selected_retests["candidate_id"].astype(str))

    remaining = annotated.loc[~annotated["candidate_id"].astype(str).isin(selected_ids)].reset_index(drop=True)
    remaining_n = max(n - sum(len(part) for part in selected_parts), 0)
    if remaining_n > 0 and not remaining.empty:
        selected_indices = _greedy_diverse_pick(
            remaining,
            remaining[score_column].to_numpy(dtype=float),
            registry.feature_names,
            n=remaining_n,
        )
        if selected_indices:
            selected_parts.append(remaining.iloc[selected_indices].copy())

    if selected_parts:
        selected = pd.concat(selected_parts, ignore_index=True)
    else:
        selected = annotated.head(0).copy()

    selected["recommendation_type"] = selected["recommendation_type"].replace("", pd.NA).fillna(default_recommendation_type)
    selected.insert(0, "selection_rank", range(1, len(selected) + 1))
    selected["selection_role"] = "round_candidate"
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
        mode = "screening_data_collection"
        seed_score = (
            minmax(pool["viability_ucb"].to_numpy(dtype=float))
            + minmax(pool["intact_patch_pass_probability"].to_numpy(dtype=float))
        )
        selected_indices = _kcenter_pick(pool, seed_score, registry.feature_names, n)
    else:
        score, mechanics_metadata = _mechanics_phase_scores(pool, models, registry, optimization_config)
        mode = mechanics_metadata["pool_selection_mode"]
        selected_indices = _greedy_diverse_pick(pool, score, registry.feature_names, n)

    selected = pool.iloc[selected_indices].copy() if selected_indices else pool.head(0).copy()
    selected = selected.head(n).copy()
    selected.insert(0, "mechanical_selection_rank", range(1, len(selected) + 1))
    selected["selection_role"] = (
        "mechanical_data_collection" if phase_resolution.active_phase == PHASE_SCREENING else "mechanical_test"
    )
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
) -> SelectionResult:
    models = train_endpoint_models(formulations, observations, registry)
    phase_resolution = resolve_phase_mode(
        formulations,
        observations,
        registry,
        optimization_config,
        requested_phase_mode=requested_phase_mode,
    )
    retest_candidates = build_retest_candidates(
        formulations,
        observations,
        models,
        registry,
        optimization_config,
    )
    combined_pool = candidate_pool.copy()
    if not retest_candidates.empty:
        combined_pool = pd.concat([combined_pool, retest_candidates], ignore_index=True, sort=False)
        combined_pool = combined_pool.drop_duplicates("candidate_id", keep="first")
    combined_pool, zero_active_filtered_count = _drop_zero_active_candidates(combined_pool, registry)
    annotated = annotate_candidates(combined_pool, models, registry, optimization_config)
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

    viability_screen = _select_round_slate(
        annotated,
        registry,
        optimization_config,
        phase_resolution,
        n=n_viability,
    )
    mechanical_tests, mechanical_metadata = select_mechanical_tests(
        viability_screen,
        models,
        registry,
        optimization_config,
        phase_resolution,
        n=n_mechanical,
    )
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
        "retest_candidate_count": int((annotated["recommendation_type"] == "retest_priority").sum()),
        "zero_active_candidate_count_filtered": zero_active_filtered_count,
    }
    return SelectionResult(
        viability_screen=viability_screen,
        mechanical_tests=mechanical_tests,
        candidate_pool=annotated,
        metadata=metadata,
    )


def _format_candidate_line(row: pd.Series, registry: IngredientRegistry) -> str:
    ingredients = []
    for column, value in row.items():
        if not (column.endswith("_M") or column.endswith("_pct")):
            continue
        if pd.isna(value) or float(value) <= 0.0:
            continue
        display_name = registry.get_by_feature(column).display_name if column in registry.feature_names else column
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
    lines = [
        "CryoMN v2 Next-Round Candidate Summary",
        "=" * 42,
        "",
        f"Batch ID: {result.metadata.get('batch_id', '')}",
        f"Active phase: {result.metadata.get('active_phase', PHASE_SCREENING)}",
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
        "3. For rows marked mechanical_test_recommended=true and intact_patch_formation_pass=true, run Instron or enter raw critical load.",
        "4. Run 03_run_round/run_round.py after the CSV is filled.",
        "",
        "Candidates:",
    ]
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
    selected["batch_id"] = batch_id
    selected["replicate_id"] = ""
    for column in wetlab_result_columns:
        if column not in selected.columns:
            selected[column] = ""
    result.metadata["batch_id"] = batch_id
    csv_columns = wetlab_result_columns + [
        column for column in selected.columns if column not in wetlab_result_columns
    ]
    selected[csv_columns].to_csv(output / "next_round_candidates.csv", index=False)

    total_pool = result.candidate_pool.copy()
    total_pool["batch_id"] = batch_id
    total_pool["active_phase"] = result.metadata.get("active_phase", PHASE_SCREENING)
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

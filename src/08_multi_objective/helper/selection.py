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
from .penalties import constraint_report
from .registry import IngredientRegistry


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

    annotated["viability_selection_score"] = (
        minmax(annotated["viability_ucb"].to_numpy())
        - annotated["acquisition_penalty"].to_numpy(dtype=float)
    )
    return annotated


def select_viability_screen(
    annotated: pd.DataFrame,
    registry: IngredientRegistry,
    n: int,
) -> pd.DataFrame:
    selected_indices = _greedy_diverse_pick(
        annotated.reset_index(drop=True),
        annotated["viability_selection_score"].to_numpy(dtype=float),
        registry.feature_names,
        n=n,
    )
    selected = annotated.reset_index(drop=True).iloc[selected_indices].copy()
    selected.insert(0, "selection_rank", range(1, len(selected) + 1))
    selected["selection_role"] = "viability_screen"
    return selected


def select_mechanical_tests(
    annotated: pd.DataFrame,
    models: EndpointModels,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    n: int,
) -> tuple[pd.DataFrame, dict]:
    threshold = float(nested_get(optimization_config, "round_policy.intact_probability_threshold", 0.50))
    startup_threshold = int(
        nested_get(optimization_config, "round_policy.mechanics_startup_observation_threshold", 8)
    )
    allow_exception = bool(nested_get(optimization_config, "round_policy.allow_one_novelty_exception", True))
    pass_pool = annotated[annotated["intact_patch_pass_probability"] >= threshold].reset_index(drop=True)
    fallback_pool = annotated.reset_index(drop=True)
    pool = pass_pool if not pass_pool.empty else fallback_pool

    mechanical_count = models.mechanical_observation_count
    if mechanical_count < startup_threshold:
        mode = "k_center_cold_start"
        seed_score = (
            minmax(pool["viability_ucb"].to_numpy(dtype=float))
            + minmax(pool["intact_patch_pass_probability"].to_numpy(dtype=float))
        )
        selected_indices = _kcenter_pick(pool, seed_score, registry.feature_names, n)
    else:
        train_frame = models.training_frame.copy()
        for objective_column in ["viability_percent", "critical_axial_load_N_per_needle"]:
            if objective_column not in train_frame.columns:
                train_frame[objective_column] = np.nan
        paired = train_frame[
            train_frame[["viability_percent", "critical_axial_load_N_per_needle"]].notna().all(axis=1)
        ].copy()
        train_x = _feature_matrix(paired, registry.feature_names) if not paired.empty else np.empty((0, len(registry.feature_names)))
        train_y = (
            paired[["viability_percent", "critical_axial_load_N_per_needle"]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
            if not paired.empty
            else np.empty((0, 2))
        )
        candidate_x = _feature_matrix(pool, registry.feature_names)
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
                pool,
                pool["viability_ucb"].to_numpy(dtype=float),
                pool["critical_axial_load_ucb"].to_numpy(dtype=float),
                reference_point=(0.0, 0.0),
            )
        else:
            mode = "qlognehvi_botorch"
        score = acquisition - pool["acquisition_penalty"].to_numpy(dtype=float)
        selected_indices = _greedy_diverse_pick(pool, score, registry.feature_names, n)

    selected = pool.iloc[selected_indices].copy() if selected_indices else pool.head(0).copy()

    if allow_exception and len(selected) < n and len(fallback_pool) > len(selected):
        already = set(selected["candidate_id"]) if "candidate_id" in selected else set()
        exception_floor = float(
            nested_get(optimization_config, "selection.novelty_exception_probability_floor", 0.25)
        )
        exception_pool = fallback_pool[
            (~fallback_pool["candidate_id"].isin(already))
            & (fallback_pool["intact_patch_pass_probability"] >= exception_floor)
        ].copy()
        if not exception_pool.empty:
            exception_pool["novelty_exception_score"] = minmax(
                exception_pool["viability_ucb"].to_numpy(dtype=float)
            ) - exception_pool["acquisition_penalty"].to_numpy(dtype=float)
            selected = pd.concat(
                [selected, exception_pool.sort_values("novelty_exception_score", ascending=False).head(n - len(selected))],
                ignore_index=True,
            )

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
    }
    if mechanical_count >= startup_threshold:
        metadata["botorch_metadata"] = botorch_metadata
    return selected, metadata


def select_next_round(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
) -> SelectionResult:
    models = train_endpoint_models(formulations, observations, registry)
    annotated = annotate_candidates(candidate_pool, models, registry, optimization_config)

    n_viability = int(nested_get(optimization_config, "round_policy.viability_screens_per_round", 12))
    n_mechanical = int(nested_get(optimization_config, "round_policy.mechanical_tests_per_round", 4))

    viability_screen = select_viability_screen(annotated, registry, n=n_viability)
    mechanical_tests, mechanical_metadata = select_mechanical_tests(
        viability_screen,
        models,
        registry,
        optimization_config,
        n=n_mechanical,
    )
    metadata = {
        "viability_screen_count": int(len(viability_screen)),
        "mechanical_test_count": int(len(mechanical_tests)),
        "mechanical_policy": mechanical_metadata,
        "objective_endpoints": ["viability_percent", "critical_axial_load_N_per_needle"],
        "secondary_endpoint": "initial_stiffness_N_per_mm_per_needle",
        "screening_gate": "intact_patch_formation_pass",
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
    lines = [
        "CryoMN v2 Next-Round Candidate Summary",
        "=" * 42,
        "",
        f"Batch ID: {result.metadata.get('batch_id', '')}",
        f"Candidates to make: {len(selected)}",
        f"Mechanical tests requested: {int(selected['mechanical_test_recommended'].sum())}",
        f"Mechanical selection mode: {result.metadata['mechanical_policy']['mechanical_selection_mode']}",
        f"Mechanical observations in database: {result.metadata['mechanical_policy']['mechanical_observation_count']}",
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
        "4. Run 03_record_results/update_from_results.py after the CSV is filled.",
        "",
        "Candidates:",
    ]
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

"""Calculation-only evidence preparation and metrics for V2 reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

from .artifacts import round_artifact_paths, validate_completed_against_proposal
from .config import load_optimization_config
from .endpoints import (
    INTACT_PATCH_ENDPOINT,
    aggregate_intact_patch_replicates,
    parse_bool,
)
from .feasibility import annotate_feasibility
from .models import build_training_frame, train_endpoint_models, paired_objective_frame
from .paths import RESULTS_V2_DIR
from .registry import IngredientRegistry

PROSPECTIVE_TABLE_COLUMNS = [
    "evaluation_policy_version",
    "mechanical_definition_id",
    "round_id",
    "round_number",
    "provenance_class",
    "formal_cohort",
    "active_phase",
    "candidate_id",
    "formulation_id",
    "recommendation_type",
    "selection_rank",
    "endpoint",
    "endpoint_role",
    "metric_type",
    "prediction_mean",
    "prediction_std",
    "observed_mean",
    "observed_unit",
    "replicate_count",
    "completed_row_count",
    "evaluation_eligible",
    "exclusion_reason",
    "formal_metric_eligible",
    "formal_exclusion_reason",
    "signed_error",
    "absolute_error",
    "squared_error",
    "standardized_residual",
    "interval_95_lower",
    "interval_95_upper",
    "interval_95_covered",
    "brier_score",
    "classification_correct",
]

METRIC_COLUMNS = [
    "mechanical_definition_id",
    "scope",
    "round_id",
    "provenance_class",
    "formal_cohort",
    "endpoint",
    "endpoint_role",
    "n_proposed",
    "n_evaluated",
    "completion_rate",
    "mae",
    "rmse",
    "bias",
    "r2",
    "interval_95_coverage",
    "interval_95_mean_width",
    "interval_95_median_width",
    "brier_score",
    "accuracy",
]

def _round_sort_key(batch_id: object) -> tuple[int, str]:
    value = str(batch_id).strip()
    if value.startswith("ROUND_"):
        suffix = value.removeprefix("ROUND_")
        if suffix.isdigit():
            return (0, f"{int(suffix):09d}")
    return (1, value)


def _boolean_numeric(value: object) -> float:
    parsed = parse_bool(value)
    return np.nan if parsed is None else float(parsed)


def _observed_endpoint_frame(formulations: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    from .group10_config import production_observations
    observations = production_observations(observations)
    if formulations.empty:
        return pd.DataFrame()
    frame = formulations.copy()
    if observations.empty:
        return frame

    obs = observations.copy()
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    pivot = obs.pivot_table(
        index="formulation_id",
        columns="endpoint",
        values="value",
        aggfunc="mean",
    )
    intact_observations = obs[obs["endpoint"].astype(str) == INTACT_PATCH_ENDPOINT]
    if not intact_observations.empty:
        pivot[INTACT_PATCH_ENDPOINT] = (
            intact_observations.groupby("formulation_id")["value"]
            .agg(aggregate_intact_patch_replicates)
        )
    frame = frame.merge(pivot, on="formulation_id", how="left")

    batch_map = (
        obs.groupby("formulation_id")["batch_id"]
        .agg(lambda values: ", ".join(sorted({str(v).strip() for v in values if str(v).strip()})))
        .rename("observed_batches")
    )
    source_map = (
        obs.groupby("formulation_id")["source_type"]
        .agg(lambda values: ", ".join(sorted({str(v).strip() for v in values if str(v).strip()})))
        .rename("observed_sources")
    )
    frame = frame.merge(batch_map, on="formulation_id", how="left")
    frame = frame.merge(source_map, on="formulation_id", how="left")
    return frame


def _paired_frame(formulations: pd.DataFrame, observations: pd.DataFrame, registry: IngredientRegistry) -> pd.DataFrame:
    frame = paired_objective_frame(build_training_frame(formulations, observations, registry))
    required = ["formulation_id", "batch_id", "viability_percent", "critical_axial_load_N_per_needle"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()
    paired = frame.dropna(subset=["viability_percent", "critical_axial_load_N_per_needle"]).copy()
    if paired.empty:
        return paired
    paired["batch_id"] = paired["batch_id"].fillna("").astype(str)
    paired["round_order"] = paired["batch_id"].map(_round_sort_key)
    return paired.sort_values(["round_order", "formulation_id"]).reset_index(drop=True)


def _pareto_frontier_mask(frame: pd.DataFrame, x_col: str, y_col: str) -> np.ndarray:
    values = frame[[x_col, y_col]].to_numpy(dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for i, current in enumerate(values):
        for j, other in enumerate(values):
            if i == j:
                continue
            if np.all(other >= current) and np.any(other > current):
                keep[i] = False
                break
    return keep


def _top_candidate_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    frame = candidates.copy()
    if "selection_rank" in frame.columns:
        frame["selection_rank"] = pd.to_numeric(frame["selection_rank"], errors="coerce")
        frame = frame.sort_values("selection_rank", ascending=True, na_position="last")
    else:
        frame["predicted_viability_percent"] = pd.to_numeric(
            frame.get("predicted_viability_percent"),
            errors="coerce",
        )
        frame = frame.sort_values(
            "predicted_viability_percent",
            ascending=False,
            na_position="last",
        )
    return frame


def _aggregate_completed_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Collapse a completed technical-replicate worksheet to its 12 candidates."""
    if candidates.empty or "candidate_id" not in candidates.columns:
        return candidates.copy()
    frame = candidates.copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    grouping = ["candidate_id"]
    if "selection_rank" in frame.columns:
        grouping.append("selection_rank")

    base = frame.groupby(grouping, dropna=False, sort=False).first().reset_index()
    if "viability_percent" in frame.columns:
        viability = (
            frame.assign(
                viability_percent=pd.to_numeric(
                    frame["viability_percent"],
                    errors="coerce",
                )
            )
            .groupby(grouping, dropna=False, sort=False)["viability_percent"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "completed_viability_mean",
                    "std": "completed_viability_sd",
                    "count": "completed_viability_replicate_count",
                }
            )
        )
        viability["completed_viability_sd"] = viability[
            "completed_viability_sd"
        ].fillna(0.0)
        base = base.merge(viability, on=grouping, how="left")
    if "intact_patch_formation_pass" in frame.columns:
        intact = (
            frame.assign(
                _intact_numeric=frame["intact_patch_formation_pass"].map(
                    _boolean_numeric
                )
            )
            .groupby(grouping, dropna=False, sort=False)["_intact_numeric"]
            .agg(
                completed_intact_pass_fraction="mean",
                completed_intact_replicate_count="count",
                completed_intact_gate_pass=aggregate_intact_patch_replicates,
            )
            .reset_index()
        )
        base = base.merge(intact, on=grouping, how="left")
    return _top_candidate_frame(base).reset_index(drop=True)


def _campaign_and_literature_observed(
    observed: pd.DataFrame,
    registry: IngredientRegistry,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if observed.empty:
        return observed.copy(), observed.copy()
    sources = observed.get(
        "observed_sources",
        pd.Series("", index=observed.index),
    ).fillna("").astype(str)
    campaign = observed.loc[
        sources.str.split(", ").apply(
            lambda values: "wetlab_feedback" in set(values)
        )
    ].copy()
    if not campaign.empty:
        campaign = annotate_feasibility(
            campaign,
            registry,
            load_optimization_config(),
            policy_active=True,
        )
        campaign = campaign.loc[
            campaign["feasibility_pass"].astype(bool)
        ].reset_index(drop=True)
    literature = observed.loc[
        sources.str.split(", ").apply(
            lambda values: "legacy_literature" in set(values)
        )
    ].copy()
    return campaign, literature


def _build_model_evaluation_frames(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
) -> dict[str, pd.DataFrame]:
    return {
        "viability_percent": _cross_validated_predictions(
            formulations,
            observations,
            registry,
            "viability_percent",
        ),
        "critical_axial_load_N_per_needle": _cross_validated_predictions(
            formulations,
            observations,
            registry,
            "critical_axial_load_N_per_needle",
        ),
        "intact_patch_formation_pass": _cross_validated_predictions(
            formulations,
            observations,
            registry,
            "intact_patch_formation_pass",
        ),
    }


def _cross_validated_predictions(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
    endpoint: str,
) -> pd.DataFrame:
    frame = build_training_frame(formulations, observations, registry)
    if frame.empty or endpoint not in frame.columns:
        return pd.DataFrame()
    valid = frame.dropna(subset=[endpoint]).copy()
    if len(valid) < 2:
        return pd.DataFrame()

    groups = valid["formulation_id"].astype(str)
    n_splits = min(5, int(groups.nunique()))
    if n_splits < 2:
        return pd.DataFrame()

    predictions: list[pd.DataFrame] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold_number, (_, test_index) in enumerate(
        splitter.split(valid, groups=groups),
        start=1,
    ):
        test_rows = valid.iloc[test_index].copy()
        holdout_formulation_ids = set(
            test_rows["formulation_id"].astype(str)
        )
        train_observations = observations.copy()
        endpoint_mask = train_observations["endpoint"].astype(str) == endpoint
        holdout_mask = (
            train_observations["formulation_id"]
            .astype(str)
            .isin(holdout_formulation_ids)
        )
        train_observations = train_observations.loc[~(endpoint_mask & holdout_mask)].copy()
        models = train_endpoint_models(formulations, train_observations, registry)
        x_test = (
            test_rows[registry.feature_names]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        if endpoint == "viability_percent":
            prediction = models.viability.predict(x_test)
            predicted = prediction.mean
            predicted_std = prediction.std
        elif endpoint == "critical_axial_load_N_per_needle":
            prediction = models.critical_load.predict(x_test)
            predicted = prediction.mean
            predicted_std = prediction.std
        elif endpoint == "intact_patch_formation_pass":
            predicted = models.intact.predict_proba(x_test)
            predicted_std = np.full(len(predicted), np.nan, dtype=float)
        else:
            continue

        predictions.append(
            pd.DataFrame(
                {
                    "formulation_id": test_rows["formulation_id"].to_numpy(),
                    "batch_id": test_rows.get("batch_id", pd.Series([""] * len(test_rows))).to_numpy(),
                    "actual": pd.to_numeric(test_rows[endpoint], errors="coerce").to_numpy(dtype=float),
                    "predicted": np.asarray(predicted, dtype=float),
                    "predicted_std": np.asarray(predicted_std, dtype=float),
                    "cv_fold": fold_number,
                }
            )
        )
    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def _normalize_frame(frame: pd.DataFrame, columns: list[str], reference: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        low = float(reference[column].min())
        high = float(reference[column].max())
        spread = max(high - low, 1e-9)
        normalized[column] = (pd.to_numeric(frame[column], errors="coerce") - low) / spread
    return normalized


def _hypervolume_2d(frontier: pd.DataFrame, x_col: str, y_col: str) -> float:
    if frontier.empty:
        return 0.0
    ordered = frontier.sort_values(x_col)
    hv = 0.0
    previous_x = 0.0
    for _, row in ordered.iterrows():
        x_value = max(float(row[x_col]), previous_x)
        y_value = max(float(row[y_col]), 0.0)
        hv += max(x_value - previous_x, 0.0) * y_value
        previous_x = x_value
    return float(hv)


def _igd_2d(true_frontier: pd.DataFrame, estimated_frontier: pd.DataFrame, x_col: str, y_col: str) -> float | None:
    if true_frontier.empty or estimated_frontier.empty:
        return None
    reference = true_frontier[[x_col, y_col]].to_numpy(dtype=float)
    estimated = estimated_frontier[[x_col, y_col]].to_numpy(dtype=float)
    distances = []
    for point in reference:
        distances.append(float(np.min(np.linalg.norm(estimated - point, axis=1))))
    return float(np.mean(distances)) if distances else None


def _round_metrics(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    rounds = sorted(paired["batch_id"].unique(), key=_round_sort_key)
    normalized_reference = _normalize_frame(
        paired,
        ["viability_percent", "critical_axial_load_N_per_needle"],
        paired,
    )
    true_frontier = normalized_reference.loc[
        _pareto_frontier_mask(normalized_reference, "viability_percent", "critical_axial_load_N_per_needle")
    ].copy()
    final_hv = _hypervolume_2d(true_frontier, "viability_percent", "critical_axial_load_N_per_needle")
    rows = []
    for round_id in rounds:
        cumulative = paired.loc[paired["batch_id"].map(_round_sort_key) <= _round_sort_key(round_id)].copy()
        normalized = _normalize_frame(
            cumulative,
            ["viability_percent", "critical_axial_load_N_per_needle"],
            paired,
        )
        frontier = normalized.loc[
            _pareto_frontier_mask(normalized, "viability_percent", "critical_axial_load_N_per_needle")
        ].copy()
        hv = _hypervolume_2d(frontier, "viability_percent", "critical_axial_load_N_per_needle")
        igd = _igd_2d(true_frontier, frontier, "viability_percent", "critical_axial_load_N_per_needle")
        rows.append(
            {
                "batch_id": round_id,
                "paired_rows_cumulative": int(len(cumulative)),
                "pareto_points_cumulative": int(len(frontier)),
                "normalized_hypervolume": float(hv / final_hv) if final_hv > 0 else np.nan,
                "igd": igd,
            }
        )
    return pd.DataFrame(rows)


def _is_blank(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _numeric(value: object) -> float | None:
    if _is_blank(value):
        return None
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)


def _round_number(batch_id: str) -> int | None:
    value = str(batch_id).strip()
    if value.startswith("ROUND_") and value.removeprefix("ROUND_").isdigit():
        return int(value.removeprefix("ROUND_"))
    return None


def _proposal_path(batch_id: str, results_root: str | Path) -> Path:
    paths = round_artifact_paths(batch_id, results_root)
    if paths.proposal_csv.exists():
        return paths.proposal_csv
    reconstructed = paths.proposal_dir / "proposal_reconstructed.csv"
    if reconstructed.exists():
        return reconstructed
    raise FileNotFoundError(f"No archived proposal exists for {batch_id}: {paths.proposal_dir}")


def _active_phase(batch_id: str, results_root: str | Path) -> str:
    metadata_path = round_artifact_paths(batch_id, results_root).proposal_metadata
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        value = str(metadata.get("active_phase", "")).strip()
        if value:
            return value
    return "screening_only"


def _provenance(
    batch_id: str,
    evaluation_config: Mapping[str, Any],
) -> tuple[str, bool]:
    provenance_config = evaluation_config.get("round_provenance", {})
    provenance = str(
        provenance_config.get(
            batch_id,
            provenance_config.get("default", "formal_frozen"),
        )
    )
    number = _round_number(batch_id)
    formal_start = int(evaluation_config.get("formal_start_round", 3))
    formal = (
        provenance == "formal_frozen"
        and number is not None
        and number >= formal_start
    )
    return provenance, formal


def _replicate_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    if "replicate_id" in frame.columns:
        values = frame["replicate_id"].dropna().astype(str).str.strip()
        values = values[values != ""]
        if not values.empty:
            return int(values.nunique())
    return int(len(frame))


def _proposal_prediction_value(
    proposal_row: pd.Series,
    definition: Mapping[str, Any],
    primary_key: str,
    fallback_key: str,
) -> float | None:
    """Read a frozen prediction with backward compatibility for old rounds."""
    primary_column = str(definition.get(primary_key, "")).strip()
    if primary_column:
        value = _numeric(proposal_row.get(primary_column))
        if value is not None:
            return value
    for column in definition.get(fallback_key, []) or []:
        value = _numeric(proposal_row.get(str(column)))
        if value is not None:
            return value
    return None


def build_round_prospective_table(
    batch_id: str,
    observations: pd.DataFrame,
    results_root: str | Path = RESULTS_V2_DIR,
    evaluation_config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build one candidate-by-endpoint table from frozen proposal predictions."""
    evaluation_config = dict(evaluation_config or {})
    results_root = Path(results_root)
    paths = round_artifact_paths(batch_id, results_root)
    proposal_path = _proposal_path(batch_id, results_root)
    if not paths.completed_csv.exists():
        raise FileNotFoundError(
            f"Completed worksheet does not exist for {batch_id}: {paths.completed_csv}"
        )
    validate_completed_against_proposal(paths.completed_csv, proposal_path)

    proposal = pd.read_csv(proposal_path)
    completed = pd.read_csv(paths.completed_csv)
    observations = observations.copy()
    if "batch_id" not in observations.columns:
        observations["batch_id"] = ""
    if "formulation_id" not in observations.columns:
        observations["formulation_id"] = ""
    if "endpoint" not in observations.columns:
        observations["endpoint"] = ""
    if "value" not in observations.columns:
        observations["value"] = np.nan
    round_observations = observations[
        observations["batch_id"].astype(str) == str(batch_id)
    ].copy()
    round_observations["value"] = pd.to_numeric(
        round_observations["value"],
        errors="coerce",
    )

    provenance, formal_cohort = _provenance(batch_id, evaluation_config)
    round_number = _round_number(batch_id)
    active_phase = _active_phase(batch_id, results_root)
    policy_version = str(
        evaluation_config.get("policy_version", "prospective_evaluation_v2")
    )
    interval_z = float(
        evaluation_config.get("prediction_interval", {}).get("z_value", 1.96)
    )
    endpoint_config = evaluation_config.get("endpoints", {})
    duplicate_formulations = set(
        proposal.loc[
            proposal["formulation_id"].astype(str).duplicated(keep=False),
            "formulation_id",
        ].astype(str)
    )
    completed_counts = (
        completed.assign(candidate_id=completed["candidate_id"].astype(str))
        .groupby("candidate_id")
        .size()
        .to_dict()
    )

    rows: list[dict[str, object]] = []
    for _, proposal_row in proposal.iterrows():
        candidate_id = str(proposal_row["candidate_id"])
        formulation_id = str(proposal_row["formulation_id"])
        for endpoint, definition in endpoint_config.items():
            prediction_mean = _proposal_prediction_value(
                proposal_row,
                definition,
                "prediction_mean_column",
                "prediction_mean_fallback_columns",
            )
            prediction_std = _proposal_prediction_value(
                proposal_row,
                definition,
                "prediction_std_column",
                "prediction_std_fallback_columns",
            )
            endpoint_observations = round_observations[
                (round_observations["formulation_id"].astype(str) == formulation_id)
                & (round_observations["endpoint"].astype(str) == str(endpoint))
            ].dropna(subset=["value"])
            from .terminal_force import MODEL_FIELD, LEGACY_DEFINITION
            endpoint_definition = ''
            if endpoint == MODEL_FIELD:
                endpoint_definition = str(proposal_row.get('mechanical_prediction_definition_id', LEGACY_DEFINITION))
                actual_definitions = endpoint_observations.get('mechanical_definition_id', pd.Series('',index=endpoint_observations.index)).fillna('').replace('',LEGACY_DEFINITION)
                endpoint_observations = endpoint_observations.loc[actual_definitions.eq(endpoint_definition)]
            if endpoint_observations.empty:
                observed_mean = None
            elif str(endpoint) == INTACT_PATCH_ENDPOINT:
                observed_mean = aggregate_intact_patch_replicates(
                    endpoint_observations["value"]
                )
            else:
                observed_mean = float(endpoint_observations["value"].mean())
            unit = (
                str(endpoint_observations.iloc[0].get("unit", ""))
                if not endpoint_observations.empty
                else ""
            )

            exclusion_reason = ""
            if formulation_id in duplicate_formulations:
                exclusion_reason = "ambiguous_duplicate_formulation"
            elif endpoint == MODEL_FIELD and proposal_row.get('mechanical_prediction_status') == 'untrained_placeholder':
                exclusion_reason = 'untrained_placeholder'
            elif prediction_mean is None:
                exclusion_reason = "missing_frozen_prediction"
            elif observed_mean is None:
                exclusion_reason = "not_measured"
            elif (
                definition.get("formal_phase")
                and str(definition.get("formal_phase")) != active_phase
            ):
                exclusion_reason = "endpoint_not_active"
            evaluation_eligible = exclusion_reason == ""

            formal_exclusion_reason = ""
            if not evaluation_eligible:
                formal_exclusion_reason = exclusion_reason
            elif not formal_cohort:
                formal_exclusion_reason = "outside_formal_cohort"
            formal_metric_eligible = formal_exclusion_reason == ""

            signed_error = None
            absolute_error = None
            squared_error = None
            standardized_residual = None
            interval_lower = None
            interval_upper = None
            interval_covered = None
            brier_score = None
            classification_correct = None
            metric_type = str(definition.get("metric_type", "continuous"))
            if evaluation_eligible and prediction_mean is not None and observed_mean is not None:
                signed_error = prediction_mean - observed_mean
                absolute_error = abs(signed_error)
                squared_error = signed_error**2
                if prediction_std is not None and prediction_std > 0:
                    standardized_residual = signed_error / prediction_std
                    interval_lower = prediction_mean - interval_z * prediction_std
                    interval_upper = prediction_mean + interval_z * prediction_std
                    interval_covered = bool(
                        interval_lower <= observed_mean <= interval_upper
                    )
                if metric_type == "binary":
                    brier_score = squared_error
                    threshold = float(definition.get("classification_threshold", 0.5))
                    classification_correct = bool(
                        (prediction_mean >= threshold) == (observed_mean >= threshold)
                    )

            rows.append(
                {
                    "evaluation_policy_version": policy_version,
                    "mechanical_definition_id": endpoint_definition,
                    "round_id": batch_id,
                    "round_number": round_number,
                    "provenance_class": provenance,
                    "formal_cohort": formal_cohort,
                    "active_phase": active_phase,
                    "candidate_id": candidate_id,
                    "formulation_id": formulation_id,
                    "recommendation_type": proposal_row.get("recommendation_type", ""),
                    "selection_rank": proposal_row.get("selection_rank", ""),
                    "endpoint": endpoint,
                    "endpoint_role": definition.get("role", ""),
                    "metric_type": metric_type,
                    "prediction_mean": prediction_mean,
                    "prediction_std": prediction_std,
                    "observed_mean": observed_mean,
                    "observed_unit": unit,
                    "replicate_count": _replicate_count(endpoint_observations),
                    "completed_row_count": int(completed_counts.get(candidate_id, 0)),
                    "evaluation_eligible": evaluation_eligible,
                    "exclusion_reason": exclusion_reason,
                    "formal_metric_eligible": formal_metric_eligible,
                    "formal_exclusion_reason": formal_exclusion_reason,
                    "signed_error": signed_error,
                    "absolute_error": absolute_error,
                    "squared_error": squared_error,
                    "standardized_residual": standardized_residual,
                    "interval_95_lower": interval_lower,
                    "interval_95_upper": interval_upper,
                    "interval_95_covered": interval_covered,
                    "brier_score": brier_score,
                    "classification_correct": classification_correct,
                }
            )
    table = pd.DataFrame(rows, columns=PROSPECTIVE_TABLE_COLUMNS)
    if "experimental_role" in proposal:
        controls = set(proposal.loc[proposal.experimental_role.eq("campaign_control"), "candidate_id"].astype(str))
        mask = table.candidate_id.astype(str).isin(controls) & ~table.endpoint.isin(
            ["critical_axial_load_N_per_needle", "initial_stiffness_N_per_mm_per_needle"]
        )
        table.loc[mask, ["evaluation_eligible", "formal_metric_eligible"]] = False
        table.loc[mask, ["exclusion_reason", "formal_exclusion_reason"]] = "reference_control_not_modeled"
    return table


def _metric_row(
    frame: pd.DataFrame,
    scope: str,
    round_id: str,
    provenance_class: str,
    formal_cohort: bool,
    eligible_column: str,
) -> dict[str, object]:
    endpoint = str(frame.iloc[0]["endpoint"])
    endpoint_role = str(frame.iloc[0]["endpoint_role"])
    eligible = frame[frame[eligible_column].astype(bool)].copy()
    actual = pd.to_numeric(eligible["observed_mean"], errors="coerce")
    predicted = pd.to_numeric(eligible["prediction_mean"], errors="coerce")
    valid = actual.notna() & predicted.notna()
    actual = actual[valid]
    predicted = predicted[valid]
    n_evaluated = int(len(actual))
    n_proposed = int(len(frame))

    mae = rmse = bias = r2 = coverage = mean_width = median_width = brier = accuracy = np.nan
    if n_evaluated:
        errors = predicted.to_numpy(dtype=float) - actual.to_numpy(dtype=float)
        metric_type = str(frame.iloc[0]["metric_type"])
        if metric_type == "continuous":
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors**2)))
            bias = float(np.mean(errors))
            if n_evaluated >= 2 and float(np.var(actual.to_numpy(dtype=float))) > 0:
                denominator = float(
                    np.sum((actual.to_numpy(dtype=float) - float(actual.mean())) ** 2)
                )
                r2 = float(1.0 - np.sum(errors**2) / denominator)
            covered = eligible.loc[valid, "interval_95_covered"].dropna()
            if not covered.empty:
                coverage = float(covered.astype(bool).mean())
            interval_lower = pd.to_numeric(
                eligible.loc[valid, "interval_95_lower"],
                errors="coerce",
            )
            interval_upper = pd.to_numeric(
                eligible.loc[valid, "interval_95_upper"],
                errors="coerce",
            )
            widths = (interval_upper - interval_lower).dropna()
            if not widths.empty:
                mean_width = float(widths.mean())
                median_width = float(widths.median())
        elif metric_type == "binary":
            brier_values = pd.to_numeric(
                eligible.loc[valid, "brier_score"],
                errors="coerce",
            ).dropna()
            correct_values = eligible.loc[valid, "classification_correct"].dropna()
            if not brier_values.empty:
                brier = float(brier_values.mean())
            if not correct_values.empty:
                accuracy = float(correct_values.astype(bool).mean())

    return {
        "scope": scope,
        "round_id": round_id,
        "provenance_class": provenance_class,
        "formal_cohort": formal_cohort,
        "endpoint": endpoint,
        "endpoint_role": endpoint_role,
        "mechanical_definition_id": frame.iloc[0].get("mechanical_definition_id", ""),
        "n_proposed": n_proposed,
        "n_evaluated": n_evaluated,
        "completion_rate": float(n_evaluated / n_proposed) if n_proposed else np.nan,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "r2": r2,
        "interval_95_coverage": coverage,
        "interval_95_mean_width": mean_width,
        "interval_95_median_width": median_width,
        "brier_score": brier,
        "accuracy": accuracy,
    }


def summarize_prospective_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Summarize each round and keep historical provenance cohorts distinct."""
    if table.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    rows: list[dict[str, object]] = []
    table = table.copy()
    if 'mechanical_definition_id' not in table:
        table['mechanical_definition_id'] = np.where(table.endpoint.eq('critical_axial_load_N_per_needle'),'legacy_curve_maximum_v1','')
    table['mechanical_definition_id'] = table.mechanical_definition_id.fillna('')
    ordered = table.assign(
        _round_sort=table["round_id"].map(_round_sort_key)
    ).sort_values(["_round_sort", "endpoint"])
    for (round_id, endpoint, endpoint_definition), frame in ordered.groupby(
        ["round_id", "endpoint", "mechanical_definition_id"],
        sort=False,
    ):
        rows.append(
            _metric_row(
                frame,
                scope="round",
                round_id=str(round_id),
                provenance_class=str(frame.iloc[0]["provenance_class"]),
                formal_cohort=bool(frame.iloc[0]["formal_cohort"]),
                eligible_column="evaluation_eligible",
            )
        )
    for (endpoint, endpoint_definition), frame in ordered.groupby(["endpoint", "mechanical_definition_id"], sort=False):
        rows.append(
            _metric_row(
                frame,
                scope="pooled_all",
                round_id="ALL_COMPLETED",
                provenance_class="mixed",
                formal_cohort=False,
                eligible_column="evaluation_eligible",
            )
        )
        for provenance_class, provenance_scope in [
            ("reconstructed", "pooled_reconstructed"),
            ("migration_frozen_supplementary", "pooled_supplementary"),
            ("formal_frozen", "pooled_formal"),
        ]:
            provenance_frame = frame[
                frame["provenance_class"].astype(str) == provenance_class
            ]
            if provenance_frame.empty:
                rows.append(
                    {
                        "mechanical_definition_id": endpoint_definition,
                        "scope": provenance_scope,
                        "round_id": (
                            "FORMAL_COHORT"
                            if provenance_scope == "pooled_formal"
                            else provenance_class
                        ),
                        "provenance_class": provenance_class,
                        "formal_cohort": provenance_scope == "pooled_formal",
                        "endpoint": endpoint,
                        "endpoint_role": str(frame.iloc[0]["endpoint_role"]),
                        "n_proposed": 0,
                        "n_evaluated": 0,
                        "completion_rate": np.nan,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "bias": np.nan,
                        "r2": np.nan,
                        "interval_95_coverage": np.nan,
                        "interval_95_mean_width": np.nan,
                        "interval_95_median_width": np.nan,
                        "brier_score": np.nan,
                        "accuracy": np.nan,
                    }
                )
                continue
            rows.append(
                _metric_row(
                    provenance_frame,
                    scope=provenance_scope,
                    round_id=(
                        "FORMAL_COHORT"
                        if provenance_scope == "pooled_formal"
                        else provenance_class
                    ),
                    provenance_class=provenance_class,
                    formal_cohort=provenance_scope == "pooled_formal",
                    eligible_column=(
                        "formal_metric_eligible"
                        if provenance_scope == "pooled_formal"
                        else "evaluation_eligible"
                    ),
                )
            )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def build_feasible_paired_objectives(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return observed paired objectives passing the intact-patch gate.

    The second table contains every excluded formulation/batch pair and one
    explicit reason, so absence from the frontier is auditable.
    """

    required_columns = {"formulation_id", "batch_id", "endpoint", "value"}
    if observations.empty or not required_columns.issubset(observations.columns):
        empty = pd.DataFrame(
            columns=[
                "formulation_id",
                "batch_id",
                "viability_percent",
                "critical_axial_load_N_per_needle",
                "intact_patch_formation_pass",
            ]
        )
        return empty.copy(), empty.assign(exclusion_reason=pd.Series(dtype=str))

    from .group10_config import paired_objective_observations
    frame = paired_objective_observations(observations)
    selected_definition = frame.attrs.get('mechanical_definition_id','legacy_curve_maximum_v1')
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    grouping = ["formulation_id", "batch_id"]
    continuous = (
        frame.loc[
            frame["endpoint"].astype(str).isin(
                ["viability_percent", "critical_axial_load_N_per_needle"]
            )
        ]
        .pivot_table(
            index=grouping,
            columns="endpoint",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    keys = frame[grouping].drop_duplicates()
    paired = keys.merge(continuous, on=grouping, how="left")
    intact = frame.loc[
        frame["endpoint"].astype(str).eq(INTACT_PATCH_ENDPOINT)
    ]
    if intact.empty:
        paired[INTACT_PATCH_ENDPOINT] = np.nan
    else:
        intact_values = (
            intact.groupby(grouping, dropna=False)["value"]
            .agg(aggregate_intact_patch_replicates)
            .rename(INTACT_PATCH_ENDPOINT)
            .reset_index()
        )
        paired = paired.merge(intact_values, on=grouping, how="left")

    if not formulations.empty and "formulation_id" in formulations.columns:
        formulation_columns = [
            column
            for column in formulations.columns
            if column not in paired.columns or column == "formulation_id"
        ]
        paired = paired.merge(
            formulations[formulation_columns].drop_duplicates("formulation_id"),
            on="formulation_id",
            how="left",
        )

    def numeric_column(column: str) -> pd.Series:
        values = (
            paired[column]
            if column in paired.columns
            else pd.Series(np.nan, index=paired.index, dtype=float)
        )
        return pd.to_numeric(values, errors="coerce")

    viability = numeric_column("viability_percent")
    load = numeric_column("critical_axial_load_N_per_needle")
    intact_value = numeric_column(INTACT_PATCH_ENDPOINT)
    reasons = np.select(
        [
            viability.isna(),
            load.isna(),
            intact_value.isna(),
            intact_value.lt(0.5),
        ],
        [
            "missing_viability",
            "missing_load",
            "unknown_intact_status",
            "failed_intact_gate",
        ],
        default="",
    )
    paired["mechanical_definition_id"] = selected_definition
    paired["exclusion_reason"] = reasons
    feasible = paired.loc[paired["exclusion_reason"].eq("")].copy()
    excluded = paired.loc[~paired["exclusion_reason"].eq("")].copy()
    return (
        feasible.drop(columns=["exclusion_reason"]).reset_index(drop=True),
        excluded.reset_index(drop=True),
    )


def compute_observed_pareto_front(
    feasible_pairs: pd.DataFrame,
    x_col: str = "viability_percent",
    y_col: str = "critical_axial_load_N_per_needle",
) -> pd.DataFrame:
    """Label observed feasible points by nondomination for two maximized objectives."""

    result = feasible_pairs.copy()
    result["is_dominated"] = pd.Series(dtype=bool)
    result["is_pareto"] = pd.Series(dtype=bool)
    result["pareto_rank"] = pd.Series(dtype="Int64")
    if result.empty:
        return result

    remaining = result.index.to_list()
    ranks = pd.Series(pd.NA, index=result.index, dtype="Int64")
    rank = 1
    while remaining:
        layer = result.loc[remaining]
        mask = _pareto_frontier_mask(layer, x_col, y_col)
        layer_indices = layer.index[np.asarray(mask, dtype=bool)].to_list()
        ranks.loc[layer_indices] = rank
        remaining = [index for index in remaining if index not in set(layer_indices)]
        rank += 1
    result["pareto_rank"] = ranks
    result["is_pareto"] = result["pareto_rank"].eq(1)
    result["is_dominated"] = ~result["is_pareto"]
    return result


def compute_fixed_reference_hypervolume(
    observed_pareto: pd.DataFrame,
    reference_point: Mapping[str, float],
    x_col: str = "viability_percent",
    y_col: str = "critical_axial_load_N_per_needle",
) -> dict[str, object]:
    """Calculate raw two-objective hypervolume against a fixed physical reference."""

    reference_x = float(reference_point[x_col])
    reference_y = float(reference_point[y_col])
    evidence_count = int(len(observed_pareto))
    frontier = observed_pareto.copy()
    if "is_pareto" in frontier.columns:
        frontier = frontier.loc[frontier["is_pareto"].astype(bool)]
    frontier[x_col] = pd.to_numeric(frontier.get(x_col), errors="coerce")
    frontier[y_col] = pd.to_numeric(frontier.get(y_col), errors="coerce")
    frontier = frontier.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
    has_valid_evidence = not frontier.empty
    frontier = frontier.loc[
        frontier[x_col].gt(reference_x) & frontier[y_col].gt(reference_y)
    ]
    if frontier.empty:
        return {
            "status": "estimated" if has_valid_evidence else "not_estimable",
            "hypervolume": 0.0 if has_valid_evidence else np.nan,
            "unit": "percent_x_N_per_needle",
            "feasible_pair_count": evidence_count,
            "pareto_point_count": 0,
            "reference_viability_percent": reference_x,
            "reference_critical_axial_load_N_per_needle": reference_y,
            "reason": (
                "Valid paired measurements contribute zero area beyond the fixed reference point."
                if has_valid_evidence else "No valid feasible paired objective measurements."
            ),
        }

    frontier = compute_observed_pareto_front(frontier, x_col=x_col, y_col=y_col)
    frontier = frontier.loc[frontier["is_pareto"]].sort_values(x_col)
    suffix_max_y = frontier[y_col][::-1].cummax()[::-1].to_numpy(dtype=float)
    x_values = frontier[x_col].to_numpy(dtype=float)
    previous = reference_x
    hypervolume = 0.0
    for x_value, y_value in zip(x_values, suffix_max_y):
        hypervolume += max(x_value - previous, 0.0) * max(y_value - reference_y, 0.0)
        previous = max(previous, x_value)
    return {
        "status": "estimated",
        "hypervolume": float(hypervolume),
        "unit": "percent_x_N_per_needle",
        "feasible_pair_count": evidence_count,
        "pareto_point_count": int(len(frontier)),
        "reference_viability_percent": reference_x,
        "reference_critical_axial_load_N_per_needle": reference_y,
        "reason": "",
    }


def compute_campaign_hypervolume_progress(
    feasible_pairs: pd.DataFrame,
    reference_point: Mapping[str, float],
) -> pd.DataFrame:
    """Compute each round using only evidence available at or before that round."""

    columns = [
        "round_id",
        "status",
        "hypervolume",
        "delta_hypervolume",
        "unit",
        "feasible_pair_count",
        "pareto_point_count",
        "reference_viability_percent",
        "reference_critical_axial_load_N_per_needle",
        "reason",
    ]
    if feasible_pairs.empty or "batch_id" not in feasible_pairs.columns:
        return pd.DataFrame(columns=columns)
    round_ids = sorted(
        feasible_pairs["batch_id"].dropna().astype(str).unique(),
        key=_round_sort_key,
    )
    rows: list[dict[str, object]] = []
    cumulative = feasible_pairs.iloc[0:0].copy()
    previous_hypervolume: float | None = None
    for round_id in round_ids:
        cumulative = pd.concat(
            [
                cumulative,
                feasible_pairs.loc[
                    feasible_pairs["batch_id"].astype(str).eq(round_id)
                ],
            ],
            ignore_index=True,
        )
        labelled = compute_observed_pareto_front(cumulative)
        metric = compute_fixed_reference_hypervolume(labelled, reference_point)
        current = (
            float(metric["hypervolume"])
            if metric["status"] == "estimated"
            else None
        )
        delta = (
            np.nan
            if current is None or previous_hypervolume is None
            else current - previous_hypervolume
        )
        rows.append(
            {
                "round_id": round_id,
                **metric,
                "delta_hypervolume": delta,
            }
        )
        if current is not None:
            previous_hypervolume = current
    return pd.DataFrame(rows, columns=columns)


def endpoint_r2_history(formulations, observations, metrics, registry):
    """Existing cumulative paired CV calculation, extracted unchanged for rendering."""
    if metrics.empty:
        return pd.DataFrame(columns=["batch_id", "viability_r2", "load_r2"])
    rounds = metrics["batch_id"].tolist()
    rows = []
    for round_id in rounds:
        cumulative_obs = observations.loc[observations["batch_id"].map(_round_sort_key) <= _round_sort_key(round_id)].copy()
        cumulative_frame = paired_objective_frame(build_training_frame(formulations, cumulative_obs, registry))
        paired_keys = cumulative_frame.dropna(subset=["viability_percent", "critical_axial_load_N_per_needle"])[["formulation_id", "batch_id"]]
        if paired_keys.empty:
            rows.append({"batch_id": round_id, "viability_r2": np.nan, "load_r2": np.nan})
            continue
        allowed = set((str(row.formulation_id), str(row.batch_id)) for row in paired_keys.itertuples())
        filtered_obs = cumulative_obs.loc[
            cumulative_obs.apply(lambda row: (str(row.get("formulation_id", "")), str(row.get("batch_id", ""))) in allowed, axis=1)
        ].copy()
        viability_predictions = _cross_validated_predictions(formulations, filtered_obs, registry, "viability_percent")
        load_predictions = _cross_validated_predictions(
            formulations,
            filtered_obs,
            registry,
            "critical_axial_load_N_per_needle",
        )
        viability_r2 = (
            float(r2_score(viability_predictions["actual"], viability_predictions["predicted"]))
            if len(viability_predictions) >= 2 and np.nanstd(viability_predictions["actual"]) > 0
            else np.nan
        )
        load_r2 = (
            float(r2_score(load_predictions["actual"], load_predictions["predicted"]))
            if len(load_predictions) >= 2 and np.nanstd(load_predictions["actual"]) > 0
            else np.nan
        )
        rows.append({"batch_id": round_id, "viability_r2": viability_r2, "load_r2": load_r2})
    return pd.DataFrame(rows, columns=["batch_id", "viability_r2", "load_r2"])

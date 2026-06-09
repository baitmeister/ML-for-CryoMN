"""Retest-priority diagnostics for the v2 multi-objective workflow."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import nested_get
from .models import EndpointModels
from .penalties import count_active_ingredients
from .registry import IngredientRegistry


RETEST_ELIGIBLE_SOURCE_TYPES = {"wetlab_feedback"}
RETEST_COMPARISON_SOURCE_TYPES = {"wetlab_feedback", "legacy_wetlab"}


def _viability_batch_frame(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()

    obs = observations.copy()
    if "endpoint" not in obs.columns:
        return pd.DataFrame()
    obs["endpoint"] = obs["endpoint"].astype(str)
    obs = obs[obs["endpoint"] == "viability_percent"].copy()
    if obs.empty:
        return obs

    if "source_type" not in obs.columns:
        obs["source_type"] = ""
    obs["source_type"] = obs["source_type"].fillna("").astype(str)
    obs = obs[obs["source_type"].isin(RETEST_COMPARISON_SOURCE_TYPES)].copy()
    if obs.empty:
        return obs

    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    obs = obs.dropna(subset=["value"]).copy()
    if obs.empty:
        return obs

    if "batch_id" not in obs.columns:
        obs["batch_id"] = ""
    obs["batch_id"] = obs["batch_id"].fillna("").astype(str)

    grouped = (
        obs.groupby(["formulation_id", "batch_id", "source_type"], dropna=False, as_index=False)
        .agg(viability_percent=("value", "mean"))
    )
    frame = formulations.merge(grouped, on="formulation_id", how="inner")
    if frame.empty:
        return frame

    for feature_name in registry.feature_names:
        if feature_name not in frame.columns:
            frame[feature_name] = 0.0
    frame["active_ingredient_count"] = frame.apply(
        lambda row: count_active_ingredients(row, registry),
        axis=1,
    )
    return frame[frame["active_ingredient_count"] > 0].reset_index(drop=True)


def build_retest_candidates(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    models: EndpointModels,
    registry: IngredientRegistry,
    optimization_config: Mapping,
) -> pd.DataFrame:
    if not bool(nested_get(optimization_config, "retest.enabled", True)):
        return pd.DataFrame()

    frame = _viability_batch_frame(formulations, observations, registry)
    if frame.empty:
        return pd.DataFrame()

    feature_names = registry.feature_names
    nearest_neighbor_count = int(nested_get(optimization_config, "retest.nearest_neighbor_count", 3))
    disagreement_threshold = float(
        nested_get(optimization_config, "retest.formulation_disagreement_threshold_percent", 15.0)
    )
    residual_threshold = float(
        nested_get(optimization_config, "retest.local_residual_threshold_percent", 20.0)
    )
    uncertainty_threshold = float(
        nested_get(optimization_config, "retest.uncertainty_percent_threshold", 12.0)
    )
    max_candidates = int(nested_get(optimization_config, "retest.max_candidates_per_round", 2))

    eligibility_rows = frame[frame["source_type"].isin(RETEST_ELIGIBLE_SOURCE_TYPES)].copy()
    if eligibility_rows.empty:
        return pd.DataFrame()

    eligibility_rows = eligibility_rows.sort_values(["formulation_id", "batch_id"])
    latest = eligibility_rows.groupby("formulation_id", as_index=False).tail(1).copy()
    if latest.empty:
        return pd.DataFrame()

    x_latest = latest[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    viability_prediction = models.viability.predict(x_latest)
    intact_prediction = models.intact.predict_proba(x_latest)
    critical_prediction = models.critical_load.predict(x_latest)

    latest["predicted_viability_percent"] = viability_prediction.mean
    latest["viability_std"] = viability_prediction.std
    latest["predicted_critical_axial_load_N_per_needle"] = critical_prediction.mean
    latest["critical_axial_load_std"] = critical_prediction.std
    latest["intact_patch_pass_probability"] = np.clip(intact_prediction, 0.0, 1.0)

    ranges = (
        frame.groupby("formulation_id")["viability_percent"]
        .agg(lambda values: float(np.nanmax(values) - np.nanmin(values)))
        .rename("same_formulation_range")
        .reset_index()
    )
    latest = latest.merge(ranges, on="formulation_id", how="left")
    latest["same_formulation_range"] = latest["same_formulation_range"].fillna(0.0)

    comparison_rows = frame.reset_index(drop=True)
    if not comparison_rows.empty:
        comparison_points = (
            comparison_rows[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        local_residuals = []
        for index, row in latest.reset_index(drop=True).iterrows():
            row_point = x_latest[index]
            distances = np.linalg.norm(comparison_points - row_point, axis=1)
            neighbor_mask = comparison_rows["formulation_id"].astype(str) != str(row["formulation_id"])
            neighbor_indices = np.flatnonzero(neighbor_mask.to_numpy())
            if len(neighbor_indices) == 0:
                local_residuals.append(0.0)
                continue
            neighbor_order = neighbor_indices[np.argsort(distances[neighbor_indices])]
            neighbor_indices = [int(i) for i in neighbor_order[:nearest_neighbor_count]]
            if neighbor_indices:
                neighbor_mean = float(comparison_rows.iloc[neighbor_indices]["viability_percent"].mean())
                local_residuals.append(abs(float(row["viability_percent"]) - neighbor_mean))
            else:
                local_residuals.append(0.0)
        latest["local_neighbor_residual"] = local_residuals
    else:
        latest["local_neighbor_residual"] = 0.0

    latest["retest_priority_score"] = (
        latest["same_formulation_range"] / max(disagreement_threshold, 1e-9)
        + latest["local_neighbor_residual"] / max(residual_threshold, 1e-9)
        + latest["viability_std"] / max(uncertainty_threshold, 1e-9)
    )
    flagged = latest[
        (latest["same_formulation_range"] >= disagreement_threshold)
        | (latest["local_neighbor_residual"] >= residual_threshold)
        | (latest["viability_std"] >= uncertainty_threshold)
    ].copy()
    if flagged.empty:
        return flagged

    flagged["recommendation_type"] = "retest_priority"
    flagged["candidate_id"] = flagged["formulation_id"].map(lambda value: f"retest_{value}")
    flagged["selection_explanation"] = flagged.apply(
        lambda row: (
            f"retest_priority: batch_disagreement={row['same_formulation_range']:.1f}, "
            f"local_residual={row['local_neighbor_residual']:.1f}, "
            f"viability_std={row['viability_std']:.1f}"
        ),
        axis=1,
    )
    columns = [
        "candidate_id",
        "formulation_id",
        "recommendation_type",
        "selection_explanation",
        "active_ingredient_count",
    ] + feature_names
    for column in columns:
        if column not in flagged.columns:
            flagged[column] = 0.0 if column in feature_names else ""
    return flagged.sort_values("retest_priority_score", ascending=False).head(max_candidates).reset_index(drop=True)

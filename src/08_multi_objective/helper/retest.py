"""Observed-evidence retest diagnostics for the v2 campaign."""

from __future__ import annotations

from collections.abc import Mapping
import re

import numpy as np
import pandas as pd

from .config import nested_get
from .models import EndpointModels
from .penalties import count_active_ingredients
from .registry import IngredientRegistry, presence_threshold


RETEST_ELIGIBLE_SOURCE_TYPES = {"wetlab_feedback"}


def _batch_sort_key(batch_id: object) -> tuple[int, str]:
    """Sort ROUND_002 after ROUND_001 while remaining stable for other IDs."""
    value = str(batch_id or "")
    numbers = re.findall(r"\d+", value)
    return (int(numbers[-1]) if numbers else -1, value)


def _viability_batch_frame(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
) -> pd.DataFrame:
    """Return one campaign-feedback row per formulation and experimental batch."""
    if observations.empty or "endpoint" not in observations.columns:
        return pd.DataFrame()

    obs = observations.copy()
    obs["endpoint"] = obs["endpoint"].astype(str)
    obs = obs[obs["endpoint"] == "viability_percent"].copy()
    if obs.empty:
        return obs

    if "source_type" not in obs.columns:
        obs["source_type"] = ""
    obs["source_type"] = obs["source_type"].fillna("").astype(str)
    obs = obs[obs["source_type"].isin(RETEST_ELIGIBLE_SOURCE_TYPES)].copy()
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
        .agg(
            viability_percent=("value", "mean"),
            viability_replicate_sd=("value", "std"),
            viability_replicate_count=("value", "size"),
        )
    )
    grouped["viability_replicate_sd"] = (
        pd.to_numeric(grouped["viability_replicate_sd"], errors="coerce")
        .fillna(0.0)
    )

    formulation_rows = formulations.drop_duplicates("formulation_id", keep="last")
    frame = formulation_rows.merge(grouped, on="formulation_id", how="inner")
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


def _bounds_normalized_matrix(
    frame: pd.DataFrame,
    registry: IngredientRegistry,
) -> np.ndarray:
    """Apply practical zero thresholds and scale by registry ranges."""
    raw = np.zeros((len(frame), len(registry.feature_names)), dtype=float)
    lower = np.zeros(len(registry.feature_names), dtype=float)
    ranges = np.ones(len(registry.feature_names), dtype=float)
    for feature_index, feature_name in enumerate(registry.feature_names):
        ingredient = registry.get_by_feature(feature_name)
        values = pd.to_numeric(
            frame.get(feature_name, pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float, copy=True)
        values[np.abs(values) < presence_threshold(feature_name)] = 0.0
        raw[:, feature_index] = values
        lower[feature_index] = ingredient.lower_bound
        ranges[feature_index] = max(
            ingredient.upper_bound - ingredient.lower_bound,
            1e-12,
        )
    return (raw - lower) / ranges


def _nearest_neighbor_diagnostics(
    latest: pd.DataFrame,
    batch_frame: pd.DataFrame,
    registry: IngredientRegistry,
    nearest_neighbor_count: int,
) -> tuple[list[float], list[float]]:
    """Compare campaign formulation means using bounds-normalized chemistry."""
    comparison = (
        batch_frame.groupby("formulation_id", as_index=False)
        .agg(viability_percent=("viability_percent", "mean"))
        .merge(
            batch_frame.drop_duplicates("formulation_id", keep="last")[
                ["formulation_id", *registry.feature_names]
            ],
            on="formulation_id",
            how="left",
        )
        .reset_index(drop=True)
    )
    if comparison.empty:
        return [0.0] * len(latest), [float("nan")] * len(latest)

    comparison_points = _bounds_normalized_matrix(comparison, registry)
    latest_points = _bounds_normalized_matrix(latest, registry)
    residuals: list[float] = []
    nearest_distances: list[float] = []
    comparison_ids = comparison["formulation_id"].astype(str).to_numpy()
    for row_index, (_, row) in enumerate(latest.iterrows()):
        distances = np.linalg.norm(comparison_points - latest_points[row_index], axis=1)
        neighbor_indices = np.flatnonzero(
            comparison_ids != str(row["formulation_id"])
        )
        if len(neighbor_indices) == 0:
            residuals.append(0.0)
            nearest_distances.append(float("nan"))
            continue
        ordered = neighbor_indices[
            np.argsort(distances[neighbor_indices], kind="mergesort")
        ]
        chosen = ordered[: max(nearest_neighbor_count, 1)]
        neighbor_mean = float(comparison.iloc[chosen]["viability_percent"].mean())
        residuals.append(abs(float(row["viability_percent"]) - neighbor_mean))
        nearest_distances.append(float(distances[int(ordered[0])]))
    return residuals, nearest_distances


def build_retest_candidates(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    models: EndpointModels,
    registry: IngredientRegistry,
    optimization_config: Mapping,
) -> pd.DataFrame:
    """Return every evidence-eligible retest; slate limits are applied later.

    Model uncertainty is deliberately excluded from eligibility. It is used
    only as the second deterministic ranking key after observed-evidence
    severity.
    """
    if not bool(nested_get(optimization_config, "retest.enabled", True)):
        return pd.DataFrame()

    frame = _viability_batch_frame(formulations, observations, registry)
    if frame.empty:
        return pd.DataFrame()

    feature_names = registry.feature_names
    nearest_neighbor_count = int(
        nested_get(optimization_config, "retest.nearest_neighbor_count", 3)
    )
    disagreement_threshold = float(
        nested_get(
            optimization_config,
            "retest.formulation_disagreement_threshold_percent",
            15.0,
        )
    )
    residual_threshold = float(
        nested_get(
            optimization_config,
            "retest.local_residual_threshold_percent",
            20.0,
        )
    )
    within_batch_threshold = float(
        nested_get(
            optimization_config,
            "retest.within_batch_std_threshold_percent",
            8.0,
        )
    )
    within_batch_min_replicates = int(
        nested_get(
            optimization_config,
            "retest.within_batch_min_replicates",
            3,
        )
    )
    one_time_confirmation = bool(
        nested_get(
            optimization_config,
            "retest.one_time_anomaly_confirmation",
            True,
        )
    )

    ordered = frame.copy()
    ordered["_batch_sort_key"] = ordered["batch_id"].map(_batch_sort_key)
    ordered = ordered.sort_values(
        ["formulation_id", "_batch_sort_key"],
        kind="mergesort",
    )
    latest = ordered.groupby("formulation_id", as_index=False).tail(1).copy()
    latest = latest.drop(columns=["_batch_sort_key"]).reset_index(drop=True)
    if latest.empty:
        return pd.DataFrame()

    batch_stats = (
        frame.groupby("formulation_id", as_index=False)
        .agg(
            feedback_batch_count=("batch_id", "nunique"),
            same_formulation_range=(
                "viability_percent",
                lambda values: float(np.nanmax(values) - np.nanmin(values)),
            ),
        )
    )
    latest = latest.merge(batch_stats, on="formulation_id", how="left")
    latest["feedback_batch_count"] = (
        pd.to_numeric(latest["feedback_batch_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    latest["same_formulation_range"] = (
        pd.to_numeric(latest["same_formulation_range"], errors="coerce")
        .fillna(0.0)
    )

    local_residuals, nearest_distances = _nearest_neighbor_diagnostics(
        latest,
        frame,
        registry,
        nearest_neighbor_count,
    )
    latest["local_neighbor_residual"] = local_residuals
    latest["_retest_nearest_neighbor_distance"] = nearest_distances

    x_latest = (
        latest[feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    viability_prediction = models.viability.predict(x_latest)
    intact_prediction = models.intact.predict_proba(x_latest)
    critical_prediction = models.critical_load.predict(x_latest)
    latest["predicted_viability_percent"] = viability_prediction.mean
    latest["viability_std"] = viability_prediction.std
    latest["predicted_critical_axial_load_N_per_needle"] = critical_prediction.mean
    latest["critical_axial_load_std"] = critical_prediction.std
    latest["intact_patch_pass_probability"] = np.clip(
        intact_prediction,
        0.0,
        1.0,
    )

    cross_batch = (
        (latest["feedback_batch_count"] >= 2)
        & (latest["same_formulation_range"] >= disagreement_threshold)
    )
    high_replicate_sd = (
        (latest["viability_replicate_count"] >= within_batch_min_replicates)
        & (latest["viability_replicate_sd"] >= within_batch_threshold)
    )
    one_batch_anomaly = (
        one_time_confirmation
        & (latest["feedback_batch_count"] == 1)
        & (latest["local_neighbor_residual"] >= residual_threshold)
    )

    range_severity = np.where(
        cross_batch,
        latest["same_formulation_range"] / max(disagreement_threshold, 1e-12),
        0.0,
    )
    replicate_severity = np.where(
        high_replicate_sd,
        latest["viability_replicate_sd"] / max(within_batch_threshold, 1e-12),
        0.0,
    )
    anomaly_severity = np.where(
        one_batch_anomaly,
        latest["local_neighbor_residual"] / max(residual_threshold, 1e-12),
        0.0,
    )
    latest["retest_priority_score"] = np.maximum.reduce(
        [range_severity, replicate_severity, anomaly_severity]
    )
    latest["_retest_cross_batch_disagreement"] = cross_batch
    latest["_retest_high_replicate_sd"] = high_replicate_sd
    latest["_retest_one_batch_anomaly"] = one_batch_anomaly

    flagged = latest[cross_batch | high_replicate_sd | one_batch_anomaly].copy()
    if flagged.empty:
        return flagged

    def eligibility_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if bool(row["_retest_cross_batch_disagreement"]):
            reasons.append("cross_batch_disagreement")
        if bool(row["_retest_high_replicate_sd"]):
            reasons.append("high_latest_batch_replicate_sd")
        if bool(row["_retest_one_batch_anomaly"]):
            reasons.append("single_batch_neighbor_anomaly_confirmation")
        return "+".join(reasons)

    flagged["_retest_eligibility_reason"] = flagged.apply(
        eligibility_reason,
        axis=1,
    )
    flagged["recommendation_type"] = "retest_priority"
    flagged["candidate_id"] = flagged["formulation_id"].map(
        lambda value: f"retest_{value}"
    )
    flagged["selection_explanation"] = flagged.apply(
        lambda row: (
            f"retest_priority: reason={row['_retest_eligibility_reason']}; "
            f"feedback_batches={int(row['feedback_batch_count'])}; "
            f"batch_range={row['same_formulation_range']:.1f}; "
            f"latest_batch_n={int(row['viability_replicate_count'])}; "
            f"latest_batch_sd={row['viability_replicate_sd']:.1f}; "
            f"neighbor_residual={row['local_neighbor_residual']:.1f}; "
            f"viability_std_tiebreak={row['viability_std']:.1f}"
        ),
        axis=1,
    )
    return flagged.sort_values(
        ["retest_priority_score", "viability_std", "formulation_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

"""Evidence-aware labels for viability surrogate outputs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .cold_start import ColdStartContext
from .config import nested_get
from .models import EndpointModels


UNKNOWN_PREFIX = "unknown_"


def is_unknown_viability_status(value: object) -> bool:
    """Return whether a public viability estimate must be withheld."""
    return str(value or "").strip().startswith(UNKNOWN_PREFIX)


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _exact_viability_evidence(observations: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "formulation_id",
        "viability_exact_observation_count",
        "viability_exact_batch_count",
        "viability_exact_observed_mean",
        "viability_exact_observed_sources",
    ]
    required = {"formulation_id", "endpoint", "value"}
    if observations.empty or not required.issubset(observations.columns):
        return pd.DataFrame(columns=columns)

    viability = observations.loc[
        observations["endpoint"].astype(str).eq("viability_percent")
    ].copy()
    viability["value"] = pd.to_numeric(viability["value"], errors="coerce")
    viability = viability.dropna(subset=["value"])
    if viability.empty:
        return pd.DataFrame(columns=columns)
    if "batch_id" not in viability.columns:
        viability["batch_id"] = ""
    if "source_type" not in viability.columns:
        viability["source_type"] = ""

    return (
        viability.groupby("formulation_id", as_index=False)
        .agg(
            viability_exact_observation_count=("value", "size"),
            viability_exact_batch_count=(
                "batch_id",
                lambda values: int(
                    pd.Series(values).fillna("").astype(str).replace("", pd.NA).nunique()
                ),
            ),
            viability_exact_observed_mean=("value", "mean"),
            viability_exact_observed_sources=(
                "source_type",
                lambda values: ";".join(
                    sorted(
                        {
                            str(value).strip()
                            for value in values
                            if str(value).strip()
                        }
                    )
                ),
            ),
        )
    )


def annotate_viability_prediction_labels(
    candidates: pd.DataFrame,
    observations: pd.DataFrame,
    models: EndpointModels,
    cold_start_context: ColdStartContext,
    optimization_config: Mapping,
    target_round_number: int | None = None,
) -> pd.DataFrame:
    """Separate raw surrogate diagnostics from public viability estimates.

    Acquisition continues to use ``viability_ucb``, which was computed from
    the raw surrogate output before this function runs. Public prediction
    columns are blanked only when evidence is insufficient, while the raw
    values remain available for frozen prospective evaluation.
    """
    annotated = candidates.copy()
    raw_mean = pd.to_numeric(
        annotated.get(
            "predicted_viability_percent",
            pd.Series(np.nan, index=annotated.index, dtype=float),
        ),
        errors="coerce",
    )
    raw_std = pd.to_numeric(
        annotated.get(
            "viability_std",
            pd.Series(np.nan, index=annotated.index, dtype=float),
        ),
        errors="coerce",
    )
    annotated["raw_surrogate_viability_mean"] = raw_mean
    annotated["raw_surrogate_viability_std"] = raw_std

    evidence = _exact_viability_evidence(observations)
    if "formulation_id" in annotated.columns:
        annotated = annotated.merge(evidence, on="formulation_id", how="left")
    else:
        for column in evidence.columns:
            if column != "formulation_id":
                annotated[column] = np.nan
    annotated["viability_exact_observation_count"] = (
        pd.to_numeric(
            annotated.get("viability_exact_observation_count", 0),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    annotated["viability_exact_batch_count"] = (
        pd.to_numeric(
            annotated.get("viability_exact_batch_count", 0),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    annotated["viability_exact_observed_sources"] = annotated.get(
        "viability_exact_observed_sources",
        pd.Series("", index=annotated.index, dtype=object),
    ).fillna("")

    labeling = nested_get(optimization_config, "prediction_labeling", {}) or {}
    start_round = int(labeling.get("start_round", 7))
    enabled = bool(labeling.get("enabled", True)) and bool(
        target_round_number is None or int(target_round_number) >= start_round
    )
    mean_tolerance = float(
        labeling.get("prior_reversion_mean_tolerance_percent", 2.5)
    )
    std_ratio_threshold = float(
        labeling.get("prior_reversion_std_ratio_threshold", 0.95)
    )
    viability_training = pd.to_numeric(
        models.training_frame.get(
            "viability_percent",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    ).dropna()
    prior_mean = float(models.viability.fallback_mean)
    prior_std = (
        max(float(viability_training.std(ddof=0)), 1e-6)
        if not viability_training.empty
        else float("nan")
    )
    annotated["viability_surrogate_prior_mean"] = prior_mean
    annotated["viability_surrogate_prior_std"] = prior_std
    annotated["viability_prior_reversion"] = (
        raw_mean.sub(prior_mean).abs().le(mean_tolerance)
        & raw_std.ge(std_ratio_threshold * prior_std)
        if np.isfinite(prior_std)
        else False
    )

    statuses: list[str] = []
    labels: list[str] = []
    reasons: list[str] = []
    for _, row in annotated.iterrows():
        observed_count = int(row.get("viability_exact_observation_count", 0))
        observed_batches = int(row.get("viability_exact_batch_count", 0))
        recommendation_type = _text(row.get("recommendation_type", ""))
        cold_ingredients = _text(row.get("cold_start_ingredients", ""))
        cold_counts = _text(row.get("cold_start_prior_evidence_counts", ""))
        support_status = _text(row.get("support_status", ""))
        prior_reversion = bool(row.get("viability_prior_reversion", False))

        if observed_count > 0:
            status = (
                "observed_retest"
                if recommendation_type == "retest_priority"
                else "observed_supported"
            )
            label = (
                "Observed formulation — retest"
                if status == "observed_retest"
                else "Observed formulation"
            )
            reason = (
                f"exact formulation has {observed_count} viability observation(s) "
                f"across {observed_batches} batch(es)"
            )
        elif not enabled:
            status = "model_supported"
            label = "Model estimate"
            reason = "prediction labeling policy is disabled"
        elif cold_ingredients:
            status = "unknown_cold_start"
            label = "Unknown — exploratory cold-start"
            threshold = cold_start_context.policy.minimum_distinct_formulations
            evidence_text = cold_counts or "no prior formulation evidence"
            reason = (
                f"cold-start ingredient(s): {cold_ingredients}; "
                f"prior evidence: {evidence_text}; graduation threshold={threshold}"
            )
        elif support_status == "boundary":
            status = "unknown_out_of_model_support"
            label = "Unknown — outside model support"
            reason = "candidate is outside the active formulation-support radius"
        elif prior_reversion:
            status = "unknown_prior_reversion"
            label = "Unknown — surrogate prior reversion"
            reason = (
                "surrogate mean and uncertainty are indistinguishable from "
                "the training prior"
            )
        elif not bool(models.viability.fitted):
            status = "unknown_no_fitted_model"
            label = "Unknown — viability model unavailable"
            reason = "viability surrogate has insufficient observations to fit"
        else:
            status = "model_supported"
            label = "Model estimate"
            reason = "candidate has a non-prior-reverted surrogate estimate"

        statuses.append(status)
        labels.append(label)
        reasons.append(reason)

    annotated["viability_prediction_status"] = statuses
    annotated["viability_prediction_label"] = labels
    annotated["viability_prediction_reason"] = reasons
    unknown = annotated["viability_prediction_status"].map(
        is_unknown_viability_status
    )
    annotated.loc[unknown, "predicted_viability_percent"] = np.nan
    annotated.loc[unknown, "viability_std"] = np.nan
    return annotated

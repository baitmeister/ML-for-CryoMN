"""Interval and outcome-stratified diagnostics for campaign forecasts."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .endpoints import parse_bool
from .terminal_force import MODEL_FIELD


VIABILITY = "viability_percent"
VIABILITY_CUTOFF_PERCENT = 40.0
VIABILITY_TOLERANCE_PP = 10.0

INTERVAL_SCORE_COLUMNS = [
    "scope", "round_id", "endpoint", "mechanical_definition_id", "interval_type",
    "nominal_coverage", "unit", "n_eligible", "n_scored", "n_covered", "coverage",
    "mean_width", "mean_interval_score", "missed_below", "missed_above",
]
VIABILITY_SEGMENT_COLUMNS = [
    "segment", "cutoff_percent", "tolerance_pp", "n", "observed_min_percent",
    "observed_max_percent", "r2", "mae_pp", "bias_pp", "n_overpredicted", "n_hits",
    "hit_rate",
]


def full_mechanics_forecasts(table: pd.DataFrame) -> pd.DataFrame:
    """Select the campaign's formal proposal-time forecasts with observed targets."""
    if "active_phase" not in table or table.empty:
        return table.iloc[:0]
    formal = table["formal_metric_eligible"].map(lambda value: parse_bool(value) is True)
    return table.loc[formal & table["active_phase"].eq("mechanics_enabled")]


def viability_segment_metrics(viability: pd.DataFrame) -> pd.DataFrame:
    """Describe errors in groups defined by measured viability, with a pooled reference."""
    if viability.empty:
        return pd.DataFrame(columns=VIABILITY_SEGMENT_COLUMNS)
    data = viability[["observed_mean", "prediction_mean"]].apply(pd.to_numeric, errors="coerce")
    data = data.loc[np.isfinite(data).all(axis=1)]
    cutoff = VIABILITY_CUTOFF_PERCENT
    tolerance = VIABILITY_TOLERANCE_PP
    below_cutoff = data["observed_mean"].lt(cutoff)
    segments = [
        ("all", data),
        ("observed_below_cutoff", data.loc[below_cutoff]),
        ("observed_at_or_above_cutoff", data.loc[~below_cutoff]),
    ]
    rows = []
    for name, frame in segments:
        actual, predicted = frame["observed_mean"], frame["prediction_mean"]
        errors = predicted - actual
        hits = errors.abs().le(tolerance)
        rows.append({
            "segment": name,
            "cutoff_percent": cutoff,
            "tolerance_pp": tolerance,
            "n": len(frame),
            "observed_min_percent": actual.min(),
            "observed_max_percent": actual.max(),
            "r2": r2_score(actual, predicted) if actual.nunique() > 1 else np.nan,
            "mae_pp": errors.abs().mean(),
            "bias_pp": errors.mean(),
            "n_overpredicted": int(errors.gt(0).sum()),
            "n_hits": int(hits.sum()),
            "hit_rate": hits.mean(),
        })
    return pd.DataFrame(rows, columns=VIABILITY_SEGMENT_COLUMNS)


def _future_viability_bands(
    viability: pd.DataFrame,
    results_root: Path,
    strategy_decisions: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Attach only archived, pre-outcome bands for the actual production point forecast."""
    keys = ["round_id", "formulation_id"]
    bounds = ["future_lower", "future_upper"]
    columns = [*keys, "predicted", *bounds]
    archives = []
    for round_id in viability["round_id"].unique():
        strategy = strategy_decisions.get(round_id, {}).get("viability_strategy")
        source = results_root / "rounds" / round_id / "gp_comparison" / "predictions.csv"
        if strategy is None or not source.is_file():
            continue
        frozen = pd.read_csv(source)
        keep = (
            frozen["batch_id"].eq(round_id)
            & frozen["strategy"].eq(strategy)
            & frozen["stage"].eq("before_batch")
            & frozen["evidence"].eq("pre_outcome_challenger_after_slate_selection")
            & frozen["is_control"].map(lambda value: parse_bool(value) is False)
        )
        forecasts = frozen.loc[keep].rename(columns={"batch_id": "round_id"})
        archives.append(forecasts[columns].drop_duplicates(keys, keep=False))

    if not archives:
        return viability.assign(future_lower=np.nan, future_upper=np.nan)
    merged = viability.merge(
        pd.concat(archives, ignore_index=True), on=keys, how="left", validate="many_to_one",
    )
    matches_production = np.isclose(
        pd.to_numeric(merged["predicted"], errors="coerce"),
        pd.to_numeric(merged["prediction_mean"], errors="coerce"),
        atol=1e-8,
        rtol=0,
    )
    merged.loc[~matches_production, bounds] = np.nan
    return merged


def _interval_summary(
    frame: pd.DataFrame,
    lower_col: str,
    upper_col: str,
    confidence: float,
    interval_type: str,
    unit: str,
    *,
    include_pool: bool,
) -> list[dict[str, object]]:
    """Average the proper interval score over cases within each requested scope."""
    rows = []
    for (endpoint, definition), cohort in frame.groupby(
        ["endpoint", "mechanical_definition_id"], dropna=False, sort=False,
    ):
        groups = list(cohort.groupby("round_id"))
        if include_pool:
            groups.append(("FULL_MECHANICS", cohort))
        for round_id, data in groups:
            values = data[["observed_mean", lower_col, upper_col]].apply(pd.to_numeric, errors="coerce")
            valid = np.isfinite(values).all(axis=1) & values[lower_col].le(values[upper_col])
            actual, lower, upper = (values.loc[valid, column] for column in values)
            miss_below = (lower - actual).clip(lower=0)
            miss_above = (actual - upper).clip(lower=0)
            covered = miss_below.eq(0) & miss_above.eq(0)
            width = upper - lower
            score = width + (2 / (1 - confidence)) * (miss_below + miss_above)
            rows.append({
                "scope": "pooled" if round_id == "FULL_MECHANICS" else "round",
                "round_id": round_id,
                "endpoint": endpoint,
                "mechanical_definition_id": definition,
                "interval_type": interval_type,
                "nominal_coverage": confidence,
                "unit": unit,
                "n_eligible": len(data),
                "n_scored": len(width),
                "n_covered": int(covered.sum()),
                "coverage": covered.mean(),
                "mean_width": width.mean(),
                "mean_interval_score": score.mean(),
                "missed_below": int(miss_below.gt(0).sum()),
                "missed_above": int(miss_above.gt(0).sum()),
            })
    return rows


def _score_intervals(
    table: pd.DataFrame,
    results_root: Path,
    strategy_decisions: Mapping[str, Mapping[str, object]],
    confidence: float,
    *,
    include_pool: bool = False,
) -> pd.DataFrame:
    table = table.reindex(columns=list(dict.fromkeys([
        *table.columns, "endpoint", "mechanical_definition_id", "round_id",
    ])))
    viability = table.loc[table["endpoint"].eq(VIABILITY)]
    mechanics = table.loc[table["endpoint"].eq(MODEL_FIELD)]
    future = _future_viability_bands(viability, results_root, strategy_decisions)
    proposal_lower, proposal_upper = "interval_95_lower", "interval_95_upper"
    rows = _interval_summary(
        viability, proposal_lower, proposal_upper, confidence, "proposal", "pp", include_pool=include_pool,
    )
    rows += _interval_summary(
        future, "future_lower", "future_upper", 0.95, "future_batch", "pp", include_pool=include_pool,
    )
    rows += _interval_summary(
        mechanics, proposal_lower, proposal_upper, confidence, "proposal", "N_per_needle", include_pool=include_pool,
    )
    return pd.DataFrame(rows, columns=INTERVAL_SCORE_COLUMNS).sort_values(
        ["scope", "round_id", "endpoint", "interval_type"], ascending=[False, True, False, False],
    )


def round_interval_scores(
    table: pd.DataFrame,
    results_root: Path,
    strategy_decisions: Mapping[str, Mapping[str, object]],
    confidence: float,
) -> pd.DataFrame:
    """Score eligible cases in a round and retain their phase and provenance."""
    eligible = table["evaluation_eligible"].map(lambda value: parse_bool(value) is True)
    scores = _score_intervals(table.loc[eligible], results_root, strategy_decisions, confidence)
    context = table[["round_id", "provenance_class", "active_phase"]].drop_duplicates()
    return scores.merge(context, on="round_id", how="left", validate="many_to_one")


def write_full_mechanics_diagnostics(
    table: pd.DataFrame,
    directory: Path,
    results_root: Path,
    strategy_decisions: Mapping[str, Mapping[str, object]],
    confidence: float,
) -> None:
    """Write a campaign interval scorecard and an exploratory split of viability errors."""
    full = full_mechanics_forecasts(table)
    scores = _score_intervals(full, results_root, strategy_decisions, confidence, include_pool=True)
    scores.to_csv(directory / "interval_scores.csv", index=False)
    viability = full if full.empty else full.loc[full["endpoint"].eq(VIABILITY)]
    viability_segment_metrics(viability).to_csv(directory / "viability_segments.csv", index=False)

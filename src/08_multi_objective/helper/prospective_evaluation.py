"""Proposal-time prospective evaluation without surrogate-model retraining."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import (
    copy_working,
    round_artifact_paths,
    validate_completed_against_proposal,
)
from .paths import RESULTS_V2_DIR


PAGE_BG = "#f7f2e8"
AX_BG = "#fffdf8"
GRID = "#d8d0c1"
TEXT = "#2d2a26"
MUTED = "#8a8175"
BLUE = "#4c78a8"
TEAL = "#4f8f6b"
GOLD = "#d7a44c"
CORAL = "#d96c5f"

PROSPECTIVE_TABLE_COLUMNS = [
    "evaluation_policy_version",
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


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.facecolor": AX_BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "figure.facecolor": PAGE_BG,
            "savefig.facecolor": PAGE_BG,
            "grid.color": GRID,
            "grid.alpha": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
        }
    )


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


def _round_sort_key(batch_id: object) -> tuple[int, str]:
    number = _round_number(str(batch_id))
    if number is not None:
        return (0, f"{number:09d}")
    return (1, str(batch_id))


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
        evaluation_config.get("policy_version", "prospective_evaluation_v1")
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
            prediction_mean = _numeric(
                proposal_row.get(str(definition.get("prediction_mean_column", "")))
            )
            prediction_std = _numeric(
                proposal_row.get(str(definition.get("prediction_std_column", "")))
            )
            endpoint_observations = round_observations[
                (round_observations["formulation_id"].astype(str) == formulation_id)
                & (round_observations["endpoint"].astype(str) == str(endpoint))
            ].dropna(subset=["value"])
            observed_mean = (
                float(endpoint_observations["value"].mean())
                if not endpoint_observations.empty
                else None
            )
            unit = (
                str(endpoint_observations.iloc[0].get("unit", ""))
                if not endpoint_observations.empty
                else ""
            )

            exclusion_reason = ""
            if formulation_id in duplicate_formulations:
                exclusion_reason = "ambiguous_duplicate_formulation"
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
    return pd.DataFrame(rows, columns=PROSPECTIVE_TABLE_COLUMNS)


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
    ordered = table.assign(
        _round_sort=table["round_id"].map(_round_sort_key)
    ).sort_values(["_round_sort", "endpoint"])
    for (round_id, endpoint), frame in ordered.groupby(
        ["round_id", "endpoint"],
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
    for endpoint, frame in ordered.groupby("endpoint", sort=False):
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


def _placeholder_plot(path: Path, title: str, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _prediction_plot(table: pd.DataFrame, path: Path) -> Path:
    continuous = table[
        table["evaluation_eligible"].astype(bool)
        & (table["metric_type"].astype(str) == "continuous")
    ].copy()
    endpoints = [
        endpoint
        for endpoint in [
            "viability_percent",
            "critical_axial_load_N_per_needle",
        ]
        if endpoint in set(continuous["endpoint"].astype(str))
    ]
    if not endpoints:
        return _placeholder_plot(
            path,
            "Frozen predictions vs observed results",
            "No eligible continuous prospective observations yet.",
        )
    fig, axes = plt.subplots(1, len(endpoints), figsize=(7 * len(endpoints), 5.5))
    axes_array = np.atleast_1d(axes)
    labels = {
        "viability_percent": "Viability (%)",
        "critical_axial_load_N_per_needle": "Critical load (N/needle)",
    }
    for ax, endpoint in zip(axes_array, endpoints):
        frame = continuous[continuous["endpoint"].astype(str) == endpoint]
        actual = pd.to_numeric(frame["observed_mean"], errors="coerce").to_numpy(dtype=float)
        predicted = pd.to_numeric(frame["prediction_mean"], errors="coerce").to_numpy(dtype=float)
        std = pd.to_numeric(frame["prediction_std"], errors="coerce").to_numpy(dtype=float)
        valid_std = np.isfinite(std) & (std >= 0)
        yerr = np.where(valid_std, 1.96 * std, 0.0)
        colors = [TEAL if bool(value) else BLUE for value in frame["formal_cohort"]]
        ax.errorbar(
            actual,
            predicted,
            yerr=yerr,
            fmt="none",
            ecolor=GRID,
            alpha=0.8,
            capsize=3,
        )
        ax.scatter(
            actual,
            predicted,
            s=70,
            c=colors,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
        )
        low = float(np.nanmin(np.concatenate([actual, predicted])))
        high = float(np.nanmax(np.concatenate([actual, predicted])))
        pad = max((high - low) * 0.08, 1e-6)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], "--", color=MUTED)
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlabel(f"Observed {labels[endpoint]}")
        ax.set_ylabel(f"Frozen prediction {labels[endpoint]}")
        ax.set_title(labels[endpoint])
    fig.suptitle("Proposal-time predictions vs observed results", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _gate_plot(table: pd.DataFrame, path: Path) -> Path:
    frame = table[
        table["evaluation_eligible"].astype(bool)
        & (table["endpoint"].astype(str) == "intact_patch_formation_pass")
    ].copy()
    if frame.empty:
        return _placeholder_plot(
            path,
            "Prospective intact-gate calibration",
            "No eligible intact-gate observations yet.",
        )
    predicted = pd.to_numeric(frame["prediction_mean"], errors="coerce")
    actual = pd.to_numeric(frame["observed_mean"], errors="coerce")
    jitter = np.linspace(-0.06, 0.06, len(frame)) if len(frame) > 1 else np.array([0.0])
    colors = [TEAL if bool(value) else BLUE for value in frame["formal_cohort"]]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        predicted,
        actual + jitter,
        s=75,
        c=colors,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.25, 1.25)
    ax.set_yticks([0, 1], labels=["Observed fail", "Observed pass"])
    ax.set_xlabel("Frozen predicted intact-pass probability")
    ax.set_title("Proposal-time intact-gate predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _error_by_round_plot(metrics: pd.DataFrame, path: Path) -> Path:
    frame = metrics[
        (metrics["scope"].astype(str) == "round")
        & (metrics["endpoint"].astype(str) == "viability_percent")
        & pd.to_numeric(metrics["mae"], errors="coerce").notna()
    ].copy()
    if frame.empty:
        return _placeholder_plot(
            path,
            "Prospective viability error by round",
            "No round-level prospective viability metrics yet.",
        )
    frame["_sort"] = frame["round_id"].map(_round_sort_key)
    frame = frame.sort_values("_sort")
    x = np.arange(len(frame))
    colors = [TEAL if bool(value) else BLUE for value in frame["formal_cohort"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, frame["mae"], color=GRID, linewidth=2)
    ax.scatter(x, frame["mae"], c=colors, s=90, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, labels=frame["round_id"], rotation=30, ha="right")
    ax.set_ylabel("Viability MAE (percentage points)")
    ax.set_title("Frozen-prediction error by completed round")
    for index, row in frame.reset_index(drop=True).iterrows():
        ax.annotate(
            str(row["provenance_class"]),
            (index, float(row["mae"])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=MUTED,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _format_metric(value: object, digits: int = 3) -> str:
    parsed = _numeric(value)
    if parsed is None or not math.isfinite(parsed):
        return "n/a"
    return f"{parsed:.{digits}f}"


def _summary_text(
    table: pd.DataFrame,
    metrics: pd.DataFrame,
    evaluation_config: Mapping[str, Any],
    campaign: bool,
) -> str:
    formal_start = int(evaluation_config.get("formal_start_round", 3))
    lines = [
        "CryoMN v2 Prospective Evaluation",
        "=" * 33,
        "",
        f"Policy version: {evaluation_config.get('policy_version', 'prospective_evaluation_v1')}",
        f"Formal prospective cohort starts: ROUND_{formal_start:03d}",
        "Prediction source: archived proposal-time means and uncertainties",
        "Technical replicate handling: mean within candidate and round",
        "",
    ]
    if campaign:
        completed_rounds = sorted(
            set(table["round_id"].astype(str)),
            key=_round_sort_key,
        )
        lines.extend(
            [
                f"Completed rounds included: {', '.join(completed_rounds) if completed_rounds else 'none'}",
                f"Formal rounds included: {int(table.loc[table['formal_cohort'].astype(bool), 'round_id'].nunique()) if not table.empty else 0}",
                "",
            ]
        )
        formal_viability = metrics[
            (metrics["scope"].astype(str) == "pooled_formal")
            & (metrics["endpoint"].astype(str) == "viability_percent")
        ]
        if formal_viability.empty or int(formal_viability.iloc[0]["n_evaluated"]) == 0:
            lines.append("Primary pooled formal viability MAE: not available; the formal cohort has not produced eligible results yet.")
        else:
            row = formal_viability.iloc[0]
            lines.append(
                "Primary pooled formal viability MAE: "
                f"{_format_metric(row['mae'])} percentage points "
                f"(n={int(row['n_evaluated'])})"
            )
            lines.append(
                "Primary pooled formal 95% interval: "
                f"coverage={_format_metric(row['interval_95_coverage'])}; "
                f"mean width={_format_metric(row['interval_95_mean_width'])} "
                "percentage points"
            )
    else:
        round_id = str(table.iloc[0]["round_id"]) if not table.empty else "unknown"
        provenance = str(table.iloc[0]["provenance_class"]) if not table.empty else "unknown"
        formal = bool(table.iloc[0]["formal_cohort"]) if not table.empty else False
        lines.extend(
            [
                f"Round: {round_id}",
                f"Provenance classification: {provenance}",
                f"Formal cohort: {'yes' if formal else 'no'}",
                "",
            ]
        )
        for _, row in metrics[metrics["scope"].astype(str) == "round"].iterrows():
            metric_name = "Brier" if row["endpoint"] == "intact_patch_formation_pass" else "MAE"
            metric_value = row["brier_score"] if metric_name == "Brier" else row["mae"]
            lines.append(
                f"- {row['endpoint']}: n={int(row['n_evaluated'])}/{int(row['n_proposed'])}, "
                f"{metric_name}={_format_metric(metric_value)}"
            )
            if str(row.get("endpoint", "")) != "intact_patch_formation_pass":
                lines.append(
                    "  95% interval: "
                    f"coverage={_format_metric(row['interval_95_coverage'])}; "
                    f"mean width={_format_metric(row['interval_95_mean_width'])}"
                )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- Reconstructed and migration-frozen rounds are reported separately from the formal pooled metric.",
            "- Cross-validation reports retrain models and answer a different question.",
            "- Missing measurements remain visible as ineligible audit rows; they are not imputed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _promote_tree(staging_dir: Path, output_dir: Path) -> list[Path]:
    generated: list[Path] = []
    for source in sorted(path for path in staging_dir.rglob("*") if path.is_file()):
        destination = output_dir / source.relative_to(staging_dir)
        copy_working(source, destination)
        generated.append(destination)
    return generated


def generate_round_prospective_artifacts(
    batch_id: str,
    observations: pd.DataFrame,
    results_root: str | Path = RESULTS_V2_DIR,
    evaluation_config: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Generate one completed round's proposal-time evaluation bundle."""
    _apply_style()
    evaluation_config = dict(evaluation_config or {})
    results_root = Path(results_root)
    reports_dir = round_artifact_paths(batch_id, results_root).reports_dir
    reports_dir.parent.mkdir(parents=True, exist_ok=True)
    table = build_round_prospective_table(
        batch_id,
        observations,
        results_root=results_root,
        evaluation_config=evaluation_config,
    )
    metrics = summarize_prospective_metrics(table)
    with tempfile.TemporaryDirectory(
        prefix=f".{batch_id}_prospective_",
        dir=reports_dir.parent,
    ) as temporary_name:
        staging = Path(temporary_name)
        (staging / "tables").mkdir(parents=True)
        (staging / "plots").mkdir(parents=True)
        table.to_csv(staging / "tables" / "prospective_evaluation_table.csv", index=False)
        metrics[metrics["scope"].astype(str) == "round"].to_csv(
            staging / "tables" / "prospective_metrics.csv",
            index=False,
        )
        (staging / "prospective_evaluation_summary.txt").write_text(
            _summary_text(table, metrics, evaluation_config, campaign=False),
            encoding="utf-8",
        )
        _prediction_plot(
            table,
            staging / "plots" / "prospective_prediction_vs_observed.png",
        )
        _gate_plot(
            table,
            staging / "plots" / "prospective_gate_calibration.png",
        )
        return _promote_tree(staging, reports_dir)


def _completed_round_ids(results_root: Path) -> list[str]:
    rounds_dir = results_root / "rounds"
    if not rounds_dir.exists():
        return []
    round_ids = [
        path.name
        for path in rounds_dir.iterdir()
        if path.is_dir() and (path / "completed" / "completed.csv").exists()
    ]
    return sorted(round_ids, key=_round_sort_key)


def generate_campaign_prospective_artifacts(
    observations: pd.DataFrame,
    results_root: str | Path = RESULTS_V2_DIR,
    evaluation_config: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Generate pooled reports over every completed round archive."""
    _apply_style()
    evaluation_config = dict(evaluation_config or {})
    results_root = Path(results_root)
    output_dir = results_root / "reports" / "prospective"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tables = [
        build_round_prospective_table(
            batch_id,
            observations,
            results_root=results_root,
            evaluation_config=evaluation_config,
        )
        for batch_id in _completed_round_ids(results_root)
    ]
    table = (
        pd.concat(tables, ignore_index=True)
        if tables
        else pd.DataFrame(columns=PROSPECTIVE_TABLE_COLUMNS)
    )
    metrics = summarize_prospective_metrics(table)
    with tempfile.TemporaryDirectory(
        prefix=".campaign_prospective_",
        dir=output_dir.parent,
    ) as temporary_name:
        staging = Path(temporary_name)
        (staging / "tables").mkdir(parents=True)
        (staging / "plots").mkdir(parents=True)
        table.to_csv(staging / "tables" / "prospective_evaluation_table.csv", index=False)
        metrics.to_csv(staging / "tables" / "prospective_metrics.csv", index=False)
        (staging / "prospective_evaluation_summary.txt").write_text(
            _summary_text(table, metrics, evaluation_config, campaign=True),
            encoding="utf-8",
        )
        _prediction_plot(
            table,
            staging / "plots" / "prospective_prediction_vs_observed.png",
        )
        _error_by_round_plot(
            metrics,
            staging / "plots" / "prospective_error_by_round.png",
        )
        return _promote_tree(staging, output_dir)

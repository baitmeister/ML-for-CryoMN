"""Proposal-time prospective evaluation without surrogate-model retraining."""

from __future__ import annotations

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
from .endpoints import INTACT_PATCH_ENDPOINT, aggregate_intact_patch_replicates
from .paths import RESULTS_V2_DIR


from .evaluation_metrics import (
    _active_phase,
    _aggregate_completed_candidates,
    _boolean_numeric,
    _build_model_evaluation_frames,
    _campaign_and_literature_observed,
    _cross_validated_predictions,
    _hypervolume_2d,
    _igd_2d,
    _is_blank,
    _metric_row,
    _normalize_frame,
    _numeric,
    _observed_endpoint_frame,
    _paired_frame,
    _pareto_frontier_mask,
    _proposal_path,
    _proposal_prediction_value,
    _provenance,
    _replicate_count,
    _round_metrics,
    _round_number,
    _round_sort_key,
    build_round_prospective_table,
    summarize_prospective_metrics,
)

from .plot_reporting import read_strategy_decisions, write_prospective
from .forecast_diagnostics import VIABILITY, round_interval_scores, write_full_mechanics_diagnostics
from .terminal_force import MODEL_FIELD


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










def _round_sort_key(batch_id: object) -> tuple[int, str]:
    number = _round_number(str(batch_id))
    if number is not None:
        return (0, f"{number:09d}")
    return (1, str(batch_id))


























def _format_metric(value: object, digits: int = 3) -> str:
    parsed = _numeric(value)
    if parsed is None or not math.isfinite(parsed):
        return "n/a"
    return f"{parsed:.{digits}f}"


def round_interval_summary(scores: pd.DataFrame) -> str:
    """Format the same interval evidence as the single-round CSV."""
    lines = ["Interval scores"]
    if not scores.empty:
        case = scores.iloc[0]
        lines.append(f"Phase {case.active_phase}. Forecast provenance {case.provenance_class}.")
    for row in scores.itertuples(index=False):
        endpoint = "Viability" if row.endpoint == VIABILITY else "Compression"
        unit = "percentage points" if row.endpoint == VIABILITY else "N per loaded needle"
        interval = row.interval_type.replace("_", " ")
        label = f"{endpoint} {interval} ({row.nominal_coverage:.0%})"
        counted = f"Scored {row.n_scored}/{row.n_eligible} eligible cases."
        if row.n_scored:
            lines.append(
                f"- {label} mean score {_format_metric(row.mean_interval_score)} {unit}. "
                f"Coverage {row.n_covered}/{row.n_scored}. Mean width {_format_metric(row.mean_width)} {unit}. "
                f"{counted}"
            )
        else:
            lines.append(f"- {label} score unavailable. {counted}")
    for endpoint, label in [(VIABILITY, "Viability"), (MODEL_FIELD, "Compression")]:
        if not scores["endpoint"].eq(endpoint).any():
            lines.append(f"- {label} score unavailable. No eligible forecast and measurement pairs.")
    lines.append("See tables/interval_scores.csv for widths, counts, and directions of misses.")
    return "\n".join(lines) + "\n"


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
        f"Policy version: {evaluation_config.get('policy_version', 'prospective_evaluation_v2')}",
        f"Formal prospective cohort starts: ROUND_{formal_start:03d}",
        "Prediction source: archived proposal-time means and uncertainties",
        "Technical replicate handling: continuous endpoints use the mean; "
        "the intact-patch gate uses all-pass within formulation and round",
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
            endpoint = str(row["endpoint"])
            metric_name = "Brier" if endpoint == INTACT_PATCH_ENDPOINT else "MAE"
            metric_value = row["brier_score"] if metric_name == "Brier" else row["mae"]
            if endpoint == INTACT_PATCH_ENDPOINT:
                gate_rows = table[
                    (table["endpoint"].astype(str) == INTACT_PATCH_ENDPOINT)
                    & table["evaluation_eligible"].astype(bool)
                ].copy()
                gate_values = pd.to_numeric(
                    gate_rows["observed_mean"],
                    errors="coerce",
                ).dropna()
                passed = int((gate_values >= 0.5).sum())
                failed = int((gate_values < 0.5).sum())
                lines.append(
                    "- Intact-patch formation gate: formulations observed="
                    f"{int(row['n_evaluated'])}/{int(row['n_proposed'])}; "
                    f"passed={passed}; failed={failed}; "
                    f"Brier={_format_metric(metric_value)}; "
                    f"accuracy={_format_metric(row['accuracy'])}"
                )
                continue
            lines.append(
                f"- {row['endpoint']}: formulations observed="
                f"{int(row['n_evaluated'])}/{int(row['n_proposed'])}, "
                f"{metric_name}={_format_metric(metric_value)}"
            )
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
    strategy_decisions = read_strategy_decisions(results_root)
    confidence = float(evaluation_config.get("prediction_interval", {}).get("confidence", 0.95))
    interval_scores = round_interval_scores(table, results_root, strategy_decisions, confidence)
    with tempfile.TemporaryDirectory(
        prefix=f".{batch_id}_prospective_",
        dir=reports_dir.parent,
    ) as temporary_name:
        staging = Path(temporary_name)
        (staging / "tables").mkdir(parents=True)
        (staging / "plots").mkdir(parents=True)
        table.to_csv(staging / "tables" / "prospective_evaluation_table.csv", index=False)
        interval_scores.to_csv(staging / "tables" / "interval_scores.csv", index=False)
        metrics[metrics["scope"].astype(str) == "round"].to_csv(
            staging / "tables" / "prospective_metrics.csv",
            index=False,
        )
        (staging / "prospective_evaluation_summary.txt").write_text(
            _summary_text(table, metrics, evaluation_config, campaign=False)
            + "\n" + round_interval_summary(interval_scores),
            encoding="utf-8",
        )
        round_observations=observations.loc[observations.batch_id.astype(str).eq(batch_id)]
        write_prospective(round_observations,table,metrics,pd.DataFrame(),staging/'plots',f'{batch_id} only; frozen prospective evidence',strategy_decisions=strategy_decisions)
        from .group10_reporting import additional_reports
        additional_reports(observations, table, results_root, staging / "tables")
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
    *, include_publication_summary: bool = False,
) -> list[Path]:
    """Generate pooled reports over every completed round archive."""
    evaluation_config = dict(evaluation_config or {})
    results_root = Path(results_root)
    output_dir = results_root / "reports" / "prospective"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    refresh_publication = include_publication_summary or (output_dir / "plots/publication_summary.png").is_file()
    strategy_decisions = read_strategy_decisions(results_root)
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
        write_full_mechanics_diagnostics(
            table, staging / "tables", results_root, strategy_decisions,
            confidence=float(evaluation_config.get("prediction_interval", {}).get("confidence", 0.95)),
        )
        write_prospective(observations,table,metrics,pd.DataFrame(),staging/'plots','Cumulative campaign; frozen prospective evidence',
                          include_publication_summary=refresh_publication,strategy_decisions=strategy_decisions)
        from .group10_reporting import additional_reports
        additional_reports(observations, table, results_root, staging / "tables")
        return _promote_tree(staging, output_dir)



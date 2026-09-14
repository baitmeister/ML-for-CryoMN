"""Generate v2 review snapshots and real multi-objective evaluation graphics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

if __package__ in (None, ""):
    V2_ROOT = Path(__file__).resolve().parents[1]
    if str(V2_ROOT) not in sys.path:
        sys.path.insert(0, str(V2_ROOT))
    from helper.config import load_optimization_config
    from helper.endpoints import INTACT_PATCH_ENDPOINT, aggregate_intact_patch_replicates, parse_bool
    from helper.feasibility import annotate_feasibility
    from helper.models import build_training_frame, train_endpoint_models
    from helper.paths import FORMULATIONS_PATH, NEXT_ROUND_CANDIDATES_PATH, OBSERVATIONS_PATH, VISUALIZATIONS_DIR
    from helper.registry import IngredientRegistry, load_registry
else:
    from .config import load_optimization_config
    from .endpoints import INTACT_PATCH_ENDPOINT, aggregate_intact_patch_replicates, parse_bool
    from .feasibility import annotate_feasibility
    from .models import build_training_frame, train_endpoint_models
    from .paths import FORMULATIONS_PATH, NEXT_ROUND_CANDIDATES_PATH, OBSERVATIONS_PATH, VISUALIZATIONS_DIR
    from .registry import IngredientRegistry, load_registry



if __package__ in (None, ""):
    from helper.evaluation_metrics import (
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
else:
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


if __package__ in (None, ""):
    from helper.evaluation_plots import _artifact_path, _format_metric
    from helper.plot_reporting import write_decision, write_pareto, write_diagnostics, write_mechanical, prepare_diagnostics
else:
    from .evaluation_plots import _artifact_path, _format_metric
    from .plot_reporting import write_decision, write_pareto, write_diagnostics, write_mechanical, prepare_diagnostics

PAGE_BG = "#f7f2e8"
AX_BG = "#fffdf8"
GRID = "#d8d0c1"
TEXT = "#2d2a26"
MUTED = "#8a8175"
BLUE = "#4c78a8"
TEAL = "#4f8f6b"
GOLD = "#d7a44c"
CORAL = "#d96c5f"
SLATE = "#6c7a89"





def _read_or_empty(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()






def _candidate_viability_text(row: pd.Series) -> str:
    status = str(row.get("viability_prediction_status", "")).strip()
    if status.startswith("unknown_"):
        return f"viability unknown ({status})"
    value = pd.to_numeric(row.get("predicted_viability_percent"), errors="coerce")
    if pd.isna(value):
        return "viability unknown"
    return f"predicted viability {float(value):.1f}%"




def _format_formulation(row: pd.Series, registry: IngredientRegistry) -> str:
    ingredients = []
    for feature_name in registry.feature_names:
        if feature_name not in row.index:
            continue
        value = pd.to_numeric(row.get(feature_name), errors="coerce")
        if pd.isna(value) or float(value) <= 0.0:
            continue
        display_name = registry.get_by_feature(feature_name).display_name
        if feature_name.endswith("_pct"):
            ingredients.append(f"{float(value):.3g}% {display_name}")
        elif float(value) >= 1.0:
            ingredients.append(f"{float(value):.3g}M {display_name}")
        else:
            ingredients.append(f"{float(value) * 1000:.3g}mM {display_name}")
    return " + ".join(ingredients) if ingredients else "No active ingredients"








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
        frame = frame.sort_values("predicted_viability_percent", ascending=False, na_position="last")
    return frame






def _write_best_performers_summary(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
    output_dir: Path,
    registry: IngredientRegistry,
    artifact_prefix: str = "",
    candidate_heading: str = "Current leading next-round candidates:",
) -> Path:
    observed = _observed_endpoint_frame(formulations, observations)
    candidate_summary = _aggregate_completed_candidates(candidates)
    show_all_candidates = "completed-round" in candidate_heading
    top_candidates = _top_candidate_frame(candidate_summary)
    if not show_all_candidates:
        top_candidates = top_candidates.head(5)
    campaign_observed, literature_observed = _campaign_and_literature_observed(
        observed,
        registry,
    )
    lines = [
        "CryoMN v2 Best Performers",
        "=" * 25,
        "",
        "Database snapshot:",
        f"- Formulations tracked: {len(formulations)}",
        f"- Observation rows: {len(observations)}",
        f"- Candidate formulations shown: {len(candidate_summary)}",
        "",
    ]

    viability_frame = campaign_observed.dropna(
        subset=["viability_percent"]
    ).sort_values(
        "viability_percent",
        ascending=False,
    )
    lines.append("Best observed viability performers:")
    lines.append("- scope: feasible wetlab_feedback campaign formulations only")
    if viability_frame.empty:
        lines.append("- none yet")
    else:
        for rank, (_, row) in enumerate(viability_frame.head(5).iterrows(), start=1):
            lines.append(
                f"- #{rank} {row['formulation_id']} | viability {_format_metric(row['viability_percent'], '{:.1f}')}% | "
                f"source {row.get('observed_sources') or row.get('source') or 'unknown'}"
            )
            lines.append(f"  formulation: {_format_formulation(row, registry)}")
    lines.append("")

    literature_viability = literature_observed.dropna(
        subset=["viability_percent"]
    ).sort_values("viability_percent", ascending=False)
    lines.append("Historical literature-reference viability leaders:")
    if literature_viability.empty:
        lines.append("- none")
    else:
        for rank, (_, row) in enumerate(
            literature_viability.head(5).iterrows(),
            start=1,
        ):
            lines.append(
                f"- #{rank} {row['formulation_id']} | viability "
                f"{_format_metric(row['viability_percent'], '{:.1f}')}%"
            )
            lines.append(f"  formulation: {_format_formulation(row, registry)}")
    lines.append("")

    load_column = "critical_axial_load_N_per_needle"
    load_frame = (
        campaign_observed.dropna(subset=[load_column]).sort_values(
            load_column,
            ascending=False,
        )
        if load_column in campaign_observed.columns
        else pd.DataFrame()
    )
    lines.append("Best observed mechanical performers:")
    lines.append("- scope: feasible wetlab_feedback campaign formulations only")
    if load_frame.empty:
        lines.append("- none yet; no critical axial load measurements are in the v2 database")
    else:
        for rank, (_, row) in enumerate(load_frame.head(5).iterrows(), start=1):
            viability_text = ""
            if pd.notna(row.get("viability_percent")):
                viability_text = f" | viability {_format_metric(row['viability_percent'], '{:.1f}')}%"
            lines.append(
                f"- #{rank} {row['formulation_id']} | critical load {_format_metric(row[load_column], '{:.3f}')} N/needle"
                f"{viability_text}"
            )
            lines.append(f"  formulation: {_format_formulation(row, registry)}")
    lines.append("")

    balanced = (
        campaign_observed.dropna(
            subset=["viability_percent", load_column]
        ).copy()
        if {"viability_percent", load_column}.issubset(
            campaign_observed.columns
        )
        else pd.DataFrame()
    )
    lines.append("Balanced multi-objective leaders:")
    lines.append("- scope: feasible wetlab_feedback campaign formulations only")
    if balanced.empty:
        lines.append("- none yet; this section appears after formulations have both viability and mechanical measurements")
    else:
        frontier = balanced.loc[_pareto_frontier_mask(balanced, "viability_percent", load_column)].copy()
        viability_scaled = frontier["viability_percent"].rank(pct=True)
        load_scaled = frontier[load_column].rank(pct=True)
        frontier["balanced_score"] = 0.5 * viability_scaled + 0.5 * load_scaled
        frontier = frontier.sort_values(["balanced_score", "viability_percent"], ascending=[False, False])
        for rank, (_, row) in enumerate(frontier.head(5).iterrows(), start=1):
            lines.append(
                f"- #{rank} {row['formulation_id']} | viability {_format_metric(row['viability_percent'], '{:.1f}')}% | "
                f"critical load {_format_metric(row[load_column], '{:.3f}')} N/needle"
            )
            lines.append(f"  formulation: {_format_formulation(row, registry)}")
    lines.append("")

    retest_mask = (
        candidate_summary.get(
            "recommendation_type",
            pd.Series(
                [""] * len(candidate_summary),
                index=candidate_summary.index,
                dtype="object",
            ),
        ).astype(str)
        == "retest_priority"
    ) if not candidate_summary.empty else pd.Series(dtype=bool)
    retest_candidates = (
        candidate_summary.loc[retest_mask].copy()
        if not candidate_summary.empty
        else pd.DataFrame()
    )
    lines.append("Retest-priority recommendations:")
    if retest_candidates.empty:
        lines.append("- none in the current slate")
    else:
        for _, row in retest_candidates.iterrows():
            lines.append(
                f"- {row.get('formulation_id', '')} | "
                f"{_candidate_viability_text(row)} | "
                f"intact probability {_format_metric(pd.to_numeric(row.get('intact_patch_pass_probability'), errors='coerce'), '{:.2f}')}"
            )
            if str(row.get("selection_explanation", "")).strip():
                lines.append(f"  note: {row['selection_explanation']}")
            lines.append(f"  formulation: {_format_formulation(row, registry)}")
    lines.append("")

    lines.append(candidate_heading)
    if top_candidates.empty:
        lines.append("- none; run Stage 02 to generate a candidate slate")
    else:
        for _, row in top_candidates.iterrows():
            selection_rank = pd.to_numeric(row.get("selection_rank"), errors="coerce")
            prefix = f"#{int(selection_rank)}" if pd.notna(selection_rank) else row.get("candidate_id", "candidate")
            lines.append(
                f"- {prefix} {row.get('formulation_id', '')} | "
                f"{_candidate_viability_text(row)} | "
                f"intact probability {_format_metric(pd.to_numeric(row.get('intact_patch_pass_probability'), errors='coerce'), '{:.2f}')}"
                f" | mechanical test {bool(row.get('mechanical_test_recommended', False))}"
            )
            if show_all_candidates:
                viability_mean = pd.to_numeric(
                    row.get("completed_viability_mean"),
                    errors="coerce",
                )
                viability_sd = pd.to_numeric(
                    row.get("completed_viability_sd"),
                    errors="coerce",
                )
                viability_count = pd.to_numeric(
                    row.get("completed_viability_replicate_count"),
                    errors="coerce",
                )
                intact_gate = pd.to_numeric(
                    row.get("completed_intact_gate_pass"),
                    errors="coerce",
                )
                lines.append(
                    "  completed result: viability "
                    f"{_format_metric(viability_mean, '{:.2f}')}% "
                    f"(sample SD {_format_metric(viability_sd, '{:.2f}')}, "
                    f"n={int(viability_count) if pd.notna(viability_count) else 0}); "
                    "intact "
                    + (
                        "pass"
                        if pd.notna(intact_gate)
                        and float(intact_gate) >= 0.5
                        else "fail"
                        if pd.notna(intact_gate)
                        else "not recorded"
                    )
                )
            lines.append(f"  formulation: {_format_formulation(row, registry)}")

    output_path = _artifact_path(output_dir, "best_performers_summary", ".txt", artifact_prefix)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path








def _write_model_evaluation_table(
    evaluation_frames: dict[str, pd.DataFrame],
    output_dir: Path,
    artifact_prefix: str = "",
) -> Path | None:
    rows: list[pd.DataFrame] = []
    for endpoint, frame in evaluation_frames.items():
        if frame.empty:
            continue
        table = frame.copy()
        table["endpoint"] = endpoint
        table["absolute_error"] = (table["predicted"] - table["actual"]).abs()
        table["squared_error"] = (table["predicted"] - table["actual"]) ** 2
        rows.append(
            table[
                [
                    "endpoint",
                    "formulation_id",
                    "batch_id",
                    "actual",
                    "predicted",
                    "predicted_std",
                    "absolute_error",
                    "squared_error",
                ]
            ]
        )
    if not rows:
        return None

    output_path = _artifact_path(output_dir, "model_evaluation_table", ".csv", artifact_prefix)
    pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)
    return output_path






























def _write_multiobjective_summary(
    output_dir: Path,
    paired: pd.DataFrame,
    metrics: pd.DataFrame,
    generated_paths: list[Path],
    artifact_prefix: str = "",
) -> Path:
    summary_path = _artifact_path(output_dir, "multiobjective_evaluation_summary", ".txt", artifact_prefix)
    lines = [
        "CryoMN v2 Multi-Objective Evaluation",
        "===================================",
        "",
        f"Paired viability/load rows: {len(paired)}",
        f"Distinct paired batches: {int(paired['batch_id'].nunique()) if not paired.empty else 0}",
        f"Distinct paired formulations: {int(paired['formulation_id'].nunique()) if not paired.empty else 0}",
        "",
    ]
    if metrics.empty:
        lines.append(
            "Not enough real paired viability-plus-load observations are currently present to compute roundwise hypervolume, IGD, or Pareto progression."
        )
    else:
        last = metrics.iloc[-1]
        lines.extend(
            [
                f"Latest cumulative normalized hypervolume: {last['normalized_hypervolume']:.4f}",
                f"Latest cumulative IGD: {last['igd']:.4f}" if pd.notna(last["igd"]) else "Latest cumulative IGD: n/a",
                "",
                "Generated files:",
            ]
        )
        lines.extend(f"- {path.name}" for path in generated_paths)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _write_visualization_summary(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
    generated: list[Path],
    output_dir: Path,
    review_label: str = "",
    artifact_prefix: str = "",
    base_name: str = "visualization_summary",
) -> Path:
    observed = _observed_endpoint_frame(formulations, observations)
    candidate_summary = _aggregate_completed_candidates(candidates)
    summary = [
        "CryoMN v2 Round Review Summary",
        "=" * 30,
        "",
        f"Review label: {review_label or 'default'}",
        f"Formulation rows: {len(formulations)}",
        f"Observation rows: {len(observations)}",
        f"Candidate worksheet rows: {len(candidates)}",
        f"Unique candidates: {len(candidate_summary)}",
        f"Retest-priority candidates in current slate: {int((candidate_summary.get('recommendation_type', pd.Series([''] * len(candidate_summary), index=candidate_summary.index, dtype='object')).astype(str) == 'retest_priority').sum()) if not candidate_summary.empty else 0}",
        f"Formulations with observed viability: {int(observed.get('viability_percent', pd.Series(dtype=float)).notna().sum()) if not observed.empty else 0}",
        f"Formulations with observed critical load: {int(observed.get('critical_axial_load_N_per_needle', pd.Series(dtype=float)).notna().sum()) if not observed.empty else 0}",
        f"Formulations with observed intact-patch gate: {int(observed.get('intact_patch_formation_pass', pd.Series(dtype=float)).notna().sum()) if not observed.empty else 0}",
        "",
        "Generated files:",
    ]
    summary.extend(f"- {path.name}" for path in generated)
    if not generated:
        summary.append("- none; not enough data for plots or reports")
    summary.append("")
    summary.append("Reader note:")
    summary.append("This review snapshot captures one specific state of the round workflow.")
    output_path = _artifact_path(output_dir, base_name, ".txt", artifact_prefix)
    output_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return output_path


def generate_visualization_artifacts(formulations, observations, candidates, output_dir,
                                     review_label="", artifact_prefix="") -> list[Path]:
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    registry=load_registry()
    frames=_build_model_evaluation_frames(formulations,observations,registry)
    generated=[p for p in [
        _write_best_performers_summary(formulations,observations,candidates,output_dir,registry,artifact_prefix=artifact_prefix),
        _write_model_evaluation_table(frames,output_dir,artifact_prefix=artifact_prefix)] if p is not None]
    generated+=write_pareto(formulations,observations,candidates,output_dir,artifact_prefix,review_label or 'Current evidence')
    generated+=write_decision(candidates,output_dir,artifact_prefix)
    generated+=write_diagnostics(prepare_diagnostics(formulations,observations,frames),output_dir,artifact_prefix,review_label or 'Current evidence')
    generated.append(_write_visualization_summary(formulations,observations,candidates,generated,output_dir,review_label,artifact_prefix))
    return generated



def generate_proposal_artifacts(candidates: pd.DataFrame, proposal_dir: str | Path) -> list[Path]:
    """Render stored proposal decisions without recomputing assignments."""
    return write_decision(candidates,Path(proposal_dir)/"plots")



def generate_completed_round_artifacts(formulations, observations, completed_candidates,
                                       reports_dir, batch_id) -> list[Path]:
    reports_dir=Path(reports_dir);plots_dir=reports_dir/'plots';tables_dir=reports_dir/'tables'
    reports_dir.mkdir(parents=True,exist_ok=True);tables_dir.mkdir(parents=True,exist_ok=True)
    registry=load_registry();frames=_build_model_evaluation_frames(formulations,observations,registry)
    generated=[p for p in [
        _write_best_performers_summary(formulations,observations,completed_candidates,reports_dir,registry,
                                      candidate_heading=f"{batch_id} completed-round candidates:"),
        _write_model_evaluation_table(frames,tables_dir)] if p is not None]
    generated+=write_pareto(formulations,observations,pd.DataFrame(),plots_dir,context=f'State after {batch_id}')
    generated+=write_mechanical(observations,completed_candidates,plots_dir,batch_id=batch_id,
                                context=f'State after {batch_id}')
    generated+=write_diagnostics(prepare_diagnostics(formulations,observations,frames),plots_dir,context=f'State after {batch_id}')
    generated.append(_write_visualization_summary(formulations,observations,completed_candidates,generated,
                     reports_dir,review_label=f'state_after_ingest_{batch_id}',base_name='report_summary'))
    return generated



def generate_multiobjective_evaluation_artifacts(formulations, observations, output_dir,
                                                artifact_prefix="") -> list[Path]:
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    inputs=prepare_diagnostics(formulations,observations)
    metrics_path=_artifact_path(output_dir,'multiobjective_round_metrics','.csv',artifact_prefix)
    inputs[2].to_csv(metrics_path,index=False)
    generated=write_diagnostics(inputs,output_dir,artifact_prefix)
    generated+=write_pareto(formulations,observations,pd.DataFrame(),output_dir,artifact_prefix)
    paired=_paired_frame(formulations,observations,load_registry())
    generated.append(_write_multiobjective_summary(output_dir,paired,inputs[2],generated+[metrics_path],artifact_prefix))
    return generated+[metrics_path]



def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["review", "multiobjective", "all"],
        default="all",
        help="Visualization entrypoint to run from the command line.",
    )
    parser.add_argument("--formulations", default=str(FORMULATIONS_PATH))
    parser.add_argument("--observations", default=str(OBSERVATIONS_PATH))
    parser.add_argument("--candidates", default=str(NEXT_ROUND_CANDIDATES_PATH))
    parser.add_argument("--output-dir", default=str(VISUALIZATIONS_DIR))
    parser.add_argument("--artifact-prefix", default="")
    parser.add_argument("--review-label", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    formulations = _read_or_empty(args.formulations)
    observations = _read_or_empty(args.observations)
    generated: list[Path] = []
    if args.mode in {"review", "all"}:
        candidates = _read_or_empty(args.candidates)
        generated.extend(
            generate_visualization_artifacts(
                formulations,
                observations,
                candidates,
                args.output_dir,
                review_label=args.review_label,
                artifact_prefix=args.artifact_prefix,
            )
        )
    if args.mode in {"multiobjective", "all"}:
        generated.extend(
            generate_multiobjective_evaluation_artifacts(
                formulations,
                observations,
                args.output_dir,
                artifact_prefix=args.artifact_prefix,
            )
        )
    print(f"Wrote {len(generated)} visualization artifact(s) to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()


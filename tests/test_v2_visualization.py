from __future__ import annotations

from pathlib import Path

import pandas as pd

from helper.registry import load_registry
from helper.visualization import (
    _apply_plot_style,
    generate_multiobjective_evaluation_artifacts,
    generate_visualization_artifacts,
    _save_candidate_plot,
    _save_model_evaluation_overview,
    _save_observed_performance_landscape,
    _write_best_performers_summary,
)


def _formulations() -> pd.DataFrame:
    registry = load_registry()
    rows = []
    for index in range(4):
        row = {feature_name: 0.0 for feature_name in registry.feature_names}
        row["formulation_id"] = f"v2_test_{index}"
        row["source"] = "wet_lab"
        row["source_row_id"] = index
        row["formulation_label"] = f"test formulation {index}"
        row["active_ingredient_count"] = 2
        row["dmso_M"] = 0.01 * (index + 1)
        row["ectoin_M"] = 0.1 * (index + 1)
        rows.append(row)
    return pd.DataFrame(rows)


def _observations() -> pd.DataFrame:
    records = []
    viability = [40.0, 58.0, 71.0, 84.0]
    intact = [0.0, 0.0, 1.0, 1.0]
    critical = [0.12, 0.24, 0.31, 0.48]
    for index in range(4):
        formulation_id = f"v2_test_{index}"
        records.extend(
            [
                {
                    "observation_id": f"{formulation_id}_viability",
                    "formulation_id": formulation_id,
                    "batch_id": f"ROUND_{index + 1:03d}",
                    "replicate_id": "rep_001",
                    "endpoint": "viability_percent",
                    "value": viability[index],
                    "unit": "percent",
                    "observation_noise": 1.0,
                    "source_type": "wet_lab",
                    "source_file": "test.csv",
                    "notes": "",
                },
                {
                    "observation_id": f"{formulation_id}_intact",
                    "formulation_id": formulation_id,
                    "batch_id": f"ROUND_{index + 1:03d}",
                    "replicate_id": "rep_001",
                    "endpoint": "intact_patch_formation_pass",
                    "value": intact[index],
                    "unit": "bool",
                    "observation_noise": 0.0,
                    "source_type": "wet_lab",
                    "source_file": "test.csv",
                    "notes": "",
                },
                {
                    "observation_id": f"{formulation_id}_load",
                    "formulation_id": formulation_id,
                    "batch_id": f"ROUND_{index + 1:03d}",
                    "replicate_id": "rep_001",
                    "endpoint": "critical_axial_load_N_per_needle",
                    "value": critical[index],
                    "unit": "N_per_needle",
                    "observation_noise": 0.02,
                    "source_type": "wet_lab",
                    "source_file": "test.csv",
                    "notes": "",
                },
            ]
        )
    return pd.DataFrame(records)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "formulation_id": "v2_test_3",
                "candidate_id": "cand_003",
                "selection_rank": 1,
                "mechanical_test_recommended": True,
                "predicted_viability_percent": 82.0,
                "intact_patch_pass_probability": 0.92,
                "predicted_critical_axial_load_N_per_needle": 0.45,
                "dmso_M": 0.04,
                "ectoin_M": 0.40,
            },
            {
                "formulation_id": "v2_test_2",
                "candidate_id": "cand_002",
                "selection_rank": 2,
                "mechanical_test_recommended": False,
                "predicted_viability_percent": 75.0,
                "intact_patch_pass_probability": 0.73,
                "predicted_critical_axial_load_N_per_needle": 0.30,
                "dmso_M": 0.03,
                "ectoin_M": 0.30,
            },
        ]
    )


def test_best_performers_summary_includes_observed_and_candidate_sections(tmp_path: Path) -> None:
    _apply_plot_style()
    registry = load_registry()
    path = _write_best_performers_summary(
        _formulations(),
        _observations(),
        _candidates(),
        tmp_path,
        registry,
    )
    text = path.read_text(encoding="utf-8")
    assert "Best observed viability performers:" in text
    assert "Best observed mechanical performers:" in text
    assert "Balanced multi-objective leaders:" in text
    assert "Current leading next-round candidates:" in text
    assert "40mM DMSO + 400mM ectoin" in text


def test_visualization_plots_are_created_with_small_complete_dataset(tmp_path: Path) -> None:
    _apply_plot_style()
    registry = load_registry()
    formulations = _formulations()
    observations = _observations()
    candidates = _candidates()

    model_eval = _save_model_evaluation_overview(formulations, observations, tmp_path, registry)
    observed_plot = _save_observed_performance_landscape(formulations, observations, tmp_path)
    candidate_plot = _save_candidate_plot(candidates, tmp_path)

    assert model_eval is not None and model_eval.exists()
    assert observed_plot is not None and observed_plot.exists()
    assert candidate_plot is not None and candidate_plot.exists()


def test_generate_visualization_artifacts_writes_prefixed_archive_and_eval_table(tmp_path: Path) -> None:
    _apply_plot_style()
    generated = generate_visualization_artifacts(
        _formulations(),
        _observations(),
        _candidates(),
        tmp_path,
        review_label="selection_state_before_update_ROUND_001",
        artifact_prefix="ROUND_001",
    )

    generated_names = {path.name for path in generated}
    assert "ROUND_001_best_performers_summary.txt" in generated_names
    assert "ROUND_001_visualization_summary.txt" in generated_names
    assert "ROUND_001_model_evaluation_table.csv" in generated_names
    assert "ROUND_001_model_evaluation_overview.png" in generated_names

    eval_table = pd.read_csv(tmp_path / "ROUND_001_model_evaluation_table.csv")
    assert set(eval_table["endpoint"]) == {
        "viability_percent",
        "critical_axial_load_N_per_needle",
        "intact_patch_formation_pass",
    }
    assert {"actual", "predicted", "predicted_std", "absolute_error", "squared_error"}.issubset(
        set(eval_table.columns)
    )


def test_generate_multiobjective_evaluation_artifacts_writes_unified_graph_set(tmp_path: Path) -> None:
    _apply_plot_style()
    generated = generate_multiobjective_evaluation_artifacts(
        _formulations(),
        _observations(),
        tmp_path,
        artifact_prefix="ROUND_004",
    )

    generated_names = {path.name for path in generated}
    assert "ROUND_004_multiobjective_paired_parity.png" in generated_names
    assert "ROUND_004_normalized_hypervolume_igd_vs_round.png" in generated_names
    assert "ROUND_004_pareto_front_progression.png" in generated_names
    assert "ROUND_004_endpoint_r2_vs_round.png" in generated_names
    assert "ROUND_004_multiobjective_round_metrics.csv" in generated_names
    assert "ROUND_004_multiobjective_evaluation_summary.txt" in generated_names

    metrics = pd.read_csv(tmp_path / "ROUND_004_multiobjective_round_metrics.csv")
    assert list(metrics["batch_id"]) == ["ROUND_001", "ROUND_002", "ROUND_003", "ROUND_004"]
    assert {"normalized_hypervolume", "igd", "pareto_points_cumulative"}.issubset(set(metrics.columns))

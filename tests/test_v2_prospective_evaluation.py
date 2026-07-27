from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from helper.prospective_evaluation import (
    build_round_prospective_table,
    generate_campaign_prospective_artifacts,
    generate_round_prospective_artifacts,
    summarize_prospective_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "08_multi_objective"
    / "04_report_campaign"
    / "report_campaign.py"
)

EVALUATION_CONFIG = {
    "policy_version": "test_policy",
    "formal_start_round": 3,
    "prediction_interval": {"z_value": 1.96},
    "round_provenance": {
        "ROUND_001": "reconstructed",
        "ROUND_002": "migration_frozen_supplementary",
        "default": "formal_frozen",
    },
    "endpoints": {
        "viability_percent": {
            "prediction_mean_column": "predicted_viability_percent",
            "prediction_std_column": "viability_std",
            "metric_type": "continuous",
            "role": "primary",
        },
        "intact_patch_formation_pass": {
            "prediction_mean_column": "intact_patch_pass_probability",
            "metric_type": "binary",
            "role": "gate_secondary",
            "classification_threshold": 0.5,
        },
        "critical_axial_load_N_per_needle": {
            "prediction_mean_column": "predicted_critical_axial_load_N_per_needle",
            "prediction_std_column": "critical_axial_load_std",
            "metric_type": "continuous",
            "role": "mechanical_secondary",
            "formal_phase": "mechanics_enabled",
        },
    },
}


def _proposal_rows(round_id: str) -> list[dict[str, object]]:
    common = {
        "batch_id": round_id,
        "recommendation_type": "screening_candidate",
        "replicate_id": "",
        "viability_percent": "",
        "intact_patch_formation_pass": "",
        "critical_axial_load_N_per_needle": "",
        "notes": "",
    }
    return [
        {
            **common,
            "candidate_id": "candidate_1",
            "formulation_id": "formulation_1",
            "selection_rank": 1,
            "dmso_M": 0.1,
            "predicted_viability_percent": 80.0,
            "viability_std": 10.0,
            "intact_patch_pass_probability": 0.8,
            "predicted_critical_axial_load_N_per_needle": 0.4,
            "critical_axial_load_std": 0.1,
        },
        {
            **common,
            "candidate_id": "candidate_2",
            "formulation_id": "formulation_2",
            "selection_rank": 2,
            "dmso_M": 0.2,
            "predicted_viability_percent": 60.0,
            "viability_std": 5.0,
            "intact_patch_pass_probability": 0.3,
            "predicted_critical_axial_load_N_per_needle": 0.2,
            "critical_axial_load_std": 0.05,
        },
        {
            **common,
            "candidate_id": "candidate_3",
            "formulation_id": "formulation_3",
            "selection_rank": 3,
            "dmso_M": 0.3,
            "predicted_viability_percent": 70.0,
            "viability_std": 8.0,
            "intact_patch_pass_probability": 0.6,
            "predicted_critical_axial_load_N_per_needle": 0.3,
            "critical_axial_load_std": 0.08,
        },
    ]


def _write_round(
    results_root: Path,
    round_id: str,
    *,
    reconstructed: bool = False,
    active_phase: str = "mechanics_enabled",
) -> None:
    proposal_dir = results_root / "rounds" / round_id / "proposal"
    completed_dir = results_root / "rounds" / round_id / "completed"
    proposal_dir.mkdir(parents=True)
    completed_dir.mkdir(parents=True)
    proposal_name = "proposal_reconstructed.csv" if reconstructed else "proposal.csv"
    proposal = pd.DataFrame(_proposal_rows(round_id))
    proposal.to_csv(proposal_dir / proposal_name, index=False)
    proposal.to_csv(completed_dir / "completed.csv", index=False)
    (proposal_dir / "selection_metadata.json").write_text(
        f'{{"active_phase": "{active_phase}"}}\n',
        encoding="utf-8",
    )


def _observations(round_id: str) -> pd.DataFrame:
    rows = [
        ("formulation_1", "rep_1", "viability_percent", 70.0, "percent"),
        ("formulation_1", "rep_2", "viability_percent", 74.0, "percent"),
        ("formulation_2", "rep_1", "viability_percent", 50.0, "percent"),
        ("formulation_1", "rep_1", "intact_patch_formation_pass", 1.0, "boolean"),
        ("formulation_1", "rep_2", "intact_patch_formation_pass", 1.0, "boolean"),
        ("formulation_2", "rep_1", "intact_patch_formation_pass", 0.0, "boolean"),
        (
            "formulation_1",
            "rep_1",
            "critical_axial_load_N_per_needle",
            0.3,
            "N/needle",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "observation_id": f"obs_{round_id}_{index}",
                "formulation_id": formulation_id,
                "batch_id": round_id,
                "replicate_id": replicate_id,
                "endpoint": endpoint,
                "value": value,
                "unit": unit,
            }
            for index, (formulation_id, replicate_id, endpoint, value, unit) in enumerate(
                rows,
                start=1,
            )
        ]
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_known_metrics_replicate_aggregation_and_missing_audit_rows(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    _write_round(results_root, "ROUND_003")

    table = build_round_prospective_table(
        "ROUND_003",
        _observations("ROUND_003"),
        results_root=results_root,
        evaluation_config=EVALUATION_CONFIG,
    )
    viability = table[table["endpoint"] == "viability_percent"].set_index(
        "candidate_id"
    )
    assert viability.loc["candidate_1", "prediction_mean"] == 80.0
    assert viability.loc["candidate_1", "observed_mean"] == 72.0
    assert viability.loc["candidate_1", "replicate_count"] == 2
    assert viability.loc["candidate_3", "evaluation_eligible"] == np.False_
    assert viability.loc["candidate_3", "exclusion_reason"] == "not_measured"

    metrics = summarize_prospective_metrics(table)
    round_viability = metrics[
        (metrics["scope"] == "round")
        & (metrics["endpoint"] == "viability_percent")
    ].iloc[0]
    assert round_viability["n_proposed"] == 3
    assert round_viability["n_evaluated"] == 2
    assert np.isclose(round_viability["mae"], 9.0)
    assert np.isclose(round_viability["rmse"], np.sqrt(82.0))
    assert np.isclose(round_viability["bias"], 9.0)
    assert np.isclose(round_viability["r2"], 1.0 - 164.0 / 242.0)
    assert np.isclose(round_viability["interval_95_coverage"], 0.5)

    round_gate = metrics[
        (metrics["scope"] == "round")
        & (metrics["endpoint"] == "intact_patch_formation_pass")
    ].iloc[0]
    assert np.isclose(round_gate["brier_score"], 0.065)
    assert np.isclose(round_gate["accuracy"], 1.0)


def test_provenance_cohorts_and_proposal_predictions_remain_separate(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    _write_round(results_root, "ROUND_001", reconstructed=True)
    _write_round(results_root, "ROUND_002")
    _write_round(results_root, "ROUND_003")
    observations = pd.concat(
        [_observations(f"ROUND_{number:03d}") for number in range(1, 4)],
        ignore_index=True,
    )

    tables = [
        build_round_prospective_table(
            f"ROUND_{number:03d}",
            observations,
            results_root=results_root,
            evaluation_config=EVALUATION_CONFIG,
        )
        for number in range(1, 4)
    ]
    combined = pd.concat(tables, ignore_index=True)
    provenance = (
        combined.groupby("round_id")["provenance_class"].first().to_dict()
    )
    assert provenance == {
        "ROUND_001": "reconstructed",
        "ROUND_002": "migration_frozen_supplementary",
        "ROUND_003": "formal_frozen",
    }
    assert combined.groupby("round_id")["formal_cohort"].first().to_dict() == {
        "ROUND_001": False,
        "ROUND_002": False,
        "ROUND_003": True,
    }
    assert set(
        summarize_prospective_metrics(combined)["scope"]
    ) >= {
        "pooled_reconstructed",
        "pooled_supplementary",
        "pooled_formal",
    }

    changed_observations = observations.copy()
    changed_observations.loc[
        changed_observations["endpoint"] == "viability_percent",
        "value",
    ] = 1.0
    rebuilt = build_round_prospective_table(
        "ROUND_003",
        changed_observations,
        results_root=results_root,
        evaluation_config=EVALUATION_CONFIG,
    )
    frozen_prediction = rebuilt.loc[
        (rebuilt["candidate_id"] == "candidate_1")
        & (rebuilt["endpoint"] == "viability_percent"),
        "prediction_mean",
    ].iloc[0]
    assert frozen_prediction == 80.0


def test_round_and_campaign_artifact_destinations(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_round(results_root, "ROUND_003")
    observations = _observations("ROUND_003")

    generate_round_prospective_artifacts(
        "ROUND_003",
        observations,
        results_root=results_root,
        evaluation_config=EVALUATION_CONFIG,
    )
    generate_campaign_prospective_artifacts(
        observations,
        results_root=results_root,
        evaluation_config=EVALUATION_CONFIG,
    )

    round_reports = results_root / "rounds" / "ROUND_003" / "reports"
    campaign_reports = results_root / "reports" / "prospective"
    for root, plot_names in [
        (
            round_reports,
            [
                "prospective_prediction_vs_observed.png",
                "prospective_gate_calibration.png",
            ],
        ),
        (
            campaign_reports,
            [
                "prospective_prediction_vs_observed.png",
                "prospective_error_by_round.png",
            ],
        ),
    ]:
        assert (root / "prospective_evaluation_summary.txt").exists()
        assert (root / "tables" / "prospective_evaluation_table.csv").exists()
        assert (root / "tables" / "prospective_metrics.csv").exists()
        for name in plot_names:
            assert (root / "plots" / name).exists()


def test_report_cli_does_not_mutate_campaign_inputs(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write_round(results_root, "ROUND_003")
    observations_path = tmp_path / "observations.csv"
    _observations("ROUND_003").to_csv(observations_path, index=False)
    evaluation_path = tmp_path / "evaluation.yaml"
    evaluation_path.write_text(
        """
policy_version: test_policy
formal_start_round: 3
prediction_interval: {z_value: 1.96}
round_provenance: {ROUND_001: reconstructed, ROUND_002: migration_frozen_supplementary, default: formal_frozen}
endpoints:
  viability_percent:
    prediction_mean_column: predicted_viability_percent
    prediction_std_column: viability_std
    metric_type: continuous
    role: primary
""".strip()
        + "\n",
        encoding="utf-8",
    )
    protected_paths = [
        observations_path,
        results_root / "rounds" / "ROUND_003" / "proposal" / "proposal.csv",
        results_root / "rounds" / "ROUND_003" / "completed" / "completed.csv",
    ]
    before = {path: _sha256(path) for path in protected_paths}

    subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--all-rounds",
            "--observations",
            str(observations_path),
            "--results-root",
            str(results_root),
            "--evaluation-config",
            str(evaluation_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert {path: _sha256(path) for path in protected_paths} == before

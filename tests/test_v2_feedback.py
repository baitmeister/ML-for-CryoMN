from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from helper.feedback import ingest_feedback
from helper.registry import load_registry
from run_round import _resolve_batch_id, _resolve_viability_noise


def test_feedback_ingests_viability_intact_and_raw_critical_load(tmp_path: Path) -> None:
    registry = load_registry()
    candidate = {feature_name: 0.0 for feature_name in registry.feature_names}
    candidate.update({"formulation_id": "v2_test", "candidate_id": "cand_test", "dmso_M": 0.01})
    candidate_file = tmp_path / "candidates.csv"
    pd.DataFrame([candidate]).to_csv(candidate_file, index=False)

    feedback = tmp_path / "feedback.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "cand_test",
                "replicate_id": "tech_1",
                "viability_percent": 88.0,
                "intact_patch_formation_pass": "yes",
                "critical_axial_load_N_per_needle": 0.25,
                "initial_stiffness_N_per_mm_per_needle": 1.2,
            }
        ]
    ).to_csv(feedback, index=False)

    formulations, observations = ingest_feedback(
        feedback,
        [candidate_file],
        pd.DataFrame(),
        pd.DataFrame(),
        registry,
        batch_id="ROUND_TEST",
    )

    assert formulations["formulation_id"].tolist() == ["v2_test"]
    assert set(observations["replicate_id"]) == {"tech_1"}
    endpoints = set(observations["endpoint"])
    assert "viability_percent" in endpoints
    assert "intact_patch_formation_pass" in endpoints
    assert "critical_axial_load_N_per_needle" in endpoints
    assert "initial_stiffness_N_per_mm_per_needle" in endpoints


def test_feedback_rejects_mechanics_for_failed_intact_patch(tmp_path: Path) -> None:
    registry = load_registry()
    candidate = {feature_name: 0.0 for feature_name in registry.feature_names}
    candidate.update({"formulation_id": "v2_test", "candidate_id": "cand_test", "dmso_M": 0.01})
    candidate_file = tmp_path / "candidates.csv"
    pd.DataFrame([candidate]).to_csv(candidate_file, index=False)

    feedback = tmp_path / "feedback.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "cand_test",
                "viability_percent": 10.0,
                "intact_patch_formation_pass": "no",
                "critical_axial_load_N_per_needle": 0.25,
            }
        ]
    ).to_csv(feedback, index=False)

    with pytest.raises(ValueError, match="non-intact"):
        ingest_feedback(
            feedback,
            [candidate_file],
            pd.DataFrame(),
            pd.DataFrame(),
            registry,
            batch_id="ROUND_TEST",
        )


def test_blank_next_round_candidates_file_does_not_create_false_intact_observations(tmp_path: Path) -> None:
    registry = load_registry()
    candidate = {feature_name: 0.0 for feature_name in registry.feature_names}
    candidate.update({"formulation_id": "v2_test", "candidate_id": "cand_test", "dmso_M": 0.01})
    candidate_file = tmp_path / "candidates.csv"
    pd.DataFrame([candidate]).to_csv(candidate_file, index=False)

    feedback = tmp_path / "feedback.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "cand_test",
                "viability_percent": "",
                "intact_patch_formation_pass": "",
                "intact_tip_count": "",
                "critical_axial_load_N_per_needle": "",
            }
        ]
    ).to_csv(feedback, index=False)

    _formulations, observations = ingest_feedback(
        feedback,
        [candidate_file],
        pd.DataFrame(),
        pd.DataFrame(),
        registry,
        batch_id="ROUND_TEST",
    )

    assert observations.empty


def test_update_cli_can_resolve_batch_id_from_next_round_candidates(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "next_round_candidates.csv"
    pd.DataFrame(
        [
            {"candidate_id": "cand_1", "batch_id": "ROUND_001"},
            {"candidate_id": "cand_2", "batch_id": "ROUND_001"},
        ]
    ).to_csv(candidates_csv, index=False)

    assert _resolve_batch_id(candidates_csv, cli_batch_id=None) == "ROUND_001"
    assert _resolve_batch_id(candidates_csv, cli_batch_id="ROUND_OVERRIDE") == "ROUND_OVERRIDE"


def test_new_feedback_noise_defaults_to_one_fifth_legacy_wetlab_noise() -> None:
    config = {
        "transfer": {"legacy_wetlab_viability_noise_percent": 5.0},
        "feedback": {"new_viability_noise_percent": 1.0},
    }
    assert _resolve_viability_noise(config, cli_viability_noise=None) == 1.0
    assert _resolve_viability_noise(config, cli_viability_noise=2.5) == 2.5


def test_feedback_preserves_technical_replicates(tmp_path: Path) -> None:
    registry = load_registry()
    candidate = {feature_name: 0.0 for feature_name in registry.feature_names}
    candidate.update({"formulation_id": "v2_test", "candidate_id": "cand_test", "dmso_M": 0.01})
    candidate_file = tmp_path / "candidates.csv"
    pd.DataFrame([candidate]).to_csv(candidate_file, index=False)

    feedback = tmp_path / "feedback.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "cand_test",
                "replicate_id": "rep_001",
                "viability_percent": 80.0,
                "intact_patch_formation_pass": "yes",
            },
            {
                "candidate_id": "cand_test",
                "replicate_id": "rep_002",
                "viability_percent": 90.0,
                "intact_patch_formation_pass": "yes",
            },
        ]
    ).to_csv(feedback, index=False)

    _formulations, observations = ingest_feedback(
        feedback,
        [candidate_file],
        pd.DataFrame(),
        pd.DataFrame(),
        registry,
        batch_id="ROUND_TEST",
    )

    viability_rows = observations[observations["endpoint"] == "viability_percent"]
    assert len(viability_rows) == 2
    assert set(viability_rows["replicate_id"]) == {"rep_001", "rep_002"}
    assert set(viability_rows["value"].astype(float)) == {80.0, 90.0}


def test_blank_replicate_ids_are_auto_numbered_per_formulation(tmp_path: Path) -> None:
    registry = load_registry()
    candidate = {feature_name: 0.0 for feature_name in registry.feature_names}
    candidate.update({"formulation_id": "v2_test", "candidate_id": "cand_test", "dmso_M": 0.01})
    candidate_file = tmp_path / "candidates.csv"
    pd.DataFrame([candidate]).to_csv(candidate_file, index=False)

    feedback = tmp_path / "feedback.csv"
    pd.DataFrame(
        [
            {"candidate_id": "cand_test", "viability_percent": 80.0},
            {"candidate_id": "cand_test", "viability_percent": 90.0},
        ]
    ).to_csv(feedback, index=False)

    _formulations, observations = ingest_feedback(
        feedback,
        [candidate_file],
        pd.DataFrame(),
        pd.DataFrame(),
        registry,
        batch_id="ROUND_TEST",
    )

    assert set(observations["replicate_id"]) == {"rep_001", "rep_002"}


def test_feedback_rejects_out_of_range_viability(tmp_path: Path) -> None:
    registry = load_registry()
    candidate = {feature_name: 0.0 for feature_name in registry.feature_names}
    candidate.update({"formulation_id": "v2_test", "candidate_id": "cand_test", "dmso_M": 0.01})
    candidate_file = tmp_path / "candidates.csv"
    pd.DataFrame([candidate]).to_csv(candidate_file, index=False)

    feedback = tmp_path / "feedback.csv"
    pd.DataFrame([{"candidate_id": "cand_test", "viability_percent": 120.0}]).to_csv(
        feedback,
        index=False,
    )

    with pytest.raises(ValueError, match="viability_percent"):
        ingest_feedback(
            feedback,
            [candidate_file],
            pd.DataFrame(),
            pd.DataFrame(),
            registry,
            batch_id="ROUND_TEST",
        )


def test_optional_preparation_feedback_is_manual_and_structured(tmp_path: Path) -> None:
    registry = load_registry()
    candidate = {feature_name: 0.0 for feature_name in registry.feature_names}
    candidate.update({"formulation_id": "v2_prep", "candidate_id": "cand_prep", "pvp_pct": 5.0})
    candidate_file = tmp_path / "candidates.csv"
    pd.DataFrame([candidate]).to_csv(candidate_file, index=False)
    feedback = tmp_path / "feedback.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "cand_prep",
                "preparation_feasibility_pass": "no",
                "homogeneous_solution_pass": "no",
                "fillability_pass": "",
                "preparation_failure_reason": "excessive_viscosity",
            }
        ]
    ).to_csv(feedback, index=False)

    _formulations, observations = ingest_feedback(
        feedback,
        [candidate_file],
        pd.DataFrame(),
        pd.DataFrame(),
        registry,
        batch_id="ROUND_002",
    )

    assert set(observations["endpoint"]) == {
        "preparation_feasibility_pass",
        "homogeneous_solution_pass",
        "preparation_failure_reason:excessive_viscosity",
    }
    assert "fillability_pass" not in set(observations["endpoint"])


def test_preparation_failure_rejects_mechanical_data(tmp_path: Path) -> None:
    registry = load_registry()
    candidate = {feature_name: 0.0 for feature_name in registry.feature_names}
    candidate.update({"formulation_id": "v2_prep", "candidate_id": "cand_prep", "pvp_pct": 5.0})
    candidate_file = tmp_path / "candidates.csv"
    pd.DataFrame([candidate]).to_csv(candidate_file, index=False)
    feedback = tmp_path / "feedback.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "cand_prep",
                "preparation_feasibility_pass": "no",
                "critical_axial_load_N_per_needle": 0.2,
            }
        ]
    ).to_csv(feedback, index=False)

    with pytest.raises(ValueError, match="preparation-failed"):
        ingest_feedback(
            feedback,
            [candidate_file],
            pd.DataFrame(),
            pd.DataFrame(),
            registry,
            batch_id="ROUND_002",
        )


@pytest.mark.parametrize(
    "failure_field,failure_value",
    [
        ("homogeneous_solution_pass", "no"),
        ("fillability_pass", "no"),
        ("preparation_failure_reason", "phase_separated"),
    ],
)
def test_any_explicit_preparation_failure_blocks_mechanics(
    tmp_path: Path,
    failure_field: str,
    failure_value: str,
) -> None:
    registry = load_registry()
    candidate = {feature_name: 0.0 for feature_name in registry.feature_names}
    candidate.update(
        {
            "formulation_id": "v2_prep",
            "candidate_id": "cand_prep",
            "pvp_pct": 5.0,
        }
    )
    candidate_file = tmp_path / "candidates.csv"
    pd.DataFrame([candidate]).to_csv(candidate_file, index=False)
    feedback = tmp_path / "feedback.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "cand_prep",
                failure_field: failure_value,
                "critical_axial_load_N_per_needle": 0.2,
            }
        ]
    ).to_csv(feedback, index=False)

    with pytest.raises(ValueError, match="preparation-failed"):
        ingest_feedback(
            feedback,
            [candidate_file],
            pd.DataFrame(),
            pd.DataFrame(),
            registry,
            batch_id="ROUND_002",
        )


def test_invalid_preparation_reason_is_rejected(tmp_path: Path) -> None:
    registry = load_registry()
    candidate = {feature_name: 0.0 for feature_name in registry.feature_names}
    candidate.update({"formulation_id": "v2_prep", "candidate_id": "cand_prep", "pvp_pct": 5.0})
    candidate_file = tmp_path / "candidates.csv"
    pd.DataFrame([candidate]).to_csv(candidate_file, index=False)
    feedback = tmp_path / "feedback.csv"
    pd.DataFrame(
        [{"candidate_id": "cand_prep", "preparation_failure_reason": "sticky"}]
    ).to_csv(feedback, index=False)

    with pytest.raises(ValueError, match="preparation_failure_reason"):
        ingest_feedback(
            feedback,
            [candidate_file],
            pd.DataFrame(),
            pd.DataFrame(),
            registry,
            batch_id="ROUND_002",
        )

from __future__ import annotations

import pandas as pd

from helper.config import load_optimization_config
from helper.models import build_training_frame, train_endpoint_models
from helper.phase import PHASE_MECHANICS, PHASE_SCREENING, resolve_phase_mode
from helper.registry import load_registry
from helper.retest import build_retest_candidates


def _formulations() -> pd.DataFrame:
    registry = load_registry()
    rows = []
    for index in range(8):
        row = {feature_name: 0.0 for feature_name in registry.feature_names}
        row["formulation_id"] = f"v2_form_{index}"
        row["source"] = "wetlab_feedback"
        row["source_row_id"] = index
        row["formulation_label"] = f"formulation {index}"
        row["active_ingredient_count"] = 1
        row["dmso_M"] = 0.01 * (index + 1)
        row["ectoin_M"] = 0.10 + 0.01 * index
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_training_frame_preserves_batch_level_rows() -> None:
    registry = load_registry()
    formulations = _formulations().head(1)
    observations = pd.DataFrame(
        [
            {
                "observation_id": "obs_1",
                "formulation_id": "v2_form_0",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 80.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test.csv",
                "notes": "",
            },
            {
                "observation_id": "obs_2",
                "formulation_id": "v2_form_0",
                "batch_id": "ROUND_002",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 62.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test.csv",
                "notes": "",
            },
        ]
    )
    frame = build_training_frame(formulations, observations, registry)
    assert len(frame) == 2
    assert sorted(frame["batch_id"].tolist()) == ["ROUND_001", "ROUND_002"]


def test_phase_resolution_auto_transitions_when_paired_thresholds_are_met() -> None:
    registry = load_registry()
    formulations = _formulations()
    records = []
    for index in range(8):
        formulation_id = f"v2_form_{index}"
        batch_id = "ROUND_001" if index < 4 else "ROUND_002"
        records.extend(
            [
                {
                    "observation_id": f"{formulation_id}_viability",
                    "formulation_id": formulation_id,
                    "batch_id": batch_id,
                    "replicate_id": "rep_001",
                    "endpoint": "viability_percent",
                    "value": 55.0 + index,
                    "unit": "percent",
                    "observation_noise": 1.0,
                    "source_type": "wetlab_feedback",
                    "source_file": "test.csv",
                    "notes": "",
                },
                {
                    "observation_id": f"{formulation_id}_load",
                    "formulation_id": formulation_id,
                    "batch_id": batch_id,
                    "replicate_id": "rep_001",
                    "endpoint": "critical_axial_load_N_per_needle",
                    "value": 0.20 + 0.01 * index,
                    "unit": "N_per_needle",
                    "observation_noise": 0.02,
                    "source_type": "wetlab_feedback",
                    "source_file": "test.csv",
                    "notes": "",
                },
            ]
        )
    resolution = resolve_phase_mode(
        formulations,
        pd.DataFrame(records),
        registry,
        load_optimization_config(),
    )
    assert resolution.active_phase == PHASE_MECHANICS
    assert resolution.paired_observation_count == 8


def test_phase_resolution_auto_stays_screening_without_enough_mechanics() -> None:
    registry = load_registry()
    formulations = _formulations().head(3)
    observations = pd.DataFrame(
        [
            {
                "observation_id": "v2_form_0_viability",
                "formulation_id": "v2_form_0",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 70.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test.csv",
                "notes": "",
            }
        ]
    )
    resolution = resolve_phase_mode(
        formulations,
        observations,
        registry,
        load_optimization_config(),
    )
    assert resolution.active_phase == PHASE_SCREENING


def test_retest_candidates_flag_offtrend_batches() -> None:
    registry = load_registry()
    formulations = _formulations().head(4).copy()
    observations = pd.DataFrame(
        [
            {
                "observation_id": "f0_round1",
                "formulation_id": "v2_form_0",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 92.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test.csv",
                "notes": "",
            },
            {
                "observation_id": "f0_round2",
                "formulation_id": "v2_form_0",
                "batch_id": "ROUND_002",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 58.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test.csv",
                "notes": "",
            },
            {
                "observation_id": "f1_round1",
                "formulation_id": "v2_form_1",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 90.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test.csv",
                "notes": "",
            },
            {
                "observation_id": "f2_round1",
                "formulation_id": "v2_form_2",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 88.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test.csv",
                "notes": "",
            },
            {
                "observation_id": "f3_round1",
                "formulation_id": "v2_form_3",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 87.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test.csv",
                "notes": "",
            },
        ]
    )
    models = train_endpoint_models(formulations, observations, registry)
    retests = build_retest_candidates(
        formulations,
        observations,
        models,
        registry,
        load_optimization_config(),
    )
    assert "v2_form_0" in set(retests["formulation_id"])
    assert (retests["recommendation_type"] == "retest_priority").all()


def test_retest_candidates_exclude_zero_active_and_legacy_only_rows() -> None:
    registry = load_registry()
    formulations = _formulations().head(3).copy()
    formulations.loc[formulations["formulation_id"] == "v2_form_0", registry.feature_names] = 0.0
    formulations.loc[formulations["formulation_id"] == "v2_form_0", "active_ingredient_count"] = 0
    observations = pd.DataFrame(
        [
            {
                "observation_id": "f0_round1",
                "formulation_id": "v2_form_0",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 95.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test.csv",
                "notes": "",
            },
            {
                "observation_id": "f1_legacy",
                "formulation_id": "v2_form_1",
                "batch_id": "legacy_wetlab",
                "replicate_id": "legacy",
                "endpoint": "viability_percent",
                "value": 25.0,
                "unit": "percent",
                "observation_noise": 5.0,
                "source_type": "legacy_wetlab",
                "source_file": "legacy.csv",
                "notes": "",
            },
            {
                "observation_id": "f2_round1",
                "formulation_id": "v2_form_2",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 88.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test.csv",
                "notes": "",
            },
        ]
    )
    models = train_endpoint_models(formulations, observations, registry)
    retests = build_retest_candidates(
        formulations,
        observations,
        models,
        registry,
        load_optimization_config(),
    )
    assert "v2_form_0" not in set(retests["formulation_id"])
    assert "v2_form_1" not in set(retests["formulation_id"])


def test_retest_candidates_can_use_transferred_validation_as_neighbor_evidence() -> None:
    registry = load_registry()
    formulations = _formulations().head(3).copy()
    base_features = {feature_name: 0.0 for feature_name in registry.feature_names}
    base_features["dmso_M"] = 0.02
    base_features["ectoin_M"] = 0.12
    for formulation_id in ["v2_form_0", "v2_form_1"]:
        for feature_name, value in base_features.items():
            formulations.loc[formulations["formulation_id"] == formulation_id, feature_name] = value
        formulations.loc[formulations["formulation_id"] == formulation_id, "active_ingredient_count"] = 2
    observations = pd.DataFrame(
        [
            {
                "observation_id": "f0_round1",
                "formulation_id": "v2_form_0",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 92.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test.csv",
                "notes": "",
            },
            {
                "observation_id": "f1_legacy",
                "formulation_id": "v2_form_1",
                "batch_id": "legacy_wetlab",
                "replicate_id": "legacy",
                "endpoint": "viability_percent",
                "value": 35.0,
                "unit": "percent",
                "observation_noise": 5.0,
                "source_type": "legacy_wetlab",
                "source_file": "legacy.csv",
                "notes": "",
            },
            {
                "observation_id": "f2_round1",
                "formulation_id": "v2_form_2",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 89.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test.csv",
                "notes": "",
            },
        ]
    )
    models = train_endpoint_models(formulations, observations, registry)
    retests = build_retest_candidates(
        formulations,
        observations,
        models,
        registry,
        load_optimization_config(),
    )
    flagged = retests[retests["formulation_id"] == "v2_form_0"]
    assert not flagged.empty
    assert float(flagged.iloc[0]["local_neighbor_residual"]) >= 20.0

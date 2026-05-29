from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from helper.endpoints import intact_patch_formation_pass
from helper.instron import metrics_to_observations, parse_instron_csv
from import_instron import update_candidate_results


def test_intact_patch_gate_uses_tip_count_and_collapse_flags() -> None:
    assert intact_patch_formation_pass({"intact_tip_count": 90, "total_tip_count": 100})
    assert not intact_patch_formation_pass({"intact_tip_count": 89, "total_tip_count": 100})
    assert not intact_patch_formation_pass(
        {"intact_tip_count": 100, "total_tip_count": 100, "no_collapse": "no"}
    )
    assert intact_patch_formation_pass({"intact_patch_formation_pass": "yes"})


def test_instron_importer_extracts_allowed_metrics() -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "instron_bluehill_sample.csv"
    metrics = parse_instron_csv(fixture, needles_compressed=10)
    assert metrics.critical_axial_load_N_per_needle == pytest.approx(0.26)
    assert metrics.initial_stiffness_N_per_mm_per_needle == pytest.approx(1.0)
    assert metrics.n_points_used_for_stiffness >= 2


def test_instron_observation_ids_preserve_batch_and_replicate() -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "instron_bluehill_sample.csv"
    metrics = parse_instron_csv(fixture, needles_compressed=10)
    rows = pd.concat(
        [
            metrics_to_observations(
                metrics,
                formulation_id="v2_test",
                batch_id="ROUND_001",
                replicate_id="rep_001",
                source_file=str(fixture),
            ),
            metrics_to_observations(
                metrics,
                formulation_id="v2_test",
                batch_id="ROUND_001",
                replicate_id="rep_002",
                source_file=str(fixture),
            ),
        ],
        ignore_index=True,
    )
    assert rows["observation_id"].is_unique
    assert set(rows["replicate_id"]) == {"rep_001", "rep_002"}


def test_import_instron_updates_next_round_candidates_file(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parent / "fixtures" / "instron_bluehill_sample.csv"
    candidates_csv = tmp_path / "next_round_candidates.csv"
    pd.DataFrame(
        [
            {
                "formulation_id": "v2_test",
                "candidate_id": "cand_test",
                "batch_id": "ROUND_001",
                "replicate_id": "",
            }
        ]
    ).to_csv(candidates_csv, index=False)

    metrics = parse_instron_csv(fixture, needles_compressed=10)
    update_candidate_results(
        candidates_csv,
        fixture,
        metrics,
        formulation_id="v2_test",
        candidate_id=None,
        batch_id="ROUND_001",
        replicate_id="rep_001",
    )

    updated = pd.read_csv(candidates_csv)
    assert updated.loc[0, "replicate_id"] == "rep_001"
    assert updated.loc[0, "instron_file"] == str(fixture)
    assert updated.loc[0, "needles_compressed"] == 10
    assert updated.loc[0, "critical_axial_load_N_per_needle"] == pytest.approx(0.26)
    assert updated.loc[0, "initial_stiffness_N_per_mm_per_needle"] == pytest.approx(1.0)

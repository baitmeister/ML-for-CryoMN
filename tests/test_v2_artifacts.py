from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from helper.artifacts import (
    ArtifactConflictError,
    ProposalValidationError,
    archive_completed,
    freeze_proposal,
    reconstruct_proposal_csv,
    round_artifact_paths,
    sha256_file,
    validate_completed_against_proposal,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _proposal_rows() -> list[dict]:
    return [
        {
            "candidate_id": "cand_001",
            "formulation_id": "form_001",
            "batch_id": "ROUND_002",
            "selection_rank": 1,
            "dmso_M": 0.5,
            "predicted_viability_percent": 70.0,
            "viability_std": 8.0,
            "viability_percent": "",
            "intact_patch_formation_pass": "",
            "replicate_id": "",
            "notes": "",
        },
        {
            "candidate_id": "cand_002",
            "formulation_id": "form_002",
            "batch_id": "ROUND_002",
            "selection_rank": 2,
            "dmso_M": 0.7,
            "predicted_viability_percent": 65.0,
            "viability_std": 9.0,
            "viability_percent": "",
            "intact_patch_formation_pass": "",
            "replicate_id": "",
            "notes": "",
        },
    ]


def test_freeze_proposal_preserves_exact_bytes_and_allows_identical_rerun(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "next_round_candidates.csv"
    candidates.write_bytes(b"candidate_id,batch_id\ncand_001,ROUND_002\n")
    summary = tmp_path / "next_round_summary.txt"
    summary.write_bytes(b"Round 2 summary\n")
    metadata = tmp_path / "next_round_metadata.json"
    metadata.write_bytes(b'{"batch_id":"ROUND_002"}\n')

    first = freeze_proposal(
        "ROUND_002",
        candidates,
        summary,
        metadata,
        results_root=tmp_path / "results",
    )
    second = freeze_proposal(
        "ROUND_002",
        candidates,
        summary,
        metadata,
        results_root=tmp_path / "results",
    )

    assert first == second
    assert first.proposal_csv.read_bytes() == candidates.read_bytes()
    assert first.proposal_summary.read_bytes() == summary.read_bytes()
    assert first.proposal_metadata.read_bytes() == metadata.read_bytes()


def test_freeze_proposal_refuses_different_bytes(tmp_path: Path) -> None:
    candidates = tmp_path / "next_round_candidates.csv"
    candidates.write_bytes(b"candidate_id\ncand_001\n")
    freeze_proposal(
        "ROUND_002",
        candidates,
        results_root=tmp_path / "results",
    )
    candidates.write_bytes(b"candidate_id\ncand_002\n")

    with pytest.raises(ArtifactConflictError, match="Refusing to overwrite"):
        freeze_proposal(
            "ROUND_002",
            candidates,
            results_root=tmp_path / "results",
        )


def test_completed_validation_allows_results_reordering_and_replicates(
    tmp_path: Path,
) -> None:
    proposal_path = tmp_path / "proposal.csv"
    pd.DataFrame(_proposal_rows()).to_csv(proposal_path, index=False)
    completed = pd.DataFrame(
        [
            {
                **_proposal_rows()[1],
                "replicate_id": "rep_001",
                "viability_percent": 61.0,
                "intact_patch_formation_pass": "yes",
            },
            {
                **_proposal_rows()[0],
                "replicate_id": "rep_001",
                "viability_percent": 72.0,
                "intact_patch_formation_pass": "yes",
            },
            {
                **_proposal_rows()[0],
                "replicate_id": "rep_002",
                "viability_percent": 74.0,
                "intact_patch_formation_pass": "yes",
            },
        ]
    )
    completed_path = tmp_path / "completed.csv"
    completed.to_csv(completed_path, index=False)

    result = validate_completed_against_proposal(completed_path, proposal_path)

    assert result == {
        "proposal_candidate_count": 2,
        "completed_row_count": 3,
        "technical_replicate_extra_rows": 1,
    }


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("dmso_M", 0.9),
        ("predicted_viability_percent", 99.0),
        ("formulation_id", "changed"),
    ],
)
def test_completed_validation_rejects_immutable_changes(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    proposal_path = tmp_path / "proposal.csv"
    pd.DataFrame(_proposal_rows()).to_csv(proposal_path, index=False)
    completed = pd.DataFrame(_proposal_rows())
    completed.loc[0, column] = value
    completed_path = tmp_path / "completed.csv"
    completed.to_csv(completed_path, index=False)

    with pytest.raises(ProposalValidationError, match="immutable column"):
        validate_completed_against_proposal(completed_path, proposal_path)


def test_completed_validation_rejects_missing_and_unknown_candidates(
    tmp_path: Path,
) -> None:
    proposal_path = tmp_path / "proposal.csv"
    pd.DataFrame(_proposal_rows()).to_csv(proposal_path, index=False)
    completed = pd.DataFrame(
        [
            _proposal_rows()[0],
            {**_proposal_rows()[1], "candidate_id": "cand_unknown"},
        ]
    )
    completed_path = tmp_path / "completed.csv"
    completed.to_csv(completed_path, index=False)

    with pytest.raises(ProposalValidationError, match="candidate IDs differ"):
        validate_completed_against_proposal(completed_path, proposal_path)


def test_archive_completed_preserves_exact_source_bytes(tmp_path: Path) -> None:
    completed_source = tmp_path / "filled.csv"
    completed_source.write_bytes(
        b"candidate_id,viability_percent\ncand_001,72.125000\n"
    )

    archived = archive_completed(
        "ROUND_002",
        completed_source,
        results_root=tmp_path / "results",
    )

    expected = round_artifact_paths(
        "ROUND_002",
        tmp_path / "results",
    ).completed_csv
    assert archived == expected
    assert archived.read_bytes() == completed_source.read_bytes()


def test_reconstruct_proposal_blanks_only_wetlab_editable_fields(
    tmp_path: Path,
) -> None:
    completed = pd.DataFrame(
        [
            {
                **_proposal_rows()[0],
                "replicate_id": "rep_001",
                "viability_percent": 72.0,
                "intact_patch_formation_pass": "yes",
                "notes": "completed",
            }
        ]
    )
    completed_path = tmp_path / "completed.csv"
    reconstructed_path = tmp_path / "proposal_reconstructed.csv"
    completed.to_csv(completed_path, index=False)

    reconstruct_proposal_csv(completed_path, reconstructed_path)

    reconstructed = pd.read_csv(reconstructed_path)
    assert pd.isna(reconstructed.loc[0, "replicate_id"])
    assert pd.isna(reconstructed.loc[0, "viability_percent"])
    assert pd.isna(reconstructed.loc[0, "intact_patch_formation_pass"])
    assert pd.isna(reconstructed.loc[0, "notes"])
    assert reconstructed.loc[0, "candidate_id"] == "cand_001"
    assert reconstructed.loc[0, "predicted_viability_percent"] == 70.0
    assert reconstructed.loc[0, "dmso_M"] == 0.5


def test_committed_migration_manifests_match_archived_bytes() -> None:
    results_root = PROJECT_ROOT / "results" / "multi_objective_v2"
    round_one_manifest = json.loads(
        (
            results_root
            / "rounds"
            / "ROUND_001"
            / "legacy"
            / "migration_manifest.json"
        ).read_text(encoding="utf-8")
    )
    round_two_manifest = json.loads(
        (
            results_root
            / "rounds"
            / "ROUND_002"
            / "proposal"
            / "migration_manifest.json"
        ).read_text(encoding="utf-8")
    )

    round_one_entries = [
        *round_one_manifest["moved_artifacts"],
        round_one_manifest["reconstructed_proposal"],
    ]
    round_two_entries = [
        *round_two_manifest["proposal_artifacts"],
        *round_two_manifest["generated_during_migration"],
    ]
    for entry in [*round_one_entries, *round_two_entries]:
        destination = PROJECT_ROOT / entry.get("destination", entry.get("path"))
        assert destination.exists()
        assert sha256_file(destination) == entry["sha256"]

    validate_completed_against_proposal(
        results_root / "rounds" / "ROUND_001" / "completed" / "completed.csv",
        results_root
        / "rounds"
        / "ROUND_001"
        / "proposal"
        / "proposal_reconstructed.csv",
    )
    assert not (results_root / "round_review").exists()
    assert not (results_root / "visualizations").exists()

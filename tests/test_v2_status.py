from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from helper.status import build_current_round_status, derive_round_tracker, write_current_round_status


def test_derive_round_tracker_ignores_legacy_batches_and_normalizes_rounds() -> None:
    observations = pd.DataFrame(
        {
            "batch_id": [
                "legacy_wetlab",
                "ROUND_001",
                "ROUND_2",
                "",
                None,
                "ROUND_010",
            ]
        }
    )

    tracker = derive_round_tracker(observations)

    assert tracker["observed_round_ids"] == ["ROUND_001", "ROUND_002", "ROUND_010"]
    assert tracker["latest_observed_round_id"] == "ROUND_010"
    assert tracker["latest_observed_round_number"] == 10
    assert tracker["next_round_id"] == "ROUND_011"
    assert tracker["non_round_batch_ids"] == ["legacy_wetlab"]


def test_build_current_round_status_records_proposed_batch_consistency() -> None:
    observations = pd.DataFrame({"batch_id": ["ROUND_001", "ROUND_001"]})

    status = build_current_round_status(
        observations=observations,
        source_observations_path="data/processed_v2/observations.csv",
        active_phase="screening_only",
        phase_reason="auto-selected screening_only",
        proposed_batch_id="ROUND_002",
    )

    assert status["latest_observed_round_id"] == "ROUND_001"
    assert status["next_round_id"] == "ROUND_002"
    assert status["proposed_batch_id"] == "ROUND_002"
    assert status["proposed_batch_matches_next_round"] is True
    assert status["active_phase"] == "screening_only"


def test_write_current_round_status_persists_override_mismatch(tmp_path: Path) -> None:
    output_path = tmp_path / "current_round_status.json"
    observations = pd.DataFrame({"batch_id": ["ROUND_003"]})

    written = write_current_round_status(
        output_path,
        observations=observations,
        source_observations_path="data/processed_v2/observations.csv",
        active_phase="mechanics_enabled",
        phase_reason="manual override requested mechanics-enabled phase",
        proposed_batch_id="ROUND_999",
        proposed_batch_override_used=True,
    )

    payload = json.loads(written.read_text(encoding="utf-8"))
    assert written == output_path
    assert payload["next_round_id"] == "ROUND_004"
    assert payload["proposed_batch_id"] == "ROUND_999"
    assert payload["proposed_batch_matches_next_round"] is False
    assert payload["proposed_batch_override_used"] is True

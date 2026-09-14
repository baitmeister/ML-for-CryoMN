from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT / "src" / "08_multi_objective"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.artifacts import validate_completed_against_proposal  # noqa: E402
from helper.mechanics_execution import validate_mechanics_execution  # noqa: E402
from helper.group10_config import (  # noqa: E402
    Group10HardStop,
    assert_frozen_group10_protocol,
    load_group10_config,
)


def _load_run_round_module():
    path = V2_ROOT / "03_run_round" / "run_round.py"
    spec = importlib.util.spec_from_file_location("v2_run_round_characterization", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V2RoundWorkflowCharacterizationTests(unittest.TestCase):
    def test_archived_round_eight_completed_sheet_matches_frozen_proposal(self) -> None:
        round_dir = (
            PROJECT_ROOT
            / "results"
            / "multi_objective_v2"
            / "rounds"
            / "ROUND_008"
        )
        validate_completed_against_proposal(
            round_dir / "completed" / "completed.csv",
            round_dir / "proposal" / "proposal.csv",
        )
        manifest = json.loads(
            (round_dir / "completed" / "mechanical_execution_manifest.json").read_text()
        )
        self.assertFalse(manifest["mechanics_active_in_proposal"])
        self.assertEqual(manifest["actual_intact_measured_count"], 12)

    def test_blank_proposal_is_not_a_completed_experiment(self) -> None:
        module = _load_run_round_module()
        proposal = pd.read_csv(
            PROJECT_ROOT
            / "results"
            / "multi_objective_v2"
            / "next_round"
            / "next_round_candidates.csv"
        )
        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "blank_proposal.csv"
            proposal.to_csv(path, index=False)
            self.assertFalse(module._round_has_new_results(path))

    def test_mechanical_data_on_unranked_row_is_rejected(self) -> None:
        proposal = pd.read_csv(
            PROJECT_ROOT
            / "results"
            / "multi_objective_v2"
            / "next_round"
            / "next_round_candidates.csv"
        )
        completed = proposal.copy()
        unranked = completed["mechanical_selection_rank"].isna().idxmax()
        completed.loc[unranked, "intact_patch_formation_pass"] = 1.0
        completed.loc[unranked, "critical_axial_load_N_per_needle"] = 1.0
        with self.assertRaisesRegex(ValueError, "unranked"):
            validate_mechanics_execution(completed, proposal, primary_capacity=4)

    def test_mechanical_data_without_confirmed_intact_pass_is_rejected(self) -> None:
        proposal = pd.read_csv(
            PROJECT_ROOT
            / "results"
            / "multi_objective_v2"
            / "next_round"
            / "next_round_candidates.csv"
        )
        completed = proposal.copy()
        ranked = completed["mechanical_selection_rank"].notna().idxmax()
        completed.loc[ranked, "critical_axial_load_N_per_needle"] = 1.0
        with self.assertRaisesRegex(ValueError, "without a measured intact pass"):
            validate_mechanics_execution(completed, proposal, primary_capacity=4)

    def test_group10_ingestion_requires_complete_matching_frozen_protocol(self) -> None:
        config = load_group10_config()
        with self.assertRaisesRegex(Group10HardStop, "no complete Group 10 effective_config"):
            assert_frozen_group10_protocol(config, 10, {})
        metadata = {"group10": {"effective_config": json.loads(json.dumps(config))}}
        metadata["group10"]["effective_config"]["mechanical_endpoint"].pop("protocol_id")
        with self.assertRaisesRegex(Group10HardStop, "settings/protocol are invalid"):
            assert_frozen_group10_protocol(config, 10, metadata)


if __name__ == "__main__":
    unittest.main()

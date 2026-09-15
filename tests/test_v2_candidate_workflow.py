from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT / "src" / "08_multi_objective"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.candidate_workflow import (  # noqa: E402
    CandidateSelectionOptions,
    run_candidate_selection,
)
from helper.paths import FORMULATIONS_PATH, OBSERVATIONS_PATH  # noqa: E402
from helper.registry import load_registry  # noqa: E402


def _load_selection_cli_module():
    path = V2_ROOT / "02_select_candidates" / "select_candidates.py"
    spec = importlib.util.spec_from_file_location("v2_select_candidates_cli", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V2CandidateWorkflowTests(unittest.TestCase):
    def setUp(self):
        # These characterization cases exercise selection before Group 9.
        # Later live ingestion must not alter their input evidence cutoff.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.observations_path = Path(temporary.name) / 'observations.csv'
        observations = pd.read_csv(OBSERVATIONS_PATH)
        rounds = pd.to_numeric(observations.batch_id.astype(str).str.extract(r'^ROUND_(\d+)$')[0], errors='coerce')
        observations = observations.loc[rounds.isna() | rounds.lt(9)]
        observations.to_csv(self.observations_path, index=False)
        self.formulations_path = Path(temporary.name) / 'formulations.csv'
        formulations = pd.read_csv(FORMULATIONS_PATH)
        formulations.loc[formulations.formulation_id.isin(observations.formulation_id)].to_csv(
            self.formulations_path, index=False
        )

    def test_options_are_immutable(self) -> None:
        options = CandidateSelectionOptions()
        with self.assertRaises(FrozenInstanceError):
            options.seed = 7  # type: ignore[misc]

    def test_cli_arguments_map_to_existing_option_names(self) -> None:
        module = _load_selection_cli_module()
        arguments = [
            "select_candidates.py",
            "--formulations",
            "formulations.csv",
            "--observations",
            "observations.csv",
            "--candidate-pool",
            "pool.csv",
            "--availability-config",
            "availability.yaml",
            "--output-dir",
            "next_round",
            "--total-candidate-pool",
            "total.csv",
            "--pool-size",
            "321",
            "--seed",
            "42",
            "--phase-mode",
            "mechanics_bootstrap",
            "--batch-id",
            "ROUND_009",
            "--supersede-unstarted-proposal",
        ]
        with patch.object(sys, "argv", arguments):
            parsed = module.parse_args()
        self.assertEqual(parsed.formulations, "formulations.csv")
        self.assertEqual(parsed.observations, "observations.csv")
        self.assertEqual(parsed.candidate_pool, "pool.csv")
        self.assertEqual(parsed.availability_config, "availability.yaml")
        self.assertEqual(parsed.output_dir, "next_round")
        self.assertEqual(parsed.total_candidate_pool, "total.csv")
        self.assertEqual(parsed.pool_size, 321)
        self.assertEqual(parsed.seed, 42)
        self.assertEqual(parsed.phase_mode, "mechanics_bootstrap")
        self.assertEqual(parsed.batch_id, "ROUND_009")
        self.assertTrue(parsed.supersede_unstarted_proposal)

    def test_empty_formulation_table_fails_before_outputs_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            empty_formulations = root / "formulations.csv"
            pd.DataFrame(columns=["formulation_id"]).to_csv(
                empty_formulations,
                index=False,
            )
            output_dir = root / "next_round"
            with self.assertRaisesRegex(SystemExit, "No v2 formulations were found"):
                run_candidate_selection(
                    CandidateSelectionOptions(
                        formulations_path=empty_formulations,
                        observations_path=self.observations_path,
                        output_dir=output_dir,
                        total_candidate_pool_path=root / "total_candidate_pool.csv",
                        seed=42,
                        batch_id="ROUND_009",
                    )
                )
            self.assertFalse(output_dir.exists())

    def test_generated_pool_run_returns_round_and_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            outcome = run_candidate_selection(
                CandidateSelectionOptions(
                    formulations_path=self.formulations_path,
                    observations_path=self.observations_path,
                    output_dir=root / "next_round",
                    total_candidate_pool_path=root / "total_candidate_pool.csv",
                    seed=42,
                    batch_id="ROUND_009",
                )
            )
            expected_ids = [
                "retest_v2_6f20e63adb1d",
                "retest_v2_7371aa062562",
                "cand_001149",
                "rescue_000027",
                "cand_001152",
                "cand_000548",
                "cand_000290",
                "cand_001885",
                "cand_001965",
                "cand_001923",
                "cand_001589",
                "cand_001515",
            ]
            self.assertEqual(outcome.round_id, "ROUND_009")
            self.assertEqual(outcome.resolved_phase, "mechanics_bootstrap")
            self.assertEqual(
                outcome.selection_result.viability_screen["candidate_id"].tolist(),
                expected_ids,
            )
            for name in (
                "working_candidates",
                "working_summary",
                "working_metadata",
                "total_candidate_pool",
                "round_status",
                "frozen_proposal",
            ):
                self.assertTrue(outcome.artifact_paths[name].exists(), name)

    def test_group10_override_hard_stops_without_group9_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            output_dir = root / "next_round"
            with self.assertRaisesRegex(RuntimeError, "GROUP 10 HARD STOP.*ROUND_009"):
                run_candidate_selection(
                    CandidateSelectionOptions(
                        formulations_path=self.formulations_path,
                        observations_path=self.observations_path,
                        output_dir=output_dir,
                        total_candidate_pool_path=root / "total_candidate_pool.csv",
                        seed=42,
                        batch_id="ROUND_010",
                    )
                )
            self.assertFalse(output_dir.exists())

    def test_supplied_pool_outside_registry_bounds_is_rejected(self) -> None:
        registry = load_registry()
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            row = {feature: 0.0 for feature in registry.feature_names}
            row.update(
                {
                    "candidate_id": "outside_bounds",
                    "formulation_id": "outside_bounds",
                    registry.feature_names[0]: 1e6,
                }
            )
            pool = root / "pool.csv"
            pd.DataFrame([row]).to_csv(pool, index=False)
            with self.assertRaisesRegex(SystemExit, "empty after applying registry bounds"):
                run_candidate_selection(
                    CandidateSelectionOptions(
                        formulations_path=self.formulations_path,
                        observations_path=self.observations_path,
                        candidate_pool_path=pool,
                        output_dir=root / "next_round",
                        total_candidate_pool_path=root / "total.csv",
                        batch_id="ROUND_009",
                    )
                )

    def test_supplied_pool_using_only_unavailable_ingredient_is_rejected(self) -> None:
        registry = load_registry()
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            row = {feature: 0.0 for feature in registry.feature_names}
            row.update(
                {
                    "candidate_id": "unavailable_only",
                    "formulation_id": "unavailable_only",
                    "human_serum_pct": 1.0,
                }
            )
            pool = root / "pool.csv"
            pd.DataFrame([row]).to_csv(pool, index=False)
            with self.assertRaisesRegex(SystemExit, "temporary availability restrictions"):
                run_candidate_selection(
                    CandidateSelectionOptions(
                        formulations_path=self.formulations_path,
                        observations_path=self.observations_path,
                        candidate_pool_path=pool,
                        output_dir=root / "next_round",
                        total_candidate_pool_path=root / "total.csv",
                        batch_id="ROUND_009",
                    )
                )


if __name__ == "__main__":
    unittest.main()

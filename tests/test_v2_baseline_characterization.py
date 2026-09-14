from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "tests" / "baselines" / "v2_current.json"
FORMULATIONS_PATH = PROJECT_ROOT / "data" / "processed_v2" / "formulations.csv"
OBSERVATIONS_PATH = PROJECT_ROOT / "data" / "processed_v2" / "observations.csv"
PROPOSAL_PATH = (
    PROJECT_ROOT
    / "results"
    / "multi_objective_v2"
    / "rounds"
    / "ROUND_009"
    / "proposal"
    / "proposal.csv"
)
CANDIDATE_POOL_PATH = (
    PROJECT_ROOT / "results" / "multi_objective_v2" / "total_candidate_pool.csv"
)
SELECTION_METADATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "multi_objective_v2"
    / "next_round"
    / "next_round_metadata.json"
)
METRICS_PATH = (
    PROJECT_ROOT
    / "results"
    / "multi_objective_v2"
    / "reports"
    / "prospective"
    / "tables"
    / "prospective_metrics.csv"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_columns_sha256(frame: pd.DataFrame) -> str:
    encoded = json.dumps(
        list(frame.columns), separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class V2BaselineCharacterizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    def test_v2_database_shape_endpoint_counts_and_hashes(self) -> None:
        formulations = pd.read_csv(FORMULATIONS_PATH)
        observations = pd.read_csv(OBSERVATIONS_PATH)

        self.assertEqual(list(formulations.shape), self.baseline["formulations_shape"])
        self.assertEqual(list(observations.shape), self.baseline["observations_shape"])
        self.assertEqual(_sha256(FORMULATIONS_PATH), self.baseline["formulations_sha256"])
        self.assertEqual(_sha256(OBSERVATIONS_PATH), self.baseline["observations_sha256"])
        self.assertEqual(
            _sha256(PROJECT_ROOT / "data" / "processed" / "parsed_formulations.csv"),
            self.baseline["legacy_literature_input_sha256"],
        )
        self.assertEqual(
            _sha256(PROJECT_ROOT / "data" / "validation" / "validation_results.csv"),
            self.baseline["legacy_validation_input_sha256"],
        )

        endpoint_counts = observations["endpoint"].value_counts().to_dict()
        for endpoint, expected in self.baseline["endpoint_counts"].items():
            self.assertEqual(int(endpoint_counts.get(endpoint, 0)), expected)
        self.assertEqual(
            observations["source_type"].value_counts().to_dict(),
            self.baseline["source_type_counts"],
        )
        self.assertTrue(formulations["formulation_id"].is_unique)
        self.assertTrue(observations["observation_id"].is_unique)

    def test_active_round_nine_proposal_matches_baseline(self) -> None:
        proposal = pd.read_csv(PROPOSAL_PATH)
        self.assertEqual(len(proposal.columns), self.baseline["proposal_column_count"])
        self.assertEqual(
            proposal["candidate_id"].tolist(),
            self.baseline["proposal_candidate_ids"],
        )
        self.assertEqual(
            proposal["mechanical_test_recommended"].astype(bool).tolist(),
            self.baseline["mechanical_test_recommended"],
        )
        self.assertEqual(
            proposal["viability_prediction_status"].tolist(),
            self.baseline["viability_prediction_status"],
        )
        np.testing.assert_array_equal(
            proposal["predicted_critical_axial_load_N_per_needle"].to_numpy(),
            np.zeros(len(proposal)),
        )
        np.testing.assert_array_equal(
            proposal["critical_axial_load_std"].to_numpy(),
            np.zeros(len(proposal)),
        )

    def test_formal_prospective_metrics_match_baseline(self) -> None:
        metrics = pd.read_csv(METRICS_PATH)
        formal = metrics.loc[metrics["scope"].eq("pooled_formal")].set_index("endpoint")
        for endpoint, expected in self.baseline["formal_prospective_metrics"].items():
            row = formal.loc[endpoint]
            self.assertEqual(int(row["n_evaluated"]), expected["n_evaluated"])
            for metric, value in expected.items():
                if metric == "n_evaluated":
                    continue
                self.assertAlmostEqual(float(row[metric]), float(value), places=10)

    def test_configuration_policy_schema_and_fixed_predictions(self) -> None:
        for name, expected_hash in self.baseline["config_sha256"].items():
            path = PROJECT_ROOT / "config_v2" / name
            actual_hash = _sha256(path)
            if name == "endpoints.yaml":
                # New reporting-only requirements precede the byte-identical
                # historical endpoint contract. Keep its original hash assertion.
                historical = "screening_gate:" + path.read_text().split("screening_gate:", 1)[1]
                actual_hash = hashlib.sha256(historical.encode()).hexdigest()
            self.assertEqual(actual_hash, expected_hash)

        proposal = pd.read_csv(PROPOSAL_PATH)
        candidate_pool = pd.read_csv(CANDIDATE_POOL_PATH)
        for frame, key in (
            (proposal, "proposal"),
            (candidate_pool, "total_candidate_pool"),
        ):
            expected = self.baseline["output_schemas"][key]
            self.assertEqual(len(frame.columns), expected["column_count"])
            self.assertEqual(
                _ordered_columns_sha256(frame),
                expected["ordered_columns_sha256"],
            )

        metadata = json.loads(SELECTION_METADATA_PATH.read_text(encoding="utf-8"))
        expected_policies = self.baseline["policy_versions"]
        self.assertEqual(
            metadata["formulation_feasibility_policy_version"],
            expected_policies["formulation_feasibility"],
        )
        self.assertEqual(
            metadata["formulation_similarity"]["policy_version"],
            expected_policies["formulation_similarity"],
        )
        self.assertEqual(
            metadata["cold_start_policy"]["policy_version"],
            expected_policies["cold_start"],
        )
        self.assertEqual(
            metadata["mechanics_transition"]["policy_version"],
            expected_policies["mechanics_transition"],
        )

        numeric_columns = [
            "predicted_viability_percent",
            "viability_std",
            "raw_surrogate_viability_mean",
            "raw_surrogate_viability_std",
            "predicted_critical_axial_load_N_per_needle",
            "critical_axial_load_std",
            "screening_phase_score",
        ]
        indexed = proposal.set_index("candidate_id")
        for expected in self.baseline["fixed_prediction_subset"]:
            row = indexed.loc[expected["candidate_id"]]
            for column in numeric_columns:
                expected_value = expected[column]
                if expected_value is None:
                    self.assertTrue(pd.isna(row[column]), column)
                else:
                    np.testing.assert_allclose(
                        float(row[column]),
                        float(expected_value),
                        rtol=1e-6,
                        atol=1e-8,
                    )
            self.assertEqual(
                bool(row["mechanical_test_recommended"]),
                expected["mechanical_test_recommended"],
            )


if __name__ == "__main__":
    unittest.main()

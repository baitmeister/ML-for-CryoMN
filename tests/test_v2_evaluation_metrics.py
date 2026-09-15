from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT / "src" / "08_multi_objective"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.evaluation_metrics import (  # noqa: E402
    _igd_2d,
    _round_metrics,
    build_feasible_paired_objectives,
    compute_campaign_hypervolume_progress,
    compute_fixed_reference_hypervolume,
    compute_observed_pareto_front,
)


REFERENCE = {
    "viability_percent": 0.0,
    "critical_axial_load_N_per_needle": 0.0,
}


class V2EvaluationMetricTests(unittest.TestCase):
    def test_no_load_fixture_is_not_estimable(self) -> None:
        formulations = pd.read_csv(
            PROJECT_ROOT / "data" / "processed_v2" / "formulations.csv"
        )
        observations = pd.read_csv(
            PROJECT_ROOT / "data" / "processed_v2" / "observations.csv"
        )
        # Construct the no-mechanics condition explicitly; the live campaign
        # now contains Group 9 mechanical measurements.
        observations = observations.loc[
            ~observations.endpoint.eq('critical_axial_load_N_per_needle')
        ].copy()
        feasible, excluded = build_feasible_paired_objectives(
            formulations,
            observations,
        )
        metric = compute_fixed_reference_hypervolume(feasible, REFERENCE)
        self.assertTrue(feasible.empty)
        self.assertFalse(excluded.empty)
        self.assertEqual(metric["status"], "not_estimable")
        self.assertTrue(np.isnan(metric["hypervolume"]))

    def test_failed_intact_rows_are_excluded_from_frontier(self) -> None:
        observations = pd.DataFrame(
            [
                ["f1", "ROUND_001", "viability_percent", 50.0],
                ["f1", "ROUND_001", "critical_axial_load_N_per_needle", 1.0],
                ["f1", "ROUND_001", "intact_patch_formation_pass", 1.0],
                ["f2", "ROUND_001", "viability_percent", 80.0],
                ["f2", "ROUND_001", "critical_axial_load_N_per_needle", 2.0],
                ["f2", "ROUND_001", "intact_patch_formation_pass", 0.0],
            ],
            columns=["formulation_id", "batch_id", "endpoint", "value"],
        )
        formulations = pd.DataFrame({"formulation_id": ["f1", "f2"]})
        feasible, excluded = build_feasible_paired_objectives(
            formulations,
            observations,
        )
        self.assertEqual(feasible["formulation_id"].tolist(), ["f1"])
        self.assertEqual(
            excluded.set_index("formulation_id").loc["f2", "exclusion_reason"],
            "failed_intact_gate",
        )
        frontier = compute_observed_pareto_front(feasible)
        self.assertTrue(frontier["is_pareto"].all())

    def test_dominance_and_fixed_reference_hypervolume_match_hand_calculation(self) -> None:
        pairs = pd.DataFrame(
            {
                "formulation_id": ["high_load", "high_viability", "dominated"],
                "batch_id": ["ROUND_001"] * 3,
                "viability_percent": [25.0, 50.0, 20.0],
                "critical_axial_load_N_per_needle": [2.0, 1.0, 0.5],
            }
        )
        labelled = compute_observed_pareto_front(pairs)
        self.assertEqual(
            labelled.set_index("formulation_id")["is_pareto"].to_dict(),
            {"high_load": True, "high_viability": True, "dominated": False},
        )
        metric = compute_fixed_reference_hypervolume(labelled, REFERENCE)
        self.assertEqual(metric["status"], "estimated")
        self.assertAlmostEqual(float(metric["hypervolume"]), 75.0)

    def test_later_round_does_not_rescale_earlier_hypervolume(self) -> None:
        early = pd.DataFrame(
            {
                "formulation_id": ["f1", "f2"],
                "batch_id": ["ROUND_001", "ROUND_002"],
                "viability_percent": [20.0, 30.0],
                "critical_axial_load_N_per_needle": [1.0, 0.8],
            }
        )
        later = pd.concat(
            [
                early,
                pd.DataFrame(
                    {
                        "formulation_id": ["f3"],
                        "batch_id": ["ROUND_003"],
                        "viability_percent": [90.0],
                        "critical_axial_load_N_per_needle": [3.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        early_progress = compute_campaign_hypervolume_progress(early, REFERENCE)
        later_progress = compute_campaign_hypervolume_progress(later, REFERENCE)
        self.assertAlmostEqual(
            float(early_progress.iloc[0]["hypervolume"]),
            float(later_progress.iloc[0]["hypervolume"]),
        )
        self.assertNotIn("igd", later_progress.columns)

    def test_compatibility_metrics_remain_importable(self) -> None:
        self.assertTrue(callable(_round_metrics))
        self.assertTrue(callable(_igd_2d))


if __name__ == "__main__":
    unittest.main()

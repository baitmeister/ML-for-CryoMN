from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT / "src" / "08_multi_objective"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.config import load_optimization_config  # noqa: E402
from helper.paths import FORMULATIONS_PATH, OBSERVATIONS_PATH  # noqa: E402
from helper.phase import resolve_phase_mode  # noqa: E402
from helper.registry import load_registry  # noqa: E402


BASELINE = PROJECT_ROOT / "tests" / "baselines" / "v2_current.json"
PROPOSAL = (
    PROJECT_ROOT
    / "results"
    / "multi_objective_v2"
    / "rounds"
    / "ROUND_009"
    / "proposal"
    / "proposal.csv"
)


class V2SelectionCharacterizationTests(unittest.TestCase):
    def test_round_nine_phase_and_frozen_selection_contract(self) -> None:
        baseline = json.loads(BASELINE.read_text())
        formulations = pd.read_csv(FORMULATIONS_PATH)
        observations = pd.read_csv(OBSERVATIONS_PATH)
        resolution = resolve_phase_mode(
            formulations,
            observations,
            load_registry(),
            load_optimization_config(),
            target_round_number=9,
        )
        proposal = pd.read_csv(PROPOSAL)
        self.assertEqual(resolution.active_phase, "mechanics_bootstrap")
        self.assertEqual(proposal.shape[0], 12)
        self.assertEqual(proposal.shape[1], 132)
        self.assertEqual(
            proposal["candidate_id"].tolist(),
            baseline["proposal_candidate_ids"],
        )
        self.assertEqual(
            proposal["mechanical_test_recommended"].astype(bool).tolist(),
            baseline["mechanical_test_recommended"],
        )
        roles = proposal["mechanical_transition_role"].fillna("").tolist()
        self.assertEqual(roles, baseline["mechanical_transition_roles"])
        self.assertEqual(int(proposal["mechanical_test_recommended"].sum()), 4)
        self.assertTrue(
            proposal["predicted_critical_axial_load_N_per_needle"].eq(0).all()
        )
        self.assertTrue(proposal["critical_axial_load_std"].eq(0).all())


if __name__ == "__main__":
    unittest.main()

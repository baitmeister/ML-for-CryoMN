from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT / "src" / "08_multi_objective"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.config import load_optimization_config  # noqa: E402
from helper.mechanics_execution import validate_mechanics_execution  # noqa: E402
from helper.phase import (  # noqa: E402
    PHASE_AUTO,
    PHASE_BOOTSTRAP,
    PHASE_HYBRID,
    PHASE_MECHANICS,
    PHASE_SCREENING,
    PhaseResolution,
    resolve_phase_mode,
)
from helper.registry import load_registry  # noqa: E402
from helper.selection import (  # noqa: E402
    SelectionResult,
    _select_bootstrap_anchor,
    select_mechanical_tests,
    write_selection_result,
)


class MechanicsTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry()
        cls.config = load_optimization_config()

    def _formulation(self, formulation_id: str, index: int = 0) -> dict:
        row = {"formulation_id": formulation_id}
        row.update({feature: 0.0 for feature in self.registry.feature_names})
        row["ectoin_M"] = 0.01 + 0.005 * index
        return row

    @staticmethod
    def _observation(
        formulation_id: str,
        batch_id: str,
        endpoint: str,
        value: float,
        replicate_id: str = "rep_001",
    ) -> dict:
        return {
            "formulation_id": formulation_id,
            "batch_id": batch_id,
            "replicate_id": replicate_id,
            "endpoint": endpoint,
            "value": value,
            "source_type": "wetlab_feedback",
        }

    def _phase_data(
        self,
        completed_screening_rounds: int,
        paired_rows: list[tuple[str, str]] = (),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        formulation_ids = {f"screen_{index}" for index in range(1, completed_screening_rounds + 1)}
        formulation_ids.update(formulation_id for formulation_id, _ in paired_rows)
        formulations = pd.DataFrame(
            [
                self._formulation(formulation_id, index)
                for index, formulation_id in enumerate(sorted(formulation_ids))
            ]
        )
        observations: list[dict] = []
        for index in range(1, completed_screening_rounds + 1):
            formulation_id = f"screen_{index}"
            batch_id = f"ROUND_{index:03d}"
            observations.extend(
                [
                    self._observation(formulation_id, batch_id, "viability_percent", 70.0),
                    self._observation(
                        formulation_id,
                        batch_id,
                        "intact_patch_formation_pass",
                        1.0,
                    ),
                ]
            )
        for formulation_id, batch_id in paired_rows:
            observations.extend(
                [
                    self._observation(formulation_id, batch_id, "viability_percent", 75.0),
                    self._observation(
                        formulation_id,
                        batch_id,
                        "intact_patch_formation_pass",
                        1.0,
                    ),
                    self._observation(
                        formulation_id,
                        batch_id,
                        "critical_axial_load_N_per_needle",
                        1.5,
                    ),
                ]
            )
        return formulations, pd.DataFrame(observations)

    def test_phase_resolution_uses_screening_and_both_evidence_gates(self) -> None:
        formulations, observations = self._phase_data(7)
        resolution = resolve_phase_mode(
            formulations, observations, self.registry, self.config
        )
        self.assertEqual(resolution.active_phase, PHASE_SCREENING)

        formulations, observations = self._phase_data(8)
        resolution = resolve_phase_mode(
            formulations, observations, self.registry, self.config
        )
        self.assertEqual(resolution.active_phase, PHASE_BOOTSTRAP)

        eight_pairs_five_formulations = [
            (f"pair_{index % 5}", f"MECH_{1 + index // 4}")
            for index in range(8)
        ]
        formulations, observations = self._phase_data(
            8, eight_pairs_five_formulations
        )
        resolution = resolve_phase_mode(
            formulations, observations, self.registry, self.config
        )
        self.assertEqual(resolution.paired_observation_count, 8)
        self.assertEqual(resolution.active_phase, PHASE_BOOTSTRAP)

        hybrid_pairs = [
            (f"pair_{index % 7}", f"MECH_{1 + index // 4}")
            for index in range(8)
        ]
        formulations, observations = self._phase_data(8, hybrid_pairs)
        resolution = resolve_phase_mode(
            formulations, observations, self.registry, self.config
        )
        self.assertEqual(resolution.active_phase, PHASE_HYBRID)
        self.assertTrue(resolution.bootstrap_gate_met)

        incomplete_full_pairs = [
            (f"pair_{index % 11}", f"MECH_{1 + index // 6}")
            for index in range(16)
        ]
        formulations, observations = self._phase_data(8, incomplete_full_pairs)
        resolution = resolve_phase_mode(
            formulations, observations, self.registry, self.config
        )
        self.assertEqual(resolution.active_phase, PHASE_HYBRID)
        self.assertFalse(resolution.full_gate_met)

        full_pairs = [
            (f"pair_{index % 13}", f"MECH_{1 + index // 6}")
            for index in range(16)
        ]
        formulations, observations = self._phase_data(8, full_pairs)
        resolution = resolve_phase_mode(
            formulations, observations, self.registry, self.config
        )
        self.assertEqual(resolution.active_phase, PHASE_MECHANICS)
        self.assertTrue(resolution.full_gate_met)

    def test_synthetic_campaign_progresses_bootstrap_hybrid_full(self) -> None:
        formulations, observations = self._phase_data(8)

        def add_mechanical_batch(
            existing_formulations: pd.DataFrame,
            existing_observations: pd.DataFrame,
            batch_id: str,
            formulation_ids: list[str],
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            known_ids = set(existing_formulations["formulation_id"].astype(str))
            additions = [
                self._formulation(formulation_id, len(known_ids) + index)
                for index, formulation_id in enumerate(formulation_ids)
                if formulation_id not in known_ids
            ]
            if additions:
                existing_formulations = pd.concat(
                    [existing_formulations, pd.DataFrame(additions)],
                    ignore_index=True,
                )
            rows = []
            for formulation_id in formulation_ids:
                rows.extend(
                    [
                        self._observation(
                            formulation_id, batch_id, "viability_percent", 80.0
                        ),
                        self._observation(
                            formulation_id,
                            batch_id,
                            "intact_patch_formation_pass",
                            1.0,
                        ),
                        self._observation(
                            formulation_id,
                            batch_id,
                            "critical_axial_load_N_per_needle",
                            1.5,
                        ),
                    ]
                )
            return existing_formulations, pd.concat(
                [existing_observations, pd.DataFrame(rows)], ignore_index=True
            )

        initial = resolve_phase_mode(
            formulations, observations, self.registry, self.config
        )
        self.assertEqual(initial.active_phase, PHASE_BOOTSTRAP)
        self.assertEqual(initial.bootstrap_batch_index, 1)

        formulations, observations = add_mechanical_batch(
            formulations, observations, "MECH_001", ["a", "b", "c", "d"]
        )
        second_bootstrap = resolve_phase_mode(
            formulations, observations, self.registry, self.config
        )
        self.assertEqual(second_bootstrap.active_phase, PHASE_BOOTSTRAP)
        self.assertEqual(second_bootstrap.bootstrap_batch_index, 2)

        formulations, observations = add_mechanical_batch(
            formulations, observations, "MECH_002", ["a", "e", "f", "g"]
        )
        hybrid = resolve_phase_mode(
            formulations, observations, self.registry, self.config
        )
        self.assertEqual(hybrid.active_phase, PHASE_HYBRID)
        self.assertEqual(hybrid.paired_observation_count, 8)
        self.assertEqual(hybrid.distinct_formulation_count, 7)
        self.assertEqual(hybrid.batch_count, 2)

        formulations, observations = add_mechanical_batch(
            formulations, observations, "MECH_003", ["h", "i", "j", "k"]
        )
        still_hybrid = resolve_phase_mode(
            formulations, observations, self.registry, self.config
        )
        self.assertEqual(still_hybrid.active_phase, PHASE_HYBRID)

        formulations, observations = add_mechanical_batch(
            formulations, observations, "MECH_004", ["l", "m", "n", "o"]
        )
        full = resolve_phase_mode(
            formulations, observations, self.registry, self.config
        )
        self.assertEqual(full.active_phase, PHASE_MECHANICS)
        self.assertEqual(full.paired_observation_count, 16)
        self.assertGreaterEqual(full.distinct_formulation_count, 12)
        self.assertGreaterEqual(full.batch_count, 3)

    def test_manual_overrides_are_audited(self) -> None:
        formulations, observations = self._phase_data(1)
        for phase in (
            PHASE_SCREENING,
            PHASE_BOOTSTRAP,
            PHASE_HYBRID,
            PHASE_MECHANICS,
        ):
            with self.subTest(phase=phase):
                resolution = resolve_phase_mode(
                    formulations,
                    observations,
                    self.registry,
                    self.config,
                    requested_phase_mode=phase,
                    target_round_number=20,
                )
                self.assertEqual(resolution.active_phase, phase)
                self.assertTrue(resolution.override_used)
                self.assertEqual(resolution.target_proposal_round, 20)

    def test_technical_replicates_do_not_inflate_pair_counts(self) -> None:
        formulations, observations = self._phase_data(
            8, [("paired", "MECH_001")]
        )
        duplicate_rows = observations.loc[
            observations["formulation_id"].eq("paired")
            & observations["endpoint"].isin(
                ["viability_percent", "critical_axial_load_N_per_needle"]
            )
        ].copy()
        duplicate_rows["replicate_id"] = "rep_002"
        observations = pd.concat(
            [observations, duplicate_rows], ignore_index=True
        )
        resolution = resolve_phase_mode(
            formulations, observations, self.registry, self.config
        )
        self.assertEqual(resolution.paired_observation_count, 1)
        self.assertEqual(resolution.distinct_formulation_count, 1)
        self.assertEqual(resolution.batch_count, 1)
        self.assertEqual(
            resolution.to_metadata()["hybrid_gate"]["remaining"][
                "paired_observations"
            ],
            7,
        )

    def _mechanical_pool(self) -> pd.DataFrame:
        rows = []
        for index in range(12):
            row = self._formulation(f"f{index}", index)
            row.update(
                {
                    "candidate_id": f"c{index}",
                    "candidate_origin": "finite_pool_fallback",
                    "recommendation_type": "screening_candidate",
                    "screening_phase_score": 12.0 - index,
                    "mechanics_phase_score": 20.0 - index,
                    "hybrid_phase_score": 16.0 - index,
                    "empirical_combination_pass_probability": 0.75,
                    "prior_mechanical_observation_count": 0,
                    "mechanical_repeat_allowed": True,
                    "mechanical_repeat_status": "unmeasured",
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _models(training_frame: pd.DataFrame | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            mechanical_observation_count=0,
            training_frame=(
                pd.DataFrame() if training_frame is None else training_frame
            ),
        )

    def test_bootstrap_uses_utility_diversity_and_filters_repeats(self) -> None:
        pool = self._mechanical_pool()
        pool.loc[0, "recommendation_type"] = "retest_priority"
        pool.loc[1, "prior_mechanical_observation_count"] = 1
        pool.loc[1, "mechanical_repeat_allowed"] = False
        pool.loc[1, "mechanical_repeat_status"] = "previously_measured"
        resolution = PhaseResolution(
            requested_phase_mode=PHASE_AUTO,
            active_phase=PHASE_BOOTSTRAP,
            paired_observation_count=0,
            distinct_formulation_count=0,
            batch_count=0,
            reason="test",
            override_used=False,
        )
        with patch(
            "helper.selection._mechanics_phase_scores",
            side_effect=AssertionError("bootstrap must not score qLogNEHVI"),
        ):
            selected, metadata = select_mechanical_tests(
                pool,
                self._models(),
                self.registry,
                self.config,
                resolution,
                n=4,
            )
        primaries = selected.loc[selected["mechanical_primary_recommended"]]
        self.assertEqual(len(primaries), 4)
        self.assertNotIn("c0", set(selected["candidate_id"]))
        self.assertNotIn("c1", set(selected["candidate_id"]))
        self.assertEqual(primaries.iloc[0]["mechanical_transition_role"], "bootstrap_utility")
        self.assertEqual(
            set(primaries.iloc[1:]["mechanical_transition_role"]),
            {"bootstrap_coverage"},
        )
        self.assertEqual(metadata["eligibility"]["retest_excluded_count"], 1)
        self.assertEqual(
            metadata["eligibility"]["prior_mechanics_excluded_count"], 1
        )

    def test_second_bootstrap_anchor_is_deterministic_and_remeasured(self) -> None:
        formulations = pd.DataFrame(
            [self._formulation("a", 0), self._formulation("b", 1)]
        )
        observations = []
        for formulation_id, viability_values, load in (
            ("a", (90.0, 90.0), 1.0),
            ("b", (78.0, 82.0), 2.0),
        ):
            for replicate, viability in enumerate(viability_values, 1):
                observations.append(
                    self._observation(
                        formulation_id,
                        "MECH_001",
                        "viability_percent",
                        viability,
                        f"rep_{replicate:03d}",
                    )
                )
            observations.extend(
                [
                    self._observation(
                        formulation_id,
                        "MECH_001",
                        "critical_axial_load_N_per_needle",
                        load,
                    ),
                    self._observation(
                        formulation_id,
                        "MECH_001",
                        "intact_patch_formation_pass",
                        1.0,
                    ),
                ]
            )
        resolution = PhaseResolution(
            requested_phase_mode=PHASE_AUTO,
            active_phase=PHASE_BOOTSTRAP,
            paired_observation_count=4,
            distinct_formulation_count=4,
            batch_count=1,
            reason="test",
            override_used=False,
            bootstrap_batch_index=2,
        )
        anchor, metadata = _select_bootstrap_anchor(
            formulations,
            pd.DataFrame(observations),
            self.registry,
            self.config,
            resolution,
            policy_active=False,
            policy_version="",
        )
        self.assertTrue(metadata["selected"])
        self.assertEqual(anchor.iloc[0]["formulation_id"], "a")
        self.assertEqual(anchor.iloc[0]["candidate_origin"], "mechanics_anchor")
        self.assertEqual(anchor.iloc[0]["recommendation_type"], "mechanics_anchor")
        self.assertEqual(anchor.iloc[0]["mechanics_anchor_source_batch"], "MECH_001")

        rejected, rejected_metadata = _select_bootstrap_anchor(
            formulations,
            pd.DataFrame(observations),
            self.registry,
            self.config,
            resolution,
            policy_active=False,
            policy_version="",
            unavailable_feature_names=("ectoin_M",),
        )
        self.assertTrue(rejected.empty)
        self.assertIn("unavailable", rejected_metadata["reason"])

    def test_second_bootstrap_ranks_anchor_then_three_fresh_primaries(self) -> None:
        pool = self._mechanical_pool()
        anchor = self._formulation("anchor", 12)
        anchor.update(
            {
                "candidate_id": "anchor_candidate",
                "candidate_origin": "mechanics_anchor",
                "recommendation_type": "mechanics_anchor",
                "screening_phase_score": -10.0,
                "mechanics_phase_score": np.nan,
                "hybrid_phase_score": np.nan,
                "empirical_combination_pass_probability": 1.0,
                "prior_mechanical_observation_count": 1,
                "mechanical_repeat_allowed": True,
                "mechanical_repeat_status": "anchor_allowed",
            }
        )
        pool = pd.concat([pd.DataFrame([anchor]), pool], ignore_index=True)
        resolution = PhaseResolution(
            requested_phase_mode=PHASE_AUTO,
            active_phase=PHASE_BOOTSTRAP,
            paired_observation_count=4,
            distinct_formulation_count=4,
            batch_count=1,
            reason="test",
            override_used=False,
            bootstrap_batch_index=2,
        )
        selected, _ = select_mechanical_tests(
            pool,
            self._models(),
            self.registry,
            self.config,
            resolution,
            n=4,
        )
        primaries = selected.loc[selected["mechanical_primary_recommended"]]
        self.assertEqual(len(primaries), 4)
        self.assertEqual(primaries.iloc[0]["candidate_id"], "anchor_candidate")
        self.assertEqual(primaries.iloc[0]["mechanical_transition_role"], "anchor")
        self.assertEqual(primaries.iloc[1]["candidate_id"], "c0")
        self.assertEqual(
            primaries.iloc[1]["mechanical_transition_role"],
            "bootstrap_utility",
        )
        self.assertEqual(
            set(primaries.iloc[1:]["prior_mechanical_observation_count"]),
            {0},
        )

    def test_hybrid_allocates_two_qlognehvi_local_and_coverage_roles(self) -> None:
        pool = self._mechanical_pool()
        pool.loc[7, "candidate_origin"] = "local_perturbation"
        resolution = PhaseResolution(
            requested_phase_mode=PHASE_AUTO,
            active_phase=PHASE_HYBRID,
            paired_observation_count=8,
            distinct_formulation_count=7,
            batch_count=2,
            reason="test",
            override_used=False,
        )
        selected, metadata = select_mechanical_tests(
            pool,
            self._models(),
            self.registry,
            self.config,
            resolution,
            n=4,
        )
        primaries = selected.loc[selected["mechanical_primary_recommended"]]
        self.assertEqual(
            primaries["mechanical_transition_role"].value_counts().to_dict(),
            {
                "hybrid_qlognehvi": 2,
                "hybrid_local": 1,
                "hybrid_coverage": 1,
            },
        )
        self.assertEqual(metadata["transition_allocation"]["role_fallbacks"], [])
        self.assertTrue(
            selected.loc[~selected["mechanical_primary_recommended"], "mechanical_transition_role"]
            .eq("ordered_backup")
            .all()
        )

    def test_hybrid_role_fallback_is_deterministic(self) -> None:
        pool = self._mechanical_pool()
        pool["candidate_origin"] = "finite_pool_fallback"
        pool["empirical_combination_pass_probability"] = 0.40
        resolution = PhaseResolution(
            requested_phase_mode=PHASE_AUTO,
            active_phase=PHASE_HYBRID,
            paired_observation_count=8,
            distinct_formulation_count=6,
            batch_count=2,
            reason="test",
            override_used=False,
        )
        selected, metadata = select_mechanical_tests(
            pool,
            self._models(),
            self.registry,
            self.config,
            resolution,
            n=4,
        )
        primaries = selected.loc[selected["mechanical_primary_recommended"]]
        self.assertEqual(len(primaries), 4)
        fallbacks = metadata["transition_allocation"]["role_fallbacks"]
        self.assertEqual(
            [entry["role"] for entry in fallbacks[:2]],
            ["hybrid_local", "hybrid_coverage"],
        )
        self.assertEqual(primaries.iloc[2]["candidate_id"], "c2")
        self.assertEqual(primaries.iloc[3]["candidate_id"], "c3")

    def test_mechanical_results_on_unranked_rows_are_rejected(self) -> None:
        proposal = pd.DataFrame(
            {
                "candidate_id": ["c1", "c2"],
                "mechanical_selection_rank": [np.nan, np.nan],
                "mechanical_test_recommended": [False, False],
            }
        )
        completed = pd.DataFrame(
            {
                "candidate_id": ["c1", "c2"],
                "intact_patch_formation_pass": [1.0, 1.0],
                "critical_axial_load_N_per_needle": [1.0, np.nan],
            }
        )
        with self.assertRaises(Exception):
            validate_mechanics_execution(completed, proposal, primary_capacity=4)

    def test_proposal_and_pool_export_transition_fields(self) -> None:
        slate = self._mechanical_pool().iloc[:4].copy()
        slate.insert(0, "selection_rank", range(1, 5))
        slate["predicted_viability_percent"] = 80.0
        slate["viability_prediction_status"] = "model_supported"
        slate["intact_patch_pass_probability"] = 0.75
        slate["predicted_critical_axial_load_N_per_needle"] = 1.5
        slate["active_ingredient_count"] = 1
        slate["selection_explanation"] = ""
        mechanical = slate.iloc[:3].copy()
        mechanical.insert(0, "mechanical_selection_rank", range(1, 4))
        mechanical["mechanical_primary_recommended"] = True
        mechanical["mechanical_selection_mode"] = "bootstrap"
        mechanical["mechanical_backup_status"] = "primary"
        mechanical["mechanical_transition_role"] = [
            "bootstrap_utility",
            "bootstrap_coverage",
            "bootstrap_coverage",
        ]
        metadata = {
            "active_phase": PHASE_BOOTSTRAP,
            "phase_resolution": PhaseResolution(
                requested_phase_mode=PHASE_AUTO,
                active_phase=PHASE_BOOTSTRAP,
                paired_observation_count=0,
                distinct_formulation_count=0,
                batch_count=0,
                reason="test",
                override_used=False,
                completed_screening_round_count=8,
                transition_policy_version="mechanics_transition_v1",
            ).to_metadata(),
            "mechanical_policy": {
                "mechanical_selection_mode": "bootstrap",
                "mechanical_observation_count": 0,
                "botorch_available": False,
                "anchor": {"selected": False},
                "transition_allocation": {},
            },
            "mechanics_transition": {
                "anchor_selection": {"reason": "first bootstrap batch"}
            },
            "continuous_qlognehvi": {
                "continuous_optimizer_reason": "disabled during bootstrap"
            },
            "optimizer_mode": "bootstrap_utility_diversity",
            "optimizer_fallback_status": "not_applicable",
            "formulation_feasibility_policy_active": False,
        }
        result = SelectionResult(
            viability_screen=slate,
            mechanical_tests=mechanical,
            candidate_pool=slate.copy(),
            metadata=metadata,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "next_round"
            pool_path = Path(directory) / "pool.csv"
            write_selection_result(
                result,
                output,
                batch_id="ROUND_009",
                total_candidate_pool_path=pool_path,
                registry=self.registry,
            )
            proposal = pd.read_csv(output / "next_round_candidates.csv")
            total_pool = pd.read_csv(pool_path)
        for column in (
            "mechanical_transition_role",
            "prior_mechanical_observation_count",
            "mechanical_repeat_status",
            "mechanical_repeat_allowed",
        ):
            self.assertIn(column, proposal.columns)
            self.assertIn(column, total_pool.columns)
        self.assertEqual(
            proposal.loc[0, "mechanical_transition_role"], "bootstrap_utility"
        )
        self.assertTrue(pd.isna(proposal.loc[3, "mechanical_selection_rank"]))


if __name__ == "__main__":
    unittest.main()

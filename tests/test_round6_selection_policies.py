from __future__ import annotations

from dataclasses import replace
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

from helper.artifacts import (  # noqa: E402
    ArtifactConflictError,
    supersede_unstarted_proposal,
)
from helper.acquisition import qlognehvi_proxy_scores  # noqa: E402
from helper.cold_start import (  # noqa: E402
    ColdStartContext,
    annotate_cold_start_candidates,
    build_cold_start_context,
    planned_graduation_allocations,
    resolve_cold_start_policy,
)
from helper.config import load_optimization_config  # noqa: E402
from helper.feedback import ingest_feedback  # noqa: E402
from helper.intact_policy import (  # noqa: E402
    annotate_intact_combination_evidence,
    build_intact_evidence,
    resolve_intact_combination_policy,
)
from helper.mechanics_execution import (  # noqa: E402
    mechanics_execution_audit,
    validate_mechanics_execution,
)
from helper.models import RegressionPrediction  # noqa: E402
from helper.phase import PHASE_MECHANICS, PHASE_SCREENING, PhaseResolution  # noqa: E402
from helper.prediction_labels import annotate_viability_prediction_labels  # noqa: E402
from helper.registry import load_registry  # noqa: E402
from helper.selection import (  # noqa: E402
    _cold_start_trial_passes_shared_constraints,
    _enforce_cold_start_policy,
    _mechanics_phase_scores,
    annotate_candidates,
    select_mechanical_tests,
)
from helper.visualization import _aggregate_completed_candidates  # noqa: E402


class Round6PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry()
        cls.config = load_optimization_config()

    def _formulation(self, formulation_id: str, **features: float) -> dict:
        row = {"formulation_id": formulation_id}
        row.update({feature: 0.0 for feature in self.registry.feature_names})
        row.update(features)
        return row

    def _observation(
        self,
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

    def test_empirical_probability_examples_and_exact_set_isolation(self) -> None:
        policy = resolve_intact_combination_policy(self.config, 6)
        formulations = pd.DataFrame(
            [
                self._formulation("failed", taurine_M=0.10),
                self._formulation("passed", taurine_M=0.10),
                self._formulation("different", taurine_M=0.10, fbs_pct=1.0),
            ]
        )
        observations = pd.DataFrame(
            [
                self._observation(
                    "failed", "ROUND_005", "intact_patch_formation_pass", 0.0
                ),
                self._observation(
                    "passed", "ROUND_005", "intact_patch_formation_pass", 1.0
                ),
                self._observation(
                    "different", "ROUND_005", "intact_patch_formation_pass", 0.0
                ),
            ]
        )
        evidence = build_intact_evidence(
            formulations, observations, self.registry, policy, 6
        )
        candidates = pd.DataFrame(
            [
                self._formulation("mixed", taurine_M=0.10),
                self._formulation("unseen", myo_inositol_M=0.10),
            ]
        )
        annotated = annotate_intact_combination_evidence(
            candidates, evidence, self.registry, policy
        )
        self.assertAlmostEqual(
            annotated.loc[0, "empirical_combination_pass_probability"], 0.5
        )
        self.assertAlmostEqual(
            annotated.loc[0, "intact_combination_screening_penalty"], 0.0
        )
        self.assertAlmostEqual(
            annotated.loc[1, "empirical_combination_pass_probability"], 0.5
        )
        self.assertEqual(
            annotated.loc[1, "empirical_combination_weighted_failures"], 0.0
        )

        failure_only = evidence.loc[
            evidence["formulation_id"].eq("failed")
        ].reset_index(drop=True)
        failure_annotated = annotate_intact_combination_evidence(
            candidates.iloc[[0]], failure_only, self.registry, policy
        )
        self.assertAlmostEqual(
            failure_annotated.iloc[0][
                "empirical_combination_pass_probability"
            ],
            1.0 / 3.0,
        )
        self.assertAlmostEqual(
            failure_annotated.iloc[0]["intact_combination_screening_penalty"],
            0.2 / 3.0,
        )

        pass_only = evidence.loc[
            evidence["formulation_id"].eq("passed")
        ].reset_index(drop=True)
        pass_annotated = annotate_intact_combination_evidence(
            candidates.iloc[[0]], pass_only, self.registry, policy
        )
        self.assertAlmostEqual(
            pass_annotated.iloc[0]["empirical_combination_pass_probability"],
            2.0 / 3.0,
        )

        two_failures = pd.concat([failure_only, failure_only], ignore_index=True)
        two_failure_annotated = annotate_intact_combination_evidence(
            candidates.iloc[[0]], two_failures, self.registry, policy
        )
        self.assertAlmostEqual(
            two_failure_annotated.iloc[0][
                "empirical_combination_pass_probability"
            ],
            0.25,
        )
        self.assertAlmostEqual(
            two_failure_annotated.iloc[0]["intact_combination_screening_penalty"],
            0.10,
        )

    def test_distance_decay_ignores_far_same_combination(self) -> None:
        policy = resolve_intact_combination_policy(self.config, 6)
        formulations = pd.DataFrame(
            [self._formulation("failed", taurine_M=0.50)]
        )
        observations = pd.DataFrame(
            [
                self._observation(
                    "failed", "ROUND_005", "intact_patch_formation_pass", 0.0
                )
            ]
        )
        evidence = build_intact_evidence(
            formulations, observations, self.registry, policy, 6
        )
        candidate = pd.DataFrame(
            [self._formulation("candidate", taurine_M=0.001)]
        )
        annotated = annotate_intact_combination_evidence(
            candidate, evidence, self.registry, policy
        )
        self.assertAlmostEqual(
            annotated.iloc[0]["empirical_combination_pass_probability"], 0.5
        )
        self.assertGreater(
            annotated.iloc[0]["nearest_matching_intact_failure_distance"],
            policy.evidence_radius,
        )

        taurine_range = (
            self.registry.get_by_feature("taurine_M").upper_bound
            - self.registry.get_by_feature("taurine_M").lower_bound
        )
        half_radius_candidate = pd.DataFrame(
            [
                self._formulation(
                    "half",
                    taurine_M=0.50 - 0.5 * policy.evidence_radius * taurine_range,
                )
            ]
        )
        half = annotate_intact_combination_evidence(
            half_radius_candidate, evidence, self.registry, policy
        )
        self.assertAlmostEqual(
            half.iloc[0]["empirical_combination_weighted_failures"],
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            half.iloc[0]["empirical_combination_pass_probability"],
            1.0 / 2.5,
            places=6,
        )

    def test_intact_technical_replicates_use_all_pass_per_formulation_batch(self) -> None:
        policy = resolve_intact_combination_policy(self.config, 6)
        formulations = pd.DataFrame(
            [self._formulation("mixed_replicates", taurine_M=0.10)]
        )
        observations = pd.DataFrame(
            [
                self._observation(
                    "mixed_replicates",
                    "ROUND_005",
                    "intact_patch_formation_pass",
                    1.0,
                    "r1",
                ),
                self._observation(
                    "mixed_replicates",
                    "ROUND_005",
                    "intact_patch_formation_pass",
                    0.0,
                    "r2",
                ),
            ]
        )
        evidence = build_intact_evidence(
            formulations, observations, self.registry, policy, 6
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence.iloc[0]["intact_patch_formation_pass"], 0.0)

    def test_cold_start_counts_distinct_formulations_not_replicates_or_batches(self) -> None:
        policy = resolve_cold_start_policy(self.config, 6)
        formulations = pd.DataFrame(
            [
                self._formulation("a", taurine_M=0.10),
                self._formulation("b", myo_inositol_M=0.10),
            ]
        )
        observations = pd.DataFrame(
            [
                self._observation("a", "ROUND_004", "viability_percent", 50, "r1"),
                self._observation("a", "ROUND_004", "viability_percent", 51, "r2"),
                self._observation("a", "ROUND_005", "viability_percent", 52, "r1"),
                self._observation("b", "ROUND_005", "viability_percent", 53, "r1"),
            ]
        )
        context = build_cold_start_context(
            formulations,
            observations,
            self.registry,
            policy,
            6,
        )
        self.assertEqual(context.evidence_counts["taurine_M"], 1)
        self.assertEqual(context.evidence_counts["myo_inositol_M"], 1)
        self.assertIn("taurine_M", context.cold_ingredients)

    def test_graduation_allocations_are_breadth_first_and_bounded(self) -> None:
        policy = resolve_cold_start_policy(self.config, 6)
        counts = {feature: 3 for feature in self.registry.feature_names}
        counts.update(
            {
                "taurine_M": 0,
                "myo_inositol_M": 1,
                "methylcellulose_pct": 2,
                "propylene_glycol_M": 2,
            }
        )
        context = ColdStartContext(
            policy=policy,
            evidence_counts=counts,
            last_campaign_round={
                feature: None for feature in self.registry.feature_names
            },
            cold_ingredients=(
                "propylene_glycol_M",
                "taurine_M",
                "myo_inositol_M",
                "methylcellulose_pct",
            ),
        )
        allocations = planned_graduation_allocations(context, self.registry)
        self.assertEqual(len(allocations), 4)
        self.assertEqual(set(allocations), set(context.cold_ingredients))

    def test_ucb_formula_and_classifier_independence_in_screening(self) -> None:
        class FixedRegression:
            def __init__(self, mean: list[float], std: list[float]):
                self.mean = np.asarray(mean, dtype=float)
                self.std = np.asarray(std, dtype=float)

            def predict(self, _x: np.ndarray) -> RegressionPrediction:
                return RegressionPrediction(self.mean, self.std)

        class FixedProbability:
            fitted = True

            def __init__(self, values: list[float]):
                self.values = np.asarray(values, dtype=float)

            def predict_proba(self, _x: np.ndarray) -> np.ndarray:
                return self.values

        candidate_rows = []
        for candidate_id in ("a", "b"):
            row = self._formulation(candidate_id, taurine_M=0.1)
            row.update(
                {
                    "candidate_id": candidate_id,
                    "support_status": "in_support",
                    "intact_combination_screening_penalty": 0.0,
                }
            )
            candidate_rows.append(row)
        candidates = pd.DataFrame(candidate_rows)

        def models(classifier_values: list[float]) -> SimpleNamespace:
            return SimpleNamespace(
                viability=FixedRegression([56.30, 40.0], [24.72, 4.0]),
                critical_load=FixedRegression([0.0, 0.0], [0.0, 0.0]),
                initial_stiffness=FixedRegression([0.0, 0.0], [0.0, 0.0]),
                intact=FixedProbability(classifier_values),
                preparation=FixedProbability([1.0, 1.0]),
            )

        low_classifier = annotate_candidates(
            candidates,
            models([0.01, 0.99]),
            self.registry,
            self.config,
            policy_active=True,
        )
        reversed_classifier = annotate_candidates(
            candidates,
            models([0.99, 0.01]),
            self.registry,
            self.config,
            policy_active=True,
        )
        self.assertAlmostEqual(
            low_classifier.iloc[0]["viability_ucb"],
            56.30 + 0.35 * 24.72,
        )
        np.testing.assert_allclose(
            low_classifier["screening_phase_score"],
            reversed_classifier["screening_phase_score"],
        )

    def test_viability_labels_hide_unknowns_but_preserve_raw_surrogate(self) -> None:
        policy = resolve_cold_start_policy(self.config, 7)
        counts = {feature: 3 for feature in self.registry.feature_names}
        counts["myo_inositol_M"] = 2
        context = ColdStartContext(
            policy=policy,
            evidence_counts=counts,
            last_campaign_round={
                feature: None for feature in self.registry.feature_names
            },
            cold_ingredients=("myo_inositol_M",),
        )
        candidates = pd.DataFrame(
            [
                {
                    **self._formulation("observed", myo_inositol_M=0.4),
                    "candidate_id": "observed",
                    "recommendation_type": "retest_priority",
                    "cold_start_ingredients": "myo_inositol_M",
                    "cold_start_prior_evidence_counts": "myo_inositol_M=2",
                    "support_status": "in_support",
                    "predicted_viability_percent": 8.2,
                    "viability_std": 1.0,
                    "viability_ucb": 8.55,
                },
                {
                    **self._formulation("cold", myo_inositol_M=0.2),
                    "candidate_id": "cold",
                    "recommendation_type": "screening_candidate",
                    "cold_start_ingredients": "myo_inositol_M",
                    "cold_start_prior_evidence_counts": "myo_inositol_M=2",
                    "support_status": "in_support",
                    "predicted_viability_percent": 50.0,
                    "viability_std": 30.0,
                    "viability_ucb": 60.5,
                },
                {
                    **self._formulation("supported", ectoin_M=0.2),
                    "candidate_id": "supported",
                    "recommendation_type": "screening_candidate",
                    "cold_start_ingredients": "",
                    "cold_start_prior_evidence_counts": "",
                    "support_status": "in_support",
                    "predicted_viability_percent": 62.0,
                    "viability_std": 5.0,
                    "viability_ucb": 63.75,
                },
                {
                    **self._formulation("prior", ectoin_M=0.1),
                    "candidate_id": "prior",
                    "recommendation_type": "screening_candidate",
                    "cold_start_ingredients": "",
                    "cold_start_prior_evidence_counts": "",
                    "support_status": "in_support",
                    "predicted_viability_percent": 50.5,
                    "viability_std": 30.0,
                    "viability_ucb": 61.0,
                },
            ]
        )
        observations = pd.DataFrame(
            [
                self._observation(
                    "observed",
                    "ROUND_006",
                    "viability_percent",
                    8.1,
                )
            ]
        )
        models = SimpleNamespace(
            viability=SimpleNamespace(fallback_mean=50.0, fitted=True),
            training_frame=pd.DataFrame(
                {"viability_percent": [20.0, 50.0, 80.0]}
            ),
        )
        labeled = annotate_viability_prediction_labels(
            candidates,
            observations,
            models,
            context,
            self.config,
            target_round_number=7,
        ).set_index("candidate_id")

        self.assertEqual(
            labeled.loc["observed", "viability_prediction_status"],
            "observed_retest",
        )
        self.assertAlmostEqual(
            labeled.loc["observed", "predicted_viability_percent"],
            8.2,
        )
        self.assertEqual(
            labeled.loc["cold", "viability_prediction_status"],
            "unknown_cold_start",
        )
        self.assertTrue(
            pd.isna(labeled.loc["cold", "predicted_viability_percent"])
        )
        self.assertTrue(pd.isna(labeled.loc["cold", "viability_std"]))
        self.assertAlmostEqual(
            labeled.loc["cold", "raw_surrogate_viability_mean"],
            50.0,
        )
        self.assertAlmostEqual(labeled.loc["cold", "viability_ucb"], 60.5)
        self.assertEqual(
            labeled.loc["supported", "viability_prediction_status"],
            "model_supported",
        )
        self.assertAlmostEqual(
            labeled.loc["supported", "predicted_viability_percent"],
            62.0,
        )
        self.assertEqual(
            labeled.loc["prior", "viability_prediction_status"],
            "unknown_prior_reversion",
        )


    def test_mechanics_adds_log_empirical_probability_not_classifier_penalty(self) -> None:
        policy = resolve_intact_combination_policy(self.config, 6)
        features = self.registry.feature_names
        rows = []
        for index, probability in enumerate([0.5, 0.25]):
            row = {feature: 0.0 for feature in features}
            row.update(
                {
                    "candidate_id": f"c{index}",
                    "viability_ucb": 50.0,
                    "critical_axial_load_ucb": 1.0,
                    "empirical_combination_pass_probability": probability,
                    "screening_acquisition_penalty": 0.0,
                    "acquisition_penalty": 999.0 if index == 0 else -999.0,
                }
            )
            rows.append(row)
        annotated = pd.DataFrame(rows)
        training = pd.DataFrame(
            {
                "formulation_id": ["a", "b"],
                "batch_id": ["ROUND_001", "ROUND_002"],
                "viability_percent": [50.0, 60.0],
                "critical_axial_load_N_per_needle": [1.0, 2.0],
                **{feature: [0.0, 0.0] for feature in features},
            }
        )
        models = SimpleNamespace(training_frame=training)
        with patch(
            "helper.selection.try_botorch_qlognehvi_scores",
            return_value=(np.array([1.0, 1.0]), {}),
        ):
            scores, metadata = _mechanics_phase_scores(
                annotated,
                models,
                self.registry,
                self.config,
                intact_policy=policy,
            )
        self.assertAlmostEqual(scores[0], 1.0 + np.log(0.5))
        self.assertAlmostEqual(scores[1], 1.0 + np.log(0.25))
        self.assertEqual(
            metadata["classifier_probability_selection_role"], "diagnostic_only"
        )

    def test_finite_pool_mechanics_proxy_multiplies_improvement_by_feasibility(self) -> None:
        viability = np.array([0.0, 1.0, 1.0])
        mechanics = np.array([0.0, 1.0, 1.0])
        probabilities = np.array([0.5, 0.5, 0.25])
        scores = qlognehvi_proxy_scores(
            pd.DataFrame(index=range(3)),
            viability,
            mechanics,
            feasibility_probability=probabilities,
        )
        self.assertAlmostEqual(scores[1], np.log1p(0.5))
        self.assertAlmostEqual(scores[2], np.log1p(0.25))

    def test_mechanics_ranks_all_rows_by_weighted_score_and_leaves_screening_blank(self) -> None:
        policy = resolve_intact_combination_policy(self.config, 6)
        rows = []
        for index in range(12):
            row = self._formulation(f"f{index}", taurine_M=0.01 + index * 0.01)
            row.update(
                {
                    "candidate_id": f"c{index:02d}",
                    "mechanics_phase_score": float(index),
                    "intact_patch_pass_probability": 0.01 if index % 2 else 0.99,
                }
            )
            rows.append(row)
        frame = pd.DataFrame(rows)
        models = SimpleNamespace(mechanical_observation_count=8)
        mechanics_phase = PhaseResolution(
            requested_phase_mode=PHASE_MECHANICS,
            active_phase=PHASE_MECHANICS,
            paired_observation_count=8,
            distinct_formulation_count=6,
            batch_count=2,
            reason="test",
            override_used=False,
        )
        ranked, metadata = select_mechanical_tests(
            frame,
            models,
            self.registry,
            self.config,
            mechanics_phase,
            n=4,
            intact_policy=policy,
        )
        self.assertEqual(len(ranked), 12)
        self.assertEqual(ranked.iloc[0]["candidate_id"], "c11")
        self.assertEqual(int(ranked["mechanical_primary_recommended"].sum()), 4)
        self.assertEqual(metadata["backup_count"], 8)

        screening_phase = replace(
            mechanics_phase,
            requested_phase_mode=PHASE_SCREENING,
            active_phase=PHASE_SCREENING,
        )
        disabled, disabled_metadata = select_mechanical_tests(
            frame,
            models,
            self.registry,
            self.config,
            screening_phase,
            n=4,
            intact_policy=policy,
        )
        self.assertTrue(disabled.empty)
        self.assertEqual(disabled_metadata["primary_recommendation_count"], 0)
        self.assertEqual(disabled_metadata["backup_count"], 0)

    def test_independent_cold_caps_count_multi_cold_rows_and_exempt_special_rows(self) -> None:
        base_policy = resolve_cold_start_policy(self.config, 6)
        policy = replace(base_policy, graduation_slots_per_round=0)
        counts = {feature: 3 for feature in self.registry.feature_names}
        counts.update({"taurine_M": 0, "myo_inositol_M": 0})
        context = ColdStartContext(
            policy=policy,
            evidence_counts=counts,
            last_campaign_round={feature: None for feature in self.registry.feature_names},
            cold_ingredients=("taurine_M", "myo_inositol_M"),
        )

        def candidate(candidate_id: str, origin: str, score: float, **features: float) -> dict:
            row = self._formulation(candidate_id, **features)
            row.update(
                {
                    "candidate_id": candidate_id,
                    "candidate_origin": origin,
                    "screening_phase_score": score,
                    "feasibility_pass": True,
                    "recommendation_type": "screening_candidate",
                    "selection_explanation": "",
                    "support_status": "in_support",
                }
            )
            return row

        selected = pd.DataFrame(
            [
                candidate("t1", "local_perturbation", 10, taurine_M=0.10),
                candidate("t2", "local_perturbation", 9, taurine_M=0.20, fbs_pct=1.0),
                candidate("both", "local_perturbation", 1, taurine_M=0.30, myo_inositol_M=0.10),
                candidate("m1", "local_perturbation", 8, myo_inositol_M=0.20, ectoin_M=0.10),
                candidate("m2", "local_perturbation", 7, myo_inositol_M=0.30, glucose_M=0.10),
                candidate("special", "rescue_dilution", 6, taurine_M=0.40, myo_inositol_M=0.40),
            ]
        )
        pool = pd.concat(
            [
                selected,
                pd.DataFrame(
                    [
                        candidate("r1", "local_perturbation", 5, ectoin_M=0.20),
                        candidate("r2", "local_perturbation", 4, glucose_M=0.20),
                    ]
                ),
            ],
            ignore_index=True,
        )
        adjusted = _enforce_cold_start_policy(
            selected,
            pool,
            self.registry,
            self.config,
            "screening_phase_score",
            context,
        )
        metadata = adjusted.attrs["cold_start_policy"]
        self.assertLessEqual(metadata["ordinary_counts_after"]["taurine_M"], 2)
        self.assertLessEqual(metadata["ordinary_counts_after"]["myo_inositol_M"], 2)
        self.assertIn("special", set(adjusted["candidate_id"]))
        self.assertEqual(len(adjusted), len(selected))

        universal_violation = pd.concat(
            [
                selected.iloc[[0]].assign(candidate_id=f"u{index}")
                for index in range(6)
            ],
            ignore_index=True,
        )
        self.assertFalse(
            _cold_start_trial_passes_shared_constraints(
                universal_violation,
                self.registry,
                self.config,
            )
        )

    def test_graduation_reassigns_an_unfillable_reserved_slot(self) -> None:
        policy = resolve_cold_start_policy(self.config, 6)
        counts = {feature: 3 for feature in self.registry.feature_names}
        counts.update(
            {
                "propylene_glycol_M": 2,
                "methylcellulose_pct": 2,
                "myo_inositol_M": 1,
                "taurine_M": 0,
                "creatine_M": 0,
            }
        )
        context = ColdStartContext(
            policy=policy,
            evidence_counts=counts,
            last_campaign_round={feature: None for feature in self.registry.feature_names},
            cold_ingredients=(
                "propylene_glycol_M",
                "methylcellulose_pct",
                "myo_inositol_M",
                "taurine_M",
                "creatine_M",
            ),
        )
        selected_rows = []
        for index, feature_name in enumerate(
            ("methylcellulose_pct", "myo_inositol_M", "taurine_M", "creatine_M")
        ):
            row = self._formulation(f"g{index}", **{feature_name: 0.2})
            row.update(
                {
                    "candidate_id": f"g{index}",
                    "candidate_origin": "local_perturbation",
                    "screening_phase_score": 10.0 - index,
                    "feasibility_pass": True,
                    "recommendation_type": "screening_candidate",
                    "selection_explanation": "",
                    "support_status": "in_support",
                }
            )
            selected_rows.append(row)
        selected = pd.DataFrame(selected_rows)
        adjusted = _enforce_cold_start_policy(
            selected,
            selected,
            self.registry,
            self.config,
            "screening_phase_score",
            context,
        )
        metadata = adjusted.attrs["cold_start_policy"]
        self.assertEqual(metadata["graduation_selected_count"], 4)
        self.assertTrue(metadata["graduation_reassignments"])
        self.assertIn(
            "creatine_M",
            {
                row["ingredient"]
                for row in metadata["selected_graduation_allocations"]
            },
        )

    def test_actual_intact_backup_promotion_and_shortfall(self) -> None:
        proposal = pd.DataFrame(
            {
                "candidate_id": [f"c{i}" for i in range(1, 13)],
                "mechanical_selection_rank": list(range(1, 13)),
                "mechanical_test_recommended": [True] * 4 + [False] * 8,
            }
        )
        pass_ids = {"c2", "c5", "c6", "c8"}
        completed = proposal[["candidate_id"]].copy()
        completed["intact_patch_formation_pass"] = completed["candidate_id"].isin(pass_ids)
        completed["critical_axial_load_N_per_needle"] = np.where(
            completed["candidate_id"].isin(pass_ids), 1.0, np.nan
        )
        audit = validate_mechanics_execution(completed, proposal, primary_capacity=4)
        self.assertEqual(audit["manifest_version"], 2)
        self.assertEqual(audit["actual_intact_measured_count"], 12)
        self.assertEqual(audit["actual_intact_pass_count"], 4)
        self.assertEqual(audit["actual_intact_fail_count"], 8)
        self.assertEqual(audit["ranked_actual_intact_pass_count"], 4)
        self.assertEqual(audit["expected_actual_intact_test_ids"], ["c2", "c5", "c6", "c8"])
        self.assertEqual(audit["promoted_backup_ids"], ["c5", "c6", "c8"])
        self.assertEqual(audit["actual_pass_shortfall"], 0)

        short = completed.copy()
        short["intact_patch_formation_pass"] = short["candidate_id"].isin({"c2", "c5"})
        short["critical_axial_load_N_per_needle"] = np.where(
            short["candidate_id"].isin({"c2", "c5"}), 1.0, np.nan
        )
        short_audit = mechanics_execution_audit(short, proposal, primary_capacity=4)
        self.assertEqual(short_audit["actual_pass_shortfall"], 2)

        wrong = completed.copy()
        wrong["critical_axial_load_N_per_needle"] = np.where(
            wrong["candidate_id"].isin({"c2", "c5", "c6", "c9"}), 1.0, np.nan
        )
        with self.assertRaises(Exception):
            validate_mechanics_execution(wrong, proposal, primary_capacity=4)

        five = completed.copy()
        five_pass_ids = {"c2", "c5", "c6", "c8", "c9"}
        five["intact_patch_formation_pass"] = five["candidate_id"].isin(five_pass_ids)
        five["critical_axial_load_N_per_needle"] = np.where(
            five["candidate_id"].isin(five_pass_ids), 1.0, np.nan
        )
        with self.assertRaises(Exception):
            validate_mechanics_execution(five, proposal, primary_capacity=4)

    def test_screening_manifest_counts_intact_outcomes_without_mechanical_ranks(self) -> None:
        proposal = pd.DataFrame(
            {
                "candidate_id": ["c1", "c2", "c3"],
                "mechanical_selection_rank": [np.nan, np.nan, np.nan],
                "mechanical_test_recommended": [False, False, False],
            }
        )
        completed = pd.DataFrame(
            {
                "candidate_id": ["c1", "c2", "c3"],
                "intact_patch_formation_pass": ["yes", "pass", "failed"],
            }
        )
        audit = mechanics_execution_audit(completed, proposal, primary_capacity=4)
        self.assertFalse(audit["mechanics_active_in_proposal"])
        self.assertEqual(audit["actual_intact_measured_count"], 3)
        self.assertEqual(audit["actual_intact_pass_count"], 2)
        self.assertEqual(audit["actual_intact_fail_count"], 1)
        self.assertEqual(audit["ranked_actual_intact_pass_count"], 0)
        self.assertEqual(audit["expected_actual_intact_test_ids"], [])

    def test_completed_report_aggregates_text_boolean_replicates(self) -> None:
        completed = pd.DataFrame(
            {
                "candidate_id": ["pass"] * 4 + ["fail"] * 4,
                "selection_rank": [1] * 4 + [2] * 4,
                "intact_patch_formation_pass": [
                    "yes",
                    "pass",
                    "TRUE",
                    "1",
                    "pass",
                    "passed",
                    "failed",
                    "yes",
                ],
            }
        )
        summary = _aggregate_completed_candidates(completed).set_index("candidate_id")
        self.assertEqual(summary.loc["pass", "completed_intact_replicate_count"], 4)
        self.assertEqual(summary.loc["pass", "completed_intact_pass_fraction"], 1.0)
        self.assertEqual(summary.loc["pass", "completed_intact_gate_pass"], 1.0)
        self.assertEqual(summary.loc["fail", "completed_intact_replicate_count"], 4)
        self.assertEqual(summary.loc["fail", "completed_intact_pass_fraction"], 0.75)
        self.assertEqual(summary.loc["fail", "completed_intact_gate_pass"], 0.0)

    def test_feedback_records_immutable_observation_source_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._formulation("f1", ectoin_M=0.1)
            candidate["candidate_id"] = "c1"
            candidate_path = root / "proposal.csv"
            pd.DataFrame([candidate]).to_csv(candidate_path, index=False)
            feedback_path = root / "active.csv"
            pd.DataFrame(
                [
                    {
                        "formulation_id": "f1",
                        "candidate_id": "c1",
                        "replicate_id": "rep_001",
                        "viability_percent": 50.0,
                        "intact_patch_formation_pass": "yes",
                    }
                ]
            ).to_csv(feedback_path, index=False)
            immutable_source = "results/multi_objective_v2/rounds/ROUND_999/completed/completed.csv"
            _, observations = ingest_feedback(
                feedback_path=feedback_path,
                candidate_files=[candidate_path],
                formulations=pd.DataFrame(),
                observations=pd.DataFrame(),
                registry=self.registry,
                batch_id="ROUND_999",
                observation_source_file=immutable_source,
            )
            self.assertEqual(set(observations["source_file"]), {immutable_source})

    def test_supersession_rejects_entered_results_and_archives_blank_proposal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposal = root / "rounds" / "ROUND_006" / "proposal"
            proposal.mkdir(parents=True)
            (proposal / "proposal.csv").write_text("candidate_id\nc1\n")
            worksheet = root / "next_round.csv"
            pd.DataFrame(
                {
                    "candidate_id": ["c1"],
                    "batch_id": ["ROUND_006"],
                    "viability_percent": [55.0],
                }
            ).to_csv(worksheet, index=False)
            with self.assertRaises(ArtifactConflictError):
                supersede_unstarted_proposal(
                    "ROUND_006",
                    observations=pd.DataFrame(),
                    active_worksheet=worksheet,
                    reason="test",
                    policy_versions=["v1"],
                    results_root=root,
                )

            pd.DataFrame(
                {
                    "candidate_id": ["c1"],
                    "batch_id": ["ROUND_006"],
                    "viability_percent": [np.nan],
                }
            ).to_csv(worksheet, index=False)
            archived = supersede_unstarted_proposal(
                "ROUND_006",
                observations=pd.DataFrame(),
                active_worksheet=worksheet,
                reason="test",
                policy_versions=["v1"],
                results_root=root,
            )
            self.assertIsNotNone(archived)
            self.assertTrue((archived / "proposal" / "proposal.csv").exists())
            self.assertTrue((archived / "supersession_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()

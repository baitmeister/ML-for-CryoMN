from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helper.config import load_optimization_config
from helper.models import EndpointModels, RegressionPrediction
from helper.phase import PHASE_MECHANICS, PHASE_SCREENING, PhaseResolution
from helper.registry import load_registry
from helper.selection import (
    _enforce_single_ingredient_spacing,
    _greedy_diverse_pick,
    _select_round_slate,
    annotate_candidates,
    select_next_round,
    select_mechanical_tests,
)


def _dummy_models(mechanical_count: int) -> EndpointModels:
    registry = load_registry()
    frame = pd.DataFrame(
        {
            "viability_percent": [60.0] * mechanical_count
            + [np.nan] * max(0, 2 - mechanical_count),
            "critical_axial_load_N_per_needle": [1.0] * mechanical_count
            + [np.nan] * max(0, 2 - mechanical_count)
        }
    )
    for feature_name in registry.feature_names:
        frame[feature_name] = 0.0
    return EndpointModels(
        feature_names=registry.feature_names,
        viability=None,
        critical_load=None,
        initial_stiffness=None,
        intact=None,
        preparation=None,
        training_frame=frame,
    )


def _phase(active_phase: str) -> PhaseResolution:
    return PhaseResolution(
        requested_phase_mode=active_phase,
        active_phase=active_phase,
        paired_observation_count=8 if active_phase == PHASE_MECHANICS else 0,
        distinct_formulation_count=6 if active_phase == PHASE_MECHANICS else 0,
        batch_count=2 if active_phase == PHASE_MECHANICS else 0,
        reason="test",
        override_used=True,
    )


def _annotated_candidates() -> pd.DataFrame:
    registry = load_registry()
    rows = []
    for index in range(8):
        row = {feature_name: 0.0 for feature_name in registry.feature_names}
        row[registry.feature_names[index % len(registry.feature_names)]] = 0.2
        row.update(
            {
                "candidate_id": f"cand_{index}",
                "viability_ucb": 50.0 + index,
                "critical_axial_load_ucb": 1.0 + index,
                "intact_patch_pass_probability": 0.9 if index < 5 else 0.2,
                "acquisition_penalty": 0.0,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_screening_phase_disables_mechanical_selection() -> None:
    registry = load_registry()
    config = load_optimization_config()
    selected, metadata = select_mechanical_tests(
        _annotated_candidates(),
        _dummy_models(mechanical_count=0),
        registry,
        config,
        _phase(PHASE_SCREENING),
        n=4,
    )
    assert metadata["mechanical_selection_mode"] == "disabled_screening_only"
    assert metadata["active_phase"] == PHASE_SCREENING
    assert len(selected) == 0


def test_seeded_mechanical_selection_switches_to_qlognehvi_path() -> None:
    registry = load_registry()
    config = load_optimization_config()
    selected, metadata = select_mechanical_tests(
        _annotated_candidates(),
        _dummy_models(mechanical_count=8),
        registry,
        config,
        _phase(PHASE_MECHANICS),
        n=3,
    )
    assert len(selected) == 3
    assert metadata["mechanical_selection_mode"].startswith("qlognehvi")


def test_single_ingredient_same_feature_candidates_are_spaced_or_replaced() -> None:
    registry = load_registry()
    config = load_optimization_config()

    def _row(candidate_id: str, feature_name: str, value: float, score: float) -> dict[str, float | str]:
        row = {name: 0.0 for name in registry.feature_names}
        row.update(
            {
                "candidate_id": candidate_id,
                feature_name: value,
                "screening_phase_score": score,
                "recommendation_type": "screening_candidate",
                "selection_explanation": "",
            }
        )
        return row

    candidate_pool = pd.DataFrame(
        [
            _row("cand_a", "trehalose_M", 0.10, 0.99),
            _row("cand_b", "trehalose_M", 0.11, 0.98),
            _row("cand_c", "glucose_M", 0.20, 0.97),
            _row("cand_d", "trehalose_M", 0.13, 0.96),
        ]
    )
    selected = candidate_pool.iloc[:2].copy()

    adjusted = _enforce_single_ingredient_spacing(
        selected,
        candidate_pool,
        registry,
        config,
        score_column="screening_phase_score",
    )

    selected_ids = set(adjusted["candidate_id"])
    assert len(adjusted) == 2
    assert "cand_c" in selected_ids
    assert not {"cand_a", "cand_b"}.issubset(selected_ids)


class _FixedRegression:
    fitted = True

    def __init__(self, mean: list[float], std: list[float]):
        self._mean = np.asarray(mean, dtype=float)
        self._std = np.asarray(std, dtype=float)

    def predict(self, _x: np.ndarray) -> RegressionPrediction:
        return RegressionPrediction(self._mean, self._std)


class _FixedProbability:
    def __init__(self, probability: float, fitted: bool):
        self.default_probability = probability
        self.fitted = fitted

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.default_probability)


def test_out_of_support_uncertainty_is_capped_for_both_objectives() -> None:
    registry = load_registry()
    config = load_optimization_config()
    rows = []
    for index, support_status in enumerate(["in_support", "in_support", "boundary"]):
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": f"candidate_{index}",
                "dmso_M": 0.01 * (index + 1),
                "support_status": support_status,
            }
        )
        rows.append(row)
    models = EndpointModels(
        feature_names=registry.feature_names,
        viability=_FixedRegression([60.0] * 3, [1.0, 2.0, 100.0]),
        critical_load=_FixedRegression([0.2] * 3, [0.1, 0.2, 10.0]),
        initial_stiffness=_FixedRegression([1.0] * 3, [0.1] * 3),
        intact=_FixedProbability(0.75, fitted=False),
        preparation=_FixedProbability(0.75, fitted=False),
        training_frame=pd.DataFrame(),
    )

    annotated = annotate_candidates(
        pd.DataFrame(rows),
        models,
        registry,
        config,
        policy_active=True,
    )

    assert annotated.loc[2, "viability_std"] == pytest.approx(1.9)
    assert annotated.loc[2, "critical_axial_load_std"] == pytest.approx(0.19)


def test_diversity_cannot_select_outside_the_competitive_utility_band() -> None:
    registry = load_registry()
    frame = pd.DataFrame(
        [
            {**{feature: 0.0 for feature in registry.feature_names}, "dmso_M": 0.01},
            {**{feature: 0.0 for feature in registry.feature_names}, "dmso_M": 0.02},
            {**{feature: 0.0 for feature in registry.feature_names}, "pvp_pct": 10.0},
        ]
    )
    selected = _greedy_diverse_pick(
        frame,
        np.array([1.0, 0.9, 0.1]),
        registry.feature_names,
        n=2,
        diversity_weight=0.05,
        competitive_utility_band=0.15,
    )
    assert selected == [0, 1]


def test_screening_slate_prefers_intact_gate_when_pool_can_fill_round() -> None:
    registry = load_registry()
    config = load_optimization_config()
    rows = []
    for candidate_id, betaine, score, intact in [
        ("low_intact_best_score", 0.10, 1.00, 0.10),
        ("pass_a", 0.20, 0.90, 0.60),
        ("pass_b", 0.30, 0.80, 0.70),
    ]:
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": candidate_id,
                "formulation_id": candidate_id,
                "betaine_M": betaine,
                "screening_phase_score": score,
                "recommendation_type": "screening_candidate",
                "selection_explanation": "",
                "intact_patch_pass_probability": intact,
            }
        )
        rows.append(row)

    selected = _select_round_slate(
        pd.DataFrame(rows),
        registry,
        config,
        _phase(PHASE_SCREENING),
        n=2,
        policy_active=True,
    )

    assert set(selected["candidate_id"]) == {"pass_a", "pass_b"}


def test_screening_slate_reserves_capped_rescue_dilution_candidates() -> None:
    registry = load_registry()
    config = load_optimization_config()
    config["candidate_generation"]["rescue_candidates_per_round"] = 1
    rows = []
    for candidate_id, origin, betaine, score, intact, ucb in [
        ("rescue_a", "rescue_dilution", 0.10, -1.00, 0.10, 100.0),
        ("rescue_b", "rescue_dilution", 0.15, -1.00, 0.10, 90.0),
        ("pass_a", "sparse_exploration", 0.20, 0.90, 0.60, 80.0),
        ("pass_b", "sparse_exploration", 0.30, 0.80, 0.70, 70.0),
    ]:
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": candidate_id,
                "formulation_id": candidate_id,
                "candidate_origin": origin,
                "betaine_M": betaine,
                "screening_phase_score": score,
                "viability_ucb": ucb,
                "recommendation_type": "rescue_candidate" if origin == "rescue_dilution" else "screening_candidate",
                "selection_explanation": "",
                "intact_patch_pass_probability": intact,
            }
        )
        rows.append(row)

    selected = _select_round_slate(
        pd.DataFrame(rows),
        registry,
        config,
        _phase(PHASE_SCREENING),
        n=3,
        policy_active=True,
    )

    assert "rescue_a" in set(selected["candidate_id"])
    assert "rescue_b" not in set(selected["candidate_id"])


def test_policy_active_retests_must_pass_formulation_feasibility() -> None:
    registry = load_registry()
    config = load_optimization_config()
    config["round_policy"]["viability_screens_per_round"] = 4

    bad_retest = {feature: 0.0 for feature in registry.feature_names}
    bad_retest.update(
        {
            "formulation_id": "v2_bad_retest",
            "source": "wetlab_feedback:ROUND_001",
            "source_row_id": "cand_bad_retest",
            "formulation_label": "bad retest",
            "pvp_pct": 11.0,
            "active_ingredient_count": 1,
        }
    )
    formulations = pd.DataFrame([bad_retest])
    observations = pd.DataFrame(
        [
            {
                "observation_id": "obs_bad_retest_legacy_viability",
                "formulation_id": "v2_bad_retest",
                "batch_id": "legacy_wetlab",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 80.0,
                "unit": "percent",
                "observation_noise": 5.0,
                "source_type": "legacy_wetlab",
                "source_file": "test",
                "notes": "",
            },
            {
                "observation_id": "obs_bad_retest_viability",
                "formulation_id": "v2_bad_retest",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 25.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test",
                "notes": "",
            }
        ]
    )
    candidate_rows = []
    for index, betaine in enumerate([0.20, 0.30, 0.40, 0.50], start=1):
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": f"cand_{index}",
                "formulation_id": f"v2_candidate_{index}",
                "betaine_M": betaine,
                "active_ingredient_count": 1,
                "feasibility_pass": True,
                "support_status": "in_support",
                "candidate_origin": "sparse_exploration",
            }
        )
        candidate_rows.append(row)

    result = select_next_round(
        formulations=formulations,
        observations=observations,
        candidate_pool=pd.DataFrame(candidate_rows),
        registry=registry,
        optimization_config=config,
        target_round_number=2,
        policy_active=True,
    )

    assert "retest_priority" not in set(result.viability_screen["recommendation_type"].astype(str))
    assert result.metadata["retest_candidate_count_rejected_by_feasibility"] == 1


def test_policy_active_retests_must_pass_intact_screening_gate() -> None:
    registry = load_registry()
    config = load_optimization_config()
    config["round_policy"]["viability_screens_per_round"] = 4

    retest = {feature: 0.0 for feature in registry.feature_names}
    retest.update(
        {
            "formulation_id": "v2_failed_intact_retest",
            "source": "wetlab_feedback:ROUND_001",
            "source_row_id": "cand_failed_intact",
            "formulation_label": "failed intact retest",
            "pvp_pct": 5.0,
            "active_ingredient_count": 1,
        }
    )
    formulations = pd.DataFrame([retest])
    observations = pd.DataFrame(
        [
            {
                "observation_id": "obs_failed_intact_legacy_viability",
                "formulation_id": "v2_failed_intact_retest",
                "batch_id": "legacy_wetlab",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 80.0,
                "unit": "percent",
                "observation_noise": 5.0,
                "source_type": "legacy_wetlab",
                "source_file": "test",
                "notes": "",
            },
            {
                "observation_id": "obs_failed_intact_viability",
                "formulation_id": "v2_failed_intact_retest",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 25.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test",
                "notes": "",
            },
            {
                "observation_id": "obs_failed_intact_gate",
                "formulation_id": "v2_failed_intact_retest",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "intact_patch_formation_pass",
                "value": 0.0,
                "unit": "binary",
                "observation_noise": "",
                "source_type": "wetlab_feedback",
                "source_file": "test",
                "notes": "",
            },
        ]
    )
    candidate_rows = []
    for index, betaine in enumerate([0.20, 0.30, 0.40, 0.50], start=1):
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": f"cand_gate_{index}",
                "formulation_id": f"v2_gate_candidate_{index}",
                "betaine_M": betaine,
                "active_ingredient_count": 1,
                "feasibility_pass": True,
                "support_status": "in_support",
                "candidate_origin": "sparse_exploration",
            }
        )
        candidate_rows.append(row)

    result = select_next_round(
        formulations=formulations,
        observations=observations,
        candidate_pool=pd.DataFrame(candidate_rows),
        registry=registry,
        optimization_config=config,
        target_round_number=2,
        policy_active=True,
    )

    assert "retest_priority" not in set(result.viability_screen["recommendation_type"].astype(str))

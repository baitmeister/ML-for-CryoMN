from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from helper.config import load_optimization_config
from helper.models import EndpointModels, RegressionPrediction
from helper.phase import PHASE_MECHANICS, PHASE_SCREENING, PhaseResolution
from helper.registry import load_registry
from helper.selection import (
    _allocate_screening_origin_quota,
    _enforce_ingredient_combination_cap,
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


def test_ingredient_combination_cap_swaps_out_overcap_pair_for_pool_replacement() -> None:
    """_enforce_ingredient_combination_cap must not let more than `cap`
    selected candidates share the exact same active-ingredient set, even
    though every one of them individually scores higher than the
    replacement candidates available in the pool.

    This is the ectoin+ethylene_glycol clustering bug: origin-bucket
    diversity alone doesn't stop every bucket from independently
    re-discovering the same favored two-ingredient (or larger) combination.
    """
    registry = load_registry()
    config = load_optimization_config()
    config["selection"]["max_candidates_per_ingredient_combination"] = 2

    def _pair_row(candidate_id: str, score: float) -> dict[str, float | str]:
        row = {name: 0.0 for name in registry.feature_names}
        row.update(
            {
                "candidate_id": candidate_id,
                "ectoin_M": 0.10,
                "ethylene_glycol_M": 0.20,
                "screening_phase_score": score,
                "recommendation_type": "screening_candidate",
                "selection_explanation": "",
            }
        )
        return row

    def _other_row(candidate_id: str, feature_name: str, score: float) -> dict[str, float | str]:
        row = {name: 0.0 for name in registry.feature_names}
        row.update(
            {
                "candidate_id": candidate_id,
                feature_name: 0.05,
                "screening_phase_score": score,
                "recommendation_type": "screening_candidate",
                "selection_explanation": "",
            }
        )
        return row

    candidate_pool = pd.DataFrame(
        [
            _pair_row("pair_a", 99.0),
            _pair_row("pair_b", 98.0),
            _pair_row("pair_c", 97.0),
            _pair_row("pair_d", 96.0),
            _other_row("other_a", "betaine_M", 50.0),
            _other_row("other_b", "glucose_M", 40.0),
        ]
    )
    # All 4 ectoin+ethylene_glycol candidates selected -- well over the cap
    # of 2 -- plus nothing else competing for those slots in the pool.
    selected = candidate_pool.iloc[:4].copy()

    adjusted = _enforce_ingredient_combination_cap(
        selected,
        candidate_pool,
        registry,
        config,
        score_column="screening_phase_score",
    )

    combo_counts = pd.Series(
        [
            frozenset(
                feature
                for feature in ["ectoin_M", "ethylene_glycol_M"]
                if float(row.get(feature, 0.0)) > 0
            )
            for _, row in adjusted.iterrows()
        ]
    ).map(lambda combo: combo == frozenset({"ectoin_M", "ethylene_glycol_M"}))
    assert len(adjusted) == 4
    assert combo_counts.sum() == 2
    selected_ids = set(adjusted["candidate_id"])
    assert {"other_a", "other_b"}.issubset(selected_ids)
    # The two highest-scoring pair candidates are kept; the two
    # lowest-scoring ones are swapped out.
    assert {"pair_a", "pair_b"}.issubset(selected_ids)
    assert not {"pair_c", "pair_d"} & selected_ids


def test_ingredient_combination_cap_limits_trio_to_one_by_default() -> None:
    """Exact 3-ingredient (and larger) combinations get a much tighter cap
    than pairs: at most 1 candidate per round may carry any given exact
    trio/four-a-kind/etc., even though the per-pair cap (3 by default)
    would otherwise let several through.
    """
    registry = load_registry()
    config = load_optimization_config()
    # Leave max_candidates_per_ingredient_combination (pair cap) and
    # max_candidates_per_larger_ingredient_combination (trio+ cap) at their
    # defaults: 3 and 1 respectively.

    def _trio_row(candidate_id: str, score: float) -> dict[str, float | str]:
        row = {name: 0.0 for name in registry.feature_names}
        row.update(
            {
                "candidate_id": candidate_id,
                "ectoin_M": 0.10,
                "ethylene_glycol_M": 0.20,
                "fbs_pct": 5.0,
                "screening_phase_score": score,
                "recommendation_type": "screening_candidate",
                "selection_explanation": "",
            }
        )
        return row

    def _other_row(candidate_id: str, feature_name: str, score: float) -> dict[str, float | str]:
        row = {name: 0.0 for name in registry.feature_names}
        row.update(
            {
                "candidate_id": candidate_id,
                feature_name: 0.05,
                "screening_phase_score": score,
                "recommendation_type": "screening_candidate",
                "selection_explanation": "",
            }
        )
        return row

    candidate_pool = pd.DataFrame(
        [
            _trio_row("trio_a", 99.0),
            _trio_row("trio_b", 98.0),
            _trio_row("trio_c", 97.0),
            _other_row("other_a", "betaine_M", 50.0),
            _other_row("other_b", "glucose_M", 40.0),
        ]
    )
    # All 3 exact-trio candidates selected -- over the default trio cap of 1.
    selected = candidate_pool.iloc[:3].copy()

    adjusted = _enforce_ingredient_combination_cap(
        selected,
        candidate_pool,
        registry,
        config,
        score_column="screening_phase_score",
    )

    trio = frozenset({"ectoin_M", "ethylene_glycol_M", "fbs_pct"})
    trio_count = sum(
        1
        for _, row in adjusted.iterrows()
        if frozenset(
            feature
            for feature in ["ectoin_M", "ethylene_glycol_M", "fbs_pct"]
            if float(row.get(feature, 0.0)) > 0
        )
        == trio
    )
    assert len(adjusted) == 3
    assert trio_count == 1
    # The single highest-scoring trio candidate survives; the pool's other
    # combinations backfill the freed slots.
    selected_ids = set(adjusted["candidate_id"])
    assert "trio_a" in selected_ids
    assert {"other_a", "other_b"}.issubset(selected_ids)


def test_ingredient_combination_cap_ignores_single_ingredient_candidates() -> None:
    """Combinations of size 0-1 are out of scope for this cap -- a single
    active ingredient repeated across candidates is governed separately by
    _enforce_single_ingredient_spacing, not this combination-size-2+ cap.
    """
    registry = load_registry()
    config = load_optimization_config()
    config["selection"]["max_candidates_per_ingredient_combination"] = 1

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
            _row("single_a", "trehalose_M", 0.10, 99.0),
            _row("single_b", "trehalose_M", 0.11, 98.0),
        ]
    )
    selected = candidate_pool.copy()

    adjusted = _enforce_ingredient_combination_cap(
        selected,
        candidate_pool,
        registry,
        config,
        score_column="screening_phase_score",
    )

    # Both single-ingredient candidates are untouched by this cap.
    assert set(adjusted["candidate_id"]) == {"single_a", "single_b"}


def test_ingredient_combination_cap_leaves_overcap_in_place_when_pool_has_no_replacement() -> None:
    """If the pool has no eligible replacement (every other candidate is
    also at/over its own combination's cap, or shares the same over-cap
    combination), the over-cap combination is left in place rather than
    shrinking the slate below its target size.
    """
    registry = load_registry()
    config = load_optimization_config()
    config["selection"]["max_candidates_per_ingredient_combination"] = 1

    def _pair_row(candidate_id: str, score: float) -> dict[str, float | str]:
        row = {name: 0.0 for name in registry.feature_names}
        row.update(
            {
                "candidate_id": candidate_id,
                "ectoin_M": 0.10,
                "ethylene_glycol_M": 0.20,
                "screening_phase_score": score,
                "recommendation_type": "screening_candidate",
                "selection_explanation": "",
            }
        )
        return row

    # The pool contains only this one combination -- no eligible swap target.
    candidate_pool = pd.DataFrame(
        [
            _pair_row("pair_a", 99.0),
            _pair_row("pair_b", 98.0),
        ]
    )
    selected = candidate_pool.copy()

    adjusted = _enforce_ingredient_combination_cap(
        selected,
        candidate_pool,
        registry,
        config,
        score_column="screening_phase_score",
    )

    assert len(adjusted) == 2
    assert set(adjusted["candidate_id"]) == {"pair_a", "pair_b"}


def test_screening_slate_applies_ingredient_combination_cap_end_to_end() -> None:
    """End-to-end through _select_round_slate: even after origin-quota
    selection picks a healthy local/sparse/boundary mix, if too many of
    those picks happen to share the same active-ingredient combination
    (the ectoin+ethylene_glycol clustering the cap exists to prevent), the
    combination cap must still trim it down by swapping in pool
    alternatives, without shrinking the slate below n.
    """
    registry = load_registry()
    config = load_optimization_config()
    config["retest"]["max_candidates_per_round"] = 0
    config["candidate_generation"]["rescue_candidates_per_round"] = 0
    config["selection"]["max_candidates_per_ingredient_combination"] = 2

    rows = []
    # All local_perturbation candidates share the ectoin+ethylene_glycol
    # combination and dominate by score -- exactly the failure mode
    # reported against the live round-2 slate.
    for index in range(6):
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": f"local_{index}",
                "formulation_id": f"local_{index}",
                "candidate_origin": "local_perturbation",
                "recommendation_type": "screening_candidate",
                "selection_explanation": "",
                "ectoin_M": 0.05 + 0.001 * index,
                "ethylene_glycol_M": 0.10 + 0.001 * index,
                "screening_phase_score": 100.0 - index,
            }
        )
        rows.append(row)
    for index in range(6):
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": f"sparse_{index}",
                "formulation_id": f"sparse_{index}",
                "candidate_origin": "sparse_exploration",
                "recommendation_type": "screening_candidate",
                "selection_explanation": "",
                "betaine_M": 0.02 * (index + 1),
                "screening_phase_score": 50.0 - index,
            }
        )
        rows.append(row)
    for index in range(6):
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": f"boundary_{index}",
                "formulation_id": f"boundary_{index}",
                "candidate_origin": "boundary_probe",
                "recommendation_type": "screening_candidate",
                "selection_explanation": "",
                "glucose_M": 0.02 * (index + 1),
                "screening_phase_score": 10.0 - index,
            }
        )
        rows.append(row)

    selected = _select_round_slate(
        pd.DataFrame(rows),
        registry,
        config,
        _phase(PHASE_SCREENING),
        n=8,
        policy_active=True,
    )

    ectoin_eg_combo_count = sum(
        1
        for _, row in selected.iterrows()
        if float(row.get("ectoin_M", 0.0)) > 0 and float(row.get("ethylene_glycol_M", 0.0)) > 0
    )
    assert len(selected) == 8
    assert ectoin_eg_combo_count <= 2


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


def test_screening_slate_ignores_intact_probability_and_selects_on_viability_score() -> None:
    """Screening selects purely on screening_phase_score (viability-driven).

    A candidate with the best score but the lowest predicted intact
    probability must still be selected during screening: intact-formation
    gating during screening is removed (it now lives only in rescue
    candidate generation and mechanics-phase scoring).
    """
    registry = load_registry()
    config = load_optimization_config()
    rows = []
    for candidate_id, betaine, score, intact in [
        ("high_score_low_intact", 0.10, 1.00, 0.10),
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

    assert set(selected["candidate_id"]) == {"high_score_low_intact", "pass_a"}


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


def test_origin_quota_splits_local_explore_probe_with_per_category_cap() -> None:
    """_allocate_screening_origin_quota must reproduce the round-policy spec:
    a fixed number of local_perturbation slots, with the rest split between
    sparse_exploration and boundary_probe capped per category.

    This guards against pure top-score selection collapsing the whole slate
    onto one origin (the bug this function exists to fix): a tight
    high-scoring local_perturbation cluster must not crowd out
    sparse_exploration/boundary_probe just because its raw score is higher.
    """
    registry = load_registry()
    rows = []
    # 5 local_perturbation candidates, all higher score than every other
    # origin -- pure top-score selection would pick only these.
    for index in range(5):
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": f"local_{index}",
                "candidate_origin": "local_perturbation",
                "betaine_M": 0.05 * (index + 1),
            }
        )
        rows.append(row)
    # 4 sparse_exploration candidates, lower score than local but higher
    # than boundary.
    for index in range(4):
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": f"sparse_{index}",
                "candidate_origin": "sparse_exploration",
                "ectoin_M": 0.05 * (index + 1),
            }
        )
        rows.append(row)
    # 4 boundary_probe candidates, lowest score.
    for index in range(4):
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": f"boundary_{index}",
                "candidate_origin": "boundary_probe",
                "glucose_M": 0.05 * (index + 1),
            }
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    score = np.array(
        [100.0 - index for index in range(5)]
        + [50.0 - index for index in range(4)]
        + [10.0 - index for index in range(4)]
    )

    selected_indices = _allocate_screening_origin_quota(
        frame,
        score,
        registry,
        n=8,
        local_quota=3,
        explore_probe_quota=5,
        explore_probe_per_category_cap=3,
        diversity_weight=0.05,
        competitive_utility_band=None,
    )

    selected_origins = frame.iloc[selected_indices]["candidate_origin"].value_counts()
    assert len(selected_indices) == 8
    assert selected_origins.get("local_perturbation", 0) == 3
    assert selected_origins.get("sparse_exploration", 0) + selected_origins.get("boundary_probe", 0) == 5
    # Per-category cap: neither sparse nor boundary may take all 5 explore/probe slots.
    assert selected_origins.get("sparse_exploration", 0) <= 3
    assert selected_origins.get("boundary_probe", 0) <= 3


def test_origin_quota_backfills_thin_categories_from_fallback() -> None:
    """If a category (e.g. boundary_probe) has fewer candidates than its
    share of the quota, the shortfall must be backfilled rather than
    shrinking the slate below n.
    """
    registry = load_registry()
    rows = []
    for index in range(3):
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": f"local_{index}",
                "candidate_origin": "local_perturbation",
                "betaine_M": 0.05 * (index + 1),
            }
        )
        rows.append(row)
    for index in range(6):
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": f"sparse_{index}",
                "candidate_origin": "sparse_exploration",
                "ectoin_M": 0.05 * (index + 1),
            }
        )
        rows.append(row)
    # Only 1 boundary_probe candidate available, fewer than its quota share.
    row = {feature: 0.0 for feature in registry.feature_names}
    row.update(
        {
            "candidate_id": "boundary_0",
            "candidate_origin": "boundary_probe",
            "glucose_M": 0.05,
        }
    )
    rows.append(row)
    frame = pd.DataFrame(rows)
    score = np.array([100.0, 99.0, 98.0] + [50.0 - index for index in range(6)] + [10.0])

    selected_indices = _allocate_screening_origin_quota(
        frame,
        score,
        registry,
        n=8,
        local_quota=3,
        explore_probe_quota=5,
        explore_probe_per_category_cap=3,
        diversity_weight=0.05,
        competitive_utility_band=None,
    )

    assert len(selected_indices) == 8


def test_screening_slate_applies_origin_quota_with_backfilled_local() -> None:
    """End-to-end through _select_round_slate: when the retest/rescue
    reserve is only partially used, the unused reserve slots become extra
    local_perturbation budget (per round_policy spec), and the remaining
    slots split sparse_exploration/boundary_probe with neither origin
    exceeding the per-category cap, even though local_perturbation has the
    best raw scores across the whole pool.
    """
    registry = load_registry()
    config = load_optimization_config()
    config["retest"]["max_candidates_per_round"] = 2
    config["candidate_generation"]["rescue_candidates_per_round"] = 2
    rows = []
    # 1 rescue candidate uses 1 of the 4 reserved slots; no retest candidates
    # exist this round, so 3 of the 4 reserved slots are unused and must be
    # backfilled with local_perturbation.
    row = {feature: 0.0 for feature in registry.feature_names}
    row.update(
        {
            "candidate_id": "rescue_0",
            "formulation_id": "rescue_0",
            "candidate_origin": "rescue_dilution",
            "recommendation_type": "rescue_candidate",
            "selection_explanation": "",
            "rescue_scale_factor": 0.25,
            "viability_ucb": 80.0,
            "screening_phase_score": -1.0,
        }
    )
    rows.append(row)
    for index in range(10):
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": f"local_{index}",
                "formulation_id": f"local_{index}",
                "candidate_origin": "local_perturbation",
                "recommendation_type": "screening_candidate",
                "selection_explanation": "",
                "betaine_M": 0.02 * (index + 1),
                "screening_phase_score": 100.0 - index,
            }
        )
        rows.append(row)
    for index in range(5):
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": f"sparse_{index}",
                "formulation_id": f"sparse_{index}",
                "candidate_origin": "sparse_exploration",
                "recommendation_type": "screening_candidate",
                "selection_explanation": "",
                "ectoin_M": 0.02 * (index + 1),
                "screening_phase_score": 50.0 - index,
            }
        )
        rows.append(row)
    for index in range(5):
        row = {feature: 0.0 for feature in registry.feature_names}
        row.update(
            {
                "candidate_id": f"boundary_{index}",
                "formulation_id": f"boundary_{index}",
                "candidate_origin": "boundary_probe",
                "recommendation_type": "screening_candidate",
                "selection_explanation": "",
                "glucose_M": 0.02 * (index + 1),
                "screening_phase_score": 10.0 - index,
            }
        )
        rows.append(row)

    selected = _select_round_slate(
        pd.DataFrame(rows),
        registry,
        config,
        _phase(PHASE_SCREENING),
        n=12,
        policy_active=True,
    )

    origin_counts = selected["candidate_origin"].value_counts()
    assert len(selected) == 12
    assert origin_counts.get("rescue_dilution", 0) == 1
    # 3 base local quota + 3 backfilled from the unused retest/rescue reserve.
    assert origin_counts.get("local_perturbation", 0) == 6
    assert origin_counts.get("sparse_exploration", 0) + origin_counts.get("boundary_probe", 0) == 5
    assert origin_counts.get("sparse_exploration", 0) <= 3
    assert origin_counts.get("boundary_probe", 0) <= 3


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


def test_policy_active_retests_are_not_excluded_by_intact_prediction() -> None:
    """A formulation flagged for retest due to viability disagreement must
    stay in the retest slate even if it failed intact-patch formation.

    Retest priority exists to resolve uncertain/disagreeing viability data;
    gating that by an unrelated intact-formation prediction could silently
    drop exactly the candidates most in need of re-testing. (Intact failure
    feeds rescue-candidate generation separately, not retest exclusion.)
    """
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

    assert "retest_priority" in set(result.viability_screen["recommendation_type"].astype(str))

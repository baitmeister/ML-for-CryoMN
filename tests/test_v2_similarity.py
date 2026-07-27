from __future__ import annotations

import pandas as pd
import pytest

from helper.config import load_optimization_config
from helper.registry import load_registry
from helper.similarity import (
    SimilarityAudit,
    SimilarityIndex,
    build_history_similarity_index,
    filter_frame_by_similarity,
    resolve_similarity_policy,
    similarity_priority_order,
    validate_selected_similarity,
)


def _row(registry, candidate_id: str, **values) -> dict:
    row = {feature: 0.0 for feature in registry.feature_names}
    row.update(
        {
            "candidate_id": candidate_id,
            "formulation_id": f"form_{candidate_id}",
            "candidate_origin": "sparse_exploration",
            "recommendation_type": "screening_candidate",
        }
    )
    row.update(values)
    return row


def _active_policy():
    return resolve_similarity_policy(load_optimization_config(), 3)


def test_policy_activates_from_round_three_only() -> None:
    config = load_optimization_config()
    assert resolve_similarity_policy(config, 2).active is False
    policy = resolve_similarity_policy(config, 3)
    assert policy.active is True
    assert policy.distance_threshold == pytest.approx(0.05)
    assert policy.single_ingredient_min_relative_difference == pytest.approx(0.50)


def test_bounds_normalized_distance_is_inclusive_at_threshold() -> None:
    registry = load_registry()
    index = SimilarityIndex(registry, _active_policy())
    reference = _row(
        registry,
        "reference",
        ectoin_M=0.10,
        trehalose_M=0.10,
    )
    index.add(reference, "reference", "history")

    # Ectoin's range is 0.5 M, so a 0.025 M change is exactly 0.05
    # in bounds-normalized formulation space.
    at_threshold = _row(
        registry,
        "at_threshold",
        ectoin_M=0.125,
        trehalose_M=0.10,
    )
    beyond_threshold = _row(
        registry,
        "beyond_threshold",
        ectoin_M=0.126,
        trehalose_M=0.10,
    )

    match = index.find_conflict(at_threshold)
    assert match is not None
    assert match.reason == "normalized_distance"
    assert match.normalized_distance == pytest.approx(0.05)
    assert index.find_conflict(beyond_threshold) is None


def test_practical_presence_threshold_zeroes_trace_ingredients() -> None:
    registry = load_registry()
    index = SimilarityIndex(registry, _active_policy())
    reference = _row(registry, "reference", ectoin_M=0.10)
    index.add(reference, "reference", "history")
    trace_added = _row(
        registry,
        "trace_added",
        ectoin_M=0.10,
        trehalose_M=0.0009,
    )
    match = index.find_conflict(trace_added)
    assert match is not None
    assert match.normalized_distance == pytest.approx(0.0)


def test_same_single_ingredient_must_pass_vector_and_relative_rules() -> None:
    registry = load_registry()
    index = SimilarityIndex(registry, _active_policy())
    index.add(
        _row(registry, "reference", ectoin_M=0.10),
        "reference",
        "history",
    )

    relative_too_close = _row(registry, "relative_too_close", ectoin_M=0.13)
    relative_match = index.find_conflict(relative_too_close)
    assert relative_match is not None
    assert relative_match.normalized_distance > 0.05
    assert relative_match.reason == "single_ingredient_relative_spacing"
    assert relative_match.single_ingredient_relative_difference == pytest.approx(0.30)

    # A 50% change passes the strict relative boundary and is also more than
    # 0.05 apart in normalized space.
    assert index.find_conflict(
        _row(registry, "passes_both", ectoin_M=0.15)
    ) is None

    low_concentration = SimilarityIndex(registry, _active_policy())
    low_concentration.add(
        _row(registry, "low_reference", ectoin_M=0.010),
        "low_reference",
        "history",
    )
    vector_too_close = low_concentration.find_conflict(
        _row(registry, "vector_too_close", ectoin_M=0.015)
    )
    assert vector_too_close is not None
    assert vector_too_close.reason == "normalized_distance"
    assert vector_too_close.single_ingredient_relative_difference == pytest.approx(0.50)


def test_different_single_and_single_vs_multi_use_general_vector_rule() -> None:
    registry = load_registry()
    index = SimilarityIndex(registry, _active_policy())
    index.add(
        _row(registry, "ectoin_reference", ectoin_M=0.10),
        "ectoin_reference",
        "history",
    )
    assert index.find_conflict(
        _row(registry, "trehalose_single", trehalose_M=0.10)
    ) is None
    assert index.find_conflict(
        _row(
            registry,
            "ectoin_plus_trehalose",
            ectoin_M=0.10,
            trehalose_M=0.10,
        )
    ) is None


def test_history_references_include_wetlab_and_exclude_literature() -> None:
    registry = load_registry()
    policy = _active_policy()
    formulations = pd.DataFrame(
        [
            _row(registry, "literature", ectoin_M=0.10)
            | {"formulation_id": "literature"},
            _row(registry, "wetlab", trehalose_M=0.20)
            | {"formulation_id": "wetlab"},
        ]
    )
    observations = pd.DataFrame(
        [
            {
                "formulation_id": "literature",
                "source_type": "legacy_literature",
            },
            {
                "formulation_id": "wetlab",
                "source_type": "wetlab_feedback",
            },
        ]
    )
    index = build_history_similarity_index(
        formulations,
        observations,
        registry,
        policy,
    )
    assert len(index) == 1
    assert index.find_conflict(
        _row(registry, "wetlab_repeat", trehalose_M=0.20)
    ) is not None
    assert index.find_conflict(
        _row(registry, "literature_repeat", ectoin_M=0.10)
    ) is None


def test_frame_filter_rejects_history_and_pool_conflicts_but_exempts_retests() -> None:
    registry = load_registry()
    policy = _active_policy()
    index = SimilarityIndex(registry, policy)
    index.add(
        _row(
            registry,
            "history",
            ectoin_M=0.10,
            trehalose_M=0.10,
        ),
        "history",
        "history",
    )
    audit = SimilarityAudit(policy, history_reference_count=1)
    frame = pd.DataFrame(
        [
            _row(
                registry,
                "retest",
                ectoin_M=0.10,
                trehalose_M=0.10,
                candidate_origin="retest",
                recommendation_type="retest_priority",
            ),
            _row(
                registry,
                "history_repeat",
                ectoin_M=0.10,
                trehalose_M=0.10,
            ),
            _row(
                registry,
                "accepted",
                ectoin_M=0.30,
                trehalose_M=0.30,
            ),
            _row(
                registry,
                "pool_neighbour",
                ectoin_M=0.31,
                trehalose_M=0.30,
            ),
        ]
    )

    accepted, rejected = filter_frame_by_similarity(frame, index, audit)
    assert set(accepted["candidate_id"]) == {"retest", "accepted"}
    assert set(rejected["candidate_id"]) == {"history_repeat", "pool_neighbour"}
    assert audit.rejections_by_reference_kind == {
        "history": 1,
        "generated_pool": 1,
    }
    assert audit.rejections_by_origin_and_reference_kind == {
        "sparse_exploration": {
            "history": 1,
            "generated_pool": 1,
        }
    }


def test_mechanics_similarity_priority_is_rescue_then_continuous_then_finite() -> None:
    frame = pd.DataFrame(
        [
            {"candidate_id": "finite_a", "candidate_origin": "finite_pool_fallback"},
            {"candidate_id": "continuous", "candidate_origin": "continuous_qlognehvi"},
            {"candidate_id": "rescue", "candidate_origin": "rescue_dilution"},
            {"candidate_id": "finite_b", "candidate_origin": "boundary_probe"},
        ]
    )
    ordered = similarity_priority_order(frame)
    assert ordered["candidate_id"].tolist() == [
        "rescue",
        "continuous",
        "finite_a",
        "finite_b",
    ]


def test_final_validation_exempts_retest_and_rejects_regular_history_repeat() -> None:
    registry = load_registry()
    policy = _active_policy()
    history = _row(registry, "history", ectoin_M=0.20)
    history["formulation_id"] = "history"
    formulations = pd.DataFrame([history])
    observations = pd.DataFrame(
        [{"formulation_id": "history", "source_type": "wetlab_feedback"}]
    )
    retest = pd.DataFrame(
        [
            history
            | {
                "candidate_id": "retest_history",
                "candidate_origin": "retest",
                "recommendation_type": "retest_priority",
            }
        ]
    )
    summary = validate_selected_similarity(
        retest,
        formulations,
        observations,
        registry,
        policy,
    )
    assert summary["selected_non_retest_count"] == 0

    regular = retest.copy()
    regular["candidate_id"] = "regular_repeat"
    regular["candidate_origin"] = "sparse_exploration"
    regular["recommendation_type"] = "screening_candidate"
    with pytest.raises(ValueError, match="violates formulation similarity"):
        validate_selected_similarity(
            regular,
            formulations,
            observations,
            registry,
            policy,
        )

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from helper.candidates import generate_support_aware_candidate_pool
from helper.config import load_optimization_config
from helper.feasibility import (
    annotate_feasibility,
    build_support_context,
    policy_activation,
)
from helper.registry import load_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_policy_activates_only_from_round_two() -> None:
    config = load_optimization_config()
    assert policy_activation(config, 1)[0] is False
    assert policy_activation(config, 2)[0] is True


def test_round_one_failed_candidates_are_rejected_by_round_two_policy() -> None:
    registry = load_registry()
    config = load_optimization_config()
    candidates = pd.read_csv(
        PROJECT_ROOT
        / "results"
        / "multi_objective_v2"
        / "next_round"
        / "next_round_candidates.csv"
    )
    failed = candidates[candidates["selection_rank"].isin([2, 5, 6, 7, 9, 11])]
    annotated = annotate_feasibility(
        failed,
        registry,
        config,
        policy_active=True,
    )
    assert (~annotated["feasibility_pass"]).all()
    assert annotated["feasibility_reasons"].astype(str).str.len().gt(0).all()


def test_campaign_caps_and_combined_load_rules() -> None:
    registry = load_registry()
    config = load_optimization_config()
    base = {feature: 0.0 for feature in registry.feature_names}
    rows = [
        base | {"candidate_id": "pvp_high", "pvp_pct": 10.1},
        base | {"candidate_id": "polymer_pair", "pvp_pct": 5.0, "dextran_pct": 2.0},
        base | {"candidate_id": "protein_high", "fbs_pct": 6.0, "hsa_pct": 5.0},
        base | {"candidate_id": "sugar_high", "pvp_pct": 5.0, "trehalose_M": 0.3, "sucrose_M": 0.3},
        base | {"candidate_id": "valid", "pvp_pct": 5.0, "trehalose_M": 0.2, "fbs_pct": 5.0},
    ]
    annotated = annotate_feasibility(
        pd.DataFrame(rows),
        registry,
        config,
        policy_active=True,
    ).set_index("candidate_id")
    assert not bool(annotated.loc["pvp_high", "feasibility_pass"])
    assert not bool(annotated.loc["polymer_pair", "feasibility_pass"])
    assert not bool(annotated.loc["protein_high", "feasibility_pass"])
    assert not bool(annotated.loc["sugar_high", "feasibility_pass"])
    assert bool(annotated.loc["valid", "feasibility_pass"])


def test_support_aware_pool_caps_local_fraction_and_redistributes_shortfall() -> None:
    registry = load_registry()
    config = load_optimization_config()
    formulations = pd.read_csv(PROJECT_ROOT / "data" / "processed_v2" / "formulations.csv")
    support = build_support_context(formulations, registry, config)
    pool = generate_support_aware_candidate_pool(
        registry,
        formulations,
        config,
        support,
        n_candidates=100,
        random_seed=7,
        unavailable_feature_names=[],
    )
    accepted = pool[pool["feasibility_pass"].astype(bool)]
    origins = accepted["candidate_origin"].value_counts()
    assert len(accepted) == 100
    assert int(origins.get("local_perturbation", 0)) == 40
    assert int(origins.get("sparse_exploration", 0)) == 35
    assert int(origins.get("boundary_probe", 0)) == 25
    assert set(accepted["candidate_origin"]).issubset(
        {"local_perturbation", "sparse_exploration", "boundary_probe"}
    )


def test_rejected_generation_attempts_remain_in_audit_pool() -> None:
    registry = load_registry()
    config = load_optimization_config()
    formulations = pd.read_csv(PROJECT_ROOT / "data" / "processed_v2" / "formulations.csv")
    support = build_support_context(formulations, registry, config)
    pool = generate_support_aware_candidate_pool(
        registry,
        formulations,
        config,
        support,
        n_candidates=80,
        random_seed=42,
        unavailable_feature_names=[],
    )
    rejected = pool[~pool["feasibility_pass"].astype(bool)]
    assert not rejected.empty
    assert rejected["feasibility_reasons"].astype(str).str.len().gt(0).all()


def test_generation_fractions_must_sum_to_one() -> None:
    registry = load_registry()
    config = load_optimization_config()
    config["candidate_generation"]["boundary_fraction"] = 0.20
    formulations = pd.read_csv(PROJECT_ROOT / "data" / "processed_v2" / "formulations.csv")
    support = build_support_context(formulations, registry, config)

    with pytest.raises(ValueError, match="must sum to 1.0"):
        generate_support_aware_candidate_pool(
            registry,
            formulations,
            config,
            support,
            n_candidates=10,
            random_seed=42,
        )


def test_round_one_rerun_preserves_legacy_artifacts_and_transferred_tables(
    tmp_path: Path,
) -> None:
    formulations_path = PROJECT_ROOT / "data" / "processed_v2" / "formulations.csv"
    observations_path = PROJECT_ROOT / "data" / "processed_v2" / "observations.csv"
    formulations_before = formulations_path.read_bytes()
    observations_before = observations_path.read_bytes()
    output_dir = tmp_path / "next_round"
    total_pool_path = tmp_path / "total_candidate_pool.csv"

    subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "src"
                / "08_multi_objective"
                / "02_select_candidates"
                / "select_candidates.py"
            ),
            "--batch-id",
            "ROUND_001",
            "--output-dir",
            str(output_dir),
            "--total-candidate-pool",
            str(total_pool_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    expected_candidates = pd.read_csv(
        PROJECT_ROOT
        / "results"
        / "multi_objective_v2"
        / "next_round"
        / "next_round_candidates.csv"
    )
    actual_candidates = pd.read_csv(output_dir / "next_round_candidates.csv")
    expected_pool = pd.read_csv(
        PROJECT_ROOT / "results" / "multi_objective_v2" / "total_candidate_pool.csv"
    )
    actual_pool = pd.read_csv(total_pool_path)

    pd.testing.assert_frame_equal(
        actual_candidates,
        expected_candidates,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(
        actual_pool,
        expected_pool,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert (output_dir / "next_round_summary.txt").read_bytes() == (
        PROJECT_ROOT
        / "results"
        / "multi_objective_v2"
        / "next_round"
        / "next_round_summary.txt"
    ).read_bytes()
    assert formulations_path.read_bytes() == formulations_before
    assert observations_path.read_bytes() == observations_before

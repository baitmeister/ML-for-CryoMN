from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from helper.candidates import generate_support_aware_candidate_pool
from helper.candidates import generate_rescue_candidate_pool
from helper.config import load_optimization_config
from helper.feasibility import (
    annotate_feasibility,
    build_support_context,
    policy_activation,
)
from helper.registry import load_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WETLAB_ENTRY_COLUMNS = {
    "replicate_id",
    "viability_percent",
    "intact_patch_formation_pass",
    "no_slurry",
    "no_collapse",
    "intact_tip_count",
    "total_tip_count",
    "instron_file",
    "needles_compressed",
    "critical_axial_load_N_per_needle",
    "critical_axial_load_N_total",
    "initial_stiffness_N_per_mm_per_needle",
    "notes",
}


def _generated_candidate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[column for column in WETLAB_ENTRY_COLUMNS if column in frame.columns])


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
    observations = pd.read_csv(PROJECT_ROOT / "data" / "processed_v2" / "observations.csv")
    support = build_support_context(formulations, registry, config, observations)
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
    observations = pd.read_csv(PROJECT_ROOT / "data" / "processed_v2" / "observations.csv")
    support = build_support_context(formulations, registry, config, observations)
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
    observations = pd.read_csv(PROJECT_ROOT / "data" / "processed_v2" / "observations.csv")
    support = build_support_context(formulations, registry, config, observations)

    with pytest.raises(ValueError, match="must sum to 1.0"):
        generate_support_aware_candidate_pool(
            registry,
            formulations,
            config,
            support,
            n_candidates=10,
            random_seed=42,
        )


def test_support_context_ignores_unobserved_formulations() -> None:
    registry = load_registry()
    config = load_optimization_config()
    formulations = pd.read_csv(PROJECT_ROOT / "data" / "processed_v2" / "formulations.csv")
    observations = pd.read_csv(PROJECT_ROOT / "data" / "processed_v2" / "observations.csv")
    support_before = build_support_context(formulations, registry, config, observations)

    extreme = {feature: 0.0 for feature in registry.feature_names}
    extreme.update(
        {
            "formulation_id": "v2_unobserved_extreme",
            "source": "test_candidate_only",
            "source_row_id": "cand_extreme",
            "formulation_label": "candidate-only extreme",
            "ethylene_glycol_M": 2.0,
            "hsa_pct": 10.0,
            "active_ingredient_count": 2,
        }
    )
    augmented_formulations = pd.concat([formulations, pd.DataFrame([extreme])], ignore_index=True)

    support_after = build_support_context(
        augmented_formulations,
        registry,
        config,
        observations,
    )

    assert support_after.radius == pytest.approx(support_before.radius)


def test_observed_round_formulations_expand_support_regardless_of_outcome() -> None:
    registry = load_registry()
    config = load_optimization_config()
    formulations = pd.read_csv(PROJECT_ROOT / "data" / "processed_v2" / "formulations.csv")
    observations = pd.read_csv(PROJECT_ROOT / "data" / "processed_v2" / "observations.csv")
    support_before = build_support_context(formulations, registry, config, observations)

    failed = {feature: 0.0 for feature in registry.feature_names}
    failed.update(
        {
            "formulation_id": "v2_failed_round_support_probe",
            "source": "wetlab_feedback:ROUND_001",
            "source_row_id": "cand_failed_probe",
            "formulation_label": "failed round probe",
            "ethylene_glycol_M": 2.0,
            "hsa_pct": 10.0,
            "active_ingredient_count": 2,
        }
    )
    failed_observations = pd.DataFrame(
        [
            {
                "observation_id": "obs_failed_probe_viability",
                "formulation_id": "v2_failed_round_support_probe",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 80.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test",
                "notes": "",
            },
            {
                "observation_id": "obs_failed_probe_intact",
                "formulation_id": "v2_failed_round_support_probe",
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

    support_after = build_support_context(
        pd.concat([formulations, pd.DataFrame([failed])], ignore_index=True),
        registry,
        config,
        pd.concat([observations, failed_observations], ignore_index=True),
    )

    assert len(support_after.observed_scaled) == len(support_before.observed_scaled) + 1


def test_boundary_style_quota_does_not_require_out_of_support_status() -> None:
    registry = load_registry()
    config = load_optimization_config()
    base = {feature: 0.0 for feature in registry.feature_names}
    base.update(
        {
            "formulation_id": "v2_single_observed_support",
            "source": "wetlab_feedback:ROUND_001",
            "source_row_id": "single_support",
            "formulation_label": "single support",
            "betaine_M": 0.25,
            "active_ingredient_count": 1,
        }
    )
    formulations = pd.DataFrame([base])
    observations = pd.DataFrame(
        [
            {
                "observation_id": "obs_single_support_viability",
                "formulation_id": "v2_single_observed_support",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 50.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test",
                "notes": "",
            }
        ]
    )
    support = build_support_context(formulations, registry, config, observations)

    pool = generate_support_aware_candidate_pool(
        registry,
        formulations,
        config,
        support,
        n_candidates=20,
        random_seed=42,
        unavailable_feature_names=[],
    )
    accepted = pool[pool["feasibility_pass"].astype(bool)]

    assert len(accepted) == 20
    assert int(accepted["candidate_origin"].value_counts().get("boundary_probe", 0)) == 5
    assert set(accepted["support_status"]) == {"in_support"}


def test_high_viability_failed_patch_generates_dilution_rescue_candidates() -> None:
    registry = load_registry()
    config = load_optimization_config()
    base = {feature: 0.0 for feature in registry.feature_names}
    base.update(
        {
            "formulation_id": "v2_high_viability_failed",
            "source": "wetlab_feedback:ROUND_001",
            "source_row_id": "failed_high",
            "formulation_label": "failed high viability",
            "ectoin_M": 0.40,
            "ethylene_glycol_M": 1.90,
            "hsa_pct": 9.0,
            "active_ingredient_count": 3,
        }
    )
    formulations = pd.DataFrame([base])
    observations = pd.DataFrame(
        [
            {
                "observation_id": "obs_failed_high_viability",
                "formulation_id": "v2_high_viability_failed",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 69.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test",
                "notes": "",
            },
            {
                "observation_id": "obs_failed_high_intact",
                "formulation_id": "v2_high_viability_failed",
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
    support = build_support_context(formulations, registry, config, observations)

    rescue = generate_rescue_candidate_pool(
        registry,
        formulations,
        observations,
        config,
        support,
        unavailable_feature_names=[],
    )

    assert not rescue.empty
    assert set(rescue["candidate_origin"]) == {"rescue_dilution"}
    assert rescue["ethylene_glycol_M"].max() < 1.90
    assert rescue["feasibility_pass"].all()


def test_round_one_rerun_is_deterministic_and_preserves_legacy_artifacts(
    tmp_path: Path,
) -> None:
    """select_candidates.py must be deterministic and side-effect-free on
    formulations/observations, run twice from identical inputs.

    This intentionally does NOT compare against the committed
    results/multi_objective_v2/next_round/next_round_candidates.csv: that
    file is round 1's real, already-completed wet-lab results (viability_percent,
    intact_patch_formation_pass, etc. filled in by hand) -- not a regenerable
    template. Round 1 is finished and is not retroactively changed by the
    screening-phase intact-gating fix (see helper/selection.py
    annotate_candidates); only round 2+ candidate generation is affected.
    Pinning a byte-for-byte comparison against round 1's historical slate
    would just re-encode the old (now intentionally removed) intact-gating
    behavior as a regression target.
    """
    formulations_path = PROJECT_ROOT / "data" / "processed_v2" / "formulations.csv"
    observations_path = PROJECT_ROOT / "data" / "processed_v2" / "observations.csv"
    formulations_before = formulations_path.read_bytes()
    observations_before = observations_path.read_bytes()

    def _run_select(output_dir: Path, total_pool_path: Path) -> None:
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
                "--seed",
                "42",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    output_dir_a = tmp_path / "run_a" / "next_round"
    total_pool_a = tmp_path / "run_a" / "total_candidate_pool.csv"
    _run_select(output_dir_a, total_pool_a)

    output_dir_b = tmp_path / "run_b" / "next_round"
    total_pool_b = tmp_path / "run_b" / "total_candidate_pool.csv"
    _run_select(output_dir_b, total_pool_b)

    candidates_a = pd.read_csv(output_dir_a / "next_round_candidates.csv")
    candidates_b = pd.read_csv(output_dir_b / "next_round_candidates.csv")
    pool_a = pd.read_csv(total_pool_a)
    pool_b = pd.read_csv(total_pool_b)

    pd.testing.assert_frame_equal(
        _generated_candidate_columns(candidates_a),
        _generated_candidate_columns(candidates_b),
        check_exact=False,
        check_dtype=False,
        rtol=1e-8,
        atol=1e-8,
    )
    pd.testing.assert_frame_equal(
        pool_a,
        pool_b,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert (output_dir_a / "next_round_summary.txt").read_bytes() == (
        output_dir_b / "next_round_summary.txt"
    ).read_bytes()

    # The currently screening-phase slate is now scored purely on predicted
    # viability (see annotate_candidates); intact-formation probability must
    # not influence which candidates are selected during screening.
    assert len(candidates_a) == 12
    assert candidates_a["recommendation_type"].eq("screening_candidate").all()

    assert formulations_path.read_bytes() == formulations_before
    assert observations_path.read_bytes() == observations_before

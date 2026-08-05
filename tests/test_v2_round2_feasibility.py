from __future__ import annotations

from pathlib import Path
import json
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
from helper.registry import load_registry, presence_threshold
from helper.similarity import (
    SimilarityAudit,
    build_history_similarity_index,
    resolve_similarity_policy,
)


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
        / "rounds"
        / "ROUND_001"
        / "completed"
        / "completed.csv"
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


def test_round_three_similarity_resampling_preserves_full_origin_targets() -> None:
    """The active gate rejects and resamples; it does not shrink the 2,000 rows."""
    registry = load_registry()
    config = load_optimization_config()
    formulations = pd.read_csv(PROJECT_ROOT / "data" / "processed_v2" / "formulations.csv")
    observations = pd.read_csv(PROJECT_ROOT / "data" / "processed_v2" / "observations.csv")
    support = build_support_context(formulations, registry, config, observations)
    policy = resolve_similarity_policy(config, 3)
    index = build_history_similarity_index(
        formulations,
        observations,
        registry,
        policy,
    )
    audit = SimilarityAudit(policy, history_reference_count=len(index))

    pool = generate_support_aware_candidate_pool(
        registry,
        formulations,
        config,
        support,
        n_candidates=2000,
        random_seed=42,
        unavailable_feature_names=[],
        similarity_index=index,
        similarity_audit=audit,
    )
    accepted = pool[pool["feasibility_pass"].astype(bool)]
    origins = accepted["candidate_origin"].value_counts()

    assert len(accepted) == 2000
    assert int(origins.get("local_perturbation", 0)) == 800
    assert int(origins.get("sparse_exploration", 0)) == 700
    assert int(origins.get("boundary_probe", 0)) == 500
    assert audit.rejection_count > 0
    assert set(audit.rejections_by_reference_kind).issubset(
        {"history", "generated_pool"}
    )


def test_round_three_similarity_resampling_is_deterministic_under_seed_42() -> None:
    registry = load_registry()
    config = load_optimization_config()
    formulations = pd.read_csv(PROJECT_ROOT / "data" / "processed_v2" / "formulations.csv")
    observations = pd.read_csv(PROJECT_ROOT / "data" / "processed_v2" / "observations.csv")
    support = build_support_context(formulations, registry, config, observations)
    policy = resolve_similarity_policy(config, 3)

    generated: list[pd.DataFrame] = []
    audits: list[dict] = []
    for _ in range(2):
        index = build_history_similarity_index(
            formulations,
            observations,
            registry,
            policy,
        )
        audit = SimilarityAudit(policy, history_reference_count=len(index))
        pool = generate_support_aware_candidate_pool(
            registry,
            formulations,
            config,
            support,
            n_candidates=100,
            random_seed=42,
            unavailable_feature_names=[],
            similarity_index=index,
            similarity_audit=audit,
        )
        generated.append(pool[pool["feasibility_pass"].astype(bool)].reset_index(drop=True))
        audits.append(audit.to_metadata())

    pd.testing.assert_frame_equal(generated[0], generated[1])
    assert audits[0] == audits[1]
    assert audits[0]["rejection_count"] > 0


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
            {
                "observation_id": "obs_failed_high_intact_second_replicate",
                "formulation_id": "v2_high_viability_failed",
                "batch_id": "ROUND_001",
                "replicate_id": "rep_002",
                "endpoint": "intact_patch_formation_pass",
                "value": 1.0,
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


def test_round_three_similarity_filters_rescue_candidates_without_exception() -> None:
    registry = load_registry()
    config = load_optimization_config()
    base = {feature: 0.0 for feature in registry.feature_names}
    base.update(
        {
            "formulation_id": "v2_single_ectoin_failed",
            "source": "wetlab_feedback:ROUND_002",
            "source_row_id": "single_failed",
            "formulation_label": "single ectoin failed",
            "ectoin_M": 0.10,
            "active_ingredient_count": 1,
        }
    )
    formulations = pd.DataFrame([base])
    observations = pd.DataFrame(
        [
            {
                "observation_id": "obs_single_failed_viability",
                "formulation_id": "v2_single_ectoin_failed",
                "batch_id": "ROUND_002",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 70.0,
                "unit": "percent",
                "observation_noise": 1.0,
                "source_type": "wetlab_feedback",
                "source_file": "test",
                "notes": "",
            },
            {
                "observation_id": "obs_single_failed_intact",
                "formulation_id": "v2_single_ectoin_failed",
                "batch_id": "ROUND_002",
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
    policy = resolve_similarity_policy(config, 3)
    index = build_history_similarity_index(
        formulations,
        observations,
        registry,
        policy,
    )
    audit = SimilarityAudit(policy, history_reference_count=len(index))

    rescue = generate_rescue_candidate_pool(
        registry,
        formulations,
        observations,
        config,
        support,
        unavailable_feature_names=[],
        similarity_index=index,
        similarity_audit=audit,
    )

    # The 0.75 dilution is exactly 0.05 from history. The 0.50 dilution is
    # exactly 0.05 from the already accepted 0.25 rescue, so within-pool
    # filtering rejects that row as well.
    assert set(rescue["rescue_scale_factor"]) == {0.25}
    assert audit.rejection_count == 2
    assert audit.rejections_by_origin["rescue_dilution"] == 2
    assert audit.rejections_by_reference_kind == {
        "generated_pool": 1,
        "history": 1,
    }


def test_round_one_rerun_is_deterministic_and_preserves_legacy_artifacts(
    tmp_path: Path,
) -> None:
    """select_candidates.py must be deterministic and side-effect-free on
    formulations/observations, run twice from identical inputs.

    This intentionally does NOT compare against the committed Round 1
    completed artifact. That file contains the real, hand-entered wet-lab
    results and is historical evidence, not a regenerable template. Pinning
    regenerated output against it would re-encode the earlier selection policy
    rather than test determinism of the current implementation.
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

    assert len(candidates_a) == 12
    assert candidates_a["candidate_id"].nunique() == 12
    assert candidates_a["active_ingredient_count"].gt(0).all()
    assert (
        output_dir_a.parent
        / "rounds"
        / "ROUND_001"
        / "proposal"
        / "proposal.csv"
    ).read_bytes() == (output_dir_a / "next_round_candidates.csv").read_bytes()

    assert formulations_path.read_bytes() == formulations_before
    assert observations_path.read_bytes() == observations_before


def test_round_three_selector_keeps_twelve_rows_and_operator_schema(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "round_three" / "next_round"
    total_pool = tmp_path / "round_three" / "total_candidate_pool.csv"
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
            "ROUND_003",
            "--output-dir",
            str(output_dir),
            "--total-candidate-pool",
            str(total_pool),
            "--pool-size",
            "200",
            "--seed",
            "42",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    round_three = pd.read_csv(output_dir / "next_round_candidates.csv")
    round_two_schema = pd.read_csv(
        PROJECT_ROOT
        / "results"
        / "multi_objective_v2"
        / "rounds"
        / "ROUND_002"
        / "proposal"
        / "proposal.csv",
        nrows=0,
    ).columns.tolist()
    metadata = json.loads(
        (output_dir / "next_round_metadata.json").read_text(encoding="utf-8")
    )

    assert len(round_three) == 12
    assert round_three.columns.tolist() == round_two_schema
    assert metadata["formulation_similarity"]["active"] is True
    assert metadata["formulation_similarity"]["start_round"] == 3
    registry = load_registry()
    ingredient_counts = {
        feature_name: int(
            (
                pd.to_numeric(
                    round_three[feature_name],
                    errors="coerce",
                )
                .fillna(0.0)
                .abs()
                >= presence_threshold(feature_name)
            ).sum()
        )
        for feature_name in registry.feature_names
    }
    assert max(ingredient_counts.values()) <= 5
    assert metadata["ingredient_frequency_diversity"]["active"] is True
    assert (
        metadata["ingredient_frequency_diversity"][
            "maximum_ingredient_frequency"
        ]
        <= 5
    )
    assert (
        metadata["formulation_similarity"]["final_validation"][
            "selected_non_retest_count"
        ]
        == int(
            (
                round_three["recommendation_type"].astype(str)
                != "retest_priority"
            ).sum()
        )
    )

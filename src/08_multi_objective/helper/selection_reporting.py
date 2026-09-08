"""Candidate scoring and next-batch selection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .acquisition import (
    botorch_available,
    minmax,
    qlognehvi_proxy_scores,
    try_botorch_optimize_qlognehvi,
    try_botorch_qlognehvi_scores,
)
from .artifacts import EDITABLE_WETLAB_COLUMNS
from .candidates import stable_formulation_id
from .cold_start import (
    ColdStartContext,
    annotate_cold_start_candidates,
    build_cold_start_context,
    cold_ingredients_in_row,
    cold_start_policy_metadata,
    graduation_allocation_attempts,
    planned_graduation_allocations,
    resolve_cold_start_policy,
)
from .config import nested_get
from .feasibility import (
    ROUND5_POLICY_VERSION,
    annotate_feasibility,
    annotate_support,
    build_support_context,
    feasibility_report,
    ingredient_upper_bound_for_policy,
    policy_activation,
)
from .models import EndpointModels, train_endpoint_models
from .intact_policy import (
    IntactCombinationPolicy,
    annotate_intact_combination_evidence,
    build_intact_evidence,
    intact_policy_metadata,
    resolve_intact_combination_policy,
)
from .phase import (
    PHASE_BOOTSTRAP,
    PHASE_HYBRID,
    PHASE_MECHANICS,
    PHASE_SCREENING,
    PhaseResolution,
    resolve_phase_mode,
)
from .penalties import constraint_report, count_active_ingredients
from .prediction_labels import (
    annotate_viability_prediction_labels,
    is_unknown_viability_status,
)
from .registry import IngredientRegistry, presence_threshold
from .retest import build_retest_candidates
from .similarity import (
    SimilarityAudit,
    build_history_similarity_index,
    filter_frame_by_similarity,
    is_retest_row,
    resolve_similarity_policy,
    similarity_priority_order,
    validate_selected_similarity,
)
from .selection_constraints import (
    _active_ingredient_set,
    _cold_start_ordinary_counts,
    _cold_start_trial_passes_shared_constraints,
    _combination_cap_for_size,
    _drop_zero_active_candidates,
    _enforce_cold_start_policy,
    _enforce_ingredient_combination_cap,
    _enforce_ingredient_frequency_cap,
    _enforce_shared_ingredient_pair_cap,
    _exact_combination_caps_pass,
    _ingredient_appearance_counts,
    _is_protected_diversity_row,
    _shared_pair_counts,
    _support_boundary_count,
)
from .selection_scoring import (
    _annotate_mechanical_history,
    _candidate_masks,
    _continuous_mechanics_candidates,
    _feature_matrix,
    _mechanical_history_counts,
    _mechanics_phase_scores,
    _registry_scaled_feature_matrix,
    _scaled_matrix,
    annotate_candidates,
)

def _format_candidate_line(row: pd.Series, registry: IngredientRegistry) -> str:
    ingredients = []
    for column in registry.feature_names:
        if column not in row.index:
            continue
        value = row.get(column)
        if pd.isna(value) or float(value) <= 0.0:
            continue
        display_name = registry.get_by_feature(column).display_name
        if column.endswith("_pct"):
            ingredients.append(f"{float(value):.3g}% {display_name}")
        elif float(value) >= 1.0:
            ingredients.append(f"{float(value):.3g}M {display_name}")
        else:
            ingredients.append(f"{float(value) * 1000:.3g}mM {display_name}")
    return " + ".join(ingredients) if ingredients else "No active ingredients"


def _write_summary(
    result: SelectionResult,
    selected: pd.DataFrame,
    output_path: Path,
    registry: IngredientRegistry,
) -> None:
    zero_active_filtered = int(result.metadata.get("candidate_pool_rows_filtered_zero_active_at_entry", 0))
    active_phase = result.metadata.get("active_phase", PHASE_SCREENING)
    phase_resolution = result.metadata.get("phase_resolution", {})
    mechanical_policy = result.metadata.get("mechanical_policy", {})
    mechanical_instruction_by_phase = {
        PHASE_SCREENING: (
            "3. Leave mechanical fields blank; rows without a numeric "
            "mechanical_selection_rank are not eligible for mechanical testing."
        ),
        PHASE_BOOTSTRAP: (
            "3. After intact results are recorded, test the first four ranked "
            "actual-intact rows. Remeasure viability, intact formation, and load "
            "for a mechanics_anchor; promote backups only in numeric rank order."
        ),
        PHASE_HYBRID: (
            "3. After intact results are recorded, test the first four ranked "
            "actual-intact rows across the hybrid roles; promote backups only in "
            "numeric rank order."
        ),
        PHASE_MECHANICS: (
            "3. After intact results are recorded, run Instron on the first four "
            "ranked actual-intact rows; skip failures and promote backups only in "
            "numeric rank order."
        ),
    }
    mechanical_instruction = mechanical_instruction_by_phase.get(
        active_phase, mechanical_instruction_by_phase[PHASE_SCREENING]
    )
    hybrid_gate = phase_resolution.get(
        "hybrid_gate", phase_resolution.get("bootstrap_gate", {})
    )
    full_gate = phase_resolution.get("full_gate", {})
    anchor = mechanical_policy.get("anchor", {})
    transition_allocation = mechanical_policy.get("transition_allocation", {})
    ranked_rows = selected.loc[
        pd.to_numeric(selected.get("mechanical_selection_rank"), errors="coerce").notna()
    ]
    lines = [
        "CryoMN v2 Next-Round Candidate Summary",
        "=" * 42,
        "",
        f"Batch ID: {result.metadata.get('batch_id', '')}",
        f"Active phase: {active_phase}",
        f"Phase reason: {result.metadata.get('phase_resolution', {}).get('reason', '')}",
        f"Candidates to make: {len(selected)}",
        f"Mechanical tests requested: {int(selected['mechanical_test_recommended'].sum())}",
        f"Mechanical selection mode: {result.metadata['mechanical_policy']['mechanical_selection_mode']}",
        f"Mechanical observations in database: {result.metadata['mechanical_policy']['mechanical_observation_count']}",
        f"Completed screening rounds: {phase_resolution.get('completed_screening_round_count', 0)}/{phase_resolution.get('minimum_completed_screening_rounds', 8)}",
        "Hybrid evidence gate: "
        f"paired={phase_resolution.get('paired_observation_count', 0)}/{hybrid_gate.get('min_paired_observations', 8)}, "
        f"formulations={phase_resolution.get('distinct_formulation_count', 0)}/{hybrid_gate.get('min_distinct_formulations', 6)}, "
        f"batches={phase_resolution.get('batch_count', 0)}/{hybrid_gate.get('min_batches', 2)}, "
        f"met={bool(phase_resolution.get('hybrid_gate_met', phase_resolution.get('bootstrap_gate_met', False)))}",
        "Full evidence gate: "
        f"paired={phase_resolution.get('paired_observation_count', 0)}/{full_gate.get('min_paired_observations', 16)}, "
        f"formulations={phase_resolution.get('distinct_formulation_count', 0)}/{full_gate.get('min_distinct_formulations', 12)}, "
        f"batches={phase_resolution.get('batch_count', 0)}/{full_gate.get('min_batches', 3)}, "
        f"met={bool(phase_resolution.get('full_gate_met', False))}",
        f"Manual phase override: {bool(phase_resolution.get('override_used', False))}",
        f"Retest-priority formulations in slate: {int((selected.get('recommendation_type', pd.Series(dtype=str)) == 'retest_priority').sum())}",
        f"Mechanically ranked rows: {len(ranked_rows)}",
        "Mechanical roles: "
        + (
            ", ".join(
                f"{row.candidate_id}={row.mechanical_transition_role or 'full_primary'}"
                for row in ranked_rows[
                    ["candidate_id", "mechanical_transition_role"]
                ].itertuples(index=False)
            )
            if not ranked_rows.empty
            else "none"
        ),
        "Anchor decision: "
        f"selected={bool(anchor.get('selected', False))}; "
        f"source_batch={anchor.get('source_batch_id', '') or 'none'}; "
        f"score={anchor.get('selection_score')}; "
        f"reason={result.metadata.get('mechanics_transition', {}).get('anchor_selection', {}).get('reason', 'not applicable')}",
        "Transition fallbacks: "
        + (
            json.dumps(transition_allocation.get("role_fallbacks", []), sort_keys=True)
            if transition_allocation.get("role_fallbacks")
            else "none"
        ),
        "qLogNEHVI status: "
        f"available={mechanical_policy.get('botorch_available', False)}; "
        f"optimizer={result.metadata.get('optimizer_mode', '')}; "
        f"fallback={result.metadata.get('optimizer_fallback_status', '')}; "
        f"reason={result.metadata.get('continuous_qlognehvi', {}).get('continuous_optimizer_reason', 'not applicable')}",
        "",
        "Main database used by selector:",
        "- data/processed_v2/formulations.csv",
        "- data/processed_v2/observations.csv",
        "",
        "Temporary selection restrictions:",
        "- "
        + (
            ", ".join(result.metadata.get("temporary_unavailable_features", []))
            if result.metadata.get("temporary_unavailable_features")
            else "none"
        ),
        "",
        "Wet-lab instructions:",
        "1. Make every formulation listed below.",
        "2. Fill viability_percent and intact_patch_formation_pass in next_round_candidates.csv.",
        mechanical_instruction,
        "4. Run 03_run_round/run_round.py after the CSV is filled.",
        "",
        "Candidates:",
    ]
    if bool(result.metadata.get("formulation_feasibility_policy_active", False)):
        policy_lines = [
            "Formulation feasibility policy:",
            f"- Version: {result.metadata.get('formulation_feasibility_policy_version', '')}",
            f"- Support radius: {float(result.metadata.get('support_radius', float('nan'))):.4g}",
            f"- Rejected pool rows: {int(result.metadata.get('candidate_pool_rows_rejected_by_feasibility', 0))}",
            f"- Optimizer mode: {result.metadata.get('optimizer_mode', '')}",
            f"- Fallback status: {result.metadata.get('optimizer_fallback_status', '')}",
            f"- Fallback reason: {result.metadata.get('continuous_qlognehvi', {}).get('continuous_optimizer_reason', 'not applicable')}",
            "",
        ]
        insertion_index = lines.index("Wet-lab instructions:")
        lines[insertion_index:insertion_index] = policy_lines
    similarity_metadata = result.metadata.get("formulation_similarity", {})
    if bool(similarity_metadata.get("enabled", False)):
        final_validation = similarity_metadata.get("final_validation", {})
        similarity_lines = [
            "Formulation similarity policy:",
            f"- Version: {similarity_metadata.get('policy_version', '')}",
            f"- Status: {'active' if similarity_metadata.get('active', False) else 'inactive'}",
            f"- Bounds-normalized distance threshold: {float(similarity_metadata.get('distance_threshold', 0.05)):.4g}",
            "- Same-single-ingredient minimum relative difference: "
            f"{float(similarity_metadata.get('single_ingredient_min_relative_difference', 0.50)):.0%}",
            f"- Historical references: {int(similarity_metadata.get('history_reference_count', 0))}",
            f"- Rejected generation/pool rows: {int(similarity_metadata.get('rejection_count', 0))}",
            "- Final minimum history distance: "
            f"{final_validation.get('minimum_history_distance')}",
            "- Final minimum within-slate distance: "
            f"{final_validation.get('minimum_within_slate_distance')}",
            "",
        ]
        insertion_index = lines.index("Wet-lab instructions:")
        lines[insertion_index:insertion_index] = similarity_lines
    ingredient_frequency_metadata = result.metadata.get(
        "ingredient_frequency_diversity",
        {},
    )
    if bool(ingredient_frequency_metadata.get("active", False)):
        frequency_lines = [
            "Marginal ingredient-frequency policy:",
            f"- Version: {ingredient_frequency_metadata.get('policy_version', '')}",
            f"- Maximum selected rows per ingredient: {int(ingredient_frequency_metadata.get('max_rows_per_ingredient', 5))}",
            "- Retest and rescue rows count toward the limit but are protected from removal.",
            f"- Frequency-driven replacements: {int(ingredient_frequency_metadata.get('replacement_count', 0))}",
            f"- Final maximum ingredient frequency: {int(ingredient_frequency_metadata.get('maximum_ingredient_frequency', 0))}",
            "",
        ]
        insertion_index = lines.index("Wet-lab instructions:")
        lines[insertion_index:insertion_index] = frequency_lines
    intact_metadata = result.metadata.get("intact_combination_policy", {})
    if bool(intact_metadata.get("active", False)):
        intact_lines = [
            "Empirical intact-combination policy:",
            f"- Version: {intact_metadata.get('policy_version', '')}",
            "- Exact active ingredient sets only; no individual-ingredient blame.",
            f"- Unseen-combination pass probability: {float(intact_metadata.get('unseen_combination_pass_probability', 0.50)):.2f}",
            f"- Screening maximum deduction: {float(intact_metadata.get('screening_max_penalty', 0.20)):.2f}",
            f"- Mechanics mode: {intact_metadata.get('mechanics_mode', '')}",
            f"- Classifier selection role: {intact_metadata.get('classifier_probability_selection_role', '')}",
            "",
        ]
        insertion_index = lines.index("Wet-lab instructions:")
        lines[insertion_index:insertion_index] = intact_lines
    cold_metadata = result.metadata.get("cold_start_policy", {})
    if bool(cold_metadata.get("active", False)):
        cold_lines = [
            "Cold-start policy:",
            f"- Version: {cold_metadata.get('policy_version', '')}",
            f"- Cold ingredients: {', '.join(cold_metadata.get('cold_ingredients', [])) or 'none'}",
            f"- Maximum ordinary rows per cold ingredient: {int(cold_metadata.get('max_ordinary_rows_per_ingredient', 2))}",
            f"- Graduation rows selected: {int(cold_metadata.get('graduation_selected_count', 0))}",
            f"- Cap-driven replacements: {int(cold_metadata.get('cap_replacement_count', 0))}",
            "- Retests and rescue dilutions are exempt from the cold cap.",
            "",
        ]
        insertion_index = lines.index("Wet-lab instructions:")
        lines[insertion_index:insertion_index] = cold_lines
    labeling_metadata = result.metadata.get("viability_prediction_labeling", {})
    if bool(labeling_metadata.get("active", False)):
        status_counts = labeling_metadata.get("selected_status_counts", {})
        unknown_count = sum(
            int(count)
            for status, count in status_counts.items()
            if is_unknown_viability_status(status)
        )
        labeling_lines = [
            "Viability prediction labeling:",
            f"- Version: {labeling_metadata.get('policy_version', '')}",
            f"- Candidates labeled unknown: {unknown_count}",
            "- Unknown candidates keep raw GP values only as acquisition/audit diagnostics.",
            "- Blank predicted_viability_percent means no reliable public viability estimate.",
            "",
        ]
        insertion_index = lines.index("Wet-lab instructions:")
        lines[insertion_index:insertion_index] = labeling_lines
    if zero_active_filtered:
        warning_lines = [
            "Warnings:",
            "- "
            f"{zero_active_filtered} zero-active candidate-pool rows were removed before scoring.",
            "- Review the supplied candidate pool or upstream candidate-generation logic.",
            "",
        ]
        insertion_index = lines.index("Wet-lab instructions:")
        lines[insertion_index:insertion_index] = warning_lines
    display_columns = [
        "selection_rank",
        "candidate_id",
        "formulation_id",
        "mechanical_test_recommended",
        "predicted_viability_percent",
        "viability_prediction_status",
        "intact_patch_pass_probability",
        "predicted_critical_axial_load_N_per_needle",
        "active_ingredient_count",
    ]
    for _, row in selected.iterrows():
        prediction_status = str(
            row.get("viability_prediction_status", "model_supported")
        )
        if is_unknown_viability_status(prediction_status):
            viability_part = "viability=UNKNOWN"
        else:
            predicted_viability = pd.to_numeric(
                row.get("predicted_viability_percent"),
                errors="coerce",
            )
            viability_part = (
                f"predicted_viability={float(predicted_viability):.1f}%"
                if pd.notna(predicted_viability)
                else "viability=UNKNOWN"
            )
        parts = [
            f"#{int(row['selection_rank'])}",
            f"candidate_id={row['candidate_id']}",
            f"formulation_id={row['formulation_id']}",
            f"recommendation_type={row.get('recommendation_type', '')}",
            f"mechanical_test={bool(row['mechanical_test_recommended'])}",
            viability_part,
            f"viability_status={prediction_status}",
            f"intact_probability={float(row['intact_patch_pass_probability']):.2f}",
        ]
        if bool(result.metadata.get("formulation_feasibility_policy_active", False)):
            parts.extend(
                [
                    f"origin={row.get('candidate_origin', 'finite_pool_fallback')}",
                    f"support={row.get('support_status', 'not_evaluated')}",
                ]
            )
        if result.metadata["mechanical_policy"]["mechanical_observation_count"] > 0 and "predicted_critical_axial_load_N_per_needle" in row and pd.notna(
            row["predicted_critical_axial_load_N_per_needle"]
        ):
            parts.append(
                "predicted_critical_load="
                f"{float(row['predicted_critical_axial_load_N_per_needle']):.3g} N/needle"
            )
        lines.append("- " + "; ".join(parts))
        lines.append(f"  formulation: {_format_candidate_line(row, registry)}")
        prediction_reason = str(row.get("viability_prediction_reason", "")).strip()
        if prediction_reason:
            raw_mean = pd.to_numeric(
                row.get("raw_surrogate_viability_mean"),
                errors="coerce",
            )
            raw_std = pd.to_numeric(
                row.get("raw_surrogate_viability_std"),
                errors="coerce",
            )
            diagnostic = ""
            if is_unknown_viability_status(prediction_status) and pd.notna(raw_mean):
                diagnostic = f"; raw surrogate diagnostic={float(raw_mean):.1f}%"
                if pd.notna(raw_std):
                    diagnostic += f" ± {float(raw_std):.1f}%"
                diagnostic += " (not a public prediction)"
            lines.append(
                f"  viability note: {prediction_reason}{diagnostic}"
            )
        explanation = row.get("selection_explanation", "")
        if pd.notna(explanation) and str(explanation).strip():
            lines.append(f"  note: {explanation}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selection_result(
    result: SelectionResult,
    output_dir: str | Path,
    batch_id: str = "",
    total_candidate_pool_path: str | Path | None = None,
    registry: IngredientRegistry | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if registry is None:
        registry = IngredientRegistry.from_config()
    selected = result.viability_screen.copy()
    primary_ids = set()
    if not result.mechanical_tests.empty:
        primary_mask = result.mechanical_tests.get(
            "mechanical_primary_recommended",
            pd.Series(True, index=result.mechanical_tests.index),
        ).astype(bool)
        primary_ids = set(
            result.mechanical_tests.loc[primary_mask, "candidate_id"].astype(str)
        )
    selected["mechanical_test_recommended"] = selected[
        "candidate_id"
    ].astype(str).isin(primary_ids)
    selected["mechanical_selection_rank"] = ""
    selected["mechanical_selection_mode"] = ""
    selected["mechanical_backup_status"] = ""
    selected["mechanical_transition_role"] = ""
    if not result.mechanical_tests.empty:
        rank_map = result.mechanical_tests.set_index("candidate_id")["mechanical_selection_rank"].to_dict()
        mode_map = result.mechanical_tests.set_index("candidate_id")["mechanical_selection_mode"].to_dict()
        backup_map = result.mechanical_tests.set_index("candidate_id")[
            "mechanical_backup_status"
        ].to_dict()
        role_map = result.mechanical_tests.set_index("candidate_id")[
            "mechanical_transition_role"
        ].to_dict()
        selected["mechanical_selection_rank"] = selected["candidate_id"].map(rank_map).fillna("")
        selected["mechanical_selection_mode"] = selected["candidate_id"].map(mode_map).fillna("")
        selected["mechanical_backup_status"] = selected["candidate_id"].map(backup_map).fillna("")
        selected["mechanical_transition_role"] = selected["candidate_id"].map(role_map).fillna("")

    wetlab_result_columns = [
        "formulation_id",
        "candidate_id",
        "selection_rank",
        "recommendation_type",
        "selection_explanation",
        "mechanical_test_recommended",
        "mechanical_selection_rank",
        "mechanical_selection_mode",
        "mechanical_backup_status",
        "mechanical_transition_role",
        "batch_id",
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
    ]
    if bool(result.metadata.get("formulation_feasibility_policy_active", False)):
        preparation_columns = [
            "preparation_feasibility_pass",
            "homogeneous_solution_pass",
            "fillability_pass",
            "preparation_failure_reason",
        ]
        notes_index = wetlab_result_columns.index("notes")
        wetlab_result_columns[notes_index:notes_index] = preparation_columns
    if "group10" in result.metadata:
        from .group10_config import METADATA_FIELDS
        wetlab_result_columns += ["experimental_role", "preparation_basis", *METADATA_FIELDS, "supplementary_analysis_file"]
    selected["batch_id"] = batch_id
    for column in wetlab_result_columns:
        if column not in selected.columns:
            selected[column] = ""
    for column in EDITABLE_WETLAB_COLUMNS:
        if column in selected.columns:
            selected[column] = ""
    result.metadata["batch_id"] = batch_id
    forward_diagnostic_columns = [
        *registry.feature_names,
        "active_ingredient_count",
        "candidate_origin",
        "rescue_scale_factor",
        "rescue_anchor_formulation_id",
        "rescue_anchor_viability_percent",
        "active_polymer_count",
        "total_polymer_pct",
        "total_serum_protein_pct",
        "total_polymer_serum_pct",
        "total_sugar_M",
        "total_nonpermeating_solute_M",
        "estimated_small_solute_g_L",
        "feasibility_pass",
        "feasibility_reasons",
        "nearest_support_distance",
        "support_status",
        "source",
        "source_row_id",
        "formulation_label",
        "source_type",
        "predicted_viability_percent",
        "viability_std",
        "viability_prediction_status",
        "viability_prediction_label",
        "viability_prediction_reason",
        "raw_surrogate_viability_mean",
        "raw_surrogate_viability_std",
        "viability_surrogate_prior_mean",
        "viability_surrogate_prior_std",
        "viability_prior_reversion",
        "viability_exact_observation_count",
        "viability_exact_batch_count",
        "viability_exact_observed_mean",
        "viability_exact_observed_sources",
        "predicted_critical_axial_load_N_per_needle",
        "critical_axial_load_std",
        "intact_patch_pass_probability",
        "empirical_combination_pass_probability",
        "empirical_combination_weighted_passes",
        "empirical_combination_weighted_failures",
        "nearest_matching_intact_pass_distance",
        "nearest_matching_intact_failure_distance",
        "intact_combination_screening_penalty",
        "intact_combination_policy_version",
        "cold_start_ingredients",
        "cold_start_ingredient_count",
        "cold_start_prior_evidence_counts",
        "cold_start_ordinary_exempt",
        "cold_start_graduation_eligible",
        "cold_start_graduation_priority",
        "cold_start_graduation_reason",
        "mechanical_feasibility_weight",
        "same_formulation_range",
        "local_neighbor_residual",
        "retest_priority_score",
        "viability_ucb",
        "critical_axial_load_ucb",
        "predicted_initial_stiffness_N_per_mm_per_needle",
        "initial_stiffness_std",
        "preparation_feasibility_probability",
        "active_ingredient_excess_above_8",
        "single_molar_excess_features",
        "single_molar_excess_total_M",
        "intact_failure_probability",
        "acquisition_penalty",
        "screening_acquisition_penalty",
        "screening_phase_score",
        "mechanics_phase_score",
        "hybrid_phase_score",
        "bootstrap_utility",
        "prior_mechanical_observation_count",
        "mechanical_repeat_status",
        "mechanical_repeat_allowed",
        "mechanics_anchor_source_batch",
        "mechanics_anchor_selection_score",
        "selection_role",
    ]
    if (
        result.metadata.get("formulation_feasibility_policy_version")
        == ROUND5_POLICY_VERSION
    ):
        diagnostic_index = forward_diagnostic_columns.index(
            "estimated_small_solute_g_L"
        )
        forward_diagnostic_columns[diagnostic_index:diagnostic_index] = [
            "total_viscosity_active_macromolecule_pct",
            "total_permeating_cpa_M",
            "crystalline_solute_saturation_burden",
            "feasibility_policy_version",
        ]
    if bool(result.metadata.get("formulation_feasibility_policy_active", False)):
        for column in forward_diagnostic_columns:
            if column not in selected.columns:
                selected[column] = ""
        csv_columns = wetlab_result_columns + forward_diagnostic_columns + [
            column
            for column in selected.columns
            if column not in wetlab_result_columns
            and column not in forward_diagnostic_columns
        ]
    else:
        csv_columns = wetlab_result_columns + [
            column
            for column in selected.columns
            if column not in wetlab_result_columns
        ]
    selected[csv_columns].to_csv(output / "next_round_candidates.csv", index=False)

    total_pool = result.candidate_pool.copy()
    total_pool["batch_id"] = batch_id
    total_pool["active_phase"] = result.metadata.get("active_phase", PHASE_SCREENING)
    if bool(result.metadata.get("formulation_feasibility_policy_active", False)):
        total_pool["formulation_feasibility_policy_active"] = True
        total_pool["formulation_feasibility_policy_version"] = result.metadata.get(
            "formulation_feasibility_policy_version",
            "",
        )
        total_pool["formulation_feasibility_policy_start_round"] = result.metadata.get(
            "formulation_feasibility_policy_start_round",
            "",
        )
        total_pool["optimizer_mode"] = result.metadata.get("optimizer_mode", "")
        total_pool["optimizer_fallback_status"] = result.metadata.get(
            "optimizer_fallback_status",
            "",
        )
    total_pool["selected_for_viability_screen"] = total_pool["candidate_id"].isin(
        set(result.viability_screen["candidate_id"])
    )
    total_pool["selected_for_mechanical_test"] = total_pool[
        "candidate_id"
    ].astype(str).isin(primary_ids)
    rank_map = result.viability_screen.set_index("candidate_id")["selection_rank"].to_dict()
    total_pool["selection_rank"] = total_pool["candidate_id"].map(rank_map).fillna("")
    total_pool["mechanical_selection_rank"] = ""
    total_pool["mechanical_selection_mode"] = ""
    total_pool["mechanical_backup_status"] = ""
    total_pool["mechanical_transition_role"] = ""
    if not result.mechanical_tests.empty:
        mech_rank_map = result.mechanical_tests.set_index("candidate_id")[
            "mechanical_selection_rank"
        ].to_dict()
        mech_mode_map = result.mechanical_tests.set_index("candidate_id")[
            "mechanical_selection_mode"
        ].to_dict()
        mech_backup_map = result.mechanical_tests.set_index("candidate_id")[
            "mechanical_backup_status"
        ].to_dict()
        mech_role_map = result.mechanical_tests.set_index("candidate_id")[
            "mechanical_transition_role"
        ].to_dict()
        total_pool["mechanical_selection_rank"] = (
            total_pool["candidate_id"].map(mech_rank_map).fillna("")
        )
        total_pool["mechanical_selection_mode"] = (
            total_pool["candidate_id"].map(mech_mode_map).fillna("")
        )
        total_pool["mechanical_backup_status"] = (
            total_pool["candidate_id"].map(mech_backup_map).fillna("")
        )
        total_pool["mechanical_transition_role"] = (
            total_pool["candidate_id"].map(mech_role_map).fillna("")
        )
    total_pool_output = (
        Path(total_candidate_pool_path)
        if total_candidate_pool_path is not None
        else output.parent / "total_candidate_pool.csv"
    )
    total_pool_output.parent.mkdir(parents=True, exist_ok=True)
    total_pool.to_csv(total_pool_output, index=False)
    _write_summary(result, selected, output / "next_round_summary.txt", registry=registry)
    if "group10" in result.metadata:
        details = result.metadata["group10"]
        with (output / "next_round_summary.txt").open("a") as handle:
            handle.write("\nGroup 10 workflow: production GP/noise/target unchanged.\n")
            handle.write("Reference readiness: " + json.dumps(details["reference_readiness"]) + "\n")
            handle.write("Mechanical budget: four formulations; specimens per formulation: " + str(details["specimens_per_formulation"]) + "\n")
            handle.write("Roles: " + json.dumps(details["realized_roles"]) + "\n")
            handle.write("Supplementary supported_load_1mm_v1 is not the production target.\n")
    if bool(result.metadata.get("formulation_feasibility_policy_active", False)):
        metadata_path = output / "next_round_metadata.json"
        metadata_path.write_text(
            json.dumps(
                result.metadata,
                indent=2,
                default=lambda value: value.item() if hasattr(value, "item") else str(value),
            )
            + "\n",
            encoding="utf-8",
        )

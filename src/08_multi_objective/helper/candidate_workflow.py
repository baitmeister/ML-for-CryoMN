"""Callable workflow for V2 candidate selection.

This module contains the ordered Stage 2 protocol.  It accepts explicit paths so
tests and replays can run without writing into the canonical campaign history.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import tempfile
from pathlib import Path
from typing import Mapping

import pandas as pd

from .candidates import (
    filter_available_candidate_pool,
    filter_candidate_pool_to_registry_bounds,
    filter_nonzero_active_candidate_pool,
    generate_random_candidate_pool,
    generate_rescue_candidate_pool,
    generate_support_aware_candidate_pool,
    load_candidate_pool,
    unavailable_features_from_config,
)
from .config import load_availability_config, load_optimization_config, nested_get
from .artifacts import copy_working, freeze_proposal, supersede_unstarted_proposal
from .feasibility import (
    annotate_feasibility,
    annotate_support,
    build_support_context,
    policy_activation,
)
from .paths import (
    AVAILABILITY_CONFIG,
    CURRENT_ROUND_STATUS_PATH,
    FORMULATIONS_PATH,
    OBSERVATIONS_PATH,
    RESULTS_V2_DIR,
    TOTAL_CANDIDATE_POOL_PATH,
)
from .registry import load_registry
from .selection import SelectionResult, select_next_round, write_selection_result
from .similarity import (
    SimilarityAudit,
    build_history_similarity_index,
    filter_frame_by_similarity,
    resolve_similarity_policy,
)
from .status import derive_round_tracker, parse_round_number, write_current_round_status
from .visualization import generate_proposal_artifacts


@dataclass(frozen=True)
class CandidateSelectionOptions:
    """Inputs corresponding one-for-one with the existing Stage 2 CLI."""

    formulations_path: str | Path = FORMULATIONS_PATH
    observations_path: str | Path = OBSERVATIONS_PATH
    candidate_pool_path: str | Path | None = None
    availability_config_path: str | Path = AVAILABILITY_CONFIG
    output_dir: str | Path = RESULTS_V2_DIR / "next_round"
    total_candidate_pool_path: str | Path = TOTAL_CANDIDATE_POOL_PATH
    pool_size: int | None = None
    seed: int | None = None
    phase_mode: str | None = None
    batch_id: str | None = None
    supersede_unstarted_proposal: bool = False
    gp_strategy_decision_path: str | Path | None = None


@dataclass(frozen=True)
class CandidateSelectionWorkflowResult:
    """Resolved Stage 2 state and every artifact path created by the workflow."""

    resolved_phase: str
    round_id: str
    selection_result: SelectionResult
    artifact_paths: Mapping[str, Path]
    superseded_proposal_path: Path | None = None


def _read_or_empty(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def _next_round_id(observations_path: str | Path = OBSERVATIONS_PATH) -> str:
    observations = _read_or_empty(observations_path)
    return str(derive_round_tracker(observations)["next_round_id"])


def run_candidate_selection(
    options: CandidateSelectionOptions,
) -> CandidateSelectionWorkflowResult:
    """Run Stage 2 in the historical transaction order."""

    registry = load_registry()
    optimization_config = load_optimization_config()
    availability_config = load_availability_config(options.availability_config_path)
    unavailable_features = unavailable_features_from_config(availability_config, registry)
    formulations = _read_or_empty(options.formulations_path)
    observations = _read_or_empty(options.observations_path)
    from .group10_config import (
        assert_group10_can_proceed,
        load_group10_config_for_round,
        production_observations,
    )
    batch_id = options.batch_id or _next_round_id(options.observations_path)
    target_round_number = parse_round_number(batch_id)
    if target_round_number >= 11:
        from .gp_strategy import validate_decision, StrategyDecisionRequired
        decision_path = options.gp_strategy_decision_path or Path(options.output_dir).parent / 'gp_strategy_decision.json'
        decision = validate_decision(decision_path, target_round_number)
        import json
        for recorded in (Path(options.output_dir).parent / 'rounds').glob('*/proposal/gp_strategy_decision.json'):
            old = json.loads(recorded.read_text())
            if old['decision_id'] == decision['decision_id'] and old['sha256'] != decision['sha256']:
                raise StrategyDecisionRequired('Changed configuration requires a new decision_id')
        optimization_config['gp_strategy_decision'] = decision
        optimization_config['_gp_target_round'] = target_round_number
        optimization_config['_gp_raw_observations'] = observations.copy()
    group10_config = load_group10_config_for_round(target_round_number)
    assert_group10_can_proceed(group10_config, target_round_number, observations)
    observations = production_observations(observations, target_round_number=target_round_number)
    policy_active, policy_version, policy_start_round = policy_activation(
        optimization_config,
        target_round_number,
    )
    support_context = build_support_context(
        formulations,
        registry,
        optimization_config,
        observations=observations,
    )

    if formulations.empty:
        raise SystemExit(
            "No v2 formulations were found. Run: python3 src/08_multi_objective/01_build_database/build_database.py"
        )
    similarity_policy = resolve_similarity_policy(
        optimization_config,
        target_round_number,
    )
    similarity_index = build_history_similarity_index(
        formulations,
        observations,
        registry,
        similarity_policy,
    )
    similarity_audit = SimilarityAudit(
        similarity_policy,
        history_reference_count=len(similarity_index),
    )
    availability_excluded_candidates = pd.DataFrame()

    if options.candidate_pool_path:
        candidate_pool = load_candidate_pool(options.candidate_pool_path, registry)
        before_filter = len(candidate_pool)
        candidate_pool = filter_candidate_pool_to_registry_bounds(candidate_pool, registry)
        bounds_filtered_count = before_filter - len(candidate_pool)
        if candidate_pool.empty:
            raise SystemExit(
                "Candidate pool is empty after applying registry bounds."
            )
        filtered_count = 0
    else:
        pool_size = options.pool_size or int(
            nested_get(optimization_config, "selection.generated_candidate_pool_size", 2000)
        )
        seed = options.seed if options.seed is not None else int(optimization_config.get("random_seed", 42))
        rescue_candidate_count = 0
        if policy_active:
            rescue_candidates = generate_rescue_candidate_pool(
                registry,
                formulations=formulations,
                observations=observations,
                optimization_config=optimization_config,
                support=support_context,
                unavailable_feature_names=unavailable_features,
                similarity_index=similarity_index,
                similarity_audit=similarity_audit,
                policy_version=policy_version,
            )
            rescue_candidate_count = int(len(rescue_candidates))
            candidate_pool = generate_support_aware_candidate_pool(
                registry,
                formulations=formulations,
                optimization_config=optimization_config,
                support=support_context,
                n_candidates=pool_size,
                random_seed=seed,
                unavailable_feature_names=unavailable_features,
                similarity_index=similarity_index,
                similarity_audit=similarity_audit,
                policy_version=policy_version,
            )
            if not rescue_candidates.empty:
                candidate_pool = pd.concat(
                    [rescue_candidates, candidate_pool],
                    ignore_index=True,
                    sort=False,
                ).drop_duplicates("formulation_id", keep="first")
        else:
            candidate_pool = generate_random_candidate_pool(
                registry,
                n_candidates=pool_size,
                random_seed=seed,
                unavailable_feature_names=unavailable_features,
            )
        bounds_filtered_count = 0
        filtered_count = 0
    if options.candidate_pool_path:
        rescue_candidate_count = 0

    if policy_active:
        candidate_pool = annotate_feasibility(
            candidate_pool,
            registry,
            optimization_config,
            policy_active=True,
            policy_version=policy_version,
        )
        candidate_pool = annotate_support(candidate_pool, registry, support_context)
        if "candidate_origin" not in candidate_pool.columns:
            candidate_pool["candidate_origin"] = "finite_pool_fallback"
        rejected_candidates = candidate_pool.loc[
            ~candidate_pool["feasibility_pass"].astype(bool)
        ].copy()
        candidate_pool = candidate_pool.loc[
            candidate_pool["feasibility_pass"].astype(bool)
        ].reset_index(drop=True)
        if candidate_pool.empty:
            raise SystemExit(
                "Candidate pool is empty after applying formulation-feasibility rules."
            )
    else:
        rejected_candidates = candidate_pool.head(0).copy()

    if options.candidate_pool_path:
        before_availability = len(candidate_pool)
        available_candidate_pool = filter_available_candidate_pool(
            candidate_pool,
            unavailable_features,
        )
        available_ids = set(available_candidate_pool["candidate_id"].astype(str))
        availability_excluded_candidates = candidate_pool.loc[
            ~candidate_pool["candidate_id"].astype(str).isin(available_ids)
        ].copy()
        if not availability_excluded_candidates.empty:
            availability_excluded_candidates["availability_eligible"] = False
            availability_excluded_candidates["availability_reasons"] = (
                "temporarily_unavailable_ingredient"
            )
        candidate_pool = available_candidate_pool
        candidate_pool["availability_eligible"] = True
        candidate_pool["availability_reasons"] = ""
        filtered_count = before_availability - len(candidate_pool)
        if candidate_pool.empty:
            raise SystemExit(
                "Candidate pool is empty after applying temporary availability restrictions."
            )

    before_zero_active_filter = len(candidate_pool)
    candidate_pool = filter_nonzero_active_candidate_pool(candidate_pool, registry)
    zero_active_filtered_count = before_zero_active_filter - len(candidate_pool)
    if candidate_pool.empty:
        raise SystemExit(
            "Candidate pool is empty after removing zero-active formulations."
        )
    if options.candidate_pool_path and similarity_policy.active:
        candidate_pool, _ = filter_frame_by_similarity(
            candidate_pool,
            similarity_index,
            similarity_audit,
            accepted_reference_kind="generated_pool",
        )
        if candidate_pool.empty:
            raise SystemExit(
                "Candidate pool is empty after applying formulation-similarity rules."
            )

    if target_round_number == 11 and optimization_config.get('gp_strategy_decision',{}).get('sparse_control_exception'):
        from .control_sensitivity import check_sensitivity
        check_sensitivity(formulations, observations, candidate_pool, registry, optimization_config,
                          Path(options.output_dir).parent / 'group11_control_sensitivity')

    result = select_next_round(
        formulations=formulations,
        observations=observations,
        candidate_pool=candidate_pool,
        registry=registry,
        optimization_config=optimization_config,
        requested_phase_mode=options.phase_mode,
        target_round_number=target_round_number,
        policy_active=policy_active,
        policy_version=policy_version,
        similarity_audit=similarity_audit,
        unavailable_feature_names=unavailable_features,
    )
    from .group10_selection import apply_group10
    result = apply_group10(result, formulations, observations, registry,
                           optimization_config, group10_config, target_round_number,
                           unavailable_features)
    audit_excluded_candidates = pd.concat(
        [rejected_candidates, availability_excluded_candidates],
        ignore_index=True,
        sort=False,
    )
    if not audit_excluded_candidates.empty:
        audit_excluded_candidates["selected_for_viability_screen"] = False
        audit_excluded_candidates["selected_for_mechanical_test"] = False
        audit_excluded_candidates["selection_rank"] = ""
        audit_excluded_candidates["mechanics_phase_score"] = float("nan")
        audit_excluded_candidates["screening_phase_score"] = float("nan")
        result = replace(
            result,
            candidate_pool=pd.concat(
                [result.candidate_pool, audit_excluded_candidates],
                ignore_index=True,
                sort=False,
            ),
        )
    result.metadata["formulation_feasibility_policy_active"] = policy_active
    result.metadata["formulation_feasibility_policy_version"] = policy_version
    result.metadata["formulation_feasibility_policy_start_round"] = policy_start_round
    result.metadata["target_round_number"] = target_round_number
    result.metadata["support_radius"] = support_context.radius
    result.metadata["candidate_pool_rows_rejected_by_feasibility"] = int(len(rejected_candidates))
    result.metadata["rescue_candidate_count"] = int(rescue_candidate_count)
    result.metadata["temporary_unavailable_features"] = unavailable_features
    result.metadata["candidate_pool_rows_filtered_by_bounds"] = bounds_filtered_count
    result.metadata["candidate_pool_rows_filtered_by_availability"] = filtered_count
    result.metadata["candidate_pool_rows_filtered_zero_active_at_entry"] = zero_active_filtered_count
    if 'gp_strategy_decision' in optimization_config:
        result.metadata['gp_strategy_decision'] = optimization_config['gp_strategy_decision']
    output_dir = Path(options.output_dir)
    results_root = output_dir.parent
    results_root.mkdir(parents=True, exist_ok=True)
    superseded_path: Path | None = None
    proposal_artifacts: list[Path] = []
    with tempfile.TemporaryDirectory(
        prefix=f".{batch_id}_selection_",
        dir=results_root,
    ) as staging_name:
        staging_root = Path(staging_name)
        staging_output = staging_root / "next_round"
        staging_pool = staging_root / "total_candidate_pool.csv"
        write_selection_result(
            result,
            staging_output,
            batch_id=batch_id,
            total_candidate_pool_path=staging_pool,
            registry=registry,
        )
        staged_candidates = staging_output / "next_round_candidates.csv"
        staged_summary = staging_output / "next_round_summary.txt"
        staged_metadata = staging_output / "next_round_metadata.json"
        if target_round_number >= 11:
            from .gp_comparison import freeze
            freeze(Path(__file__).resolve().parents[3], batch_id, staging_root / 'gp_comparison',
                slate_path=staged_candidates, forms=formulations,
                obs=optimization_config.get('_gp_raw_observations', observations), registry=registry)
        if options.supersede_unstarted_proposal:
            superseded_path = supersede_unstarted_proposal(
                batch_id,
                observations=observations,
                active_worksheet=output_dir / "next_round_candidates.csv",
                reason=(
                    "Explicit forward-only selection-policy update for an "
                    "unstarted proposal"
                ),
                policy_versions=[
                    result.metadata.get("intact_combination_policy", {}).get(
                        "policy_version", ""
                    ),
                    result.metadata.get("cold_start_policy", {}).get(
                        "policy_version", ""
                    ),
                    result.metadata.get("viability_prediction_labeling", {}).get(
                        "policy_version", ""
                    ),
                ],
                total_candidate_pool=options.total_candidate_pool_path,
                results_root=results_root,
            )
        round_paths = freeze_proposal(
            batch_id,
            staged_candidates,
            summary_path=staged_summary,
            metadata_path=staged_metadata if staged_metadata.exists() else None,
            results_root=results_root,
        )
        if 'gp_strategy_decision' in optimization_config:
            import json
            (round_paths.proposal_dir / 'gp_strategy_decision.json').write_text(json.dumps(optimization_config['gp_strategy_decision'], indent=2)+'\n')
        if target_round_number >= 11:
            import shutil
            comparison_dir = round_paths.proposal_dir.parent / 'gp_comparison'
            if comparison_dir.exists():
                raise FileExistsError('Existing challenger evidence cannot be overwritten: '+str(comparison_dir))
            shutil.move(str(staging_root / 'gp_comparison'), comparison_dir)
        proposal_artifacts = generate_proposal_artifacts(
            pd.read_csv(staged_candidates),
            round_paths.proposal_dir,
        )

        copy_working(staged_candidates, output_dir / "next_round_candidates.csv")
        copy_working(staged_summary, output_dir / "next_round_summary.txt")
        if staged_metadata.exists():
            copy_working(staged_metadata, output_dir / "next_round_metadata.json")
        copy_working(staging_pool, options.total_candidate_pool_path)

    status_path = write_current_round_status(
        Path(options.output_dir).parent / CURRENT_ROUND_STATUS_PATH.name,
        observations=observations,
        source_observations_path=options.observations_path,
        active_phase=result.metadata["active_phase"],
        phase_reason=result.metadata.get("phase_resolution", {}).get("reason", ""),
        phase_resolution=result.metadata.get("phase_resolution", {}),
        proposal_policy={
            "mechanics_transition": result.metadata.get("mechanics_transition", {}),
            "mechanical_policy": result.metadata.get("mechanical_policy", {}),
            "continuous_qlognehvi": result.metadata.get("continuous_qlognehvi", {}),
            "optimizer_mode": result.metadata.get("optimizer_mode", ""),
            "optimizer_fallback_status": result.metadata.get(
                "optimizer_fallback_status", ""
            ),
            "ranked_candidates": (
                result.mechanical_tests[
                    [
                        "candidate_id",
                        "mechanical_selection_rank",
                        "mechanical_transition_role",
                        "mechanical_backup_status",
                    ]
                ].to_dict(orient="records")
                if not result.mechanical_tests.empty
                else []
            ),
        },
        proposed_batch_id=batch_id,
        proposed_batch_override_used=options.batch_id is not None,
    )

    artifact_paths: dict[str, Path] = {
        "working_candidates": Path(options.output_dir) / "next_round_candidates.csv",
        "working_summary": Path(options.output_dir) / "next_round_summary.txt",
        "working_metadata": Path(options.output_dir) / "next_round_metadata.json",
        "total_candidate_pool": Path(options.total_candidate_pool_path),
        "round_status": status_path,
        "frozen_proposal": round_paths.proposal_csv,
        "frozen_summary": round_paths.proposal_summary,
        "frozen_metadata": round_paths.proposal_metadata,
        "proposal_plots_dir": round_paths.proposal_plots_dir,
    }
    for index, path in enumerate(proposal_artifacts, start=1):
        artifact_paths[f"proposal_artifact_{index}"] = Path(path)
    if superseded_path is not None:
        artifact_paths["superseded_proposal"] = Path(superseded_path)

    return CandidateSelectionWorkflowResult(
        resolved_phase=str(result.metadata["active_phase"]),
        round_id=batch_id,
        selection_result=result,
        artifact_paths=artifact_paths,
        superseded_proposal_path=(
            None if superseded_path is None else Path(superseded_path)
        ),
    )

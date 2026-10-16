"""Callable workflow for V2 candidate selection.

This module contains the ordered Stage 2 protocol.  It accepts explicit paths so
tests and replays can run without writing into the canonical campaign history.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import tempfile
import json
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
from .artifacts import (supersede_unstarted_proposal, round_artifact_paths,
                        validate_completed_against_proposal, EDITABLE_WETLAB_COLUMNS)
from .persistence import (serialized, recover_transactions, journal_root, commit_files,
                          file_hash, digest_bytes, json_bytes, PersistenceConflict)
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
from .automatic_repeats import repeat_policy_enabled, load_completed_retest_history


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
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def _next_round_id(observations_path: str | Path = OBSERVATIONS_PATH) -> str:
    observations = _read_or_empty(observations_path)
    return str(derive_round_tracker(observations)["next_round_id"])


@serialized
def run_candidate_selection(
    options: CandidateSelectionOptions,
) -> CandidateSelectionWorkflowResult:
    """Run Stage 2 in the historical transaction order."""

    recover_transactions(journal_root(options.formulations_path))
    recover_transactions(Path(options.output_dir).parent / '.v2-transactions')
    from .paths import CONFIG_DIR
    input_paths = [Path(options.formulations_path), Path(options.observations_path),
                   Path(options.availability_config_path), *sorted(CONFIG_DIR.rglob('*.yaml'))]
    if options.candidate_pool_path:
        input_paths.append(Path(options.candidate_pool_path))
    input_hashes = {str(p.resolve()): file_hash(p) for p in input_paths}
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
        input_hashes[str(Path(decision_path).resolve())] = file_hash(decision_path)
        decision = validate_decision(decision_path, target_round_number)
        for recorded in (Path(options.output_dir).parent / 'rounds').glob('*/proposal/gp_strategy_decision.json'):
            old = json.loads(recorded.read_text())
            if old['decision_id'] == decision['decision_id'] and old['sha256'] != decision['sha256']:
                raise StrategyDecisionRequired('Changed configuration requires a new decision_id')
        optimization_config['gp_strategy_decision'] = decision
        optimization_config['_gp_target_round'] = target_round_number
        optimization_config['_gp_raw_observations'] = observations.copy()
        from .gp_strategy import validate_search_config
        validate_search_config(optimization_config)
    group10_config = load_group10_config_for_round(target_round_number)
    assert_group10_can_proceed(group10_config, target_round_number, observations)
    observations = production_observations(observations, target_round_number=target_round_number)
    if repeat_policy_enabled(optimization_config):
        history, history_inputs = load_completed_retest_history(
            observations, Path(options.output_dir).parent / 'rounds', target_round_number)
        optimization_config['_completed_automatic_retest_history'] = history
        input_hashes.update(history_inputs)
    selection_inputs = dict(
        files=dict(input_hashes),
        pool_size=options.pool_size or nested_get(optimization_config, 'selection.generated_candidate_pool_size', 2000),
        seed=options.seed if options.seed is not None else optimization_config.get('random_seed', 42),
        phase_mode=options.phase_mode or 'auto', batch_id=batch_id,
        decision_sha256=optimization_config.get('gp_strategy_decision', {}).get('sha256'))
    round_paths = round_artifact_paths(batch_id, Path(options.output_dir).parent)
    receipt_path = round_paths.proposal_dir / 'selection_commit.json'
    if receipt_path.exists() and not options.supersede_unstarted_proposal:
        receipt = json.loads(receipt_path.read_text())
        if receipt['inputs'] != selection_inputs:
            raise PersistenceConflict('Frozen selection inputs/decision differ; refusing to reuse or overwrite proposal.')
        for name, expected in receipt['hashes'].items():
            if file_hash(round_paths.round_dir / name) != expected:
                raise PersistenceConflict('Frozen selection artifact changed: ' + name)
        state = round_paths.proposal_dir / 'selection_state'
        result = SelectionResult(
            _read_or_empty(state / 'viability_screen.csv'), _read_or_empty(state / 'mechanical_tests.csv'),
            _read_or_empty(state / 'candidate_pool.csv'), json.loads(round_paths.proposal_metadata.read_text()))
        _publish_working(options, batch_id, round_paths, input_hashes)
        artifacts = generate_proposal_artifacts(pd.read_csv(round_paths.proposal_csv), round_paths.proposal_dir)
        return _finish_selection(options, observations, batch_id, result, round_paths, artifacts, None)
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
    working_path = Path(options.output_dir) / 'next_round_candidates.csv'
    input_hashes[str(working_path.resolve())] = file_hash(working_path)

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
            candidate_pool["candidate_origin"] = "finite_pool"
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

    sensitivity_required = target_round_number == 11 and optimization_config.get('gp_strategy_decision', {}).get('sparse_control_exception')
    if sensitivity_required:
        from .control_sensitivity import check_sensitivity
        sensitivity_dir = Path(options.output_dir).parent / 'group11_control_sensitivity'
        if not sensitivity_dir.exists():
            # Keep a failed check's diagnostics available for the operator to review.
            check_sensitivity(formulations, observations, candidate_pool, registry, optimization_config,
                              sensitivity_dir)
            sensitivity_files = {p.name: p.read_bytes() for p in sensitivity_dir.iterdir() if p.is_file()}
        else:
            with tempfile.TemporaryDirectory(prefix='cryomn-sensitivity-') as sensitivity_temp:
                try:
                    check_sensitivity(formulations, observations, candidate_pool, registry, optimization_config,
                                      Path(sensitivity_temp))
                except Exception:
                    import uuid
                    failed = sensitivity_dir.parent / ('group11_control_sensitivity_failed_' + uuid.uuid4().hex[:12])
                    outputs = [(failed / p.name, p.read_bytes(), True)
                               for p in Path(sensitivity_temp).iterdir() if p.is_file()]
                    if outputs:
                        commit_files(journal_root(options.formulations_path), outputs, label='failed control sensitivity diagnostic')
                        print('Control sensitivity diagnostics: ' + str(failed))
                    raise
                sensitivity_files = {p.name: p.read_bytes() for p in Path(sensitivity_temp).iterdir() if p.is_file()}

    try:
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
    except RuntimeError as exc:
        if hasattr(exc, 'metadata'):
            failure = Path(options.output_dir).parent / 'continuous_search_failure.json'
            commit_files(journal_root(options.formulations_path),
                         [(failure, json_bytes({'error': str(exc), **exc.metadata}), False)],
                         label='continuous search failure diagnostic')
        raise
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
        if not staged_metadata.exists():
            staged_metadata.write_text(json.dumps(result.metadata, indent=2,
                default=lambda value: value.item() if hasattr(value, 'item') else str(value)) + '\n')
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
                    result.metadata.get('automatic_repeat_policy', {}).get('policy_version', ''),
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
        round_paths = round_artifact_paths(batch_id, results_root)
        frozen_outputs = [
            (round_paths.proposal_csv, staged_candidates.read_bytes(), True),
            (round_paths.proposal_summary, staged_summary.read_bytes(), True),
            (round_paths.proposal_metadata, staged_metadata.read_bytes(), True),
        ]
        if 'gp_strategy_decision' in optimization_config:
            # Preserve the established serialization of historical decision files.
            frozen_outputs.append((round_paths.proposal_dir / 'gp_strategy_decision.json',
                (json.dumps(optimization_config['gp_strategy_decision'], indent=2) + '\n').encode(), True))
        if target_round_number >= 11:
            comparison_dir = round_paths.round_dir / 'gp_comparison'
            if comparison_dir.exists():
                _validate_existing_comparison(comparison_dir, staging_root / 'gp_comparison')
                # No regenerated timestamps or forecasts replace historical evidence.
                comparison_source = comparison_dir
            else:
                comparison_source = staging_root / 'gp_comparison'
            evidence_names = set(json.loads((comparison_source / 'manifest.json').read_text())['hashes']) | {'manifest.json'}
            frozen_outputs.extend((comparison_dir / p.relative_to(comparison_source), p.read_bytes(),
                                   str(p.relative_to(comparison_source)) in evidence_names)
                                  for p in sorted(comparison_source.rglob('*')) if p.is_file())
        state_dir = round_paths.proposal_dir / 'selection_state'
        for name in ('viability_screen', 'mechanical_tests', 'candidate_pool'):
            frame = getattr(result, name)
            frozen_outputs.append((state_dir / (name + '.csv'), frame.to_csv(index=False).encode(), True))
        frozen_outputs.append((state_dir / 'total_candidate_pool.csv', staging_pool.read_bytes(), True))
        receipt = dict(version=1, inputs=selection_inputs,
                       hashes={str(p.relative_to(round_paths.round_dir)): digest_bytes(data)
                               for p, data, frozen in frozen_outputs if frozen})
        frozen_outputs.append((round_paths.proposal_dir / 'selection_commit.json', json_bytes(receipt), True))
        working_options = replace(options, supersede_unstarted_proposal=superseded_path is not None)
        working_outputs = _working_outputs(working_options, batch_id, round_paths, staged_candidates.read_bytes(),
                                          staged_summary.read_bytes(), staged_metadata.read_bytes(),
                                          staging_pool.read_bytes())
        if sensitivity_required:
            sensitivity_dir = results_root / 'group11_control_sensitivity'
            # Existing diagnostics remain immutable. Conflicting runs fail clearly.
            frozen_outputs.extend((sensitivity_dir / name, data, True) for name, data in sensitivity_files.items())
        commit_files(journal_root(options.formulations_path), [*frozen_outputs, *working_outputs],
                     inputs=input_hashes, label=f'{batch_id} selection')
        proposal_artifacts = generate_proposal_artifacts(pd.read_csv(staged_candidates), round_paths.proposal_dir)

    return _finish_selection(options, observations, batch_id, result, round_paths, proposal_artifacts, superseded_path)


def _finish_selection(options, observations, batch_id, result, round_paths, proposal_artifacts, superseded_path):
    status_path = write_current_round_status(
        Path(options.output_dir).parent / CURRENT_ROUND_STATUS_PATH.name,
        observations=observations,
        source_observations_path=options.observations_path,
        active_phase=result.metadata["active_phase"],
        phase_reason=result.metadata.get("phase_resolution", {}).get("reason", ""),
        phase_resolution=result.metadata.get("phase_resolution", {}),
        proposal_policy={
            'automatic_repeat_policy': result.metadata.get('automatic_repeat_policy', {}),
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


def _validate_existing_comparison(existing, staged):
    """Adopt pre-journal evidence only when a reproduced snapshot agrees."""
    old = json.loads((existing / 'manifest.json').read_text())
    new = json.loads((staged / 'manifest.json').read_text())
    for name, expected in old['hashes'].items():
        if file_hash(existing / name) != expected:
            raise PersistenceConflict('Frozen challenger artifact changed: ' + name)
    for key in ('batch_id', 'cutoff', 'features', 'reference_ids', 'models',
                'proposal_sha256', 'training_sha256', 'assumptions'):
        if old[key] != new[key]:
            raise PersistenceConflict('Frozen challenger inputs differ: ' + key)
    for name in old['hashes']:
        if name == 'predictions.csv':
            left = pd.read_csv(existing / name).drop(columns='created_at')
            right = pd.read_csv(staged / name).drop(columns='created_at')
            if not left.equals(right):
                raise PersistenceConflict('Reproduced challenger predictions differ from frozen evidence.')
        elif file_hash(existing / name) != file_hash(staged / name):
            raise PersistenceConflict('Reproduced challenger snapshot differs: ' + name)


def _working_outputs(options, batch_id, round_paths, candidates, summary, metadata, pool):
    working = Path(options.output_dir) / 'next_round_candidates.csv'
    preserve = False
    if working.exists() and working.stat().st_size:
        frame = pd.read_csv(working)
        batches = set(frame.get('batch_id', pd.Series(dtype=str)).dropna().astype(str))
        if batches == {batch_id} and not options.supersede_unstarted_proposal:
            # Compare immutable identity with the proposed bytes, retaining all lab edits.
            with tempfile.NamedTemporaryFile(suffix='.csv') as temporary:
                temporary.write(candidates)
                temporary.flush()
                validate_completed_against_proposal(working, temporary.name)
            preserve = True
        elif batches != {batch_id}:
            from .artifacts import _values_equivalent
            if len(batches) != 1:
                raise PersistenceConflict('Working worksheet has ambiguous batch identity; refusing to replace it.')
            previous = round_artifact_paths(next(iter(batches)), Path(options.output_dir).parent)
            if file_hash(previous.completed_csv) != file_hash(working):
                if not previous.proposal_csv.exists():
                    raise PersistenceConflict('Working worksheet has no matching frozen proposal; refusing replacement.')
                proposal = pd.read_csv(previous.proposal_csv).set_index('candidate_id')
                for _, row in frame.iterrows():
                    candidate = row.get('candidate_id')
                    if candidate not in proposal.index or any(
                        not _values_equivalent(row.get(column), proposal.loc[candidate].get(column))
                        for column in EDITABLE_WETLAB_COLUMNS
                    ):
                        raise PersistenceConflict('Working worksheet contains uncommitted lab edits; refusing replacement.')
    outputs = [
        (Path(options.output_dir) / 'next_round_summary.txt', summary, False),
        (Path(options.output_dir) / 'next_round_metadata.json', metadata, False),
        (Path(options.total_candidate_pool_path), pool, False),
    ]
    if not preserve:
        outputs.insert(0, (working, candidates, False))
    return outputs


def _publish_working(options, batch_id, round_paths, input_hashes):
    working = Path(options.output_dir) / 'next_round_candidates.csv'
    guards = {**input_hashes, str(working.resolve()): file_hash(working)}
    outputs = _working_outputs(
        options, batch_id, round_paths, round_paths.proposal_csv.read_bytes(),
        round_paths.proposal_summary.read_bytes(), round_paths.proposal_metadata.read_bytes(),
        (round_paths.proposal_dir / 'selection_state/total_candidate_pool.csv').read_bytes())
    # Only restore missing/stale mutable exports; frozen artifacts are never regenerated.
    outputs = [(p, data, frozen) for p, data, frozen in outputs if file_hash(p) != digest_bytes(data)]
    if outputs:
        commit_files(journal_root(options.formulations_path), outputs, inputs=guards,
                     label=f'{batch_id} restore working exports')

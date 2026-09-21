#!/usr/bin/env python3
"""Validate, ingest, report, and advance one v2 wet-lab round."""

from __future__ import annotations

import argparse
import json
import io
import math
import pandas as pd
import re
import subprocess
import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = V2_ROOT.parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.paths import (
    AVAILABILITY_CONFIG,
    EVALUATION_CONFIG,
    FORMULATIONS_PATH,
    OBSERVATIONS_PATH,
    NEXT_ROUND_CANDIDATES_PATH,
    NEXT_ROUND_SUMMARY_PATH,
    portable_source_path,
    RESULTS_V2_DIR,
    TOTAL_CANDIDATE_POOL_PATH,
)
from helper.artifacts import (
    assert_completed_archive_compatible,
    round_artifact_paths,
    validate_completed_against_proposal,
    validate_no_unconfirmed_carried_results,
)
from helper.config import (
    load_evaluation_config,
    load_optimization_config,
    nested_get,
)
from helper.feedback import ingest_feedback, validate_numeric_feedback
from helper.finalization import finalization_manifest
from helper.persistence import (writer_lock, recover_transactions, journal_root, commit_files,
                                file_hash, json_bytes, PersistenceConflict)
from helper.group10_config import (
    Group10HardStop,
    assert_frozen_group10_protocol,
    load_group10_config_for_round,
    load_round_mechanical_endpoint,
)
from helper.mechanics_execution import (
    validate_mechanics_execution,
)
from helper.phase import resolve_phase_mode
from helper.prospective_evaluation import (
    generate_campaign_prospective_artifacts,
    generate_round_prospective_artifacts,
)
from helper.registry import load_registry
from helper.status import write_current_round_status
from helper.visualization import generate_completed_round_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidates_csv",
        nargs="?",
        default=str(NEXT_ROUND_CANDIDATES_PATH),
        help=(
            "Filled next_round_candidates.csv. Defaults to "
            f"{NEXT_ROUND_CANDIDATES_PATH} if omitted."
        ),
    )
    parser.add_argument("--batch-id", default=None, help="Optional batch/round override.")
    parser.add_argument("--batch-date", default="", help="Optional batch date.")
    parser.add_argument("--candidate-file", action="append", default=None, help="Candidate CSV lookup(s).")
    parser.add_argument("--formulations", default=str(FORMULATIONS_PATH))
    parser.add_argument("--observations", default=str(OBSERVATIONS_PATH))
    parser.add_argument("--default-needles-compressed", type=int, default=None)
    parser.add_argument("--viability-noise", type=float, default=None)
    parser.add_argument("--availability-config", default=str(AVAILABILITY_CONFIG))
    parser.add_argument("--evaluation-config", default=str(EVALUATION_CONFIG))
    parser.add_argument("--output-dir", default=str(RESULTS_V2_DIR / "next_round"))
    parser.add_argument("--total-candidate-pool", default=str(TOTAL_CANDIDATE_POOL_PATH))
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--phase-mode",
        choices=[
            "auto",
            "screening_only",
            "mechanics_bootstrap",
            "mechanics_hybrid",
            "mechanics_enabled",
        ],
        default=None,
        help="Optional phase override for debugging/audits. Default behavior is automatic.",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip post-ingestion round reports (advanced/debug use only).",
    )
    parser.add_argument("--gp-strategy-decision", default=None, help="Recorded strategy JSON passed to next proposal generation.")
    parser.add_argument("--skip-generate", action="store_true", help="Skip Stage 02 candidate generation.")
    parser.add_argument('--finalize', action='store_true',
                        help='Attest that all planned measurements are finished and commit this round.')
    parser.add_argument('--unmeasured-reason', default='',
                        help='Document why the listed unresolved outcomes will remain unmeasured.')
    parser.add_argument(
        '--validate-only', action='store_true',
        help=(
            'Validate and parse the completed worksheet in memory, then exit before '
            'archival, canonical table writes, reports, candidate generation, or round advancement.'
        ),
    )
    parser.add_argument(
        '--validation-output-dir', type=Path, default=None,
        help='Optional noncanonical directory for mechanical validation-preview plots; requires --validate-only.',
    )
    parser.add_argument(
        "--supersede-unstarted-proposal",
        action="store_true",
        help=(
            "Forward the guarded Stage-02 option that archives and replaces "
            "an existing unstarted proposal."
        ),
    )
    return parser.parse_args()


def _run(script_path: Path, extra_args: list[str]) -> None:
    command = [sys.executable, str(script_path), *extra_args]
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _is_blank(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _read_or_empty(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


# Columns that helper.feedback.ingest_feedback reads to produce new
# observation rows. If none of these are filled in across every row of the
# feedback CSV, the round has not actually progressed yet (no new wet-lab
# results were recorded) -- see helper/feedback.py for the parsing logic
# that consumes each of these.
RESULT_COLUMNS = (
    "mechanical_test_attempted",
    "supplementary_analysis_file",
    "viability_percent",
    "intact_patch_formation_pass",
    "intact_tip_count",
    "preparation_feasibility_pass",
    "homogeneous_solution_pass",
    "fillability_pass",
    "preparation_failure_reason",
    "instron_file",
    "critical_axial_load_N_per_needle",
    "critical_axial_load_N_total",
    "initial_stiffness_N_per_mm_per_needle",
)

# viability_percent is the one result column that next_round_candidates.csv
# can legitimately pre-fill on generation: retest_priority rows carry the
# formulation's prior observed viability forward as context for the person
# re-running the test (see helper/retest.py). That carried-over value is not
# a new wet-lab result, so it must not by itself count as round progress.
_CARRIED_OVER_RECOMMENDATION_TYPES = ("retest_priority",)


def _round_has_new_results(feedback_path: str | Path) -> bool:
    """Check whether the feedback CSV has any filled-in result columns.

    Returns False (round not progressed) if the file is missing/empty, has
    none of the known result columns, or every result column is blank in
    every row -- after discounting viability_percent values that
    next_round_candidates.csv pre-fills for retest_priority rows as
    historical context rather than as a new observation. In that case
    run_round.py should not snapshot or ingest, but should still regenerate
    candidates from the current state.
    """
    path = Path(feedback_path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    feedback = pd.read_csv(path)
    present_columns = [column for column in RESULT_COLUMNS if column in feedback.columns]
    if not present_columns:
        return False

    is_carried_over_row = (
        feedback["recommendation_type"].isin(_CARRIED_OVER_RECOMMENDATION_TYPES)
        if "recommendation_type" in feedback.columns
        else pd.Series(False, index=feedback.index)
    )

    for column in present_columns:
        blank_mask = feedback[column].map(_is_blank)
        if column == "viability_percent":
            # A filled viability_percent on a retest_priority row is just the
            # carried-over historical value, not a new result; only count it
            # as "filled" for rows that are not retest_priority.
            blank_mask = blank_mask | is_carried_over_row
        if not blank_mask.all():
            return True
    return False


def _resolve_current_summary_path(candidates_csv: str | Path) -> Path:
    candidate_path = Path(candidates_csv)
    sibling_summary = candidate_path.with_name("next_round_summary.txt")
    if sibling_summary.exists():
        return sibling_summary
    return NEXT_ROUND_SUMMARY_PATH


def _resolve_batch_id(candidates_csv: str | Path, cli_batch_id: str | None) -> str:
    if not _is_blank(cli_batch_id):
        return str(cli_batch_id).strip()

    results = pd.read_csv(candidates_csv)
    if "batch_id" not in results.columns:
        raise SystemExit("Provide --batch-id or fill the batch_id column in next_round_candidates.csv.")

    values = [
        str(value).strip()
        for value in results["batch_id"].dropna().tolist()
        if str(value).strip() != ""
    ]
    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        return unique_values[0]
    if not unique_values:
        raise SystemExit("batch_id is blank. Fill next_round_candidates.csv batch_id or pass --batch-id ROUND_ID.")
    raise SystemExit(
        "next_round_candidates.csv has multiple batch_id values. Split the file by batch or pass one --batch-id."
    )


def _resolve_viability_noise(optimization_config: dict, cli_viability_noise: float | None) -> float:
    if cli_viability_noise is not None:
        noise = float(cli_viability_noise)
    else:
        fallback_legacy_noise = float(
            nested_get(
                optimization_config,
                "transfer.legacy_wetlab_viability_noise_percent",
                nested_get(optimization_config, "transfer.wetlab_viability_noise_percent", 5.0),
            )
        )
        noise = float(
            nested_get(
                optimization_config,
                "feedback.new_viability_noise_percent",
                fallback_legacy_noise / 5.0,
            )
        )
    if not math.isfinite(noise) or noise <= 0.0:
        raise SystemExit("viability noise must be finite and > 0.")
    return noise


def main() -> None:
    args = parse_args()
    with writer_lock():
        recover_transactions(journal_root(args.formulations), read_only=args.validate_only)
        recover_transactions(Path(args.output_dir).parent / '.v2-transactions', read_only=args.validate_only)
        selection = _run_locked(args)
    # The selector takes the same lock in its own process and rereads committed tables.
    if selection is not None:
        _run(*selection)
        print(f"Round status: {(Path(args.output_dir).parent / 'current_round_status.json').resolve()}")
        print('Round progression complete.')


def _run_locked(args):
    if args.validation_output_dir is not None and not args.validate_only:
        raise SystemExit('--validation-output-dir requires --validate-only')
    select_script = V2_ROOT / "02_select_candidates" / "select_candidates.py"
    registry = load_registry()
    optimization_config = load_optimization_config()
    evaluation_config = load_evaluation_config(args.evaluation_config)
    batch_id = _resolve_batch_id(args.candidates_csv, args.batch_id)
    candidate_files = args.candidate_file or [str(args.candidates_csv or NEXT_ROUND_CANDIDATES_PATH)]
    validate_numeric_feedback(args.candidates_csv)
    input_hashes = {str(Path(p).resolve()): file_hash(p)
                    for p in [args.candidates_csv, args.formulations, args.observations, *candidate_files]}
    current_candidates = _read_or_empty(args.candidates_csv)
    raw_inputs = {}
    for column in ('instron_file', 'supplementary_analysis_file'):
        for value in current_candidates.get(column, pd.Series(dtype=str)).dropna():
            if str(value).strip():
                path = Path(str(value)).resolve()
                raw_inputs[str(path)] = file_hash(path)
    input_hashes.update(raw_inputs)
    current_formulations = _read_or_empty(args.formulations)
    current_observations = _read_or_empty(args.observations)
    results_root = Path(args.output_dir).parent
    round_paths = round_artifact_paths(batch_id, results_root)
    proposal_metadata = {}
    if round_paths.proposal_metadata.exists():
        proposal_metadata = json.loads(
            round_paths.proposal_metadata.read_text(encoding="utf-8")
        )
    round_mechanical_endpoint = load_round_mechanical_endpoint(
        round_paths.proposal_dir, batch_id
    )
    round_match = re.fullmatch(r'ROUND_(\d+)', batch_id)
    round_number = int(round_match.group(1)) if round_match else None
    if round_number is not None and round_number >= 10:
        assert_frozen_group10_protocol(
            load_group10_config_for_round(round_number),
            round_number,
            proposal_metadata,
        )

    receipt_path = round_paths.completed_dir / 'ingestion_commit.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    settings = dict(batch_id=batch_id, batch_date=args.batch_date,
                    default_needles_compressed=args.default_needles_compressed,
                    viability_noise=_resolve_viability_noise(optimization_config, args.viability_noise),
                    proposal_sha256=file_hash(round_paths.proposal_csv),
                    metadata_sha256=file_hash(round_paths.proposal_metadata),
                    mechanical_endpoint=round_mechanical_endpoint,
                    candidate_lookup_sha256=[file_hash(p) for p in candidate_files
                        if Path(p).resolve() != Path(args.candidates_csv).resolve()])
    round_progressed = _round_has_new_results(args.candidates_csv) or args.finalize or receipt is not None
    if not round_progressed:
        if args.validate_only:
            print(
                f'No new wet-lab results found in {Path(args.candidates_csv).resolve()}. '
                'Validation-only complete: no files were changed and no candidates were generated.'
            )
            return
        print(
            f"No new wet-lab results found in {Path(args.candidates_csv).resolve()}; "
            "round has not progressed. Skipping completed-sheet archival, reports, and "
            "formulations/observations ingest. Candidates will still be regenerated "
            "from the current data."
        )

    if round_progressed:
        validation = validate_completed_against_proposal(
            args.candidates_csv,
            round_paths.proposal_csv,
        )
        validate_no_unconfirmed_carried_results(
            args.candidates_csv,
            round_paths.proposal_csv,
        )
        assert_completed_archive_compatible(
            args.candidates_csv,
            round_paths.completed_csv,
        )
        print(
            "Validated completed worksheet against frozen proposal: "
            f"{validation['proposal_candidate_count']} candidates, "
            f"{validation['completed_row_count']} completed row(s)."
        )
        mechanics_audit = validate_mechanics_execution(
            current_candidates,
            pd.read_csv(round_paths.proposal_csv),
            primary_capacity=int(
                nested_get(
                    optimization_config,
                    "mechanics_transition.bootstrap.mechanical_capacity",
                    4,
                )
            ),
            proposal_metadata=proposal_metadata,
        )

    if round_progressed:
        if receipt is not None:
            if (receipt['worksheet_sha256'] != file_hash(args.candidates_csv)
                    or receipt['settings'] != settings):
                raise PersistenceConflict('Committed ingestion inputs differ; refusing to reinterpret the round.')
            if args.unmeasured_reason.strip():
                completion = json.loads((round_paths.completed_dir / 'finalization.json').read_text())
                if completion['unmeasured_reason'] != args.unmeasured_reason.strip():
                    raise PersistenceConflict('Committed unmeasured-outcome reason differs.')
            for path, expected in receipt['frozen_outputs'].items():
                if file_hash(path) != expected:
                    raise PersistenceConflict('Committed round evidence changed: ' + path)
            for path, expected in receipt.get('raw_inputs', {}).items():
                if file_hash(path) != expected:
                    raise PersistenceConflict('Committed raw input changed: ' + path)
            existing_batch = current_observations.loc[current_observations.batch_id.eq(batch_id)]
            expected_batch = pd.read_csv(io.StringIO(receipt['batch_observations_csv']))
            try:
                pd.testing.assert_frame_equal(
                    existing_batch.sort_values('observation_id').reset_index(drop=True),
                    expected_batch.sort_values('observation_id').reset_index(drop=True),
                    check_dtype=False, rtol=1e-12, atol=1e-12)
            except AssertionError as exc:
                raise PersistenceConflict('Committed batch observations changed; cannot resume.') from exc
            expected_forms = pd.read_csv(io.StringIO(receipt['formulations_csv']))
            actual_forms = current_formulations.loc[
                current_formulations.formulation_id.isin(expected_forms.formulation_id), expected_forms.columns]
            try:
                pd.testing.assert_frame_equal(
                    actual_forms.sort_values('formulation_id').reset_index(drop=True),
                    expected_forms.sort_values('formulation_id').reset_index(drop=True),
                    check_dtype=False, rtol=1e-12, atol=1e-12)
            except AssertionError as exc:
                raise PersistenceConflict('Committed formulation concentrations changed; cannot resume.') from exc
            formulations, observations = current_formulations, current_observations
            print('Identical ingestion already committed; resuming remaining workflow steps.')
        else:
            formulations, observations = ingest_feedback(
                feedback_path=args.candidates_csv,
                candidate_files=candidate_files,
                formulations=current_formulations,
                observations=current_observations,
                registry=registry,
                batch_id=batch_id,
                batch_date=args.batch_date,
                default_needles_compressed=args.default_needles_compressed,
                viability_noise=_resolve_viability_noise(optimization_config, args.viability_noise),
                observation_source_file=portable_source_path(round_paths.completed_csv),
                proposal_metadata=proposal_metadata,
                round_mechanical_endpoint=round_mechanical_endpoint,
            )

        from helper.mechanics_execution import update_endpoint_workload
        mechanics_audit = update_endpoint_workload(mechanics_audit, observations, batch_id)
        completion = None if receipt is not None else finalization_manifest(
            current_candidates, mechanics_audit, finalize=args.finalize,
            unmeasured_reason=args.unmeasured_reason, validate_only=args.validate_only)
        if args.validate_only:
            current = observations.loc[observations.batch_id.astype(str).eq(batch_id)]
            counts = current.groupby('endpoint').size().to_dict()
            print('Validation-only prospective observation counts:')
            for endpoint in sorted(counts):
                print(f'  {endpoint}: {counts[endpoint]}')
            mechanical = current.loc[current.endpoint.eq('critical_axial_load_N_per_needle')]
            if not mechanical.empty:
                print('Mechanical replicate counts by formulation:')
                for formulation_id, group in mechanical.groupby('formulation_id', sort=True):
                    print(f'  {formulation_id}: {group.replicate_id.astype(str).nunique()}')
            if args.validation_output_dir is not None:
                from helper.plot_reporting import write_mechanical
                paths = write_mechanical(
                    observations, current_candidates, args.validation_output_dir,
                    batch_id=batch_id, context=f'Validation preview for {batch_id}',
                )
                print(
                    f'Generated {len(paths)} noncanonical mechanical validation artifact(s): '
                    f'{args.validation_output_dir.resolve()}'
                )
            print('Validation-only complete: no canonical files were changed and the round was not advanced.')
            return
        if receipt is None:
            from helper.persistence import digest_bytes
            batch_observations = observations.loc[observations.batch_id.eq(batch_id)]
            if batch_observations.empty:
                raise ValueError('No observations to commit; all-unmeasured round advancement needs an explicit campaign decision.')
            frozen_outputs = [
                (round_paths.completed_csv, Path(args.candidates_csv).read_bytes(), True),
                (round_paths.completed_dir / 'mechanical_execution_manifest.json', json_bytes(mechanics_audit), True),
                (round_paths.completed_dir / 'finalization.json', json_bytes(completion), True),
            ]
            receipt = dict(version=1, worksheet_sha256=file_hash(args.candidates_csv), settings=settings,
                           raw_inputs=raw_inputs,
                           batch_observations_csv=batch_observations.to_csv(index=False),
                           formulations_csv=formulations.loc[
                               formulations.formulation_id.isin(batch_observations.formulation_id),
                               ['formulation_id', *registry.feature_names]].to_csv(index=False),
                           frozen_outputs={str(p.resolve()): digest_bytes(data) for p, data, _ in frozen_outputs})
            commit_files(journal_root(args.formulations), [
                *frozen_outputs,
                (Path(args.formulations), formulations.to_csv(index=False).encode(), False),
                (Path(args.observations), observations.to_csv(index=False).encode(), False),
                (receipt_path, json_bytes(receipt), True),
            ], inputs={**input_hashes, **raw_inputs}, label=f'{batch_id} ingestion')
            print(f'Committed completed worksheet and database: {round_paths.completed_csv.resolve()}')
        if not args.skip_review:
            report_paths = generate_completed_round_artifacts(
                formulations,
                observations,
                current_candidates,
                round_paths.reports_dir,
                batch_id=batch_id,
            )
            print(
                f"Generated {len(report_paths)} completed-round report file(s): "
                f"{round_paths.reports_dir.resolve()}"
            )
            round_prospective_paths = generate_round_prospective_artifacts(
                batch_id,
                observations,
                results_root=results_root,
                evaluation_config=evaluation_config,
            )
            print(
                f"Generated {len(round_prospective_paths)} round prospective "
                f"evaluation file(s): {round_paths.reports_dir.resolve()}"
            )
            campaign_prospective_paths = generate_campaign_prospective_artifacts(
                observations,
                results_root=results_root,
                evaluation_config=evaluation_config,
            )
            print(
                f"Generated {len(campaign_prospective_paths)} campaign prospective "
                f"evaluation file(s): "
                f"{(results_root / 'reports' / 'prospective').resolve()}"
            )
    else:
        formulations, observations = current_formulations, current_observations

    # Score immutable challenger forecasts before any next-round refit/generation.
    if round_progressed:
        from helper.gp_comparison import evaluate
        comparison_root = Path(args.output_dir).parent / 'rounds' / str(batch_id)
        frozen_comparison = comparison_root / 'gp_comparison'
        comparison_evaluation = comparison_root / 'reports' / 'gp_comparison'
        if frozen_comparison.exists():
            evaluate(frozen_comparison, observations, comparison_evaluation)

    status_path = Path(args.output_dir).parent / "current_round_status.json"
    if args.skip_generate:
        phase_resolution = resolve_phase_mode(
            formulations,
            observations,
            registry,
            optimization_config,
            requested_phase_mode=args.phase_mode,
        )
        status_path = write_current_round_status(
            status_path,
            observations=observations,
            source_observations_path=args.observations,
            active_phase=phase_resolution.active_phase,
            phase_reason=phase_resolution.reason,
            phase_resolution=phase_resolution.to_metadata(),
        )

    if not args.skip_generate:
        select_args = [
            "--formulations",
            args.formulations,
            "--observations",
            args.observations,
            "--availability-config",
            args.availability_config,
            "--output-dir",
            args.output_dir,
            "--total-candidate-pool",
            args.total_candidate_pool,
        ]
        if args.gp_strategy_decision:
            select_args.extend(["--gp-strategy-decision", args.gp_strategy_decision])
        if args.pool_size is not None:
            select_args.extend(["--pool-size", str(args.pool_size)])
        if args.seed is not None:
            select_args.extend(["--seed", str(args.seed)])
        if args.phase_mode is not None:
            select_args.extend(["--phase-mode", args.phase_mode])
        if args.supersede_unstarted_proposal:
            select_args.append("--supersede-unstarted-proposal")
        return select_script, select_args

    print(f"Round status: {status_path.resolve()}")
    print("Round progression complete.")


if __name__ == "__main__":
    try:
        main()
    except Group10HardStop as exc:
        raise SystemExit(str(exc)) from None

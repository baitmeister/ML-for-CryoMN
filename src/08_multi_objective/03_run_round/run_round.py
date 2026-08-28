#!/usr/bin/env python3
"""Validate, ingest, report, and advance one v2 wet-lab round."""

from __future__ import annotations

import argparse
import pandas as pd
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
    RESULTS_V2_DIR,
    TOTAL_CANDIDATE_POOL_PATH,
)
from helper.artifacts import (
    archive_completed,
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
from helper.feedback import ingest_feedback
from helper.mechanics_execution import (
    freeze_mechanics_execution_manifest,
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
        choices=["auto", "screening_only", "mechanics_enabled"],
        default=None,
        help="Optional phase override for debugging/audits. Default behavior is automatic.",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip post-ingestion round reports (advanced/debug use only).",
    )
    parser.add_argument("--skip-generate", action="store_true", help="Skip Stage 02 candidate generation.")
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


def _portable_source_path(path: str | Path) -> str:
    """Return a stable repository-relative path when the artifact is local."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


# Columns that helper.feedback.ingest_feedback reads to produce new
# observation rows. If none of these are filled in across every row of the
# feedback CSV, the round has not actually progressed yet (no new wet-lab
# results were recorded) -- see helper/feedback.py for the parsing logic
# that consumes each of these.
RESULT_COLUMNS = (
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
    if noise <= 0.0:
        raise SystemExit("viability noise must be > 0.")
    return noise


def main() -> None:
    args = parse_args()
    select_script = V2_ROOT / "02_select_candidates" / "select_candidates.py"
    registry = load_registry()
    optimization_config = load_optimization_config()
    evaluation_config = load_evaluation_config(args.evaluation_config)
    batch_id = _resolve_batch_id(args.candidates_csv, args.batch_id)
    candidate_files = args.candidate_file or [str(args.candidates_csv or NEXT_ROUND_CANDIDATES_PATH)]
    current_candidates = _read_or_empty(args.candidates_csv)
    current_formulations = _read_or_empty(args.formulations)
    current_observations = _read_or_empty(args.observations)
    results_root = Path(args.output_dir).parent
    round_paths = round_artifact_paths(batch_id, results_root)

    round_progressed = _round_has_new_results(args.candidates_csv)
    if not round_progressed:
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
                    "intact_combination_policy.mechanics_primary_test_count",
                    nested_get(
                        optimization_config,
                        "round_policy.mechanical_tests_per_round",
                        4,
                    ),
                )
            ),
        )

    if round_progressed:
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
            observation_source_file=_portable_source_path(round_paths.completed_csv),
        )

        completed_path = archive_completed(
            batch_id,
            args.candidates_csv,
            results_root=results_root,
        )
        print(f"Archived completed worksheet: {completed_path.resolve()}")
        Path(args.formulations).parent.mkdir(parents=True, exist_ok=True)
        formulations.to_csv(args.formulations, index=False)
        observations.to_csv(args.observations, index=False)
        print(f"Updated formulations: {Path(args.formulations).resolve()} ({len(formulations)} rows)")
        print(f"Updated observations: {Path(args.observations).resolve()} ({len(observations)} rows)")
        mechanics_manifest = freeze_mechanics_execution_manifest(
            mechanics_audit,
            round_paths.completed_dir / "mechanical_execution_manifest.json",
        )
        print(f"Archived mechanical execution audit: {mechanics_manifest.resolve()}")
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
        if args.pool_size is not None:
            select_args.extend(["--pool-size", str(args.pool_size)])
        if args.seed is not None:
            select_args.extend(["--seed", str(args.seed)])
        if args.phase_mode is not None:
            select_args.extend(["--phase-mode", args.phase_mode])
        if args.supersede_unstarted_proposal:
            select_args.append("--supersede-unstarted-proposal")
        _run(select_script, select_args)

    print(f"Round status: {status_path.resolve()}")
    print("Round progression complete.")


if __name__ == "__main__":
    main()

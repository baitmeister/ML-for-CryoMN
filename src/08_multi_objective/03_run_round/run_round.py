#!/usr/bin/env python3
"""Advance one v2 wet-lab round with a pre-update review, ingest, and next-slate generation."""

from __future__ import annotations

import argparse
import pandas as pd
import shutil
import subprocess
import sys
from pathlib import Path


V2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = V2_ROOT.parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.paths import (
    AVAILABILITY_CONFIG,
    FORMULATIONS_PATH,
    OBSERVATIONS_PATH,
    NEXT_ROUND_CANDIDATES_PATH,
    NEXT_ROUND_SUMMARY_PATH,
    ROUND_REVIEW_DIR,
    RESULTS_V2_DIR,
    TOTAL_CANDIDATE_POOL_PATH,
)
from helper.config import load_optimization_config, nested_get
from helper.feedback import ingest_feedback
from helper.phase import resolve_phase_mode
from helper.registry import load_registry
from helper.status import write_current_round_status
from helper.visualization import generate_visualization_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_csv", help="Filled next_round_candidates.csv.")
    parser.add_argument("--batch-id", default=None, help="Optional batch/round override.")
    parser.add_argument("--batch-date", default="", help="Optional batch date.")
    parser.add_argument("--candidate-file", action="append", default=None, help="Candidate CSV lookup(s).")
    parser.add_argument("--formulations", default=str(FORMULATIONS_PATH))
    parser.add_argument("--observations", default=str(OBSERVATIONS_PATH))
    parser.add_argument("--default-needles-compressed", type=int, default=None)
    parser.add_argument("--viability-noise", type=float, default=None)
    parser.add_argument("--availability-config", default=str(AVAILABILITY_CONFIG))
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
    parser.add_argument("--skip-review", action="store_true", help="Skip the pre-update round review.")
    parser.add_argument("--skip-generate", action="store_true", help="Skip Stage 02 candidate generation.")
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


def _copy_if_present(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    if not source_path.exists() or source_path.stat().st_size == 0:
        return
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)


def _archived_name(batch_id: str, base_name: str, suffix: str) -> str:
    return f"{batch_id}_{base_name}{suffix}"


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
    batch_id = _resolve_batch_id(args.candidates_csv, args.batch_id)
    candidate_files = args.candidate_file or [str(args.candidates_csv or NEXT_ROUND_CANDIDATES_PATH)]
    current_candidates = _read_or_empty(args.candidates_csv)
    current_formulations = _read_or_empty(args.formulations)
    current_observations = _read_or_empty(args.observations)

    round_progressed = _round_has_new_results(args.candidates_csv)
    if not round_progressed:
        print(
            f"No new wet-lab results found in {Path(args.candidates_csv).resolve()}; "
            "round has not progressed. Skipping round-review snapshot and "
            "formulations/observations ingest. Candidates will still be regenerated "
            "from the current data."
        )

    if round_progressed and not args.skip_review:
        review_dir = ROUND_REVIEW_DIR / batch_id
        pre_paths = generate_visualization_artifacts(
            current_formulations,
            current_observations,
            current_candidates,
            review_dir,
            review_label=f"selection_state_before_update_{batch_id}",
            artifact_prefix=batch_id,
        )
        _copy_if_present(
            args.candidates_csv,
            review_dir / _archived_name(batch_id, "next_round_candidates", ".csv"),
        )
        _copy_if_present(
            _resolve_current_summary_path(args.candidates_csv),
            review_dir / _archived_name(batch_id, "next_round_summary", ".txt"),
        )
        _copy_if_present(
            args.total_candidate_pool,
            review_dir / _archived_name(batch_id, "total_candidate_pool", ".csv"),
        )
        print(f"Generated {len(pre_paths)} round review file(s): {review_dir.resolve()}")

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
        )

        Path(args.formulations).parent.mkdir(parents=True, exist_ok=True)
        formulations.to_csv(args.formulations, index=False)
        observations.to_csv(args.observations, index=False)
        print(f"Updated formulations: {Path(args.formulations).resolve()} ({len(formulations)} rows)")
        print(f"Updated observations: {Path(args.observations).resolve()} ({len(observations)} rows)")
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
        _run(select_script, select_args)

    print(f"Round status: {status_path.resolve()}")
    print("Round progression complete.")


if __name__ == "__main__":
    main()

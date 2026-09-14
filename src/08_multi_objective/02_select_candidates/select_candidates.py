#!/usr/bin/env python3
"""Select the next v2 CryoMN viability screen and mechanical test subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.candidate_workflow import (
    CandidateSelectionOptions,
    CandidateSelectionWorkflowResult,
    run_candidate_selection,
)
from helper.group10_config import Group10HardStop
from helper.paths import (
    AVAILABILITY_CONFIG,
    FORMULATIONS_PATH,
    OBSERVATIONS_PATH,
    RESULTS_V2_DIR,
    TOTAL_CANDIDATE_POOL_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulations", default=str(FORMULATIONS_PATH), help="v2 formulations table.")
    parser.add_argument("--observations", default=str(OBSERVATIONS_PATH), help="v2 observations table.")
    parser.add_argument("--candidate-pool", default=None, help="Optional CSV candidate pool to score.")
    parser.add_argument(
        "--availability-config",
        default=str(AVAILABILITY_CONFIG),
        help="YAML file listing temporarily unavailable ingredients to exclude from this selection run.",
    )
    parser.add_argument("--output-dir", default=str(RESULTS_V2_DIR / "next_round"), help="Output directory.")
    parser.add_argument(
        "--total-candidate-pool",
        default=str(TOTAL_CANDIDATE_POOL_PATH),
        help="Output CSV for the full generated/scored candidate pool.",
    )
    parser.add_argument("--pool-size", type=int, default=None, help="Generated candidate pool size.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generated pool.")
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
        help="Optional override for the automatic selection phase.",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Wet-lab round ID to prefill in next_round_candidates.csv. Defaults to next ROUND_###.",
    )
    parser.add_argument(
        "--supersede-unstarted-proposal",
        action="store_true",
        help=(
            "Explicitly archive and replace an existing frozen proposal only "
            "when the batch has no observations/completed artifact and the "
            "active worksheet has no entered wet-lab results."
        ),
    )
    return parser.parse_args()


def _present_completion(
    args: argparse.Namespace,
    workflow: CandidateSelectionWorkflowResult,
) -> None:
    """Print the existing command-line completion summary."""
    result = workflow.selection_result
    metadata = result.metadata
    if workflow.superseded_proposal_path is not None:
        print(
            "Archived superseded unstarted proposal: "
            f"{workflow.superseded_proposal_path.resolve()}"
        )
    print(f"Selected {len(result.viability_screen)} viability-screen candidates.")
    print(
        "Selected "
        f"{int(metadata.get('mechanical_test_count', 0))} primary "
        "mechanical-test candidate(s)."
    )
    print(f"Batch ID: {workflow.round_id}")
    print(f"Active phase: {workflow.resolved_phase}")
    policy_start = int(metadata.get("formulation_feasibility_policy_start_round", 0))
    print(
        "Formulation feasibility policy: "
        f"{metadata.get('formulation_feasibility_policy_version', '')} "
        f"({'active' if metadata.get('formulation_feasibility_policy_active') else 'inactive'}, "
        f"starts ROUND_{policy_start:03d})"
    )
    similarity = metadata.get("formulation_similarity", {})
    similarity_start = int(similarity.get("start_round", 0))
    print(
        "Formulation similarity policy: "
        f"{similarity.get('policy_version', '')} "
        f"({'active' if similarity.get('active') else 'inactive'}, "
        f"starts ROUND_{similarity_start:03d})"
    )
    print(
        "Mechanical selection mode: "
        f"{metadata['mechanical_policy']['mechanical_selection_mode']}"
    )
    unavailable = metadata.get("temporary_unavailable_features", [])
    if unavailable:
        print("Temporary ingredient restrictions: " + ", ".join(unavailable))
    bounds_count = int(metadata.get("candidate_pool_rows_filtered_by_bounds", 0))
    availability_count = int(
        metadata.get("candidate_pool_rows_filtered_by_availability", 0)
    )
    zero_count = int(
        metadata.get("candidate_pool_rows_filtered_zero_active_at_entry", 0)
    )
    if bounds_count:
        print(
            f"Filtered {bounds_count} externally supplied candidate-pool rows "
            "by registry bounds."
        )
    if availability_count:
        print(
            f"Filtered {availability_count} externally supplied candidate-pool "
            "rows by availability."
        )
    if zero_count:
        print(
            "WARNING: "
            f"filtered {zero_count} zero-active candidate-pool rows before scoring. "
            "This usually means the supplied candidate pool or upstream "
            "candidate-generation logic needs review."
        )
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    print(f"Total candidate pool: {Path(args.total_candidate_pool).resolve()}")
    print(f"Round status: {workflow.artifact_paths['round_status'].resolve()}")


def main() -> None:
    args = parse_args()
    options = CandidateSelectionOptions(
        formulations_path=args.formulations,
        observations_path=args.observations,
        candidate_pool_path=args.candidate_pool,
        availability_config_path=args.availability_config,
        output_dir=args.output_dir,
        total_candidate_pool_path=args.total_candidate_pool,
        pool_size=args.pool_size,
        seed=args.seed,
        phase_mode=args.phase_mode,
        batch_id=args.batch_id,
        supersede_unstarted_proposal=args.supersede_unstarted_proposal,
    )
    workflow = run_candidate_selection(options)
    _present_completion(args, workflow)


if __name__ == "__main__":
    try:
        main()
    except Group10HardStop as exc:
        raise SystemExit(str(exc)) from None

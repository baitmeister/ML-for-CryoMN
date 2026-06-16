#!/usr/bin/env python3
"""Select the next v2 CryoMN viability screen and mechanical test subset."""

from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path

import pandas as pd

V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.candidates import (
    filter_available_candidate_pool,
    filter_candidate_pool_to_registry_bounds,
    filter_nonzero_active_candidate_pool,
    generate_random_candidate_pool,
    generate_rescue_candidate_pool,
    generate_support_aware_candidate_pool,
    load_candidate_pool,
    unavailable_features_from_config,
)
from helper.config import load_availability_config, load_optimization_config, nested_get
from helper.feasibility import (
    annotate_feasibility,
    annotate_support,
    build_support_context,
    policy_activation,
)
from helper.paths import (
    AVAILABILITY_CONFIG,
    CURRENT_ROUND_STATUS_PATH,
    FORMULATIONS_PATH,
    OBSERVATIONS_PATH,
    RESULTS_V2_DIR,
    TOTAL_CANDIDATE_POOL_PATH,
)
from helper.registry import load_registry
from helper.selection import select_next_round, write_selection_result
from helper.status import derive_round_tracker, write_current_round_status
from helper.status import parse_round_number


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
        choices=["auto", "screening_only", "mechanics_enabled"],
        default=None,
        help="Optional override for the automatic selection phase.",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Wet-lab round ID to prefill in next_round_candidates.csv. Defaults to next ROUND_###.",
    )
    return parser.parse_args()


def _read_or_empty(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def _next_round_id(observations_path: str | Path = OBSERVATIONS_PATH) -> str:
    observations = _read_or_empty(observations_path)
    return str(derive_round_tracker(observations)["next_round_id"])


def main() -> None:
    args = parse_args()
    registry = load_registry()
    optimization_config = load_optimization_config()
    availability_config = load_availability_config(args.availability_config)
    unavailable_features = unavailable_features_from_config(availability_config, registry)
    formulations = _read_or_empty(args.formulations)
    observations = _read_or_empty(args.observations)
    batch_id = args.batch_id or _next_round_id(args.observations)
    target_round_number = parse_round_number(batch_id)
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

    if args.candidate_pool:
        candidate_pool = load_candidate_pool(args.candidate_pool, registry)
        before_filter = len(candidate_pool)
        candidate_pool = filter_candidate_pool_to_registry_bounds(candidate_pool, registry)
        bounds_filtered_count = before_filter - len(candidate_pool)
        candidate_pool = filter_available_candidate_pool(candidate_pool, unavailable_features)
        if candidate_pool.empty:
            raise SystemExit(
                "Candidate pool is empty after applying registry bounds and temporary availability restrictions."
            )
        filtered_count = before_filter - bounds_filtered_count - len(candidate_pool)
    else:
        pool_size = args.pool_size or int(
            nested_get(optimization_config, "selection.generated_candidate_pool_size", 2000)
        )
        seed = args.seed if args.seed is not None else int(optimization_config.get("random_seed", 42))
        rescue_candidate_count = 0
        if policy_active:
            candidate_pool = generate_support_aware_candidate_pool(
                registry,
                formulations=formulations,
                optimization_config=optimization_config,
                support=support_context,
                n_candidates=pool_size,
                random_seed=seed,
                unavailable_feature_names=unavailable_features,
            )
            rescue_candidates = generate_rescue_candidate_pool(
                registry,
                formulations=formulations,
                observations=observations,
                optimization_config=optimization_config,
                support=support_context,
                unavailable_feature_names=unavailable_features,
            )
            rescue_candidate_count = int(len(rescue_candidates))
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
    if args.candidate_pool:
        rescue_candidate_count = 0

    if policy_active:
        candidate_pool = annotate_feasibility(
            candidate_pool,
            registry,
            optimization_config,
            policy_active=True,
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

    before_zero_active_filter = len(candidate_pool)
    candidate_pool = filter_nonzero_active_candidate_pool(candidate_pool, registry)
    zero_active_filtered_count = before_zero_active_filter - len(candidate_pool)
    if candidate_pool.empty:
        raise SystemExit(
            "Candidate pool is empty after removing zero-active formulations."
        )

    result = select_next_round(
        formulations=formulations,
        observations=observations,
        candidate_pool=candidate_pool,
        registry=registry,
        optimization_config=optimization_config,
        requested_phase_mode=args.phase_mode,
        target_round_number=target_round_number,
        policy_active=policy_active,
    )
    if not rejected_candidates.empty:
        rejected_candidates["selected_for_viability_screen"] = False
        rejected_candidates["selected_for_mechanical_test"] = False
        rejected_candidates["selection_rank"] = ""
        rejected_candidates["mechanics_phase_score"] = float("nan")
        rejected_candidates["screening_phase_score"] = float("nan")
        result = replace(
            result,
            candidate_pool=pd.concat(
                [result.candidate_pool, rejected_candidates],
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
    write_selection_result(
        result,
        args.output_dir,
        batch_id=batch_id,
        total_candidate_pool_path=args.total_candidate_pool,
        registry=registry,
    )
    status_path = write_current_round_status(
        Path(args.output_dir).parent / CURRENT_ROUND_STATUS_PATH.name,
        observations=observations,
        source_observations_path=args.observations,
        active_phase=result.metadata["active_phase"],
        phase_reason=result.metadata.get("phase_resolution", {}).get("reason", ""),
        proposed_batch_id=batch_id,
        proposed_batch_override_used=args.batch_id is not None,
    )
    print(f"Selected {len(result.viability_screen)} viability-screen candidates.")
    print(f"Selected {len(result.mechanical_tests)} mechanical-test candidates.")
    print(f"Batch ID: {batch_id}")
    print(f"Active phase: {result.metadata['active_phase']}")
    print(
        "Formulation feasibility policy: "
        f"{policy_version} ({'active' if policy_active else 'inactive'}, starts ROUND_{policy_start_round:03d})"
    )
    print(f"Mechanical selection mode: {result.metadata['mechanical_policy']['mechanical_selection_mode']}")
    if unavailable_features:
        print("Temporary ingredient restrictions: " + ", ".join(unavailable_features))
    if bounds_filtered_count:
        print(f"Filtered {bounds_filtered_count} externally supplied candidate-pool rows by registry bounds.")
    if filtered_count:
        print(f"Filtered {filtered_count} externally supplied candidate-pool rows by availability.")
    if zero_active_filtered_count:
        print(
            "WARNING: "
            f"filtered {zero_active_filtered_count} zero-active candidate-pool rows before scoring. "
            "This usually means the supplied candidate pool or upstream candidate-generation logic needs review."
        )
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    print(f"Total candidate pool: {Path(args.total_candidate_pool).resolve()}")
    print(f"Round status: {status_path.resolve()}")


if __name__ == "__main__":
    main()

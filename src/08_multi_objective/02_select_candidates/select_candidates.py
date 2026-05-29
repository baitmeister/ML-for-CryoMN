#!/usr/bin/env python3
"""Select the next v2 CryoMN viability screen and mechanical test subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.candidates import (
    filter_available_candidate_pool,
    generate_random_candidate_pool,
    load_candidate_pool,
    unavailable_features_from_config,
)
from helper.config import load_availability_config, load_optimization_config, nested_get
from helper.paths import (
    AVAILABILITY_CONFIG,
    FORMULATIONS_PATH,
    OBSERVATIONS_PATH,
    RESULTS_V2_DIR,
    TOTAL_CANDIDATE_POOL_PATH,
)
from helper.registry import load_registry
from helper.selection import select_next_round, write_selection_result


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
    if observations.empty or "batch_id" not in observations.columns:
        return "ROUND_001"
    max_round = 0
    for value in observations["batch_id"].dropna().astype(str):
        value = value.strip()
        if not value.startswith("ROUND_"):
            continue
        suffix = value.removeprefix("ROUND_")
        if suffix.isdigit():
            max_round = max(max_round, int(suffix))
    return f"ROUND_{max_round + 1:03d}"


def main() -> None:
    args = parse_args()
    registry = load_registry()
    optimization_config = load_optimization_config()
    availability_config = load_availability_config(args.availability_config)
    unavailable_features = unavailable_features_from_config(availability_config, registry)
    formulations = _read_or_empty(args.formulations)
    observations = _read_or_empty(args.observations)

    if formulations.empty:
        raise SystemExit(
            "No v2 formulations were found. Run: python3 src/08_multi_objective/01_build_database/build_database.py"
        )

    if args.candidate_pool:
        candidate_pool = load_candidate_pool(args.candidate_pool, registry)
        before_filter = len(candidate_pool)
        candidate_pool = filter_available_candidate_pool(candidate_pool, unavailable_features)
        if candidate_pool.empty:
            raise SystemExit(
                "Candidate pool is empty after applying temporary availability restrictions."
            )
        filtered_count = before_filter - len(candidate_pool)
    else:
        pool_size = args.pool_size or int(
            nested_get(optimization_config, "selection.generated_candidate_pool_size", 2000)
        )
        seed = args.seed if args.seed is not None else int(optimization_config.get("random_seed", 42))
        candidate_pool = generate_random_candidate_pool(
            registry,
            n_candidates=pool_size,
            random_seed=seed,
            unavailable_feature_names=unavailable_features,
        )
        filtered_count = 0

    result = select_next_round(
        formulations=formulations,
        observations=observations,
        candidate_pool=candidate_pool,
        registry=registry,
        optimization_config=optimization_config,
    )
    batch_id = args.batch_id or _next_round_id(args.observations)
    result.metadata["temporary_unavailable_features"] = unavailable_features
    result.metadata["candidate_pool_rows_filtered_by_availability"] = filtered_count
    write_selection_result(
        result,
        args.output_dir,
        batch_id=batch_id,
        total_candidate_pool_path=args.total_candidate_pool,
    )
    print(f"Selected {len(result.viability_screen)} viability-screen candidates.")
    print(f"Selected {len(result.mechanical_tests)} mechanical-test candidates.")
    print(f"Batch ID: {batch_id}")
    print(f"Mechanical selection mode: {result.metadata['mechanical_policy']['mechanical_selection_mode']}")
    if unavailable_features:
        print("Temporary ingredient restrictions: " + ", ".join(unavailable_features))
    if filtered_count:
        print(f"Filtered {filtered_count} externally supplied candidate-pool rows by availability.")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    print(f"Total candidate pool: {Path(args.total_candidate_pool).resolve()}")


if __name__ == "__main__":
    main()

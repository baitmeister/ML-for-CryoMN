#!/usr/bin/env python3
"""Update v2 tables from one wet-lab feedback CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.config import load_optimization_config, nested_get
from helper.feedback import ingest_feedback
from helper.paths import FORMULATIONS_PATH, OBSERVATIONS_PATH, RESULTS_V2_DIR
from helper.registry import load_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates_csv", help="Filled next_round_candidates.csv.")
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Batch/round identifier. Optional if next_round_candidates.csv has one non-empty batch_id value.",
    )
    parser.add_argument("--batch-date", default="", help="Optional batch date.")
    parser.add_argument(
        "--candidate-file",
        action="append",
        default=None,
        help="Candidate output CSV. May be provided multiple times.",
    )
    parser.add_argument("--formulations", default=str(FORMULATIONS_PATH))
    parser.add_argument("--observations", default=str(OBSERVATIONS_PATH))
    parser.add_argument("--default-needles-compressed", type=int, default=None)
    parser.add_argument(
        "--viability-noise",
        type=float,
        default=None,
        help="Override the observation_noise assigned to new viability_percent rows.",
    )
    return parser.parse_args()


def _is_blank(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _read_or_empty(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


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
    registry = load_registry()
    optimization_config = load_optimization_config()
    batch_id = _resolve_batch_id(args.candidates_csv, args.batch_id)
    candidate_files = args.candidate_file or [
        str(RESULTS_V2_DIR / "next_round" / "next_round_candidates.csv"),
    ]

    formulations, observations = ingest_feedback(
        feedback_path=args.candidates_csv,
        candidate_files=candidate_files,
        formulations=_read_or_empty(args.formulations),
        observations=_read_or_empty(args.observations),
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


if __name__ == "__main__":
    main()

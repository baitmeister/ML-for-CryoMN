#!/usr/bin/env python3
"""Regenerate proposal-time prospective reports without changing campaign state."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.config import load_evaluation_config
from helper.paths import EVALUATION_CONFIG, OBSERVATIONS_PATH, RESULTS_V2_DIR
from helper.prospective_evaluation import (
    generate_campaign_prospective_artifacts,
    generate_round_prospective_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--round",
        dest="round_id",
        help="Regenerate this completed round before refreshing the campaign report.",
    )
    selection.add_argument(
        "--all-rounds",
        action="store_true",
        help="Regenerate every completed round before refreshing the campaign report.",
    )
    parser.add_argument("--observations", default=str(OBSERVATIONS_PATH))
    parser.add_argument("--results-root", default=str(RESULTS_V2_DIR))
    parser.add_argument("--evaluation-config", default=str(EVALUATION_CONFIG))
    return parser.parse_args()


def _completed_round_ids(results_root: Path) -> list[str]:
    rounds_dir = results_root / "rounds"
    if not rounds_dir.exists():
        return []
    return sorted(
        path.name
        for path in rounds_dir.iterdir()
        if path.is_dir() and (path / "completed" / "completed.csv").exists()
    )


def main() -> None:
    args = parse_args()
    observations_path = Path(args.observations)
    results_root = Path(args.results_root)
    if not observations_path.exists():
        raise SystemExit(f"Observations file does not exist: {observations_path}")
    observations = pd.read_csv(observations_path)
    evaluation_config = load_evaluation_config(args.evaluation_config)

    if args.round_id:
        generated = generate_round_prospective_artifacts(
            args.round_id,
            observations,
            results_root=results_root,
            evaluation_config=evaluation_config,
        )
        print(
            f"Regenerated {len(generated)} prospective file(s) for "
            f"{args.round_id}."
        )
    elif args.all_rounds:
        for round_id in _completed_round_ids(results_root):
            generated = generate_round_prospective_artifacts(
                round_id,
                observations,
                results_root=results_root,
                evaluation_config=evaluation_config,
            )
            print(
                f"Regenerated {len(generated)} prospective file(s) for "
                f"{round_id}."
            )

    campaign_generated = generate_campaign_prospective_artifacts(
        observations,
        results_root=results_root,
        evaluation_config=evaluation_config,
    )
    print(
        f"Regenerated {len(campaign_generated)} campaign prospective file(s): "
        f"{(results_root / 'reports' / 'prospective').resolve()}"
    )


if __name__ == "__main__":
    main()

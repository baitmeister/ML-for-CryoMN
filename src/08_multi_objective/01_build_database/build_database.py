#!/usr/bin/env python3
"""Build v2 decoupled tables from legacy viability-only project data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.config import load_optimization_config
from helper.paths import LEGACY_LITERATURE_PATH, LEGACY_VALIDATION_PATH, PROCESSED_V2_DIR
from helper.registry import load_registry
from helper.transfer import build_v2_tables_from_legacy, write_v2_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--literature", default=str(LEGACY_LITERATURE_PATH), help="Legacy parsed literature CSV.")
    parser.add_argument("--validation", default=str(LEGACY_VALIDATION_PATH), help="Legacy wet-lab validation CSV.")
    parser.add_argument("--output-dir", default=str(PROCESSED_V2_DIR), help="Output directory for v2 tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_registry()
    optimization_config = load_optimization_config()
    formulations, observations = build_v2_tables_from_legacy(
        literature_path=Path(args.literature),
        validation_path=Path(args.validation),
        registry=registry,
        optimization_config=optimization_config,
    )
    write_v2_tables(formulations, observations, args.output_dir)
    print(f"Wrote {len(formulations)} formulations and {len(observations)} observations.")
    print(f"Output directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

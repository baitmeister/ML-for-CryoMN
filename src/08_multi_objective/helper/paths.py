"""Shared filesystem paths for the v2 multi-objective lane."""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
CONFIG_DIR = PROJECT_ROOT / "config_v2"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_V2_DIR = DATA_DIR / "processed_v2"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_V2_DIR = RESULTS_DIR / "multi_objective_v2"
NEXT_ROUND_DIR = RESULTS_V2_DIR / "next_round"
ROUND_REVIEW_DIR = RESULTS_V2_DIR / "round_review"
NEXT_ROUND_CANDIDATES_PATH = NEXT_ROUND_DIR / "next_round_candidates.csv"
NEXT_ROUND_SUMMARY_PATH = NEXT_ROUND_DIR / "next_round_summary.txt"
TOTAL_CANDIDATE_POOL_PATH = RESULTS_V2_DIR / "total_candidate_pool.csv"

INGREDIENTS_CONFIG = CONFIG_DIR / "ingredients.yaml"
ENDPOINTS_CONFIG = CONFIG_DIR / "endpoints.yaml"
OPTIMIZATION_CONFIG = CONFIG_DIR / "optimization.yaml"
AVAILABILITY_CONFIG = CONFIG_DIR / "availability.yaml"

FORMULATIONS_PATH = PROCESSED_V2_DIR / "formulations.csv"
OBSERVATIONS_PATH = PROCESSED_V2_DIR / "observations.csv"

LEGACY_LITERATURE_PATH = DATA_DIR / "processed" / "parsed_formulations.csv"
LEGACY_VALIDATION_PATH = DATA_DIR / "validation" / "validation_results.csv"

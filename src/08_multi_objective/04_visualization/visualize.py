#!/usr/bin/env python3
"""Visualize v2 multi-objective database and next-round candidate state."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.paths import FORMULATIONS_PATH, NEXT_ROUND_CANDIDATES_PATH, OBSERVATIONS_PATH, RESULTS_V2_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulations", default=str(FORMULATIONS_PATH))
    parser.add_argument("--observations", default=str(OBSERVATIONS_PATH))
    parser.add_argument("--candidates", default=str(NEXT_ROUND_CANDIDATES_PATH))
    parser.add_argument("--output-dir", default=str(RESULTS_V2_DIR / "visualizations"))
    return parser.parse_args()


def _read_or_empty(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def _save_endpoint_counts(observations: pd.DataFrame, output_dir: Path) -> Path | None:
    if observations.empty or "endpoint" not in observations.columns:
        return None
    counts = observations["endpoint"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts.plot(kind="barh", ax=ax, color="#4c78a8")
    ax.set_xlabel("Observation rows")
    ax.set_ylabel("Endpoint")
    ax.set_title("v2 Observation Coverage")
    fig.tight_layout()
    path = output_dir / "endpoint_observation_counts.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_objective_scatter(observations: pd.DataFrame, output_dir: Path) -> Path | None:
    if observations.empty:
        return None
    obs = observations.copy()
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    pivot = obs.pivot_table(
        index="formulation_id",
        columns="endpoint",
        values="value",
        aggfunc="mean",
    )
    required = {"viability_percent", "critical_axial_load_N_per_needle"}
    if not required.issubset(set(pivot.columns)):
        return None
    plot_data = pivot.dropna(subset=list(required))
    if plot_data.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        plot_data["viability_percent"],
        plot_data["critical_axial_load_N_per_needle"],
        s=48,
        color="#59a14f",
        alpha=0.85,
    )
    ax.set_xlabel("Viability (%)")
    ax.set_ylabel("Critical axial load (N/needle)")
    ax.set_title("Observed Pareto Objectives")
    fig.tight_layout()
    path = output_dir / "observed_objective_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _save_candidate_plot(candidates: pd.DataFrame, output_dir: Path) -> Path | None:
    required = {"predicted_viability_percent", "intact_patch_pass_probability"}
    if candidates.empty or not required.issubset(set(candidates.columns)):
        return None
    frame = candidates.copy()
    frame["predicted_viability_percent"] = pd.to_numeric(
        frame["predicted_viability_percent"],
        errors="coerce",
    )
    frame["intact_patch_pass_probability"] = pd.to_numeric(
        frame["intact_patch_pass_probability"],
        errors="coerce",
    )
    frame = frame.dropna(subset=list(required))
    if frame.empty:
        return None
    if "mechanical_test_recommended" in frame.columns:
        mechanical_flags = frame["mechanical_test_recommended"].astype(bool)
    else:
        mechanical_flags = pd.Series(False, index=frame.index)
    colors = mechanical_flags.map({True: "#e15759", False: "#4c78a8"})
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(
        frame["predicted_viability_percent"],
        frame["intact_patch_pass_probability"],
        s=60,
        c=colors,
        alpha=0.85,
    )
    ax.set_xlabel("Predicted viability (%)")
    ax.set_ylabel("Predicted intact-patch probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Next-Round Candidate Screen")
    fig.tight_layout()
    path = output_dir / "next_round_candidate_screen.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formulations = _read_or_empty(args.formulations)
    observations = _read_or_empty(args.observations)
    candidates = _read_or_empty(args.candidates)

    generated = [
        path
        for path in [
            _save_endpoint_counts(observations, output_dir),
            _save_objective_scatter(observations, output_dir),
            _save_candidate_plot(candidates, output_dir),
        ]
        if path is not None
    ]
    summary = [
        "CryoMN v2 Visualization Summary",
        "=" * 34,
        "",
        f"Formulation rows: {len(formulations)}",
        f"Observation rows: {len(observations)}",
        f"Candidate rows: {len(candidates)}",
        "",
        "Generated files:",
    ]
    summary.extend(f"- {path.name}" for path in generated)
    if not generated:
        summary.append("- none; not enough data for plots")
    (output_dir / "visualization_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Generated {len(generated)} visualization file(s).")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

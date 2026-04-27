#!/usr/bin/env python3
"""
Generate stage-specific predicted-vs-measured R^2 plots.

Default outputs:
- literature_only_r2_predicted_vs_actual.png
- iteration_1_wetlab_r2_predicted_vs_actual.png
- iteration_2_wetlab_r2_predicted_vs_actual.png
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = os.path.join(tempfile.gettempdir(), "cryomn-mpl")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_config_dir

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EVALUATION_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "evaluation_data.csv")
VALIDATION_PATH = os.path.join(PROJECT_ROOT, "data", "validation", "validation_results.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "explainability", "stage_r2")

VALIDATION_LOOP_DIR = os.path.join(PROJECT_ROOT, "src", "04_validation_loop")
HELPER_DIR = os.path.join(PROJECT_ROOT, "src", "helper")
if VALIDATION_LOOP_DIR not in sys.path:
    sys.path.insert(0, VALIDATION_LOOP_DIR)
if HELPER_DIR not in sys.path:
    sys.path.insert(0, HELPER_DIR)

# Required so composite_model.pkl can be unpickled safely.
from update_model_weighted_prior import CompositeGP  # noqa: F401,E402
from formulation_formatting import normalize_formulation_matrix  # noqa: E402


FONT_BUMP = 2


@dataclass
class PlotConfig:
    figsize_small: Tuple[int, int] = (9, 6)
    dpi: int = 170
    line_primary: str = "#0b5d7a"
    color_literature: str = "#0072b2"
    color_wetlab: str = "#e69f00"
    marker_literature: str = "o"
    marker_wetlab: str = "^"


def apply_palette_profile(config: PlotConfig, profile: str):
    """Apply color settings consistent with explainability palette profiles."""
    normalized = profile.strip().lower()
    if normalized == "colorblind":
        config.color_literature = "#0072b2"
        config.color_wetlab = "#e69f00"
        config.marker_literature = "o"
        config.marker_wetlab = "^"
        return
    if normalized == "legacy":
        config.color_literature = "#6a7f8f"
        config.color_wetlab = "#d55d3e"
        config.marker_literature = "o"
        config.marker_wetlab = "o"
        return
    raise ValueError(f"Unsupported palette profile: {profile}")


def apply_publication_style():
    """Match the publication-style plotting defaults from explainability.py."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfd",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#c4ccd4",
            "axes.linewidth": 1.0,
            "grid.color": "#d8dde3",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.35,
            "axes.titleweight": "bold",
            "axes.labelsize": 11 + FONT_BUMP,
            "axes.titlesize": 13 + FONT_BUMP,
            "font.size": 11 + FONT_BUMP,
            "legend.frameon": False,
            "legend.fontsize": 9 + FONT_BUMP,
            "xtick.color": "#33414f",
            "ytick.color": "#33414f",
        }
    )
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", context="talk", font_scale=0.90)


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Generate stage-specific predicted-vs-measured R^2 plots."
    )
    parser.add_argument(
        "--palette-profile",
        choices=("colorblind", "legacy"),
        default="colorblind",
        help="Choose color palette profile for plots.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated PNG files.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("post_update_cutoff", "prospective_batch"),
        default="prospective_batch",
        help=(
            "post_update_cutoff: evaluate on wet-lab rows up to model updated_at date. "
            "prospective_batch: literature plot stays literature-only, and wet-lab plots are "
            "stage-indexed as iteration_0(stage-0 batch by literature model), "
            "iteration_1(stage<=1 by iteration_1 model), etc."
        ),
    )
    return parser.parse_args(argv)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle(path: str):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def parse_validation_dates(series: pd.Series) -> pd.Series:
    """Parse wet-lab dates using repository month/day/year format."""
    parsed = pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
    if parsed.isna().any():
        fallback_mask = parsed.isna()
        parsed.loc[fallback_mask] = pd.to_datetime(
            series.loc[fallback_mask], errors="coerce"
        )
    return parsed


def stage_from_experiment_id(experiment_id: str) -> Optional[int]:
    """Map EXP IDs to project stages."""
    digits = "".join(ch for ch in str(experiment_id) if ch.isdigit())
    if not digits:
        return None
    value = int(digits)
    if value < 1000:
        return 0
    return value // 1000


def stage_from_iteration_dir(iteration_dir: str) -> Optional[int]:
    """Extract iteration number from directory names like iteration_2_prior_mean."""
    raw = str(iteration_dir)
    if not raw.startswith("iteration_"):
        return None
    tail = raw[len("iteration_") :]
    digits = []
    for char in tail:
        if char.isdigit():
            digits.append(char)
        else:
            break
    if not digits:
        return None
    return int("".join(digits))


def discover_iteration_models() -> List[Tuple[int, str]]:
    """Discover available iteration model directories sorted by iteration number."""
    discovered: List[Tuple[int, str]] = []
    for entry in sorted(os.listdir(MODELS_DIR)):
        if not entry.startswith("iteration_"):
            continue
        metadata_path = os.path.join(MODELS_DIR, entry, "model_metadata.json")
        if not os.path.exists(metadata_path):
            continue
        iteration = stage_from_iteration_dir(entry)
        if iteration is None:
            continue
        discovered.append((iteration, entry))
    discovered.sort(key=lambda item: item[0])
    return discovered


def select_feature_matrix(df: pd.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    aligned = df.reindex(columns=list(feature_names), fill_value=0.0).fillna(0.0)
    return normalize_formulation_matrix(
        aligned.to_numpy(dtype=float),
        list(feature_names),
    )


def predict_values(model, scaler, X_raw: np.ndarray, is_composite: bool) -> np.ndarray:
    if is_composite:
        return np.asarray(model.predict(X_raw), dtype=float)
    X_scaled = scaler.transform(X_raw)
    return np.asarray(model.predict(X_scaled), dtype=float)


def predict_with_uncertainty(model, scaler, X_raw: np.ndarray, is_composite: bool) -> Tuple[np.ndarray, np.ndarray]:
    if is_composite:
        y_pred, y_std = model.predict(X_raw, return_std=True)
        return np.asarray(y_pred, dtype=float), np.asarray(y_std, dtype=float)
    X_scaled = scaler.transform(X_raw)
    y_pred, y_std = model.predict(X_scaled, return_std=True)
    return np.asarray(y_pred, dtype=float), np.asarray(y_std, dtype=float)


def load_model_bundle(model_subdir: str):
    model_dir = os.path.join(MODELS_DIR, model_subdir)
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    metadata = load_json(metadata_path)
    feature_names = list(metadata["feature_names"])
    composite_path = os.path.join(model_dir, "composite_model.pkl")
    if os.path.exists(composite_path):
        return load_pickle(composite_path), None, True, feature_names, metadata
    model_path = os.path.join(model_dir, "gp_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    return load_pickle(model_path), load_pickle(scaler_path), False, feature_names, metadata


def r2_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return None
    return float(1.0 - (ss_res / ss_tot))


def plot_predicted_vs_actual(
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    output_path: str,
    title: str,
    marker: str,
    legend_label: str,
    marker_color: str,
    config: PlotConfig,
    x_label: str = "Measured Viability (%)",
):
    """Create one mean-predicted-vs-actual scatter with uncertainty color encoding."""
    annotation_scale = 1.5
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if len(y_actual) == 0:
        fig, ax = plt.subplots(figsize=config.figsize_small)
        ax.axis("off")
        ax.text(
            0.5,
            0.52,
            "No rows available for this stage/data slice.",
            ha="center",
            va="center",
            fontsize=12 + FONT_BUMP,
            color="#33414f",
        )
        ax.text(
            0.5,
            0.40,
            "Predicted-vs-actual R² plot not generated from data points.",
            ha="center",
            va="center",
            fontsize=10 + FONT_BUMP,
            color="#55616d",
        )
        fig.suptitle(title, fontsize=16 + FONT_BUMP, fontweight="bold", y=0.95)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print(f"  ✓ Saved: {output_path} (n=0, R²=N/A)")
        return

    r2 = r2_score_manual(y_actual, y_pred)
    r2_text = "N/A" if r2 is None else f"{r2:.3f}"

    fig, ax = plt.subplots(figsize=config.figsize_small)
    ax.plot([0, 100], [0, 100], linestyle="--", color="#4a5966", alpha=0.7, linewidth=2)

    scatter = ax.scatter(
        y_actual,
        y_pred,
        c=np.asarray(y_std, dtype=float),
        cmap="plasma",
        marker=marker,
        s=80,
        alpha=0.90,
        edgecolors="white",
        linewidths=0.55,
    )
    plt.colorbar(scatter, ax=ax, label="Prediction Uncertainty (std)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Predicted Viability (%)")
    ax.set_title(title, fontsize=14 + FONT_BUMP, fontweight="bold", pad=10)

    legend_handle = Line2D(
        [0],
        [0],
        marker=marker,
        color="none",
        markerfacecolor=marker_color,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label=legend_label,
        markersize=8,
        alpha=0.95,
    )
    ax.legend(handles=[legend_handle], loc="lower right")

    annotation_transform = ax.transAxes + mtransforms.ScaledTranslation(
        20.0 / fig.dpi,
        -20.0 / fig.dpi,
        fig.dpi_scale_trans,
    )
    stats_box = AnchoredText(
        f"n = {len(y_actual)}\nR² = {r2_text}",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        bbox_transform=annotation_transform,
        prop={"size": (10 + FONT_BUMP) * annotation_scale},
        frameon=True,
        borderpad=0.18 * annotation_scale,
        pad=0.18 * annotation_scale,
    )
    stats_box.patch.set_facecolor("white")
    stats_box.patch.set_alpha(0.82)
    stats_box.patch.set_edgecolor("#c4ccd4")
    stats_box.patch.set_boxstyle(f"round,pad={0.28 * annotation_scale}")
    ax.add_artist(stats_box)

    fig.tight_layout()
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  ✓ Saved: {output_path} (n={len(y_actual)}, R²={r2_text})")


def run_literature_plot(output_dir: str, config: PlotConfig):
    model, scaler, is_composite, feature_names, _ = load_model_bundle("literature_only")

    df = pd.read_csv(EVALUATION_DATA_PATH)
    source = df.get("source", pd.Series([""] * len(df))).astype(str).str.strip().str.lower()
    df = df[source == "literature"].copy()
    df["viability_percent"] = pd.to_numeric(df["viability_percent"], errors="coerce")
    df = df[df["viability_percent"].notna()].copy()

    X = select_feature_matrix(df, feature_names)
    y = df["viability_percent"].to_numpy(dtype=float)
    y_pred, y_std = predict_with_uncertainty(model, scaler, X, is_composite=is_composite)

    output_path = os.path.join(output_dir, "literature_only_r2_predicted_vs_actual.png")
    plot_predicted_vs_actual(
        y_actual=y,
        y_pred=y_pred,
        y_std=y_std,
        output_path=output_path,
        title="Literature-Only Predicted vs Reported",
        marker=config.marker_literature,
        legend_label="Literature",
        marker_color=config.color_literature,
        config=config,
        x_label="Reported Viability (%)",
    )


def _updated_date_from_metadata(metadata: dict) -> datetime.date:
    raw = metadata.get("updated_at") or metadata.get("trained_at")
    if not raw:
        raise ValueError("model metadata is missing both updated_at and trained_at")
    return datetime.fromisoformat(str(raw)).date()


def _load_validation_df_with_dates() -> pd.DataFrame:
    df = pd.read_csv(VALIDATION_PATH)
    df = df[df["viability_measured"].notna()].copy()
    df["parsed_date"] = parse_validation_dates(df["experiment_date"])
    df = df[df["parsed_date"].notna()].copy()
    df["parsed_date"] = df["parsed_date"].dt.date
    df["viability_measured"] = pd.to_numeric(df["viability_measured"], errors="coerce")
    df = df[df["viability_measured"].notna()].copy()
    df["stage"] = df["experiment_id"].map(stage_from_experiment_id)
    return df


def _validation_batch_for_stage(validation_df: pd.DataFrame, stage: int) -> pd.DataFrame:
    return validation_df[validation_df["stage"] == int(stage)].copy()


def _validation_cumulative_for_stage(validation_df: pd.DataFrame, stage: int) -> pd.DataFrame:
    """Return wet-lab rows from stage 0 up to and including target stage."""
    return validation_df[validation_df["stage"].notna() & (validation_df["stage"] <= int(stage))].copy()


def run_wetlab_plot(
    model_subdir: str,
    output_filename: str,
    title: str,
    output_dir: str,
    config: PlotConfig,
    evaluation_mode: str,
    stage: int,
):
    model, scaler, is_composite, feature_names, metadata = load_model_bundle(model_subdir)
    validation_df = _load_validation_df_with_dates()
    if evaluation_mode == "prospective_batch":
        sliced_df = _validation_cumulative_for_stage(validation_df, stage=stage)
    else:
        cutoff_date = _updated_date_from_metadata(metadata)
        sliced_df = validation_df[validation_df["parsed_date"] <= cutoff_date].copy()
    X = select_feature_matrix(sliced_df, feature_names)
    y = sliced_df["viability_measured"].to_numpy(dtype=float)
    y_pred, y_std = predict_with_uncertainty(model, scaler=scaler, X_raw=X, is_composite=is_composite)

    output_path = os.path.join(output_dir, output_filename)
    plot_predicted_vs_actual(
        y_actual=y,
        y_pred=y_pred,
        y_std=y_std,
        output_path=output_path,
        title=title,
        marker=config.marker_wetlab,
        legend_label="Wet Lab",
        marker_color=config.color_wetlab,
        config=config,
    )


def run_all_wetlab_plots(output_dir: str, config: PlotConfig, evaluation_mode: str):
    """Generate wet-lab plots for all discovered iterations."""
    discovered = discover_iteration_models()
    if not discovered:
        print("  ! No iteration_* model directories found; wet-lab plots skipped.")
        return

    if evaluation_mode == "prospective_batch":
        # iteration_0 corresponds to stage-0 wet-lab batch evaluated by literature-only model.
        run_wetlab_plot(
            model_subdir="literature_only",
            output_filename="iteration_0_wetlab_r2_predicted_vs_actual.png",
            title="Iteration 0 Wet-Lab Predicted vs Actual (Prospective Cumulative)",
            output_dir=output_dir,
            config=config,
            evaluation_mode=evaluation_mode,
            stage=0,
        )
        iteration_to_dir = {iteration: directory for iteration, directory in discovered}
        max_iteration = max(iteration_to_dir)
        for stage_index in range(1, max_iteration):
            model_dir = iteration_to_dir.get(stage_index)
            if not model_dir:
                print(
                    f"  ! Missing model for iteration_{stage_index}; "
                    f"skipping iteration_{stage_index} wet-lab plot."
                )
                continue
            run_wetlab_plot(
                model_subdir=model_dir,
                output_filename=f"iteration_{stage_index}_wetlab_r2_predicted_vs_actual.png",
                title=f"Iteration {stage_index} Wet-Lab Predicted vs Actual (Prospective Cumulative)",
                output_dir=output_dir,
                config=config,
                evaluation_mode=evaluation_mode,
                stage=stage_index,
            )
        return

    # post_update_cutoff mode: each iteration uses its own saved model and date cutoff.
    for iteration, model_dir in discovered:
        run_wetlab_plot(
            model_subdir=model_dir,
            output_filename=f"iteration_{iteration}_wetlab_r2_predicted_vs_actual.png",
            title=f"Iteration {iteration} Wet-Lab Predicted vs Actual",
            output_dir=output_dir,
            config=config,
            evaluation_mode=evaluation_mode,
            stage=iteration,
        )


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    config = PlotConfig()
    apply_palette_profile(config, args.palette_profile)
    apply_publication_style()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("Stage-Specific Predicted-vs-Actual R² Plots")
    print("=" * 80)
    print(f"Saving outputs to: {output_dir}")
    print(f"Evaluation mode: {args.evaluation_mode}")

    run_literature_plot(output_dir=output_dir, config=config)
    run_all_wetlab_plots(
        output_dir=output_dir,
        config=config,
        evaluation_mode=args.evaluation_mode,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

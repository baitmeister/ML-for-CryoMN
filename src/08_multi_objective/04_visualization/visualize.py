#!/usr/bin/env python3
"""Visualize v2 multi-objective database, best performers, and model fit."""

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
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.models import train_endpoint_models
from helper.paths import FORMULATIONS_PATH, NEXT_ROUND_CANDIDATES_PATH, OBSERVATIONS_PATH, RESULTS_V2_DIR
from helper.registry import IngredientRegistry, load_registry


PAGE_BG = "#f7f2e8"
AX_BG = "#fffdf8"
GRID = "#d8d0c1"
TEXT = "#2d2a26"
MUTED = "#8a8175"
BLUE = "#4c78a8"
TEAL = "#4f8f6b"
GOLD = "#d7a44c"
CORAL = "#d96c5f"
SLATE = "#6c7a89"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulations", default=str(FORMULATIONS_PATH))
    parser.add_argument("--observations", default=str(OBSERVATIONS_PATH))
    parser.add_argument("--candidates", default=str(NEXT_ROUND_CANDIDATES_PATH))
    parser.add_argument("--output-dir", default=str(RESULTS_V2_DIR / "visualizations"))
    return parser.parse_args()


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.facecolor": AX_BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "figure.facecolor": PAGE_BG,
            "savefig.facecolor": PAGE_BG,
            "grid.color": GRID,
            "grid.alpha": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
        }
    )


def _read_or_empty(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
    return pd.DataFrame()


def _format_metric(value: float | None, fmt: str = "{:.2f}") -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return fmt.format(float(value))


def _format_formulation(row: pd.Series, registry: IngredientRegistry) -> str:
    ingredients = []
    for feature_name in registry.feature_names:
        if feature_name not in row.index:
            continue
        value = pd.to_numeric(row.get(feature_name), errors="coerce")
        if pd.isna(value) or float(value) <= 0.0:
            continue
        display_name = registry.get_by_feature(feature_name).display_name
        if feature_name.endswith("_pct"):
            ingredients.append(f"{float(value):.3g}% {display_name}")
        elif float(value) >= 1.0:
            ingredients.append(f"{float(value):.3g}M {display_name}")
        else:
            ingredients.append(f"{float(value) * 1000:.3g}mM {display_name}")
    return " + ".join(ingredients) if ingredients else "No active ingredients"


def _observed_endpoint_frame(formulations: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    if formulations.empty:
        return pd.DataFrame()
    frame = formulations.copy()
    if observations.empty:
        return frame

    obs = observations.copy()
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    pivot = obs.pivot_table(
        index="formulation_id",
        columns="endpoint",
        values="value",
        aggfunc="mean",
    )
    frame = frame.merge(pivot, on="formulation_id", how="left")

    batch_map = (
        obs.groupby("formulation_id")["batch_id"]
        .agg(lambda values: ", ".join(sorted({str(v).strip() for v in values if str(v).strip()})))
        .rename("observed_batches")
    )
    source_map = (
        obs.groupby("formulation_id")["source_type"]
        .agg(lambda values: ", ".join(sorted({str(v).strip() for v in values if str(v).strip()})))
        .rename("observed_sources")
    )
    frame = frame.merge(batch_map, on="formulation_id", how="left")
    frame = frame.merge(source_map, on="formulation_id", how="left")
    return frame


def _pareto_frontier_mask(frame: pd.DataFrame, x_col: str, y_col: str) -> np.ndarray:
    values = frame[[x_col, y_col]].to_numpy(dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for i, current in enumerate(values):
        for j, other in enumerate(values):
            if i == j:
                continue
            if np.all(other >= current) and np.any(other > current):
                keep[i] = False
                break
    return keep


def _top_candidate_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    frame = candidates.copy()
    if "selection_rank" in frame.columns:
        frame["selection_rank"] = pd.to_numeric(frame["selection_rank"], errors="coerce")
        frame = frame.sort_values("selection_rank", ascending=True, na_position="last")
    else:
        frame["predicted_viability_percent"] = pd.to_numeric(
            frame.get("predicted_viability_percent"),
            errors="coerce",
        )
        frame = frame.sort_values("predicted_viability_percent", ascending=False, na_position="last")
    return frame


def _write_best_performers_summary(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
    output_dir: Path,
    registry: IngredientRegistry,
) -> Path:
    observed = _observed_endpoint_frame(formulations, observations)
    top_candidates = _top_candidate_frame(candidates).head(5)
    lines = [
        "CryoMN v2 Best Performers",
        "=" * 25,
        "",
        "Database snapshot:",
        f"- Formulations tracked: {len(formulations)}",
        f"- Observation rows: {len(observations)}",
        f"- Current next-round candidates: {len(candidates)}",
        "",
    ]

    viability_frame = observed.dropna(subset=["viability_percent"]).sort_values(
        "viability_percent",
        ascending=False,
    )
    lines.append("Best observed viability performers:")
    if viability_frame.empty:
        lines.append("- none yet")
    else:
        for rank, (_, row) in enumerate(viability_frame.head(5).iterrows(), start=1):
            lines.append(
                f"- #{rank} {row['formulation_id']} | viability {_format_metric(row['viability_percent'], '{:.1f}')}% | "
                f"source {row.get('observed_sources') or row.get('source') or 'unknown'}"
            )
            lines.append(f"  formulation: {_format_formulation(row, registry)}")
    lines.append("")

    load_column = "critical_axial_load_N_per_needle"
    load_frame = (
        observed.dropna(subset=[load_column]).sort_values(load_column, ascending=False)
        if load_column in observed.columns
        else pd.DataFrame()
    )
    lines.append("Best observed mechanical performers:")
    if load_frame.empty:
        lines.append("- none yet; no critical axial load measurements are in the v2 database")
    else:
        for rank, (_, row) in enumerate(load_frame.head(5).iterrows(), start=1):
            viability_text = ""
            if pd.notna(row.get("viability_percent")):
                viability_text = f" | viability {_format_metric(row['viability_percent'], '{:.1f}')}%"
            lines.append(
                f"- #{rank} {row['formulation_id']} | critical load {_format_metric(row[load_column], '{:.3f}')} N/needle"
                f"{viability_text}"
            )
            lines.append(f"  formulation: {_format_formulation(row, registry)}")
    lines.append("")

    balanced = (
        observed.dropna(subset=["viability_percent", load_column]).copy()
        if {"viability_percent", load_column}.issubset(observed.columns)
        else pd.DataFrame()
    )
    lines.append("Balanced multi-objective leaders:")
    if balanced.empty:
        lines.append("- none yet; this section appears after formulations have both viability and mechanical measurements")
    else:
        frontier = balanced.loc[_pareto_frontier_mask(balanced, "viability_percent", load_column)].copy()
        viability_scaled = frontier["viability_percent"].rank(pct=True)
        load_scaled = frontier[load_column].rank(pct=True)
        frontier["balanced_score"] = 0.5 * viability_scaled + 0.5 * load_scaled
        frontier = frontier.sort_values(["balanced_score", "viability_percent"], ascending=[False, False])
        for rank, (_, row) in enumerate(frontier.head(5).iterrows(), start=1):
            lines.append(
                f"- #{rank} {row['formulation_id']} | viability {_format_metric(row['viability_percent'], '{:.1f}')}% | "
                f"critical load {_format_metric(row[load_column], '{:.3f}')} N/needle"
            )
            lines.append(f"  formulation: {_format_formulation(row, registry)}")
    lines.append("")

    lines.append("Current leading next-round candidates:")
    if top_candidates.empty:
        lines.append("- none; run Stage 02 to generate a candidate slate")
    else:
        for _, row in top_candidates.iterrows():
            selection_rank = pd.to_numeric(row.get("selection_rank"), errors="coerce")
            prefix = f"#{int(selection_rank)}" if pd.notna(selection_rank) else row.get("candidate_id", "candidate")
            lines.append(
                f"- {prefix} {row.get('formulation_id', '')} | predicted viability "
                f"{_format_metric(pd.to_numeric(row.get('predicted_viability_percent'), errors='coerce'), '{:.1f}')}% | "
                f"intact probability {_format_metric(pd.to_numeric(row.get('intact_patch_pass_probability'), errors='coerce'), '{:.2f}')}"
                f" | mechanical test {bool(row.get('mechanical_test_recommended', False))}"
            )
            lines.append(f"  formulation: {_format_formulation(row, registry)}")

    output_path = output_dir / "best_performers_summary.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _save_endpoint_counts(observations: pd.DataFrame, output_dir: Path) -> Path | None:
    if observations.empty or "endpoint" not in observations.columns:
        return None
    counts = observations["endpoint"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor(PAGE_BG)
    bars = ax.barh(counts.index, counts.values, color=[BLUE, GOLD, TEAL, CORAL][: len(counts)])
    ax.set_xlabel("Observation rows")
    ax.set_title("Where the v2 evidence currently lives", pad=14)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_width() + max(counts.values) * 0.02, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center")
    fig.tight_layout()
    path = output_dir / "endpoint_observation_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_observed_performance_landscape(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    frame = _observed_endpoint_frame(formulations, observations)
    required = {"viability_percent", "critical_axial_load_N_per_needle"}
    if frame.empty or not required.issubset(frame.columns):
        return None
    plot_data = frame.dropna(subset=list(required)).copy()
    if plot_data.empty:
        return None

    frontier_mask = _pareto_frontier_mask(
        plot_data,
        "viability_percent",
        "critical_axial_load_N_per_needle",
    )
    frontier = plot_data.loc[frontier_mask].sort_values("viability_percent")

    fig, ax = plt.subplots(figsize=(8.2, 6))
    fig.patch.set_facecolor(PAGE_BG)
    ax.axvspan(plot_data["viability_percent"].quantile(0.75), plot_data["viability_percent"].max() + 2, color=TEAL, alpha=0.08)
    ax.scatter(
        plot_data["viability_percent"],
        plot_data["critical_axial_load_N_per_needle"],
        s=70,
        color=SLATE,
        alpha=0.35,
        edgecolor="white",
        linewidth=0.6,
        label="Observed formulations",
    )
    ax.scatter(
        frontier["viability_percent"],
        frontier["critical_axial_load_N_per_needle"],
        s=90,
        color=TEAL,
        edgecolor="white",
        linewidth=0.8,
        label="Pareto frontier",
        zorder=3,
    )
    if len(frontier) > 1:
        ax.plot(
            frontier["viability_percent"],
            frontier["critical_axial_load_N_per_needle"],
            color=TEAL,
            linewidth=2,
            alpha=0.8,
            zorder=2,
        )
    ax.set_xlabel("Observed viability (%)")
    ax.set_ylabel("Observed critical axial load (N/needle)")
    ax.set_title("Observed performance landscape", pad=14)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    path = output_dir / "observed_performance_landscape.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _cross_validated_predictions(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
    endpoint: str,
) -> pd.DataFrame:
    frame = _observed_endpoint_frame(formulations, observations)
    if frame.empty or endpoint not in frame.columns:
        return pd.DataFrame()
    valid = frame.dropna(subset=[endpoint]).copy()
    if len(valid) < 2:
        return pd.DataFrame()

    n_splits = min(5, len(valid))
    if n_splits < 2:
        return pd.DataFrame()

    predictions: list[pd.DataFrame] = []
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for _, test_index in splitter.split(valid):
        test_rows = valid.iloc[test_index].copy()
        test_ids = set(test_rows["formulation_id"])
        train_observations = observations.loc[
            ~(
                observations["formulation_id"].isin(test_ids)
                & (observations["endpoint"].astype(str) == endpoint)
            )
        ].copy()
        models = train_endpoint_models(formulations, train_observations, registry)
        x_test = (
            test_rows[registry.feature_names]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        if endpoint == "viability_percent":
            prediction = models.viability.predict(x_test)
            predicted = prediction.mean
            predicted_std = prediction.std
        elif endpoint == "critical_axial_load_N_per_needle":
            prediction = models.critical_load.predict(x_test)
            predicted = prediction.mean
            predicted_std = prediction.std
        elif endpoint == "intact_patch_formation_pass":
            predicted = models.intact.predict_proba(x_test)
            predicted_std = np.full(len(predicted), np.nan, dtype=float)
        else:
            continue

        predictions.append(
            pd.DataFrame(
                {
                    "formulation_id": test_rows["formulation_id"].to_numpy(),
                    "actual": pd.to_numeric(test_rows[endpoint], errors="coerce").to_numpy(dtype=float),
                    "predicted": np.asarray(predicted, dtype=float),
                    "predicted_std": np.asarray(predicted_std, dtype=float),
                }
            )
        )
    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def _plot_parity_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    unit_label: str,
    color: str,
) -> dict[str, float | int | None]:
    if frame.empty:
        ax.text(0.5, 0.5, "Not enough data yet", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(f"Observed {unit_label}")
        ax.set_ylabel(f"Predicted {unit_label}")
        return {"n": 0, "mae": None, "r2": None}

    actual = frame["actual"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    low = float(np.nanmin(np.concatenate([actual, predicted])))
    high = float(np.nanmax(np.concatenate([actual, predicted])))
    pad = max((high - low) * 0.08, 1e-6)

    ax.scatter(actual, predicted, s=75, color=color, alpha=0.85, edgecolor="white", linewidth=0.7)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color=MUTED, linestyle="--", linewidth=1.5)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_title(title)
    ax.set_xlabel(f"Observed {unit_label}")
    ax.set_ylabel(f"Predicted {unit_label}")

    mae = float(mean_absolute_error(actual, predicted))
    r2: float | None
    if len(frame) >= 2 and np.nanstd(actual) > 0:
        r2 = float(r2_score(actual, predicted))
    else:
        r2 = None
    ax.text(
        0.03,
        0.97,
        f"n={len(frame)}\nMAE={_format_metric(mae)}\nR²={_format_metric(r2)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": AX_BG, "edgecolor": GRID},
    )
    return {"n": len(frame), "mae": mae, "r2": r2}


def _plot_probability_axis(ax: plt.Axes, frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty:
        ax.text(0.5, 0.5, "Not enough pass/fail data yet", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_title("Intact-patch pass probabilities")
        ax.set_xlabel("Predicted pass probability")
        return {"n": 0, "brier": None, "accuracy": None}

    plot_data = frame.copy()
    jitter = np.linspace(-0.08, 0.08, len(plot_data)) if len(plot_data) > 1 else np.array([0.0])
    plot_data["y"] = plot_data["actual"] + jitter
    colors = plot_data["actual"].map({0.0: CORAL, 1.0: TEAL}).fillna(BLUE)
    ax.scatter(
        plot_data["predicted"],
        plot_data["y"],
        s=75,
        c=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.set_title("Intact-patch pass probabilities")
    ax.set_xlabel("Predicted pass probability")
    ax.set_yticks([0, 1], labels=["Observed fail", "Observed pass"])
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.4, 1.4)

    actual = plot_data["actual"].to_numpy(dtype=float)
    predicted = plot_data["predicted"].to_numpy(dtype=float)
    accuracy = float(accuracy_score(actual, predicted >= 0.5))
    brier = float(brier_score_loss(actual, predicted))
    ax.text(
        0.03,
        0.97,
        f"n={len(plot_data)}\nAccuracy={_format_metric(accuracy)}\nBrier={_format_metric(brier)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": AX_BG, "edgecolor": GRID},
    )
    return {"n": len(plot_data), "brier": brier, "accuracy": accuracy}


def _save_model_evaluation_overview(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    output_dir: Path,
    registry: IngredientRegistry,
) -> Path | None:
    if formulations.empty or observations.empty:
        return None

    viability_cv = _cross_validated_predictions(formulations, observations, registry, "viability_percent")
    load_cv = _cross_validated_predictions(formulations, observations, registry, "critical_axial_load_N_per_needle")
    intact_cv = _cross_validated_predictions(formulations, observations, registry, "intact_patch_formation_pass")
    if viability_cv.empty and load_cv.empty and intact_cv.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    fig.patch.set_facecolor(PAGE_BG)
    ax_viability, ax_load, ax_intact, ax_scorecard = axes.flatten()

    viability_metrics = _plot_parity_axis(ax_viability, viability_cv, "Viability model fit", "viability (%)", BLUE)
    load_metrics = _plot_parity_axis(ax_load, load_cv, "Mechanical model fit", "critical load (N/needle)", GOLD)
    intact_metrics = _plot_probability_axis(ax_intact, intact_cv)

    ax_scorecard.axis("off")
    score_lines = [
        "Reader notes",
        "",
        "These panels use cross-validated predictions",
        "built from the current v2 database.",
        "",
        f"Viability rows evaluated: {viability_metrics['n']}",
        f"Mechanical rows evaluated: {load_metrics['n']}",
        f"Intact pass/fail rows evaluated: {intact_metrics['n']}",
        "",
        f"Viability MAE: {_format_metric(viability_metrics['mae'])}",
        f"Mechanical MAE: {_format_metric(load_metrics['mae'])}",
        f"Intact accuracy: {_format_metric(intact_metrics['accuracy'])}",
        "",
        "Use this figure to judge whether the model",
        "is tracking the current data cleanly enough",
        "to support the next round of experiments.",
    ]
    ax_scorecard.text(
        0.02,
        0.98,
        "\n".join(score_lines),
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": AX_BG, "edgecolor": GRID},
    )
    fig.suptitle("V2 model evaluation overview", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = output_dir / "model_evaluation_overview.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_candidate_plot(candidates: pd.DataFrame, output_dir: Path) -> Path | None:
    required = {"predicted_viability_percent", "intact_patch_pass_probability"}
    if candidates.empty or not required.issubset(set(candidates.columns)):
        return None
    frame = candidates.copy()
    frame["predicted_viability_percent"] = pd.to_numeric(frame["predicted_viability_percent"], errors="coerce")
    frame["intact_patch_pass_probability"] = pd.to_numeric(frame["intact_patch_pass_probability"], errors="coerce")
    frame["predicted_critical_axial_load_N_per_needle"] = pd.to_numeric(
        frame.get("predicted_critical_axial_load_N_per_needle"),
        errors="coerce",
    )
    frame["selection_rank"] = pd.to_numeric(frame.get("selection_rank"), errors="coerce")
    frame = frame.dropna(subset=list(required))
    if frame.empty:
        return None

    mechanical_flags = (
        frame["mechanical_test_recommended"].astype(bool)
        if "mechanical_test_recommended" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    sizes = frame["predicted_critical_axial_load_N_per_needle"].fillna(0.0).to_numpy(dtype=float)
    if np.nanmax(sizes) > 0:
        sizes = 80 + 160 * (sizes / max(np.nanmax(sizes), 1e-9))
    else:
        sizes = np.full(len(frame), 100.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(PAGE_BG)
    high_viability = float(frame["predicted_viability_percent"].quantile(0.75))
    high_pass = float(frame["intact_patch_pass_probability"].quantile(0.75))
    ax.axvspan(high_viability, frame["predicted_viability_percent"].max() + 1, color=GOLD, alpha=0.08)
    ax.axhspan(high_pass, 1.03, color=TEAL, alpha=0.08)

    screen_only = frame.loc[~mechanical_flags]
    mechanical = frame.loc[mechanical_flags]
    if not screen_only.empty:
        ax.scatter(
            screen_only["predicted_viability_percent"],
            screen_only["intact_patch_pass_probability"],
            s=sizes[~mechanical_flags.to_numpy()],
            color=BLUE,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
            label="Viability screen",
        )
    if not mechanical.empty:
        ax.scatter(
            mechanical["predicted_viability_percent"],
            mechanical["intact_patch_pass_probability"],
            s=sizes[mechanical_flags.to_numpy()],
            color=CORAL,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
            label="Mechanical follow-up",
        )
    for _, row in frame.nsmallest(5, "selection_rank").iterrows():
        if pd.isna(row["selection_rank"]):
            continue
        ax.annotate(
            f"#{int(row['selection_rank'])}",
            (row["predicted_viability_percent"], row["intact_patch_pass_probability"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color=TEXT,
        )
    ax.set_xlabel("Predicted viability (%)")
    ax.set_ylabel("Predicted intact-patch probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Next-round candidate screen", pad=14)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path = output_dir / "next_round_candidate_screen.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_visualization_summary(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
    generated: list[Path],
    output_dir: Path,
) -> Path:
    observed = _observed_endpoint_frame(formulations, observations)
    summary = [
        "CryoMN v2 Visualization Summary",
        "=" * 32,
        "",
        f"Formulation rows: {len(formulations)}",
        f"Observation rows: {len(observations)}",
        f"Candidate rows: {len(candidates)}",
        f"Formulations with observed viability: {int(observed.get('viability_percent', pd.Series(dtype=float)).notna().sum()) if not observed.empty else 0}",
        f"Formulations with observed critical load: {int(observed.get('critical_axial_load_N_per_needle', pd.Series(dtype=float)).notna().sum()) if not observed.empty else 0}",
        f"Formulations with observed intact-patch gate: {int(observed.get('intact_patch_formation_pass', pd.Series(dtype=float)).notna().sum()) if not observed.empty else 0}",
        "",
        "Generated files:",
    ]
    summary.extend(f"- {path.name}" for path in generated)
    if not generated:
        summary.append("- none; not enough data for plots or reports")
    summary.append("")
    summary.append("Reader note:")
    summary.append("Stage 04 now produces a best-performers report and reader-friendly model evaluation graphics.")
    output_path = output_dir / "visualization_summary.txt"
    output_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    _apply_plot_style()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formulations = _read_or_empty(args.formulations)
    observations = _read_or_empty(args.observations)
    candidates = _read_or_empty(args.candidates)
    registry = load_registry()

    generated = [
        path
        for path in [
            _write_best_performers_summary(formulations, observations, candidates, output_dir, registry),
            _save_endpoint_counts(observations, output_dir),
            _save_observed_performance_landscape(formulations, observations, output_dir),
            _save_model_evaluation_overview(formulations, observations, output_dir, registry),
            _save_candidate_plot(candidates, output_dir),
        ]
        if path is not None
    ]
    generated.append(_write_visualization_summary(formulations, observations, candidates, generated, output_dir))
    print(f"Generated {len(generated)} visualization file(s).")
    print(f"Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

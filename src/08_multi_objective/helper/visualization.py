"""Generate v2 review snapshots and real multi-objective evaluation graphics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

if __package__ in (None, ""):
    V2_ROOT = Path(__file__).resolve().parents[1]
    if str(V2_ROOT) not in sys.path:
        sys.path.insert(0, str(V2_ROOT))
    from helper.models import build_training_frame, train_endpoint_models
    from helper.paths import FORMULATIONS_PATH, NEXT_ROUND_CANDIDATES_PATH, OBSERVATIONS_PATH, VISUALIZATIONS_DIR
    from helper.registry import IngredientRegistry, load_registry
else:
    from .models import build_training_frame, train_endpoint_models
    from .paths import FORMULATIONS_PATH, NEXT_ROUND_CANDIDATES_PATH, OBSERVATIONS_PATH, VISUALIZATIONS_DIR
    from .registry import IngredientRegistry, load_registry


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


def _artifact_path(
    output_dir: Path,
    base_name: str,
    suffix: str,
    artifact_prefix: str = "",
) -> Path:
    prefix = f"{artifact_prefix}_" if str(artifact_prefix).strip() else ""
    return output_dir / f"{prefix}{base_name}{suffix}"

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


def _round_sort_key(batch_id: object) -> tuple[int, str]:
    value = str(batch_id).strip()
    if value.startswith("ROUND_"):
        suffix = value.removeprefix("ROUND_")
        if suffix.isdigit():
            return (0, f"{int(suffix):09d}")
    return (1, value)


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


def _paired_frame(formulations: pd.DataFrame, observations: pd.DataFrame, registry: IngredientRegistry) -> pd.DataFrame:
    frame = build_training_frame(formulations, observations, registry)
    required = ["formulation_id", "batch_id", "viability_percent", "critical_axial_load_N_per_needle"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()
    paired = frame.dropna(subset=["viability_percent", "critical_axial_load_N_per_needle"]).copy()
    if paired.empty:
        return paired
    paired["batch_id"] = paired["batch_id"].fillna("").astype(str)
    paired["round_order"] = paired["batch_id"].map(_round_sort_key)
    return paired.sort_values(["round_order", "formulation_id"]).reset_index(drop=True)


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
    artifact_prefix: str = "",
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

    retest_mask = (
        candidates.get(
            "recommendation_type",
            pd.Series([""] * len(candidates), index=candidates.index, dtype="object"),
        ).astype(str)
        == "retest_priority"
    ) if not candidates.empty else pd.Series(dtype=bool)
    retest_candidates = candidates.loc[retest_mask].copy() if not candidates.empty else pd.DataFrame()
    lines.append("Retest-priority recommendations:")
    if retest_candidates.empty:
        lines.append("- none in the current slate")
    else:
        for _, row in retest_candidates.iterrows():
            lines.append(
                f"- {row.get('formulation_id', '')} | predicted viability "
                f"{_format_metric(pd.to_numeric(row.get('predicted_viability_percent'), errors='coerce'), '{:.1f}')}% | "
                f"intact probability {_format_metric(pd.to_numeric(row.get('intact_patch_pass_probability'), errors='coerce'), '{:.2f}')}"
            )
            if str(row.get("selection_explanation", "")).strip():
                lines.append(f"  note: {row['selection_explanation']}")
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

    output_path = _artifact_path(output_dir, "best_performers_summary", ".txt", artifact_prefix)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _save_endpoint_counts(
    observations: pd.DataFrame,
    output_dir: Path,
    artifact_prefix: str = "",
) -> Path | None:
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
    path = _artifact_path(output_dir, "endpoint_observation_counts", ".png", artifact_prefix)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_observed_performance_landscape(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    output_dir: Path,
    artifact_prefix: str = "",
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
    path = _artifact_path(output_dir, "observed_performance_landscape", ".png", artifact_prefix)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _build_model_evaluation_frames(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
) -> dict[str, pd.DataFrame]:
    return {
        "viability_percent": _cross_validated_predictions(
            formulations,
            observations,
            registry,
            "viability_percent",
        ),
        "critical_axial_load_N_per_needle": _cross_validated_predictions(
            formulations,
            observations,
            registry,
            "critical_axial_load_N_per_needle",
        ),
        "intact_patch_formation_pass": _cross_validated_predictions(
            formulations,
            observations,
            registry,
            "intact_patch_formation_pass",
        ),
    }


def _write_model_evaluation_table(
    evaluation_frames: dict[str, pd.DataFrame],
    output_dir: Path,
    artifact_prefix: str = "",
) -> Path | None:
    rows: list[pd.DataFrame] = []
    for endpoint, frame in evaluation_frames.items():
        if frame.empty:
            continue
        table = frame.copy()
        table["endpoint"] = endpoint
        table["absolute_error"] = (table["predicted"] - table["actual"]).abs()
        table["squared_error"] = (table["predicted"] - table["actual"]) ** 2
        rows.append(
            table[
                [
                    "endpoint",
                    "formulation_id",
                    "batch_id",
                    "actual",
                    "predicted",
                    "predicted_std",
                    "absolute_error",
                    "squared_error",
                ]
            ]
        )
    if not rows:
        return None

    output_path = _artifact_path(output_dir, "model_evaluation_table", ".csv", artifact_prefix)
    pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)
    return output_path


def _cross_validated_predictions(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
    endpoint: str,
) -> pd.DataFrame:
    frame = build_training_frame(formulations, observations, registry)
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
        holdout_keys = {
            (str(row["formulation_id"]), str(row.get("batch_id", "")))
            for _, row in test_rows.iterrows()
        }
        train_observations = observations.copy()
        if "batch_id" not in train_observations.columns:
            train_observations["batch_id"] = ""
        endpoint_mask = train_observations["endpoint"].astype(str) == endpoint
        holdout_mask = train_observations.apply(
            lambda row: (str(row.get("formulation_id", "")), str(row.get("batch_id", ""))) in holdout_keys,
            axis=1,
        )
        train_observations = train_observations.loc[~(endpoint_mask & holdout_mask)].copy()
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
                    "batch_id": test_rows.get("batch_id", pd.Series([""] * len(test_rows))).to_numpy(),
                    "actual": pd.to_numeric(test_rows[endpoint], errors="coerce").to_numpy(dtype=float),
                    "predicted": np.asarray(predicted, dtype=float),
                    "predicted_std": np.asarray(predicted_std, dtype=float),
                }
            )
        )
    return pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def _normalize_frame(frame: pd.DataFrame, columns: list[str], reference: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        low = float(reference[column].min())
        high = float(reference[column].max())
        spread = max(high - low, 1e-9)
        normalized[column] = (pd.to_numeric(frame[column], errors="coerce") - low) / spread
    return normalized


def _hypervolume_2d(frontier: pd.DataFrame, x_col: str, y_col: str) -> float:
    if frontier.empty:
        return 0.0
    ordered = frontier.sort_values(x_col)
    hv = 0.0
    previous_x = 0.0
    for _, row in ordered.iterrows():
        x_value = max(float(row[x_col]), previous_x)
        y_value = max(float(row[y_col]), 0.0)
        hv += max(x_value - previous_x, 0.0) * y_value
        previous_x = x_value
    return float(hv)


def _igd_2d(true_frontier: pd.DataFrame, estimated_frontier: pd.DataFrame, x_col: str, y_col: str) -> float | None:
    if true_frontier.empty or estimated_frontier.empty:
        return None
    reference = true_frontier[[x_col, y_col]].to_numpy(dtype=float)
    estimated = estimated_frontier[[x_col, y_col]].to_numpy(dtype=float)
    distances = []
    for point in reference:
        distances.append(float(np.min(np.linalg.norm(estimated - point, axis=1))))
    return float(np.mean(distances)) if distances else None


def _save_placeholder(
    output_dir: Path,
    base_name: str,
    title: str,
    message: str,
    artifact_prefix: str = "",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(PAGE_BG)
    ax.axis("off")
    ax.set_title(title, pad=16)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14, color=MUTED, transform=ax.transAxes)
    fig.tight_layout()
    path = _artifact_path(output_dir, base_name, ".png", artifact_prefix)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


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
    artifact_prefix: str = "",
) -> Path | None:
    if formulations.empty or observations.empty:
        return None

    evaluation_frames = _build_model_evaluation_frames(formulations, observations, registry)
    viability_cv = evaluation_frames["viability_percent"]
    load_cv = evaluation_frames["critical_axial_load_N_per_needle"]
    intact_cv = evaluation_frames["intact_patch_formation_pass"]
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

    path = _artifact_path(output_dir, "model_evaluation_overview", ".png", artifact_prefix)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_multiobjective_parity_plot(
    viability_predictions: pd.DataFrame,
    load_predictions: pd.DataFrame,
    output_dir: Path,
    artifact_prefix: str = "",
) -> Path:
    if viability_predictions.empty and load_predictions.empty:
        return _save_placeholder(
            output_dir,
            "multiobjective_paired_parity",
            "Multi-objective parity overview",
            "Not enough real paired data yet.",
            artifact_prefix=artifact_prefix,
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor(PAGE_BG)
    panels = [
        (axes[0], viability_predictions, "Viability parity", "Observed viability (%)", "Predicted viability (%)", BLUE),
        (axes[1], load_predictions, "Critical-load parity", "Observed critical load (N/needle)", "Predicted critical load (N/needle)", GOLD),
    ]
    for ax, frame, title, xlabel, ylabel, color in panels:
        if frame.empty:
            ax.text(0.5, 0.5, "Not enough data yet", ha="center", va="center", color=MUTED, transform=ax.transAxes)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            continue
        actual = frame["actual"].to_numpy(dtype=float)
        predicted = frame["predicted"].to_numpy(dtype=float)
        low = float(np.nanmin(np.concatenate([actual, predicted])))
        high = float(np.nanmax(np.concatenate([actual, predicted])))
        pad = max((high - low) * 0.08, 1e-6)
        ax.scatter(actual, predicted, s=70, color=color, alpha=0.85, edgecolor="white", linewidth=0.7)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], linestyle="--", color=MUTED, linewidth=1.4)
        r2 = float(r2_score(actual, predicted)) if len(frame) >= 2 and np.nanstd(actual) > 0 else np.nan
        ax.text(
            0.03,
            0.97,
            f"n={len(frame)}\nR²={r2:.3f}" if np.isfinite(r2) else f"n={len(frame)}\nR²=n/a",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": AX_BG, "edgecolor": GRID},
        )
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.suptitle("Observed vs predicted fits from the real multi-objective database", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = _artifact_path(output_dir, "multiobjective_paired_parity", ".png", artifact_prefix)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_candidate_plot(
    candidates: pd.DataFrame,
    output_dir: Path,
    artifact_prefix: str = "",
) -> Path | None:
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
    path = _artifact_path(output_dir, "next_round_candidate_screen", ".png", artifact_prefix)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _round_metrics(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    rounds = sorted(paired["batch_id"].unique(), key=_round_sort_key)
    normalized_reference = _normalize_frame(
        paired,
        ["viability_percent", "critical_axial_load_N_per_needle"],
        paired,
    )
    true_frontier = normalized_reference.loc[
        _pareto_frontier_mask(normalized_reference, "viability_percent", "critical_axial_load_N_per_needle")
    ].copy()
    final_hv = _hypervolume_2d(true_frontier, "viability_percent", "critical_axial_load_N_per_needle")
    rows = []
    for round_id in rounds:
        cumulative = paired.loc[paired["batch_id"].map(_round_sort_key) <= _round_sort_key(round_id)].copy()
        normalized = _normalize_frame(
            cumulative,
            ["viability_percent", "critical_axial_load_N_per_needle"],
            paired,
        )
        frontier = normalized.loc[
            _pareto_frontier_mask(normalized, "viability_percent", "critical_axial_load_N_per_needle")
        ].copy()
        hv = _hypervolume_2d(frontier, "viability_percent", "critical_axial_load_N_per_needle")
        igd = _igd_2d(true_frontier, frontier, "viability_percent", "critical_axial_load_N_per_needle")
        rows.append(
            {
                "batch_id": round_id,
                "paired_rows_cumulative": int(len(cumulative)),
                "pareto_points_cumulative": int(len(frontier)),
                "normalized_hypervolume": float(hv / final_hv) if final_hv > 0 else np.nan,
                "igd": igd,
            }
        )
    return pd.DataFrame(rows)


def _save_hv_igd_plot(metrics: pd.DataFrame, output_dir: Path, artifact_prefix: str = "") -> Path:
    if metrics.empty:
        return _save_placeholder(
            output_dir,
            "normalized_hypervolume_igd_vs_round",
            "Normalized hypervolume and IGD vs round",
            "No paired viability/load rounds yet.",
            artifact_prefix=artifact_prefix,
        )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    fig.patch.set_facecolor(PAGE_BG)
    x = np.arange(1, len(metrics) + 1)
    axes[0].plot(x, metrics["normalized_hypervolume"], marker="o", color=TEAL, linewidth=2)
    axes[0].set_title("Normalized hypervolume vs round")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Normalized hypervolume")
    axes[0].set_xticks(x, metrics["batch_id"], rotation=30, ha="right")

    axes[1].plot(x, metrics["igd"], marker="o", color=CORAL, linewidth=2)
    axes[1].set_title("IGD vs round")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("IGD")
    axes[1].set_xticks(x, metrics["batch_id"], rotation=30, ha="right")
    fig.tight_layout()
    path = _artifact_path(output_dir, "normalized_hypervolume_igd_vs_round", ".png", artifact_prefix)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_pareto_progression_plot(paired: pd.DataFrame, output_dir: Path, artifact_prefix: str = "") -> Path:
    if paired.empty:
        return _save_placeholder(
            output_dir,
            "pareto_front_progression",
            "Pareto-front progression",
            "No paired viability/load observations yet.",
            artifact_prefix=artifact_prefix,
        )
    rounds = sorted(paired["batch_id"].unique(), key=_round_sort_key)
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(rounds)))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(PAGE_BG)
    ax.scatter(
        paired["viability_percent"],
        paired["critical_axial_load_N_per_needle"],
        s=40,
        color=SLATE,
        alpha=0.18,
        edgecolor="none",
        label="All paired observations",
    )
    for color, round_id in zip(colors, rounds):
        cumulative = paired.loc[paired["batch_id"].map(_round_sort_key) <= _round_sort_key(round_id)].copy()
        frontier = cumulative.loc[
            _pareto_frontier_mask(cumulative, "viability_percent", "critical_axial_load_N_per_needle")
        ].sort_values("viability_percent")
        if frontier.empty:
            continue
        ax.plot(
            frontier["viability_percent"],
            frontier["critical_axial_load_N_per_needle"],
            color=color,
            linewidth=2,
            marker="o",
            label=round_id,
        )
    ax.set_title("Pareto-front progression across real multi-objective rounds")
    ax.set_xlabel("Observed viability (%)")
    ax.set_ylabel("Observed critical load (N/needle)")
    ax.legend(frameon=False, loc="best", fontsize=8)
    fig.tight_layout()
    path = _artifact_path(output_dir, "pareto_front_progression", ".png", artifact_prefix)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_endpoint_r2_plot(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    registry: IngredientRegistry,
    artifact_prefix: str = "",
) -> Path:
    if metrics.empty:
        return _save_placeholder(
            output_dir,
            "endpoint_r2_vs_round",
            "Endpoint R² vs round",
            "No paired viability/load rounds yet.",
            artifact_prefix=artifact_prefix,
        )

    rounds = metrics["batch_id"].tolist()
    rows = []
    for round_id in rounds:
        cumulative_obs = observations.loc[observations["batch_id"].map(_round_sort_key) <= _round_sort_key(round_id)].copy()
        cumulative_frame = build_training_frame(formulations, cumulative_obs, registry)
        paired_keys = cumulative_frame.dropna(subset=["viability_percent", "critical_axial_load_N_per_needle"])[["formulation_id", "batch_id"]]
        if paired_keys.empty:
            rows.append({"batch_id": round_id, "viability_r2": np.nan, "load_r2": np.nan})
            continue
        allowed = set((str(row.formulation_id), str(row.batch_id)) for row in paired_keys.itertuples())
        filtered_obs = cumulative_obs.loc[
            cumulative_obs.apply(lambda row: (str(row.get("formulation_id", "")), str(row.get("batch_id", ""))) in allowed, axis=1)
        ].copy()
        viability_predictions = _cross_validated_predictions(formulations, filtered_obs, registry, "viability_percent")
        load_predictions = _cross_validated_predictions(
            formulations,
            filtered_obs,
            registry,
            "critical_axial_load_N_per_needle",
        )
        viability_r2 = (
            float(r2_score(viability_predictions["actual"], viability_predictions["predicted"]))
            if len(viability_predictions) >= 2 and np.nanstd(viability_predictions["actual"]) > 0
            else np.nan
        )
        load_r2 = (
            float(r2_score(load_predictions["actual"], load_predictions["predicted"]))
            if len(load_predictions) >= 2 and np.nanstd(load_predictions["actual"]) > 0
            else np.nan
        )
        rows.append({"batch_id": round_id, "viability_r2": viability_r2, "load_r2": load_r2})
    r2_frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.patch.set_facecolor(PAGE_BG)
    x = np.arange(1, len(r2_frame) + 1)
    ax.plot(x, r2_frame["viability_r2"], marker="o", linewidth=2, color=BLUE, label="Viability R²")
    ax.plot(x, r2_frame["load_r2"], marker="o", linewidth=2, color=GOLD, label="Critical-load R²")
    ax.set_title("Cross-validated endpoint R² vs cumulative round")
    ax.set_xlabel("Round")
    ax.set_ylabel("R²")
    ax.set_xticks(x, r2_frame["batch_id"], rotation=30, ha="right")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    path = _artifact_path(output_dir, "endpoint_r2_vs_round", ".png", artifact_prefix)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_multiobjective_summary(
    output_dir: Path,
    paired: pd.DataFrame,
    metrics: pd.DataFrame,
    generated_paths: list[Path],
    artifact_prefix: str = "",
) -> Path:
    summary_path = _artifact_path(output_dir, "multiobjective_evaluation_summary", ".txt", artifact_prefix)
    lines = [
        "CryoMN v2 Multi-Objective Evaluation",
        "===================================",
        "",
        f"Paired viability/load rows: {len(paired)}",
        f"Distinct paired batches: {int(paired['batch_id'].nunique()) if not paired.empty else 0}",
        f"Distinct paired formulations: {int(paired['formulation_id'].nunique()) if not paired.empty else 0}",
        "",
    ]
    if metrics.empty:
        lines.append(
            "Not enough real paired viability-plus-load observations are currently present to compute roundwise hypervolume, IGD, or Pareto progression."
        )
    else:
        last = metrics.iloc[-1]
        lines.extend(
            [
                f"Latest cumulative normalized hypervolume: {last['normalized_hypervolume']:.4f}",
                f"Latest cumulative IGD: {last['igd']:.4f}" if pd.notna(last["igd"]) else "Latest cumulative IGD: n/a",
                "",
                "Generated files:",
            ]
        )
        lines.extend(f"- {path.name}" for path in generated_paths)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _write_visualization_summary(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
    generated: list[Path],
    output_dir: Path,
    review_label: str = "",
    artifact_prefix: str = "",
) -> Path:
    observed = _observed_endpoint_frame(formulations, observations)
    summary = [
        "CryoMN v2 Round Review Summary",
        "=" * 30,
        "",
        f"Review label: {review_label or 'default'}",
        f"Formulation rows: {len(formulations)}",
        f"Observation rows: {len(observations)}",
        f"Candidate rows: {len(candidates)}",
        f"Retest-priority rows in current slate: {int((candidates.get('recommendation_type', pd.Series([''] * len(candidates), index=candidates.index, dtype='object')).astype(str) == 'retest_priority').sum()) if not candidates.empty else 0}",
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
    summary.append("This review snapshot captures one specific state of the round workflow.")
    output_path = _artifact_path(output_dir, "visualization_summary", ".txt", artifact_prefix)
    output_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return output_path


def generate_visualization_artifacts(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
    output_dir: str | Path,
    review_label: str = "",
    artifact_prefix: str = "",
) -> list[Path]:
    _apply_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    evaluation_frames = _build_model_evaluation_frames(formulations, observations, registry)

    generated = [
        path
        for path in [
            _write_best_performers_summary(
                formulations,
                observations,
                candidates,
                output_dir,
                registry,
                artifact_prefix=artifact_prefix,
            ),
            _save_endpoint_counts(observations, output_dir, artifact_prefix=artifact_prefix),
            _save_observed_performance_landscape(
                formulations,
                observations,
                output_dir,
                artifact_prefix=artifact_prefix,
            ),
            _write_model_evaluation_table(
                evaluation_frames,
                output_dir,
                artifact_prefix=artifact_prefix,
            ),
            _save_model_evaluation_overview(
                formulations,
                observations,
                output_dir,
                registry,
                artifact_prefix=artifact_prefix,
            ),
            _save_candidate_plot(candidates, output_dir, artifact_prefix=artifact_prefix),
        ]
        if path is not None
    ]
    generated.append(
        _write_visualization_summary(
            formulations,
            observations,
            candidates,
            generated,
            output_dir,
            review_label=review_label,
            artifact_prefix=artifact_prefix,
        )
    )
    return generated


def generate_multiobjective_evaluation_artifacts(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    output_dir: str | Path,
    artifact_prefix: str = "",
) -> list[Path]:
    _apply_plot_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry()

    paired = _paired_frame(formulations, observations, registry)
    metrics = _round_metrics(paired)
    metrics_path = _artifact_path(output_dir, "multiobjective_round_metrics", ".csv", artifact_prefix)
    metrics.to_csv(metrics_path, index=False)

    paired_obs = pd.DataFrame()
    if not paired.empty:
        paired_keys = set((str(row.formulation_id), str(row.batch_id)) for row in paired.itertuples())
        paired_obs = observations.loc[
            observations.apply(
                lambda row: (str(row.get("formulation_id", "")), str(row.get("batch_id", ""))) in paired_keys,
                axis=1,
            )
        ].copy()

    viability_predictions = (
        _cross_validated_predictions(formulations, paired_obs, registry, "viability_percent")
        if not paired_obs.empty
        else pd.DataFrame()
    )
    load_predictions = (
        _cross_validated_predictions(formulations, paired_obs, registry, "critical_axial_load_N_per_needle")
        if not paired_obs.empty
        else pd.DataFrame()
    )

    generated = [
        _save_multiobjective_parity_plot(
            viability_predictions,
            load_predictions,
            output_dir,
            artifact_prefix=artifact_prefix,
        ),
        _save_hv_igd_plot(metrics, output_dir, artifact_prefix=artifact_prefix),
        _save_pareto_progression_plot(paired, output_dir, artifact_prefix=artifact_prefix),
        _save_endpoint_r2_plot(
            formulations,
            observations,
            metrics,
            output_dir,
            registry,
            artifact_prefix=artifact_prefix,
        ),
    ]
    generated.append(
        _write_multiobjective_summary(
            output_dir,
            paired,
            metrics,
            generated + [metrics_path],
            artifact_prefix=artifact_prefix,
        )
    )
    generated.append(metrics_path)
    return generated


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["review", "multiobjective", "all"],
        default="all",
        help="Visualization entrypoint to run from the command line.",
    )
    parser.add_argument("--formulations", default=str(FORMULATIONS_PATH))
    parser.add_argument("--observations", default=str(OBSERVATIONS_PATH))
    parser.add_argument("--candidates", default=str(NEXT_ROUND_CANDIDATES_PATH))
    parser.add_argument("--output-dir", default=str(VISUALIZATIONS_DIR))
    parser.add_argument("--artifact-prefix", default="")
    parser.add_argument("--review-label", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    formulations = _read_or_empty(args.formulations)
    observations = _read_or_empty(args.observations)
    generated: list[Path] = []
    if args.mode in {"review", "all"}:
        candidates = _read_or_empty(args.candidates)
        generated.extend(
            generate_visualization_artifacts(
                formulations,
                observations,
                candidates,
                args.output_dir,
                review_label=args.review_label,
                artifact_prefix=args.artifact_prefix,
            )
        )
    if args.mode in {"multiobjective", "all"}:
        generated.extend(
            generate_multiobjective_evaluation_artifacts(
                formulations,
                observations,
                args.output_dir,
                artifact_prefix=args.artifact_prefix,
            )
        )
    print(f"Wrote {len(generated)} visualization artifact(s) to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()

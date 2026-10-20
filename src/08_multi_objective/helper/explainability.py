"""Report-only endpoint explanations with explicit historical training cutoffs.

Frozen viability comparison GPs can be explained directly. Other models are
retrospective refits, never replacements for the frozen prospective forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import textwrap
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from .batch_gp import BatchGP, matern
from .campaign_plots import BLUE, RED, finish_axis, save_png, style
from .config import load_optimization_config
from .gp_comparison import attach_roles, round_numbers
from .gp_strategy import fit_viability, viability_rows
from .group10_config import production_observations
from .models import build_training_frame, _fit_classifier, _fit_regression
from .registry import FEATURE_NEGLIGIBLE_THRESHOLD, IngredientRegistry


ENDPOINTS = {
    "viability_percent": ("Viability", "percentage points", "objective"),
    "critical_axial_load_N_per_needle": ("Mechanical resistance", "N per needle", "objective"),
    "intact_patch_formation_pass": ("Intact patch probability", "probability", "diagnostic gate model"),
    "initial_stiffness_N_per_mm_per_needle": ("Initial stiffness", "N/mm per needle", "diagnostic"),
    "preparation_feasibility_pass": ("Preparation probability", "probability", "diagnostic gate model"),
}
METHOD_URL = "https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html"


@dataclass(frozen=True)
class ExplanationSettings:
    max_explain: int = 48
    max_background: int = 16
    permutations: int = 16
    top_features: int = 12
    seed: int = 42

    def __post_init__(self):
        for key in ("max_explain", "max_background", "permutations", "top_features"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preceding_observations(observations: pd.DataFrame, target_round: int) -> pd.DataFrame:
    """Keep baseline legacy evidence and strictly preceding campaign rounds."""
    numbers = round_numbers(observations)
    return observations.loc[numbers.isna() | numbers.lt(target_round)].copy()


def _check_cutoff(observations: pd.DataFrame, target_round: int) -> None:
    if round_numbers(observations).ge(target_round).any():
        raise ValueError("Historical training evidence contains target/future outcomes")


def load_round_evidence(root: Path, target_round: int, observations: pd.DataFrame,
                        features: list[str]) -> tuple[pd.DataFrame, BatchGP | None, dict]:
    directory = root / "results/multi_objective_v2/rounds" / f"ROUND_{target_round:03d}"
    decision_path = directory / "proposal/gp_strategy_decision.json"
    decision = json.loads(decision_path.read_text()) if decision_path.exists() else {}
    if target_round >= 11 and not decision:
        raise ValueError(f"Missing recorded GP strategy decision for ROUND_{target_round:03d}")
    strategy = decision.get("viability_strategy", "current")
    info = {"viability_strategy": strategy, "decision_id": decision.get("decision_id", "historical_current"),
            "sources": {}, "training_evidence": "canonical_data_at_cutoff"}
    if decision_path.exists():
        info["sources"][str(decision_path.relative_to(root))] = digest(decision_path)
    prior = preceding_observations(observations, target_round)
    frozen = directory / "gp_comparison"
    snapshot = None
    manifest_path = frozen / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["cutoff"] != target_round - 1 or manifest["batch_id"] != directory.name
                or manifest["features"] != features):
            raise ValueError("Frozen GP comparison cutoff, round or feature schema mismatch")
        training = frozen / "training_observations.csv"
        if digest(training) != manifest["training_sha256"]:
            raise ValueError("Frozen training evidence hash mismatch")
        prior = pd.read_csv(training)
        info["training_evidence"] = "archived_training_observations"
        info["sources"][str(training.relative_to(root))] = digest(training)
        info["sources"][str(manifest_path.relative_to(root))] = digest(manifest_path)
        model_path = frozen / (strategy + ".json")
        if model_path.exists():
            if digest(model_path) != manifest["hashes"][model_path.name]:
                raise ValueError("Frozen GP model hash mismatch")
            snapshot = BatchGP.load(model_path)
            if snapshot.strategy != strategy or snapshot.x.shape[1] != len(features):
                raise ValueError("Frozen GP model strategy or feature schema mismatch")
            if round_numbers(pd.DataFrame({"batch_id": snapshot.batches})).ge(target_round).any():
                raise ValueError("Frozen GP model contains target/future outcomes")
            info["sources"][str(model_path.relative_to(root))] = digest(model_path)
    _check_cutoff(prior, target_round)
    # Entirely blank historical CSV fields otherwise load as float under pandas.
    prior["experimental_role"] = prior.get("experimental_role", pd.Series("", index=prior.index)).fillna("").astype(str)
    return prior, snapshot, info


def batch_gp_mean(model: BatchGP) -> Callable:
    """Same formulation posterior mean as BatchGP.predict, without an NxN covariance.

    Ordinary formulations only: fresh batch offsets have expectation zero, and
    the control intercept is not attributed to formulation ingredients.
    """
    ordinary = ~model.controls
    train = (model.x[ordinary] - model.center) / model.scale
    coefficients = model.alpha[ordinary]

    def predict(x):
        x = np.asarray(x, dtype=float)
        return model.mean + model.amplitude ** 2 * matern((x - model.center) / model.scale, train) @ coefficients
    return predict


def sample_rows(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if len(frame) <= count:
        return frame.reset_index(drop=True)
    indices = np.sort(np.random.default_rng(seed).choice(len(frame), count, replace=False))
    return frame.iloc[indices].reset_index(drop=True)


def compute_shap(predict: Callable, explained: np.ndarray, background: np.ndarray,
                 settings: ExplanationSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is required for this report; install the project's requirements.txt") from exc
    # SHAP seeds NumPy's global RNG; preserve callers' state in this report helper.
    state = np.random.get_state()
    try:
        masker = shap.maskers.Independent(background, max_samples=len(background))
        explainer = shap.PermutationExplainer(predict, masker, seed=settings.seed)
        explanation = explainer(explained, max_evals=(2 * explained.shape[1] + 1) * settings.permutations,
                                batch_size=1024, silent=True)
    finally:
        np.random.set_state(state)
    values = np.asarray(explanation.values, float)
    base = np.broadcast_to(np.asarray(explanation.base_values, float), (len(explained),)).copy()
    predicted = np.asarray(predict(explained), float)
    if not np.isfinite(values).all() or not np.isfinite(base).all() or not np.isfinite(predicted).all():
        raise ValueError("Nonfinite SHAP explanation")
    if not np.allclose(base + values.sum(axis=1), predicted, rtol=1e-6, atol=1e-8):
        raise ValueError("SHAP additivity check failed")
    return values, base, predicted


def endpoint_predictors(forms: pd.DataFrame, prior: pd.DataFrame, registry: IngredientRegistry,
                        target_round: int, snapshot: BatchGP | None, strategy: str, config: dict):
    clean = production_observations(prior, target_round_number=target_round)
    frame = build_training_frame(forms, clean, registry)
    features = registry.feature_names
    result = {}
    for endpoint in ENDPOINTS:
        classifier = endpoint.endswith("_pass")
        rows = frame.loc[pd.to_numeric(frame.get(endpoint, pd.Series(index=frame.index, dtype=float)),
                                       errors="coerce").notna()].copy()
        evidence = "retrospective_refit"
        if endpoint == "viability_percent" and snapshot is not None:
            if strategy != "current":
                rows = viability_rows(forms, prior, features)
                rows = rows.loc[~rows.is_control].copy()
            # Retain real formulation/batch identities and verify their exact alignment.
            if (len(rows) != (~snapshot.controls).sum()
                    or not np.allclose(rows[features].to_numpy(float), snapshot.x[~snapshot.controls], rtol=1e-12, atol=1e-12)
                    or not np.array_equal(rows.batch_id.to_numpy(str), snapshot.batches[~snapshot.controls])):
                raise ValueError("Archived GP training rows do not match reconstructed formulation identities")
            predict = batch_gp_mean(snapshot)
            evidence = "frozen_gp_comparison_snapshot"
        elif endpoint == "viability_percent" and strategy != "current":
            rows = viability_rows(forms, prior, features)
            rows = rows.loc[~rows.is_control].copy()
            model = fit_viability(forms, prior, features, strategy)
            predict = batch_gp_mean(model)
        else:
            minimum = (int(config.get("preparation_model", {}).get("min_labels", 8))
                       if endpoint == "preparation_feasibility_pass" else 4 if classifier else 2)
            if len(rows) < minimum:
                result[endpoint] = {"status": "unavailable", "reason": f"{len(rows)} training rows; requires {minimum}",
                                    "training_rows": len(rows), "model_evidence": evidence}
                continue
            x = rows[features].to_numpy(float)
            if classifier:
                if rows[endpoint].nunique() < 2:
                    result[endpoint] = {"status": "unavailable", "reason": "Only one observed class; constant gate model",
                                        "training_rows": len(rows), "model_evidence": evidence}
                    continue
                model = _fit_classifier(x, rows[endpoint], min_samples=minimum, require_both_classes=True)
                predict = model.predict_proba
            else:
                model = _fit_regression(x, rows[endpoint], default_mean=50. if endpoint == "viability_percent" else 0.,
                                        y_noise=rows.get(endpoint + "__noise"))
                predict = model.model.predict
        result[endpoint] = {"status": "available", "rows": rows, "predict": predict,
                            "training_rows": len(rows), "model_evidence": evidence,
                            "mechanical_definition_id": clean.attrs.get("mechanical_definition_id")
                                if endpoint in ("critical_axial_load_N_per_needle", "initial_stiffness_N_per_mm_per_needle") else None}
    return result


def feature_label(registry: IngredientRegistry, feature: str) -> str:
    ingredient = registry.get_by_feature(feature)
    if feature == "methoxyphenyl_beta_d_glucopyranoside_M":
        return "4-methoxyphenyl\nβ-D-glucopyranoside"
    return textwrap.fill(ingredient.display_name, width=20, break_long_words=False)


def summary_figure(importance: pd.DataFrame, statuses: dict, registry: IngredientRegistry,
                   context: str, top_features: int):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.7))
    for ax, endpoint in zip(axes, list(ENDPOINTS)[:3]):
        title, unit, _ = ENDPOINTS[endpoint]
        status = statuses[endpoint]
        ax.set_title(title, fontsize=12, weight="bold", pad=16)
        if status["status"] != "available":
            ax.text(.5, .5, "Unavailable\n" + status["reason"], transform=ax.transAxes,
                    ha="center", va="center", wrap=True, fontsize=10)
            ax.set_axis_off()
            continue
        table = importance.loc[importance.endpoint.eq(endpoint)].nlargest(top_features, "mean_abs_shap").iloc[::-1]
        ax.barh(range(len(table)), table.mean_abs_shap, color=BLUE, height=.68)
        ax.set_yticks(range(len(table)), [feature_label(registry, f) for f in table.feature])
        ax.set_xlabel(f"Mean |SHAP| ({unit})")
        ax.set_xlim(left=0)
        finish_axis(ax, "x")
        ax.text(0, 1.015, f"n={status['explained_rows']} explained; {status['training_rows']} training rows",
                transform=ax.transAxes, fontsize=8, color="#666666")
    fig.suptitle("Ingredient influence on endpoint predictions", fontsize=17, weight="bold", y=.98)
    fig.text(.5, .915, context, ha="center", fontsize=10)
    fig.text(.04, .025, "Separate axis scales and endpoint-specific training backgrounds. Mean |SHAP| measures model influence, not causality.\n"
             "Intact classifier is diagnostic; active selection can use a separate empirical feasibility policy.", fontsize=9)
    fig.subplots_adjust(left=.12, right=.985, bottom=.19, top=.81, wspace=.85)
    return fig


def beeswarm_figure(values: np.ndarray, x: np.ndarray, features: list[str], registry: IngredientRegistry,
                    endpoint: str, context: str, top_features: int):
    style()
    order = np.argsort(np.mean(np.abs(values), axis=0))[-top_features:]
    fig, ax = plt.subplots(figsize=(9, 6.7))
    rng = np.random.default_rng(42)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("feature_value", [BLUE, "#dedede", RED])
    for row, column in enumerate(order):
        raw = x[:, column]
        span = np.ptp(raw)
        colors = (raw - raw.min()) / span if span > 0 else np.full(len(raw), .5)
        ax.scatter(values[:, column], row + rng.uniform(-.23, .23, len(x)), c=colors, cmap=cmap,
                   vmin=0, vmax=1, s=23, alpha=.8, edgecolors="none")
    ax.axvline(0, color="#666666", lw=.8)
    ax.set_yticks(range(len(order)), [feature_label(registry, features[i]) for i in order])
    ax.set_xlabel(f"SHAP contribution ({ENDPOINTS[endpoint][1]})")
    ax.set_title(ENDPOINTS[endpoint][0] + " | directional ingredient influence", fontsize=13, weight="bold", pad=35)
    ax.text(.5, 1.035, context, transform=ax.transAxes, ha="center", fontsize=9)
    finish_axis(ax, "x")
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=ax, pad=.025, fraction=.035)
    colorbar.set_ticks([0, 1], labels=["Low", "High"])
    colorbar.set_label("Ingredient value (within feature)")
    fig.text(.04, .025, "Positive contributions raise the prediction relative to the endpoint's background mean.\n"
             "Interventional masking can combine correlated ingredients into untested mixtures; these are model explanations.", fontsize=9)
    fig.subplots_adjust(left=.31, right=.88, bottom=.18, top=.81)
    return fig


def explain_round(root: Path, forms: pd.DataFrame, observations: pd.DataFrame, registry: IngredientRegistry,
                  target_round: int, output: Path, settings: ExplanationSettings) -> dict:
    features = registry.feature_names
    prior, snapshot, provenance = load_round_evidence(root, target_round, observations, features)
    prior = attach_roles(prior, root)
    predictors = endpoint_predictors(forms, prior, registry, target_round, snapshot,
                                     provenance["viability_strategy"], load_optimization_config(root / "config_v2/optimization.yaml"))
    output.mkdir(parents=True, exist_ok=True)
    cutoff_label = f"data through Round {target_round - 1}" if target_round > 1 else "baseline legacy evidence"
    context = f"Before Round {target_round} | {cutoff_label} | viability GP: {provenance['viability_strategy']}"
    statuses, tables, contributions = {}, [], []
    for endpoint, item in predictors.items():
        status = {key: value for key, value in item.items() if key not in ("predict", "rows")}
        statuses[endpoint] = status
        if item["status"] != "available":
            continue
        rows = item["rows"].reset_index(drop=True)
        explained = sample_rows(rows, settings.max_explain, settings.seed)
        background = sample_rows(rows, settings.max_background, settings.seed + 1)
        x = explained[features].to_numpy(float)
        values, base, predicted = compute_shap(item["predict"], x, background[features].to_numpy(float), settings)
        status.update(explained_rows=len(explained), background_rows=len(background),
                      additivity_max_absolute_error=float(np.max(np.abs(base + values.sum(1) - predicted))),
                      background_prediction=float(base[0]))
        magnitude = np.abs(values).mean(axis=0)
        total = float(magnitude.sum())
        table = pd.DataFrame({"endpoint": endpoint, "feature": features, "mean_abs_shap": magnitude,
                              "mean_signed_shap": values.mean(axis=0),
                              "relative_importance_percent": magnitude / total * 100 if total > 0 else np.zeros(len(features))})
        tables.append(table)
        for row in range(len(explained)):
            contributions.append(pd.DataFrame({"endpoint": endpoint, "row_index": row,
                "formulation_id": explained.iloc[row].formulation_id, "batch_id": explained.iloc[row].batch_id,
                "feature": features, "feature_value": x[row], "shap_value": values[row],
                "base_value": base[row], "prediction": predicted[row]}))
        explained.assign(base_value=base, prediction=predicted).to_csv(output / (endpoint + "_explained_rows.csv"), index=False)
        background.to_csv(output / (endpoint + "_background.csv"), index=False)
        fig = beeswarm_figure(values, x, features, registry, endpoint, context, settings.top_features)
        save_png(fig, output / (endpoint + "_shap_summary.png"))
    importance = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=["endpoint", "feature", "mean_abs_shap", "mean_signed_shap", "relative_importance_percent"])
    importance.insert(0, "target_round", target_round)
    importance["viability_strategy"] = provenance["viability_strategy"]
    importance.to_csv(output / "feature_importance.csv", index=False)
    shap_values = pd.concat(contributions, ignore_index=True) if contributions else pd.DataFrame()
    if not shap_values.empty:
        shap_values.to_csv(output / "shap_values.csv", index=False)
    save_png(summary_figure(importance, statuses, registry, context, settings.top_features), output / "feature_importance.png")
    sources = {str(p.relative_to(root)): digest(p) for p in
               [root / "data/processed_v2/formulations.csv", root / "data/processed_v2/observations.csv",
                root / "config_v2/ingredients.yaml", root / "config_v2/optimization.yaml"]}
    manifest = {"report_version": 1, "target_round": target_round, "training_cutoff": target_round - 1,
                **provenance, "sources": {**sources, **provenance["sources"]},
                "method": "SHAP PermutationExplainer; independent empirical background; identity link",
                "method_reference": METHOD_URL, "shap_version": __import__("shap").__version__,
                "code_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                ("explainability.py", "models.py", "batch_gp.py", "gp_strategy.py", "group10_config.py")},
                "settings": settings.__dict__, "endpoints": statuses,
                "limitations": ["Attributions explain model predictions, not causal effects or predictive accuracy.",
                    "No acquisition function, uncertainty bonus, penalties or empirical gate policy is explained.",
                    "Retrospective refits use today's implementation/configuration and preceding labels, not a preserved historical production model.",
                    "Frozen GPs are archived comparison models matching the recorded strategy; their provenance is separate from production proposal forecasts.",
                    "Legacy baseline evidence is assumed available before campaign Round 1, consistent with existing retrospective audits.",
                    "Backgrounds and explained training populations vary across rounds and endpoints; rankings are descriptive, not a controlled longitudinal comparison.",
                    "Interventional masking can create untested or infeasible mixtures and does not preserve ingredient correlations.",
                    "Permutation estimates are sampled; no statistical confidence intervals or causal claims are implied."],
                "generated_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest": manifest, "importance": importance, "shap_values": shap_values,
            "shap_source": str(output / "shap_values.csv")}


def load_saved_reports(directory: Path, include_shap_values: bool = False) -> list[dict]:
    """Read verified SHAP summaries for rendering without loading or fitting models."""
    reports = []
    targets = set()
    for path in sorted(directory.glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        source = path.parent / "feature_importance.csv"
        if digest(source) != manifest["generated_sha256"][source.name]:
            raise ValueError(f"Stored feature importance hash mismatch: {source}")
        table = pd.read_csv(source)
        target = manifest["target_round"]
        if target in targets or (not table.empty and set(table.target_round) != {target}):
            raise ValueError("Stored explanation rounds are duplicated or inconsistent")
        targets.add(target)
        report = {"manifest": manifest, "importance": table}
        if include_shap_values:
            shap_source = path.parent / "shap_values.csv"
            expected = manifest["generated_sha256"].get(shap_source.name)
            if expected is None or not shap_source.is_file() or digest(shap_source) != expected:
                raise ValueError(f"Stored row-level SHAP hash mismatch: {shap_source}")
            report.update(shap_values=pd.read_csv(shap_source), shap_source=str(shap_source))
        reports.append(report)
    if not reports:
        raise ValueError(f"No saved explanation reports in {directory}; run --all-rounds first")
    return sorted(reports, key=lambda report: report["manifest"]["target_round"])


def signed_history_table(reports: list[dict]) -> pd.DataFrame:
    """Net mean SHAP, scaled by total absolute influence per endpoint and round."""
    table = pd.concat([report["importance"] for report in reports], ignore_index=True)
    total = table.groupby(["target_round", "endpoint"]).mean_abs_shap.transform("sum").to_numpy(float)
    table["signed_relative_contribution_percent"] = np.divide(
        100 * table.mean_signed_shap.to_numpy(float), total,
        out=np.zeros(len(table)), where=total > 0,
    )
    return table


def write_history(reports: list[dict], directory: Path, registry: IngredientRegistry, settings: ExplanationSettings):
    """Signed endpoint-normalized means; unavailable rounds remain missing."""
    table = signed_history_table(reports)
    table.to_csv(directory / "feature_importance_history.csv", index=False)
    style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.8))
    rounds = [report["manifest"]["target_round"] for report in reports]
    shown = table.loc[table.endpoint.isin(list(ENDPOINTS)[:3])]
    maximum = float(shown.signed_relative_contribution_percent.abs().max()) if not shown.empty else 0.
    color_limit = min(100., max(10., np.ceil(maximum / 10) * 10))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("signed_influence", [BLUE, "#ffffff", RED]).with_extremes(bad="#eeeeee")
    for ax, endpoint in zip(axes, list(ENDPOINTS)[:3]):
        data = table.loc[table.endpoint.eq(endpoint)]
        ax.set_title(ENDPOINTS[endpoint][0], fontsize=12, weight="bold", pad=15)
        if data.empty:
            ax.text(.5, .5, "No estimable models", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        top = data.groupby("feature").relative_importance_percent.mean().nlargest(settings.top_features).index
        matrix = data.pivot(index="feature", columns="target_round", values="signed_relative_contribution_percent").reindex(index=top, columns=rounds)
        ax.imshow(np.ma.masked_invalid(matrix.to_numpy(float)), aspect="auto", cmap=cmap, vmin=-color_limit, vmax=color_limit)
        ax.set_yticks(range(len(top)), [feature_label(registry, f) for f in top])
        ax.set_xticks(range(len(rounds)), rounds)
        ax.set_xlabel("Model before round")
        if endpoint == "viability_percent":
            strategies = [report["manifest"]["viability_strategy"] for report in reports]
            for index in range(1, len(strategies)):
                if strategies[index] != strategies[index - 1]:
                    ax.axvline(index - .5, color="#555555", ls="--", lw=1.1)
    color_axis = fig.add_axes([.94, .24, .012, .55])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(-color_limit, color_limit), cmap=cmap), cax=color_axis,
                 label="Mean SHAP / total mean |SHAP| (%)")
    fig.suptitle("Signed ingredient influence across historical model cutoffs", fontsize=17, weight="bold", y=.98)
    fig.text(.04, .025, "Blue: lowers predictions; orange: raises predictions relative to the background. Grey: model unavailable.\n"
             "Colour = 100 × mean(SHAP) / total mean |SHAP|. Features selected by absolute influence; opposing contributions can cancel.\n"
             "Dashed line: viability GP strategy change. Background populations also change; this is a descriptive comparison.", fontsize=9)
    fig.subplots_adjust(left=.12, right=.91, bottom=.17, top=.87, wspace=.75)
    save_png(fig, directory / "feature_importance_history.png")
    metadata = {"report_version": 2, "source_sha256": digest(directory / "feature_importance_history.csv"),
                "value_column": "signed_relative_contribution_percent",
                "value_definition": "100 * mean(SHAP_j) / sum_k mean(abs(SHAP_k)), per endpoint and target round",
                "feature_selection": "Largest average relative absolute importance across rounds",
                "interpretation": "Net contribution relative to the model background; opposing row contributions can cancel; signed values do not sum to 100%",
                "target_rounds": rounds, "color_scale_percent": [-color_limit, color_limit],
                "color_legend": {"negative": "blue; lowers prediction", "zero": "white; zero net contribution",
                                 "positive": "orange; raises prediction"},
                "renderer_sha256": digest(Path(__file__)),
                "missing_cells": "model unavailable", "format": "transparent 300 dpi PNG",
                "model_evidence": [{key: report["manifest"][key] for key in
                                    ("target_round", "training_cutoff", "viability_strategy", "training_evidence")} for report in reports]}
    (directory / "feature_importance_history_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def presence_history_table(reports: list[dict], registry: IngredientRegistry) -> pd.DataFrame:
    """Condition existing SHAP values on presence; count explained formulation/rounds.

    Backgrounds and original SHAP values are retained. Conditioning selects rows;
    it does not redefine the SHAP game or estimate an ingredient intervention.
    """
    records = []
    for report in reports:
        manifest = report["manifest"]
        shap = report["shap_values"]
        if not shap.empty and shap.duplicated(["endpoint", "row_index", "feature"]).any():
            raise ValueError("Duplicate row-level SHAP contributions would inflate presence counts")
        for endpoint in list(ENDPOINTS)[:2]:
            status = manifest["endpoints"][endpoint]
            available = status["status"] == "available"
            for feature in registry.feature_names:
                item = {"target_round": manifest["target_round"], "training_cutoff": manifest["training_cutoff"],
                        "viability_strategy": manifest["viability_strategy"], "endpoint": endpoint,
                        "feature": feature, "ingredient": registry.get_by_feature(feature).display_name,
                        "ingredient_unit": registry.get_by_feature(feature).unit, "shap_unit": ENDPOINTS[endpoint][1],
                        "model_evidence": status["model_evidence"], "training_rows": status["training_rows"],
                        "n_explained": status.get("explained_rows", 0), "n_present": 0,
                        "n_distinct_formulations_present": 0, "mean_shap_present": np.nan,
                        "mean_abs_shap_present": np.nan, "positive_shap_fraction_present": np.nan,
                        "feature_value_min_present": np.nan, "feature_value_max_present": np.nan,
                        "status": "model_unavailable" if not available else "no_present_explained_rows"}
                if available:
                    rows = shap.loc[shap.endpoint.eq(endpoint) & shap.feature.eq(feature)]
                    if (len(rows) != item["n_explained"]
                            or not np.isfinite(rows[["feature_value", "shap_value"]].to_numpy(float)).all()):
                        raise ValueError(f"Incomplete or nonfinite SHAP rows for {endpoint}/{feature}")
                    present = rows.loc[rows.feature_value.gt(FEATURE_NEGLIGIBLE_THRESHOLD)]
                    if not present.empty:
                        item.update(status="estimated", n_present=len(present),
                                    n_distinct_formulations_present=present.formulation_id.nunique(),
                                    mean_shap_present=present.shap_value.mean(),
                                    mean_abs_shap_present=present.shap_value.abs().mean(),
                                    positive_shap_fraction_present=present.shap_value.gt(0).mean(),
                                    feature_value_min_present=present.feature_value.min(),
                                    feature_value_max_present=present.feature_value.max())
                records.append(item)
    return pd.DataFrame(records)


def presence_history_figure(table: pd.DataFrame, registry: IngredientRegistry, endpoints: list[str]):
    """Every ingredient, native endpoint units, and a presence count in every cell."""
    style()
    features = registry.feature_names
    rounds = {}
    for endpoint in endpoints:
        available = table.loc[table.endpoint.eq(endpoint) & table.status.ne("model_unavailable")]
        rounds[endpoint] = sorted(available.target_round.unique())
    widths = [max(len(rounds[endpoint]), 5) for endpoint in endpoints]
    fig = plt.figure(figsize=(max(12, sum(widths) * .9 + 4), len(features) * .43 + 3.7))
    grid = fig.add_gridspec(1, len(endpoints), width_ratios=widths,
                           left=.14 if len(endpoints) > 1 else .25, right=.96,
                           bottom=.21, top=.84, wspace=.035)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("presence_influence", [BLUE, "#ffffff", RED]).with_extremes(bad="#e5e5e5")
    scales = {}
    axes = []
    for index, endpoint in enumerate(endpoints):
        ax = fig.add_subplot(grid[0, index], sharey=axes[0] if axes else None)
        axes.append(ax)
        data = table.loc[table.endpoint.eq(endpoint)]
        ax.set_title(ENDPOINTS[endpoint][0], fontsize=15, weight="bold", pad=16)
        if not rounds[endpoint]:
            ax.text(.5, .5, "No available model cutoffs", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        values = data.pivot(index="feature", columns="target_round", values="mean_shap_present").reindex(index=features, columns=rounds[endpoint])
        counts = data.pivot(index="feature", columns="target_round", values="n_present").reindex(index=features, columns=rounds[endpoint])
        finite = values.to_numpy(float)
        maximum = float(np.nanmax(np.abs(finite))) if np.isfinite(finite).any() else 0.
        limit = max(5., np.ceil(maximum / 5) * 5) if endpoint == "viability_percent" else max(.01, np.ceil(maximum / .01) * .01)
        scales[endpoint] = [-limit, limit]
        norm = Normalize(-limit, limit)
        plotted = ax.imshow(np.ma.masked_invalid(finite), aspect="auto", cmap=cmap, norm=norm)
        for row in range(len(features)):
            for column in range(len(rounds[endpoint])):
                value = finite[row, column]
                count = int(counts.iloc[row, column])
                if np.isfinite(value):
                    precision = 2 if endpoint == "viability_percent" else 4
                    label = f"{value:+.{precision}f}\nn={count}"
                    color = "white" if np.dot(cmap(norm(value))[:3], [.2126, .7152, .0722]) < .45 else "#242424"
                else:
                    label, color = "—\nn=0", "#666666"
                ax.text(column, row, label, ha="center", va="center", fontsize=9.5, color=color, linespacing=1.3)
        total = data.groupby("target_round").n_explained.first()
        ax.set_xticks(range(len(rounds[endpoint])), [f"{target}\nN={int(total.loc[target])}" for target in rounds[endpoint]])
        ax.set_xlabel("Model before round | N = total explained rows", labelpad=11)
        ax.set_yticks(range(len(features)), [feature_label(registry, feature) for feature in features])
        ax.tick_params(axis="y", labelleft=index == 0, length=0)
        ax.set_xticks(np.arange(-.5, len(rounds[endpoint]), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(features), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        if endpoint == "viability_percent":
            strategies = data.groupby("target_round").viability_strategy.first()
            for column in range(1, len(rounds[endpoint])):
                if strategies.loc[rounds[endpoint][column]] != strategies.loc[rounds[endpoint][column - 1]]:
                    ax.axvline(column - .5, color="#333333", ls="--", lw=1.3)
        bounds = ax.get_position()
        color_axis = fig.add_axes([bounds.x0, .115, bounds.width, .014])
        fig.colorbar(plotted, cax=color_axis, orientation="horizontal",
                     label=f"Mean SHAP when ingredient is present ({ENDPOINTS[endpoint][1]})")
    title = ("Historical ingredient contributions | present formulations only" if len(endpoints) > 1
             else ENDPOINTS[endpoints[0]][0] + " | SHAP when ingredient is present")
    fig.suptitle(title, fontsize=18, weight="bold", y=.975)
    fig.text(.5, .925, "Each cell shows the signed mean SHAP contribution and its sample count (n). Endpoint colour scales differ.",
             ha="center", fontsize=11)
    fig.text(.04, .018,
             "Orange: raises predictions relative to the original model background; blue: lowers them. Grey: no present rows in the explained sample.\n"
             "Presence = stored ingredient value > 0.000001. n counts formulation–round rows, not independent batches or technical replicates.\n"
             "SHAP samples are reused; cohorts/backgrounds change and successive models share data. Dashed line: viability GP strategy change.\n"
             "Mechanical columns begin when models are available. These are conditional model contributions, not causal ingredient effects.",
             fontsize=9)
    return fig, scales


def write_presence_history(reports: list[dict], directory: Path, registry: IngredientRegistry):
    table = presence_history_table(reports, registry)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "feature_presence_shap_history.csv"
    table.to_csv(source, index=False)
    endpoints = list(ENDPOINTS)[:2]
    fig, scales = presence_history_figure(table, registry, endpoints)
    generated = [save_png(fig, directory / "feature_presence_shap_history.png")]
    for endpoint in endpoints:
        fig, _ = presence_history_figure(table, registry, [endpoint])
        generated.append(save_png(fig, directory / (endpoint + "_presence_shap_history.png")))
    sources = {}
    for report in reports:
        path = Path(report["shap_source"])
        sources[str(path.relative_to(directory))] = digest(path)
    metadata = {"report_version": 1, "source_sha256": digest(source), "shap_source_sha256": sources,
                "value_definition": "mean(SHAP_j | feature_value_j > presence_threshold); original backgrounds retained",
                "presence_threshold": FEATURE_NEGLIGIBLE_THRESHOLD,
                "presence_threshold_unit": "each ingredient's stored unit (M or pct); numerical zero tolerance",
                "count_definition": "Explained formulation-round rows containing the ingredient; not independent biological replicates",
                "missing_value": "NaN; distinguish model_unavailable from no_present_explained_rows in CSV",
                "feature_selection": "All active registry ingredients, including those absent from explained samples",
                "color_scales": scales, "color_legend": {"negative": "blue", "zero": "white", "positive": "orange", "unavailable": "grey"},
                "format": "transparent 300 dpi PNG", "renderer_sha256": digest(Path(__file__)),
                "target_rounds": sorted(table.target_round.unique().tolist()),
                "limitations": ["Small n and recurring formulations limit interpretation; n is not a count of independent confirmations.",
                                "Presence conditioning does not remove ingredient, dose, formulation or round confounding.",
                                "Counts concern the stored explained samples, which can be smaller than the training cohorts.",
                                "Positive contribution is relative to the original background; no causal intervention is estimated."],
                "generated_sha256": {path.name: digest(path) for path in generated}}
    (directory / "feature_presence_shap_history_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return table

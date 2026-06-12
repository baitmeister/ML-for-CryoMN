"""Small surrogate models for v2 candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .registry import IngredientRegistry


@dataclass
class RegressionPrediction:
    mean: np.ndarray
    std: np.ndarray


class RegressionSurrogate:
    """Predictive wrapper with a stable mean/std interface."""

    def __init__(self, model: object, residual_std: float, fitted: bool, fallback_mean: float):
        self.model = model
        self.residual_std = float(max(residual_std, 1e-6))
        self.fitted = bool(fitted)
        self.fallback_mean = float(fallback_mean)

    def predict(self, x: np.ndarray) -> RegressionPrediction:
        if hasattr(self.model, "predict"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    mean, std = self.model.predict(x, return_std=True)
                    mean = np.asarray(mean, dtype=float)
                    std = np.asarray(std, dtype=float)
                except TypeError:
                    mean = np.asarray(self.model.predict(x), dtype=float)
                    std = np.full(x.shape[0], self.residual_std, dtype=float)
            mean = np.where(np.isfinite(mean), mean, self.fallback_mean)
            std = np.where(np.isfinite(std) & (std >= 0.0), std, self.residual_std)
            return RegressionPrediction(mean, std)
        raise TypeError("Regression surrogate model does not implement predict().")


class ProbabilitySurrogate:
    """Predictive wrapper for intact-patch pass probability."""

    def __init__(self, model: object, fitted: bool, default_probability: float):
        self.model = model
        self.fitted = bool(fitted)
        self.default_probability = float(default_probability)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.full(x.shape[0], self.default_probability, dtype=float)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)
            if proba.shape[1] == 1:
                classes = getattr(self.model, "classes_", None)
                if classes is not None and int(classes[0]) == 1:
                    return np.asarray(proba[:, 0], dtype=float)
                return np.zeros(x.shape[0], dtype=float)
            return np.asarray(proba[:, 1], dtype=float)
        return np.full(x.shape[0], self.default_probability, dtype=float)


@dataclass
class EndpointModels:
    feature_names: list[str]
    viability: RegressionSurrogate
    critical_load: RegressionSurrogate
    initial_stiffness: RegressionSurrogate
    intact: ProbabilitySurrogate
    preparation: ProbabilitySurrogate
    training_frame: pd.DataFrame

    @property
    def mechanical_observation_count(self) -> int:
        if "critical_axial_load_N_per_needle" not in self.training_frame.columns:
            return 0
        return int(self.training_frame["critical_axial_load_N_per_needle"].notna().sum())

    @property
    def preparation_observation_count(self) -> int:
        if "preparation_feasibility_pass" not in self.training_frame.columns:
            return 0
        return int(self.training_frame["preparation_feasibility_pass"].notna().sum())


def _pivot_observations(formulations: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    if observations.empty:
        frame = formulations.copy()
        if "batch_id" not in frame.columns:
            frame["batch_id"] = ""
        return frame
    obs = observations.copy()
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    if "observation_noise" not in obs.columns:
        obs["observation_noise"] = np.nan
    obs["observation_noise"] = pd.to_numeric(obs["observation_noise"], errors="coerce")
    if "batch_id" not in obs.columns:
        obs["batch_id"] = ""
    obs["batch_id"] = obs["batch_id"].fillna("").astype(str)
    aggregated = (
        obs.groupby(["formulation_id", "batch_id", "endpoint"], dropna=False, as_index=False)
        .agg(
            value=("value", "mean"),
            observation_noise=("observation_noise", "mean"),
        )
    )
    pivot = aggregated.pivot_table(
        index=["formulation_id", "batch_id"],
        columns="endpoint",
        values="value",
        aggfunc="mean",
    ).reset_index()
    noise_pivot = aggregated.pivot_table(
        index=["formulation_id", "batch_id"],
        columns="endpoint",
        values="observation_noise",
        aggfunc="mean",
    ).add_suffix("__noise").reset_index()
    return formulations.merge(pivot, on="formulation_id", how="left").merge(
        noise_pivot,
        on=["formulation_id", "batch_id"],
        how="left",
    )


def build_training_frame(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
) -> pd.DataFrame:
    frame = _pivot_observations(formulations, observations)
    for feature_name in registry.feature_names:
        if feature_name not in frame.columns:
            frame[feature_name] = 0.0
    if "batch_id" not in frame.columns:
        frame["batch_id"] = ""
    frame["batch_id"] = frame["batch_id"].fillna("").astype(str)
    return frame


def _fit_regression(
    x: np.ndarray,
    y: pd.Series,
    default_mean: float = 0.0,
    y_noise: pd.Series | None = None,
) -> RegressionSurrogate:
    valid = pd.to_numeric(y, errors="coerce").notna().to_numpy()
    if np.sum(valid) < 2:
        model = DummyRegressor(strategy="constant", constant=float(default_mean))
        model.fit(np.zeros((1, x.shape[1])), [float(default_mean)])
        return RegressionSurrogate(model, residual_std=25.0, fitted=False, fallback_mean=float(default_mean))

    x_valid = x[valid]
    y_valid = pd.to_numeric(y[valid], errors="coerce").to_numpy(dtype=float)
    target_std = max(float(np.std(y_valid)), 1.0)
    alpha: float | np.ndarray = 1.0
    if y_noise is not None:
        noise_valid = pd.to_numeric(y_noise[valid], errors="coerce").to_numpy(dtype=float)
        noise_valid = np.where(np.isfinite(noise_valid) & (noise_valid > 0.0), noise_valid, target_std)
        alpha = np.square(noise_valid / target_std)
    if len(np.unique(y_valid)) == 1:
        model = DummyRegressor(strategy="constant", constant=float(y_valid[0]))
        model.fit(x_valid, y_valid)
        return RegressionSurrogate(model, residual_std=1.0, fitted=True, fallback_mean=float(y_valid[0]))

    kernel = Matern(length_scale=np.ones(x_valid.shape[1]), nu=2.5) + WhiteKernel(noise_level=5.0)
    model = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            alpha=alpha,
            optimizer=None,
            random_state=42,
        ),
    )
    model.fit(x_valid, y_valid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        predictions = model.predict(x_valid)
    predictions = np.where(np.isfinite(predictions), predictions, float(np.mean(y_valid)))
    residual_std = float(np.std(y_valid - predictions)) if len(y_valid) > 2 else float(np.std(y_valid))
    return RegressionSurrogate(
        model,
        residual_std=max(residual_std, 1.0),
        fitted=True,
        fallback_mean=float(np.mean(y_valid)),
    )


def _fit_classifier(
    x: np.ndarray,
    y: pd.Series,
    min_samples: int = 4,
    require_both_classes: bool = False,
) -> ProbabilitySurrogate:
    valid = pd.to_numeric(y, errors="coerce").notna().to_numpy()
    if np.sum(valid) < min_samples:
        default = float(np.nanmean(pd.to_numeric(y, errors="coerce"))) if np.sum(valid) else 0.75
        model = DummyClassifier(strategy="constant", constant=int(default >= 0.5))
        model.fit(np.zeros((1, x.shape[1])), [int(default >= 0.5)])
        return ProbabilitySurrogate(model, fitted=False, default_probability=default)

    x_valid = x[valid]
    y_valid = (pd.to_numeric(y[valid], errors="coerce").to_numpy(dtype=float) >= 0.5).astype(int)
    if len(np.unique(y_valid)) < 2:
        default = float(y_valid[0])
        model = DummyClassifier(strategy="constant", constant=int(y_valid[0]))
        model.fit(x_valid, y_valid)
        return ProbabilitySurrogate(
            model,
            fitted=not require_both_classes,
            default_probability=default,
        )

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(x_valid, y_valid)
    return ProbabilitySurrogate(model, fitted=True, default_probability=float(np.mean(y_valid)))


def train_endpoint_models(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: dict | None = None,
) -> EndpointModels:
    """Train lightweight surrogates for selection and screening."""
    frame = build_training_frame(formulations, observations, registry)
    x = frame[registry.feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    viability = _fit_regression(
        x,
        frame.get("viability_percent", pd.Series(dtype=float)),
        default_mean=50.0,
        y_noise=frame.get("viability_percent__noise", pd.Series(index=frame.index, dtype=float)),
    )
    critical_load = _fit_regression(
        x,
        frame.get("critical_axial_load_N_per_needle", pd.Series(dtype=float)),
        default_mean=0.0,
        y_noise=frame.get(
            "critical_axial_load_N_per_needle__noise",
            pd.Series(index=frame.index, dtype=float),
        ),
    )
    initial_stiffness = _fit_regression(
        x,
        frame.get("initial_stiffness_N_per_mm_per_needle", pd.Series(dtype=float)),
        default_mean=0.0,
        y_noise=frame.get(
            "initial_stiffness_N_per_mm_per_needle__noise",
            pd.Series(index=frame.index, dtype=float),
        ),
    )
    intact = _fit_classifier(x, frame.get("intact_patch_formation_pass", pd.Series(dtype=float)))
    preparation_min_labels = int(
        ((optimization_config or {}).get("preparation_model") or {}).get("min_labels", 8)
    )
    preparation = _fit_classifier(
        x,
        frame.get("preparation_feasibility_pass", pd.Series(dtype=float)),
        min_samples=preparation_min_labels,
        require_both_classes=True,
    )

    return EndpointModels(
        feature_names=registry.feature_names,
        viability=viability,
        critical_load=critical_load,
        initial_stiffness=initial_stiffness,
        intact=intact,
        preparation=preparation,
        training_frame=frame,
    )

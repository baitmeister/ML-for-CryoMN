"""Acquisition scoring for finite candidate pools."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd


def botorch_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("botorch") is not None
        and importlib.util.find_spec("gpytorch") is not None
    )


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def qlognehvi_proxy_scores(
    candidate_frame: pd.DataFrame,
    viability_ucb: np.ndarray,
    critical_load_ucb: np.ndarray,
    reference_point: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Finite-pool proxy for qLogNEHVI when BoTorch is unavailable.

    The production target is BoTorch qLogNEHVI. This proxy keeps local runs
    executable by scoring log hypervolume-like improvement over a reference
    point using normalized UCB estimates for the two Pareto objectives.
    """
    del candidate_frame
    v = minmax(viability_ucb)
    m = minmax(critical_load_ucb)
    ref_v, ref_m = reference_point
    improvement = np.maximum(v - ref_v, 0.0) * np.maximum(m - ref_m, 0.0)
    return np.log1p(improvement)


def try_botorch_qlognehvi_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    candidate_x: np.ndarray,
    reference_point: tuple[float, float],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Score candidates with BoTorch qLogNEHVI when optional deps are installed."""
    metadata: dict[str, Any] = {"botorch_attempted": False, "botorch_error": ""}
    if not botorch_available():
        metadata["botorch_error"] = "torch/gpytorch/botorch not importable"
        return None, metadata
    if train_x.shape[0] < 2 or train_y.shape[0] < 2:
        metadata["botorch_error"] = "at least two paired objective observations are required"
        return None, metadata

    metadata["botorch_attempted"] = True
    try:
        import torch
        from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        train_x = np.asarray(train_x, dtype=float)
        candidate_x = np.asarray(candidate_x, dtype=float)
        train_y = np.asarray(train_y, dtype=float)

        low = np.nanmin(np.vstack([train_x, candidate_x]), axis=0)
        high = np.nanmax(np.vstack([train_x, candidate_x]), axis=0)
        spread = np.where((high - low) < 1e-12, 1.0, high - low)
        train_x_scaled = (train_x - low) / spread
        candidate_x_scaled = (candidate_x - low) / spread

        train_X = torch.tensor(train_x_scaled, dtype=torch.double)
        train_Y = torch.tensor(train_y, dtype=torch.double)
        candidate_X = torch.tensor(candidate_x_scaled, dtype=torch.double)

        model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=train_Y.shape[-1]))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acquisition = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=list(reference_point),
            X_baseline=train_X,
        )
        scores = []
        with torch.no_grad():
            for candidate in candidate_X:
                value = acquisition(candidate.view(1, 1, -1))
                scores.append(float(value.detach().cpu().item()))
        return np.asarray(scores, dtype=float), metadata
    except Exception as exc:  # pragma: no cover - depends on optional BoTorch stack
        metadata["botorch_error"] = f"{type(exc).__name__}: {exc}"
        return None, metadata

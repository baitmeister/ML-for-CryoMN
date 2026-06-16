from __future__ import annotations

import numpy as np
import pandas as pd

import helper.acquisition as acquisition
import helper.selection as selection
from helper.config import load_optimization_config
from helper.models import EndpointModels
from helper.registry import load_registry


def test_continuous_qlognehvi_has_explicit_finite_pool_fallback(monkeypatch) -> None:
    monkeypatch.setattr(acquisition, "botorch_available", lambda: False)
    candidates, metadata = acquisition.try_botorch_optimize_qlognehvi(
        train_x=np.zeros((2, 3)),
        train_y=np.array([[50.0, 0.1], [60.0, 0.2]]),
        lower_bounds=np.zeros(3),
        upper_bounds=np.ones(3),
        active_masks=[(0,), (1, 2)],
        reference_point=(0.0, 0.0),
        n_candidates=2,
        feasibility_callback=lambda _candidate: True,
    )
    assert candidates is None
    assert "not importable" in metadata["botorch_error"]


class _UnfittedPreparation:
    fitted = False


def test_mechanics_continuous_path_applies_caps_before_finite_pool_fallback(
    monkeypatch,
) -> None:
    registry = load_registry()
    config = load_optimization_config()
    training = pd.DataFrame(
        {
            "viability_percent": np.linspace(50.0, 80.0, 8),
            "critical_axial_load_N_per_needle": np.linspace(0.1, 0.8, 8),
        }
    )
    for index, feature in enumerate(registry.feature_names):
        training[feature] = 0.01 * ((np.arange(8) + index) % 3)
    models = EndpointModels(
        feature_names=registry.feature_names,
        viability=None,
        critical_load=None,
        initial_stiffness=None,
        intact=None,
        preparation=_UnfittedPreparation(),
        training_frame=training,
    )
    candidate = {feature: 0.0 for feature in registry.feature_names}
    candidate.update(
        {
            "candidate_id": "candidate",
            "formulation_id": "formulation",
            "trehalose_M": 0.2,
        }
    )
    candidate_pool = pd.DataFrame([candidate])
    captured: dict = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return None, {
            "botorch_attempted": True,
            "botorch_error": "synthetic optimizer failure",
        }

    monkeypatch.setattr(selection, "try_botorch_optimize_qlognehvi", fake_optimize)
    generated, metadata = selection._continuous_mechanics_candidates(
        candidate_pool,
        training,
        pd.DataFrame(),
        models,
        registry,
        config,
        policy_active=True,
    )

    pvp_index = registry.feature_names.index("pvp_pct")
    assert captured["upper_bounds"][pvp_index] == 10.0
    invalid = np.zeros(len(registry.feature_names))
    invalid[pvp_index] = 10.1
    assert captured["feasibility_callback"](invalid) is False
    assert generated.empty
    assert metadata["continuous_optimizer_enabled"] is True
    assert metadata["continuous_optimizer_fallback"] is True
    assert metadata["continuous_optimizer_reason"] == "synthetic optimizer failure"

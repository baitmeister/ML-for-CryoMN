from __future__ import annotations

import numpy as np
import pandas as pd

from helper.models import _fit_classifier, _pivot_observations


def test_training_frame_uses_all_pass_for_intact_replicates() -> None:
    formulations = pd.DataFrame([{"formulation_id": "v2_gate"}])
    observations = pd.DataFrame(
        [
            {
                "formulation_id": "v2_gate",
                "batch_id": "ROUND_TEST",
                "replicate_id": "rep_001",
                "endpoint": "viability_percent",
                "value": 80.0,
                "observation_noise": 1.0,
            },
            {
                "formulation_id": "v2_gate",
                "batch_id": "ROUND_TEST",
                "replicate_id": "rep_002",
                "endpoint": "viability_percent",
                "value": 90.0,
                "observation_noise": 1.0,
            },
            {
                "formulation_id": "v2_gate",
                "batch_id": "ROUND_TEST",
                "replicate_id": "rep_001",
                "endpoint": "intact_patch_formation_pass",
                "value": 1.0,
                "observation_noise": np.nan,
            },
            {
                "formulation_id": "v2_gate",
                "batch_id": "ROUND_TEST",
                "replicate_id": "rep_002",
                "endpoint": "intact_patch_formation_pass",
                "value": 0.0,
                "observation_noise": np.nan,
            },
        ]
    )

    frame = _pivot_observations(formulations, observations)

    assert frame.loc[0, "viability_percent"] == 85.0
    assert frame.loc[0, "intact_patch_formation_pass"] == 0.0


def test_preparation_classifier_requires_eight_labels_and_both_classes() -> None:
    x = np.zeros((8, 2), dtype=float)

    one_class = _fit_classifier(
        x,
        pd.Series([1.0] * 8),
        min_samples=8,
        require_both_classes=True,
    )
    both_classes = _fit_classifier(
        x,
        pd.Series([0.0, 1.0] * 4),
        min_samples=8,
        require_both_classes=True,
    )

    assert one_class.fitted is False
    assert both_classes.fitted is True

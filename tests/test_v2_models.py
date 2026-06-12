from __future__ import annotations

import numpy as np
import pandas as pd

from helper.models import _fit_classifier


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

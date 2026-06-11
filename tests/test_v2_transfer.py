from __future__ import annotations

from pathlib import Path

import pandas as pd

from helper.config import load_optimization_config
from helper.registry import load_registry
from helper.transfer import build_v2_tables_from_legacy


def test_legacy_transfer_is_viability_only(tmp_path: Path) -> None:
    literature = tmp_path / "literature.csv"
    validation = tmp_path / "validation.csv"
    pd.DataFrame(
        [
            {
                "formulation_id": 1,
                "viability_percent": 70.0,
                "dmso_M": 0.01,
                "trehalose_M": 0.1,
            }
        ]
    ).to_csv(literature, index=False)
    pd.DataFrame(
        [
            {
                "experiment_id": "EXP1",
                "experiment_date": "2026-01-01",
                "viability_measured": 80.0,
                "dmso_M": 0.02,
            }
        ]
    ).to_csv(validation, index=False)

    formulations, observations = build_v2_tables_from_legacy(
        literature,
        validation,
        load_registry(),
        load_optimization_config(),
    )

    assert len(formulations) == 2
    assert set(observations["endpoint"]) == {"viability_percent"}
    assert "critical_axial_load_N_per_needle" not in set(observations["endpoint"])


def test_literature_noise_is_10x_wetlab_noise(tmp_path: Path) -> None:
    literature = tmp_path / "literature.csv"
    validation = tmp_path / "validation.csv"
    pd.DataFrame(
        [{"formulation_id": 1, "viability_percent": 70.0, "dmso_M": 0.01}]
    ).to_csv(literature, index=False)
    pd.DataFrame(
        [{"experiment_id": "EXP1", "viability_measured": 80.0, "dmso_M": 0.02}]
    ).to_csv(validation, index=False)

    _, observations = build_v2_tables_from_legacy(
        literature,
        validation,
        load_registry(),
        load_optimization_config(),
    )

    noise_by_source = observations.set_index("source_type")["observation_noise"].astype(float)
    assert noise_by_source["legacy_literature"] == 10 * noise_by_source["legacy_wetlab"]


def test_legacy_transfer_preserves_rows_outside_registry_bounds(tmp_path: Path) -> None:
    literature = tmp_path / "literature.csv"
    validation = tmp_path / "validation.csv"
    pd.DataFrame(
        [
            {
                "formulation_id": 1,
                "viability_percent": 70.0,
                "creatine_M": 0.02,
                "hyaluronic_acid_pct": 0.5,
            },
            {
                "formulation_id": 2,
                "viability_percent": 60.0,
                "creatine_M": 0.04,
            },
            {
                "formulation_id": 3,
                "viability_percent": 50.0,
                "hyaluronic_acid_pct": 1.2,
            },
        ]
    ).to_csv(literature, index=False)
    pd.DataFrame(
        [
            {
                "experiment_id": "EXP1",
                "experiment_date": "2026-01-01",
                "viability_measured": 80.0,
                "creatine_M": 0.031,
            }
        ]
    ).to_csv(validation, index=False)

    formulations, observations = build_v2_tables_from_legacy(
        literature,
        validation,
        load_registry(),
        load_optimization_config(),
    )

    assert len(formulations) == 4
    assert set(formulations["formulation_id"]) == {
        "legacy_lit_1.0",
        "legacy_lit_2.0",
        "legacy_lit_3.0",
        "legacy_wetlab_EXP1",
    }
    assert len(observations) == 4

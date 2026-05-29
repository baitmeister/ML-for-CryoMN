"""Legacy viability-only data transfer into the v2 decoupled schema."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .config import nested_get
from .penalties import count_active_ingredients
from .registry import IngredientRegistry


FORMULATION_BASE_COLUMNS = [
    "formulation_id",
    "source",
    "source_row_id",
    "formulation_label",
    "active_ingredient_count",
]
OBSERVATION_COLUMNS = [
    "observation_id",
    "formulation_id",
    "batch_id",
    "replicate_id",
    "endpoint",
    "value",
    "unit",
    "observation_noise",
    "source_type",
    "source_file",
    "notes",
]
def _safe_float(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)


def _feature_payload(row: pd.Series, registry: IngredientRegistry) -> dict[str, float]:
    return {
        feature_name: _safe_float(row.get(feature_name, 0.0))
        for feature_name in registry.feature_names
    }


def _formulation_label(row: pd.Series, registry: IngredientRegistry) -> str:
    parts = []
    for feature_name in registry.feature_names:
        value = _safe_float(row.get(feature_name, 0.0))
        if value <= 0.0:
            continue
        if feature_name.endswith("_pct"):
            parts.append(f"{value:.3g}% {feature_name.removesuffix('_pct')}")
        elif value >= 1.0:
            parts.append(f"{value:.3g}M {feature_name.removesuffix('_M')}")
        else:
            parts.append(f"{value * 1000:.3g}mM {feature_name.removesuffix('_M')}")
    return " + ".join(parts) if parts else "base cryopreservation medium only"


def _append_formulation(
    formulations: list[dict],
    row: pd.Series,
    registry: IngredientRegistry,
    formulation_id: str,
    source: str,
    source_row_id: str,
) -> None:
    payload = _feature_payload(row, registry)
    payload.update(
        {
            "formulation_id": formulation_id,
            "source": source,
            "source_row_id": source_row_id,
            "formulation_label": _formulation_label(row, registry),
            "active_ingredient_count": count_active_ingredients(row, registry),
        }
    )
    formulations.append(payload)


def build_v2_tables_from_legacy(
    literature_path: str | Path,
    validation_path: str | Path,
    registry: IngredientRegistry,
    optimization_config: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transfer old project data as viability-only evidence.

    No mechanical labels are inferred from legacy rows.
    """
    formulations: list[dict] = []
    observations: list[dict] = []
    literature_noise = float(
        nested_get(optimization_config, "transfer.literature_viability_noise_percent", 15.0)
    )
    wetlab_noise = float(
        nested_get(
            optimization_config,
            "transfer.legacy_wetlab_viability_noise_percent",
            nested_get(optimization_config, "transfer.wetlab_viability_noise_percent", 5.0),
        )
    )

    literature_path = Path(literature_path)
    if literature_path.exists():
        literature = pd.read_csv(literature_path)
        for index, row in literature.iterrows():
            formulation_id = f"legacy_lit_{row.get('formulation_id', index + 1)}"
            _append_formulation(
                formulations,
                row,
                registry,
                formulation_id=formulation_id,
                source="legacy_literature",
                source_row_id=str(row.get("formulation_id", index + 1)),
            )
            observations.append(
                {
                    "observation_id": f"obs_{formulation_id}_viability",
                    "formulation_id": formulation_id,
                    "batch_id": "legacy_literature",
                    "replicate_id": "legacy",
                    "endpoint": "viability_percent",
                    "value": float(row["viability_percent"]),
                    "unit": "percent",
                    "observation_noise": literature_noise,
                    "source_type": "legacy_literature",
                    "source_file": str(literature_path),
                    "notes": "Transferred from previous viability-only project.",
                }
            )

    validation_path = Path(validation_path)
    if validation_path.exists():
        validation = pd.read_csv(validation_path)
        for index, row in validation.iterrows():
            experiment_id = str(row.get("experiment_id", index + 1))
            formulation_id = f"legacy_wetlab_{experiment_id}"
            _append_formulation(
                formulations,
                row,
                registry,
                formulation_id=formulation_id,
                source="legacy_wetlab",
                source_row_id=experiment_id,
            )
            observations.append(
                {
                    "observation_id": f"obs_{formulation_id}_viability",
                    "formulation_id": formulation_id,
                    "batch_id": "legacy_wetlab",
                    "replicate_id": "legacy",
                    "endpoint": "viability_percent",
                    "value": float(row["viability_measured"]),
                    "unit": "percent",
                    "observation_noise": wetlab_noise,
                    "source_type": "legacy_wetlab",
                    "source_file": str(validation_path),
                    "notes": "Transferred from previous wet-lab validation as viability-only evidence.",
                }
            )

    formulation_columns = FORMULATION_BASE_COLUMNS + registry.feature_names
    formulations_df = pd.DataFrame(formulations)
    observations_df = pd.DataFrame(observations)
    for column in formulation_columns:
        if column not in formulations_df.columns:
            formulations_df[column] = ""
    for column in OBSERVATION_COLUMNS:
        if column not in observations_df.columns:
            observations_df[column] = ""
    return (
        formulations_df[formulation_columns],
        observations_df[OBSERVATION_COLUMNS],
    )


def write_v2_tables(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    formulations.to_csv(output / "formulations.csv", index=False)
    observations.to_csv(output / "observations.csv", index=False)

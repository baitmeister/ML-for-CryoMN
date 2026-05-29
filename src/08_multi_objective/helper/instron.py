"""Instron/Bluehill CSV importer for CryoMN mechanical endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FORCE_PATTERNS = ("load", "force")
DISPLACEMENT_PATTERNS = ("extension", "displacement", "compressive extension", "position")


@dataclass(frozen=True)
class InstronMetrics:
    critical_axial_load_N_per_needle: float
    initial_stiffness_N_per_mm_per_needle: float
    critical_axial_load_N_total: float
    initial_stiffness_N_per_mm_total: float
    needles_compressed: int
    force_column: str
    displacement_column: str
    n_points_used_for_stiffness: int


def _normalize_column_name(name: object) -> str:
    return str(name).strip().lower().replace("_", " ")


def _find_column(columns: Iterable[str], patterns: Iterable[str]) -> str | None:
    normalized = [(column, _normalize_column_name(column)) for column in columns]
    for pattern in patterns:
        pattern = pattern.lower()
        for original, clean in normalized:
            if pattern in clean:
                return original
    return None


def _read_bluehill_csv(path: str | Path) -> pd.DataFrame:
    """Read Bluehill-like CSVs, allowing short metadata preambles."""
    path = Path(path)
    try:
        df = pd.read_csv(path)
        if _find_column(df.columns, FORCE_PATTERNS) and _find_column(df.columns, DISPLACEMENT_PATTERNS):
            return df
    except pd.errors.ParserError:
        pass

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines[:100]):
        clean = line.lower()
        if any(pattern in clean for pattern in FORCE_PATTERNS) and any(
            pattern in clean for pattern in DISPLACEMENT_PATTERNS
        ):
            return pd.read_csv(path, skiprows=index)
    raise ValueError(f"Could not find force and displacement columns in {path}")


def _positive_compressive_force(force: pd.Series) -> np.ndarray:
    values = pd.to_numeric(force, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    positive_peak = np.nanmax(values)
    negative_peak = abs(np.nanmin(values))
    if negative_peak > positive_peak:
        return -values
    return values


def compute_instron_metrics(
    frame: pd.DataFrame,
    needles_compressed: int,
    force_column: str | None = None,
    displacement_column: str | None = None,
    stiffness_force_window: tuple[float, float] = (0.05, 0.30),
    min_stiffness_points: int = 5,
) -> InstronMetrics:
    """Compute critical load and initial stiffness from one load-displacement curve."""
    if needles_compressed <= 0:
        raise ValueError("needles_compressed must be positive.")

    force_column = force_column or _find_column(frame.columns, FORCE_PATTERNS)
    displacement_column = displacement_column or _find_column(frame.columns, DISPLACEMENT_PATTERNS)
    if force_column is None or displacement_column is None:
        raise ValueError("Could not identify force and displacement columns.")

    force = _positive_compressive_force(frame[force_column])
    displacement = pd.to_numeric(frame[displacement_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(force) & np.isfinite(displacement)
    force = force[valid]
    displacement = displacement[valid]
    if force.size < 3:
        raise ValueError("Instron CSV must contain at least three numeric load-displacement points.")

    force = force - force[0]
    displacement = displacement - displacement[0]
    positive = force >= 0.0
    force = force[positive]
    displacement = displacement[positive]
    if force.size < 3:
        raise ValueError("Instron curve has too few positive compressive-force points.")

    critical_total = float(np.nanmax(force))
    if critical_total <= 0:
        raise ValueError("Critical load is non-positive after baseline correction.")

    lower, upper = stiffness_force_window
    stiffness_mask = (force >= lower * critical_total) & (force <= upper * critical_total)
    if int(np.sum(stiffness_mask)) < min_stiffness_points:
        max_displacement = float(np.nanmax(displacement))
        stiffness_mask = (displacement >= 0.0) & (displacement <= 0.20 * max_displacement)
    if int(np.sum(stiffness_mask)) < 2:
        raise ValueError("Could not identify enough early linear points for stiffness.")

    x = displacement[stiffness_mask]
    y = force[stiffness_mask]
    slope, _intercept = np.polyfit(x, y, deg=1)
    stiffness_total = float(abs(slope))

    return InstronMetrics(
        critical_axial_load_N_per_needle=critical_total / needles_compressed,
        initial_stiffness_N_per_mm_per_needle=stiffness_total / needles_compressed,
        critical_axial_load_N_total=critical_total,
        initial_stiffness_N_per_mm_total=stiffness_total,
        needles_compressed=int(needles_compressed),
        force_column=str(force_column),
        displacement_column=str(displacement_column),
        n_points_used_for_stiffness=int(np.sum(stiffness_mask)),
    )


def parse_instron_csv(
    path: str | Path,
    needles_compressed: int,
    force_column: str | None = None,
    displacement_column: str | None = None,
) -> InstronMetrics:
    frame = _read_bluehill_csv(path)
    return compute_instron_metrics(
        frame,
        needles_compressed=needles_compressed,
        force_column=force_column,
        displacement_column=displacement_column,
    )


def metrics_to_observations(
    metrics: InstronMetrics,
    formulation_id: str,
    batch_id: str,
    source_file: str,
    replicate_id: str = "rep_001",
    observation_id_prefix: str = "instron",
) -> pd.DataFrame:
    """Return observation rows for the two allowed mechanical endpoints."""
    observation_prefix = f"{observation_id_prefix}_{batch_id}_{formulation_id}_{replicate_id}"
    rows = [
        {
            "observation_id": f"{observation_prefix}_critical_load",
            "formulation_id": formulation_id,
            "batch_id": batch_id,
            "replicate_id": replicate_id,
            "endpoint": "critical_axial_load_N_per_needle",
            "value": metrics.critical_axial_load_N_per_needle,
            "unit": "N_per_needle",
            "observation_noise": "",
            "source_type": "instron_5942",
            "source_file": source_file,
            "notes": f"needles_compressed={metrics.needles_compressed}",
        },
        {
            "observation_id": f"{observation_prefix}_initial_stiffness",
            "formulation_id": formulation_id,
            "batch_id": batch_id,
            "replicate_id": replicate_id,
            "endpoint": "initial_stiffness_N_per_mm_per_needle",
            "value": metrics.initial_stiffness_N_per_mm_per_needle,
            "unit": "N_per_mm_per_needle",
            "observation_noise": "",
            "source_type": "instron_5942",
            "source_file": source_file,
            "notes": f"needles_compressed={metrics.needles_compressed}",
        },
    ]
    return pd.DataFrame(rows)

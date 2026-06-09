"""Instron/Bluehill CSV parsing and round-sheet update helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .paths import NEXT_ROUND_CANDIDATES_PATH, OBSERVATIONS_PATH
except ImportError:  # pragma: no cover - direct script execution fallback
    V2_ROOT = Path(__file__).resolve().parents[1]
    if str(V2_ROOT) not in sys.path:
        sys.path.insert(0, str(V2_ROOT))
    from helper.paths import NEXT_ROUND_CANDIDATES_PATH, OBSERVATIONS_PATH


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


TEXT_RESULT_COLUMNS = [
    "batch_id",
    "replicate_id",
    "instron_file",
]


def _is_blank(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _string_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.strip()


def _ensure_text_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        frame[column] = pd.Series([""] * len(frame), index=frame.index, dtype="object")
    elif frame[column].dtype != "object":
        frame[column] = frame[column].astype("object")
    frame[column] = frame[column].where(frame[column].notna(), "").astype("object")


def _ensure_numeric_column(frame: pd.DataFrame, column: str, dtype: str = "Float64") -> None:
    if column not in frame.columns:
        frame[column] = pd.Series([pd.NA] * len(frame), index=frame.index, dtype=dtype)
        return
    numeric = pd.to_numeric(frame[column], errors="coerce")
    frame[column] = pd.Series(numeric, index=frame.index, dtype=dtype)


def _matching_base_rows(
    frame: pd.DataFrame,
    formulation_id: str | None,
    candidate_id: str | None,
    batch_id: str | None,
) -> pd.Index:
    if _is_blank(formulation_id) and _is_blank(candidate_id):
        raise ValueError("Provide --formulation-id or --candidate-id to identify the candidate row.")

    mask = pd.Series(True, index=frame.index)
    if not _is_blank(formulation_id):
        mask &= _string_series(frame, "formulation_id") == str(formulation_id).strip()
    if not _is_blank(candidate_id):
        mask &= _string_series(frame, "candidate_id") == str(candidate_id).strip()
    if not _is_blank(batch_id):
        batch_values = _string_series(frame, "batch_id")
        mask &= (batch_values == str(batch_id).strip()) | (batch_values == "")

    matches = frame.index[mask]
    if len(matches) == 0:
        raise ValueError("No matching row found in next_round_candidates.csv.")
    return matches


def _resolve_candidate_results_row(
    frame: pd.DataFrame,
    formulation_id: str | None,
    candidate_id: str | None,
    batch_id: str | None,
    replicate_id: str | None,
) -> tuple[pd.DataFrame, int]:
    matches = _matching_base_rows(frame, formulation_id, candidate_id, batch_id)
    replicate_values = _string_series(frame, "replicate_id")

    if not _is_blank(replicate_id):
        replicate_id = str(replicate_id).strip()
        exact = [idx for idx in matches if replicate_values.loc[idx] == replicate_id]
        if len(exact) == 1:
            return frame, int(exact[0])
        if len(exact) > 1:
            raise ValueError("Multiple matching candidate rows have the same replicate_id.")

        blank = [idx for idx in matches if replicate_values.loc[idx] == ""]
        if blank:
            return frame, int(blank[0])

        new_row = frame.loc[matches[0]].copy()
        new_row["replicate_id"] = replicate_id
        frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
        return frame, int(frame.index[-1])

    if len(matches) == 1:
        return frame, int(matches[0])

    blank = [idx for idx in matches if replicate_values.loc[idx] == ""]
    if len(blank) == 1:
        return frame, int(blank[0])

    raise ValueError(
        "Multiple candidate rows match. Provide --replicate-id, or make replicate_id values unique."
    )


def update_candidate_results(
    candidates_csv_path: str | Path,
    instron_csv_path: str | Path,
    metrics: InstronMetrics,
    formulation_id: str | None,
    candidate_id: str | None,
    batch_id: str | None,
    replicate_id: str | None,
) -> Path:
    path = Path(candidates_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"next_round_candidates.csv not found: {path}")
    frame = pd.read_csv(path)
    frame, row_index = _resolve_candidate_results_row(
        frame,
        formulation_id=formulation_id,
        candidate_id=candidate_id,
        batch_id=batch_id,
        replicate_id=replicate_id,
    )
    for column in TEXT_RESULT_COLUMNS:
        _ensure_text_column(frame, column)
    _ensure_numeric_column(frame, "needles_compressed", dtype="Int64")
    for column in [
        "critical_axial_load_N_per_needle",
        "critical_axial_load_N_total",
        "initial_stiffness_N_per_mm_per_needle",
    ]:
        _ensure_numeric_column(frame, column, dtype="Float64")

    if not _is_blank(batch_id):
        frame.at[row_index, "batch_id"] = str(batch_id).strip()
    if not _is_blank(replicate_id):
        frame.at[row_index, "replicate_id"] = str(replicate_id).strip()
    frame.at[row_index, "instron_file"] = str(Path(instron_csv_path))
    frame.at[row_index, "needles_compressed"] = int(metrics.needles_compressed)
    frame.at[row_index, "critical_axial_load_N_per_needle"] = float(metrics.critical_axial_load_N_per_needle)
    frame.at[row_index, "critical_axial_load_N_total"] = float(metrics.critical_axial_load_N_total)
    frame.at[row_index, "initial_stiffness_N_per_mm_per_needle"] = (
        metrics.initial_stiffness_N_per_mm_per_needle
    )
    frame.to_csv(path, index=False)
    return path


def append_observations(
    output_path: str | Path,
    metrics: InstronMetrics,
    formulation_id: str,
    batch_id: str,
    replicate_id: str,
    source_file: str,
) -> Path:
    new_rows = metrics_to_observations(
        metrics,
        formulation_id=formulation_id,
        batch_id=batch_id,
        replicate_id=replicate_id,
        source_file=source_file,
    )
    output = Path(output_path)
    if output.exists() and output.stat().st_size > 0:
        existing = pd.read_csv(output)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates("observation_id", keep="last")
    else:
        combined = new_rows
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Instron/Bluehill CSV export.")
    parser.add_argument("--formulation-id", default=None, help="Existing v2 formulation_id.")
    parser.add_argument("--candidate-id", default=None, help="Candidate row ID from next_round_candidates.csv.")
    parser.add_argument("--batch-id", default=None, help="Batch ID to match or fill in next_round_candidates.csv.")
    parser.add_argument("--needles-compressed", required=True, type=int, help="Number of needles under compression.")
    parser.add_argument("--replicate-id", default="rep_001", help="Technical replicate ID for this Instron file.")
    parser.add_argument(
        "--candidates-csv",
        default=str(NEXT_ROUND_CANDIDATES_PATH),
        help="next_round_candidates.csv to update.",
    )
    parser.add_argument(
        "--write-observations",
        action="store_true",
        help="Advanced: write directly to observations.csv instead of next_round_candidates.csv.",
    )
    parser.add_argument(
        "--output",
        default=str(OBSERVATIONS_PATH),
        help="Observations CSV to append/update, only used with --write-observations.",
    )
    parser.add_argument("--force-column", default=None, help="Optional force/load column override.")
    parser.add_argument("--displacement-column", default=None, help="Optional displacement/extension column override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = parse_instron_csv(
        args.csv,
        needles_compressed=args.needles_compressed,
        force_column=args.force_column,
        displacement_column=args.displacement_column,
    )
    if args.write_observations:
        if _is_blank(args.formulation_id) or _is_blank(args.batch_id):
            raise SystemExit("--write-observations requires --formulation-id and --batch-id.")
        output_path = append_observations(
            args.output,
            metrics,
            formulation_id=str(args.formulation_id),
            batch_id=str(args.batch_id),
            replicate_id=str(args.replicate_id),
            source_file=str(Path(args.csv)),
        )
        destination = f"Updated observations: {output_path.resolve()}"
    else:
        output_path = update_candidate_results(
            args.candidates_csv,
            args.csv,
            metrics,
            formulation_id=args.formulation_id,
            candidate_id=args.candidate_id,
            batch_id=args.batch_id,
            replicate_id=args.replicate_id,
        )
        destination = f"Updated candidate results: {output_path.resolve()}"

    print(
        "Imported Instron metrics: "
        f"critical_load={metrics.critical_axial_load_N_per_needle:.6g} N/needle, "
        f"initial_stiffness={metrics.initial_stiffness_N_per_mm_per_needle:.6g} N/mm/needle"
    )
    print(destination)


if __name__ == "__main__":
    main()

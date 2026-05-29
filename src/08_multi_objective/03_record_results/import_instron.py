#!/usr/bin/env python3
"""Parse one Instron/Bluehill CSV into next_round_candidates.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.instron import metrics_to_observations, parse_instron_csv
from helper.paths import NEXT_ROUND_CANDIDATES_PATH, OBSERVATIONS_PATH


WETLAB_RESULT_COLUMNS = [
    "instron_file",
    "needles_compressed",
    "critical_axial_load_N_per_needle",
    "critical_axial_load_N_total",
    "initial_stiffness_N_per_mm_per_needle",
]


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


def _is_blank(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _string_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.strip()


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
    metrics,
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
    for column in WETLAB_RESULT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    if "batch_id" not in frame.columns:
        frame["batch_id"] = ""
    if "replicate_id" not in frame.columns:
        frame["replicate_id"] = ""
    for column in ["batch_id", "replicate_id", "instron_file"]:
        if column in frame.columns and frame[column].dtype != "object":
            frame[column] = frame[column].astype("object")

    if not _is_blank(batch_id):
        frame.loc[row_index, "batch_id"] = str(batch_id).strip()
    if not _is_blank(replicate_id):
        frame.loc[row_index, "replicate_id"] = str(replicate_id).strip()
    frame.loc[row_index, "instron_file"] = str(Path(instron_csv_path))
    frame.loc[row_index, "needles_compressed"] = metrics.needles_compressed
    frame.loc[row_index, "critical_axial_load_N_per_needle"] = metrics.critical_axial_load_N_per_needle
    frame.loc[row_index, "critical_axial_load_N_total"] = metrics.critical_axial_load_N_total
    frame.loc[row_index, "initial_stiffness_N_per_mm_per_needle"] = (
        metrics.initial_stiffness_N_per_mm_per_needle
    )
    frame.to_csv(path, index=False)
    return path


def append_observations(
    output_path: str | Path,
    metrics,
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

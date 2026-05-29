"""Wet-lab feedback ingestion for the simplified v2 optimization loop."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

import pandas as pd

from .endpoints import intact_patch_formation_pass, parse_bool
from .instron import parse_instron_csv
from .penalties import count_active_ingredients
from .registry import IngredientRegistry
from .transfer import FORMULATION_BASE_COLUMNS, OBSERVATION_COLUMNS


def _blank(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _safe_float(value: object) -> float | None:
    if _blank(value):
        return None
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)


def _require_range(
    value: float | None,
    column: str,
    row_number: int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if value is None:
        return
    if minimum is not None and value < minimum:
        raise ValueError(f"Row {row_number} {column} must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"Row {row_number} {column} must be <= {maximum}.")


def _safe_id(value: object, fallback: str) -> str:
    if _blank(value):
        value = fallback
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def load_candidate_lookup(candidate_files: Iterable[str | Path], registry: IngredientRegistry) -> pd.DataFrame:
    frames = []
    for candidate_file in candidate_files:
        path = Path(candidate_file)
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for feature_name in registry.feature_names:
            if feature_name not in frame.columns:
                frame[feature_name] = 0.0
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    candidates = pd.concat(frames, ignore_index=True)
    candidates = candidates.drop_duplicates(
        [column for column in ["formulation_id", "candidate_id"] if column in candidates.columns],
        keep="last",
    )
    return candidates


def _resolve_candidate(row: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    if "formulation_id" in row and not _blank(row.get("formulation_id")):
        formulation_id = str(row["formulation_id"])
        hit = candidates[candidates.get("formulation_id", pd.Series(dtype=str)).astype(str) == formulation_id]
        if not hit.empty:
            return hit.iloc[-1]
        resolved = row.copy()
        resolved["formulation_id"] = formulation_id
        return resolved

    if "candidate_id" in row and not _blank(row.get("candidate_id")) and "candidate_id" in candidates.columns:
        candidate_id = str(row["candidate_id"])
        hit = candidates[candidates["candidate_id"].astype(str) == candidate_id]
        if not hit.empty:
            return hit.iloc[-1]

    raise ValueError("Each feedback row needs a formulation_id or a candidate_id present in the candidate file.")


def _upsert_formulation(
    formulations: pd.DataFrame,
    candidate: pd.Series,
    registry: IngredientRegistry,
    source: str,
) -> pd.DataFrame:
    formulation_id = str(candidate["formulation_id"])
    payload = {column: "" for column in FORMULATION_BASE_COLUMNS + registry.feature_names}
    payload.update(
        {
            "formulation_id": formulation_id,
            "source": source,
            "source_row_id": str(candidate.get("candidate_id", formulation_id)),
            "formulation_label": str(candidate.get("formulation", "")),
        }
    )
    for feature_name in registry.feature_names:
        payload[feature_name] = _safe_float(candidate.get(feature_name, 0.0)) or 0.0
    payload["active_ingredient_count"] = count_active_ingredients(payload, registry)

    for column in payload:
        if column not in formulations.columns:
            formulations[column] = ""
    if "formulation_id" in formulations.columns and formulation_id in set(formulations["formulation_id"].astype(str)):
        formulations = formulations.copy()
        mask = formulations["formulation_id"].astype(str) == formulation_id
        for column, value in payload.items():
            if column in formulations.columns and formulations[column].dtype != "object":
                formulations[column] = formulations[column].astype("object")
            formulations.loc[mask, column] = value
        return formulations
    if formulations.empty:
        return pd.DataFrame([payload])
    return pd.concat([formulations, pd.DataFrame([payload])], ignore_index=True)


def _observation_row(
    observation_id: str,
    formulation_id: str,
    batch_id: str,
    replicate_id: str,
    endpoint: str,
    value: float,
    unit: str,
    source_type: str,
    source_file: str,
    notes: str = "",
    observation_noise: float | str = "",
) -> dict:
    return {
        "observation_id": observation_id,
        "formulation_id": formulation_id,
        "batch_id": batch_id,
        "replicate_id": replicate_id,
        "endpoint": endpoint,
        "value": value,
        "unit": unit,
        "observation_noise": observation_noise,
        "source_type": source_type,
        "source_file": source_file,
        "notes": notes,
    }


def ingest_feedback(
    feedback_path: str | Path,
    candidate_files: Iterable[str | Path],
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
    batch_id: str,
    batch_date: str = "",
    default_needles_compressed: int | None = None,
    viability_noise: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append one wet-lab feedback CSV into the v2 tables."""
    feedback_path = Path(feedback_path)
    feedback = pd.read_csv(feedback_path)
    candidates = load_candidate_lookup(candidate_files, registry)
    new_observations: list[dict] = []
    auto_replicate_counts: dict[str, int] = {}

    for index, row in feedback.iterrows():
        row_number = int(index) + 1
        candidate = _resolve_candidate(row, candidates)
        formulations = _upsert_formulation(formulations, candidate, registry, source=f"wetlab_feedback:{batch_id}")
        formulation_id = str(candidate["formulation_id"])
        if _blank(row.get("replicate_id")):
            auto_replicate_counts[formulation_id] = auto_replicate_counts.get(formulation_id, 0) + 1
            replicate_id = f"rep_{auto_replicate_counts[formulation_id]:03d}"
        else:
            replicate_id = _safe_id(row.get("replicate_id"), "rep_001")
        observation_prefix = f"obs_{_safe_id(batch_id, 'batch')}_{_safe_id(formulation_id, 'formulation')}_{replicate_id}"
        notes = "" if _blank(row.get("notes")) else str(row.get("notes"))

        viability = _safe_float(row.get("viability_percent"))
        _require_range(viability, "viability_percent", row_number, minimum=0.0, maximum=100.0)
        if viability is not None:
            new_observations.append(
                _observation_row(
                    f"{observation_prefix}_viability",
                    formulation_id,
                    batch_id,
                    replicate_id,
                    "viability_percent",
                    viability,
                    "percent",
                    "wetlab_feedback",
                    str(feedback_path),
                    notes=notes,
                    observation_noise=viability_noise,
                )
            )

        intact = None
        has_intact_feedback = any(
            column in row.index and not _blank(row.get(column))
            for column in ["intact_patch_formation_pass", "intact_tip_count", "no_slurry", "no_collapse"]
        )
        if has_intact_feedback:
            if not _blank(row.get("intact_patch_formation_pass")) and parse_bool(
                row.get("intact_patch_formation_pass")
            ) is None:
                raise ValueError(
                    f"Row {row_number} intact_patch_formation_pass must be yes/no, true/false, pass/fail, or 1/0."
                )
            intact_tip_count = _safe_float(row.get("intact_tip_count"))
            total_tip_count = _safe_float(row.get("total_tip_count"))
            _require_range(intact_tip_count, "intact_tip_count", row_number, minimum=0.0)
            _require_range(total_tip_count, "total_tip_count", row_number, minimum=1.0)
            if intact_tip_count is not None and total_tip_count is not None and intact_tip_count > total_tip_count:
                raise ValueError(f"Row {row_number} intact_tip_count cannot exceed total_tip_count.")
            intact = intact_patch_formation_pass(row)
            new_observations.append(
                _observation_row(
                    f"{observation_prefix}_intact_patch",
                    formulation_id,
                    batch_id,
                    replicate_id,
                    "intact_patch_formation_pass",
                    1.0 if intact else 0.0,
                    "binary",
                    "wetlab_feedback",
                    str(feedback_path),
                    notes=notes,
                )
            )
        else:
            intact = parse_bool(row.get("intact_patch_formation_pass"))

        has_mechanical = any(
            not _blank(row.get(column))
            for column in [
                "instron_file",
                "critical_axial_load_N_per_needle",
                "critical_axial_load_N_total",
                "initial_stiffness_N_per_mm_per_needle",
            ]
        )
        if has_mechanical and intact is False:
            raise ValueError(
                f"Row {row_number} supplies mechanical data for non-intact formulation {formulation_id}."
            )

        needles = _safe_float(row.get("needles_compressed"))
        _require_range(needles, "needles_compressed", row_number, minimum=1.0)
        if needles is None and default_needles_compressed is not None:
            needles = float(default_needles_compressed)

        instron_file = row.get("instron_file")
        parsed_instron_file = False
        if not _blank(instron_file):
            if needles is None:
                raise ValueError(f"Row {row_number} needs needles_compressed for Instron import.")
            metrics = parse_instron_csv(instron_file, needles_compressed=int(needles))
            parsed_instron_file = True
            new_observations.append(
                _observation_row(
                    f"{observation_prefix}_critical_load",
                    formulation_id,
                    batch_id,
                    replicate_id,
                    "critical_axial_load_N_per_needle",
                    metrics.critical_axial_load_N_per_needle,
                    "N_per_needle",
                    "instron_5942",
                    str(instron_file),
                    notes=notes,
                )
            )
            new_observations.append(
                _observation_row(
                    f"{observation_prefix}_initial_stiffness",
                    formulation_id,
                    batch_id,
                    replicate_id,
                    "initial_stiffness_N_per_mm_per_needle",
                    metrics.initial_stiffness_N_per_mm_per_needle,
                    "N_per_mm_per_needle",
                    "instron_5942",
                    str(instron_file),
                    notes=notes,
                )
            )

        if not parsed_instron_file:
            critical_per_needle = _safe_float(row.get("critical_axial_load_N_per_needle"))
            _require_range(
                critical_per_needle,
                "critical_axial_load_N_per_needle",
                row_number,
                minimum=0.0,
            )
            if critical_per_needle is None:
                critical_total = _safe_float(row.get("critical_axial_load_N_total"))
                _require_range(critical_total, "critical_axial_load_N_total", row_number, minimum=0.0)
                if critical_total is not None:
                    if needles is None or needles <= 0:
                        raise ValueError(f"Row {row_number} needs needles_compressed for total critical load.")
                    critical_per_needle = critical_total / needles
            if critical_per_needle is not None:
                new_observations.append(
                    _observation_row(
                        f"{observation_prefix}_critical_load_raw",
                        formulation_id,
                        batch_id,
                        replicate_id,
                        "critical_axial_load_N_per_needle",
                        critical_per_needle,
                        "N_per_needle",
                        "wetlab_feedback_raw",
                        str(feedback_path),
                        notes=notes,
                    )
                )

            stiffness = _safe_float(row.get("initial_stiffness_N_per_mm_per_needle"))
            _require_range(
                stiffness,
                "initial_stiffness_N_per_mm_per_needle",
                row_number,
                minimum=0.0,
            )
            if stiffness is not None:
                new_observations.append(
                    _observation_row(
                        f"{observation_prefix}_initial_stiffness_raw",
                        formulation_id,
                        batch_id,
                        replicate_id,
                        "initial_stiffness_N_per_mm_per_needle",
                        stiffness,
                        "N_per_mm_per_needle",
                        "wetlab_feedback_raw",
                        str(feedback_path),
                        notes=notes,
                    )
                )

    for column in OBSERVATION_COLUMNS:
        if column not in observations.columns:
            observations[column] = ""
    new_observations_frame = pd.DataFrame(new_observations)
    if observations.empty:
        combined_observations = new_observations_frame
    elif new_observations_frame.empty:
        combined_observations = observations
    else:
        combined_observations = pd.concat([observations, new_observations_frame], ignore_index=True)
    for column in OBSERVATION_COLUMNS:
        if column not in combined_observations.columns:
            combined_observations[column] = ""
    combined_observations = combined_observations.drop_duplicates("observation_id", keep="last")
    return formulations, combined_observations[OBSERVATION_COLUMNS]

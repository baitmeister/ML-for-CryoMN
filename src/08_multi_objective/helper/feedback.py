"""Wet-lab feedback ingestion for the simplified v2 optimization loop."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

import pandas as pd

from .endpoints import intact_patch_formation_pass, parse_bool
from .instron import parse_instron_csv
from .paths import portable_source_path
from .penalties import count_active_ingredients
from .registry import IngredientRegistry
from .transfer import FORMULATION_BASE_COLUMNS, OBSERVATION_COLUMNS


PREPARATION_FAILURE_REASONS = {
    "insoluble_or_precipitated",
    "phase_separated",
    "excessive_viscosity",
    "incomplete_polymer_hydration",
    "other_preparation_failure",
}


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
    observation_source_file: str | Path | None = None,
    proposal_metadata: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append one wet-lab feedback CSV into the v2 tables."""
    feedback_path = Path(feedback_path)
    source_file = portable_source_path(
        Path(observation_source_file)
        if observation_source_file is not None
        else feedback_path
    )
    feedback = pd.read_csv(feedback_path)
    candidates = load_candidate_lookup(candidate_files, registry)
    new_observations: list[dict] = []
    auto_replicate_counts: dict[str, int] = {}

    metadata_by_prefix = {}
    terminal_source_hashes = set()
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
        from .group10_config import METADATA_FIELDS
        if "experimental_role" in candidate:
            metadata_by_prefix[observation_prefix] = {
                "experimental_role": str(candidate.get("experimental_role", "ordinary")),
                **{key: "" if _blank(row.get(key)) else row.get(key) for key in METADATA_FIELDS},
            }


        frozen_endpoint = (proposal_metadata or {}).get("group10", {}).get("effective_config", {}).get("mechanical_endpoint", {})
        from .terminal_force import DEFINITION as TERMINAL_DEFINITION
        terminal_policy = frozen_endpoint.get("definition_id") == TERMINAL_DEFINITION
        if terminal_policy:
            from .terminal_force import validate_settings
            validate_settings(frozen_endpoint)
            batch_match = re.fullmatch(r"ROUND_(\d+)", batch_id)
            if batch_match is None or int(batch_match.group(1)) < 10:
                raise ValueError("Terminal endpoint policy cannot reinterpret a pre-Group-10 worksheet")

        if not terminal_policy and not _blank(row.get("supplementary_analysis_file")):
            import json, hashlib, math
            analysis = json.loads(Path(str(row["supplementary_analysis_file"])).read_text())
            frozen = (proposal_metadata or {}).get("group10", {}).get("effective_config", {}).get("mechanical_endpoint")
            if frozen is None or analysis.get("settings") != frozen:
                raise ValueError("Supplementary analysis must match frozen proposal detector settings")
            source = Path(analysis["source_file"])
            if hashlib.sha256(source.read_bytes()).hexdigest() != analysis.get("source_file_hash"):
                raise ValueError("Supplementary raw source hash mismatch")
            if _blank(row.get("instron_file")) or Path(str(row["instron_file"])).resolve() != source.resolve():
                raise ValueError("Supplementary analysis source must match this row's instron_file")
            from .mechanical_events import analyze_curve
            from .instron import _read_bluehill_csv
            count = _safe_float(row.get("needles_compressed"))
            if count is None or int(count) != count or int(count) != analysis.get("loaded_needle_count"):
                raise ValueError("Supplementary loaded needle count must match worksheet")
            raw = _read_bluehill_csv(source)
            checked = analyze_curve(raw[analysis['force_column']], raw[analysis['displacement_column']], frozen, int(count))
            for field in ('status', 'endpoint_N_total', 'endpoint_N_per_needle', 'event_displacement_mm'):
                if checked.get(field) != analysis.get(field):
                    raise ValueError("Supplementary result does not reproduce from source: " + field)
            record = _observation_row(f"{observation_prefix}_supplementary_status", formulation_id, batch_id, replicate_id,
                "supported_axial_load_1mm_status", 1.0 if analysis['status'] in ('event_detected','no_event_detected') else 0.0,
                "status_flag", "wetlab_feedback_supplementary", source_file, notes=analysis['status'])
            record.update(mechanical_definition_id='supported_load_1mm_v1',detector_version=analysis['detector_version'],
                          analysis_provenance=json.dumps(analysis,sort_keys=True))
            new_observations.append(record)
            if analysis['status'] in ('event_detected','no_event_detected'):
                value=analysis['endpoint_N_per_needle']
                if not isinstance(value,(float,int)) or not math.isfinite(value) or value<0:
                    raise ValueError("Invalid supplementary endpoint")
                new_observations.append({**record,"observation_id":f"{observation_prefix}_supported_load",
                    "endpoint":"supported_axial_load_1mm_N_per_needle","unit":"N_per_needle","value":value})

        preparation_values: dict[str, bool] = {}
        for column in [
            "preparation_feasibility_pass",
            "homogeneous_solution_pass",
            "fillability_pass",
        ]:
            if _blank(row.get(column)):
                continue
            parsed = parse_bool(row.get(column))
            if parsed is None:
                raise ValueError(
                    f"Row {row_number} {column} must be yes/no, true/false, pass/fail, or 1/0."
                )
            preparation_values[column] = parsed
            new_observations.append(
                _observation_row(
                    f"{observation_prefix}_{column}",
                    formulation_id,
                    batch_id,
                    replicate_id,
                    column,
                    1.0 if parsed else 0.0,
                    "binary",
                    "wetlab_feedback",
                    source_file,
                    notes=notes,
                )
            )

        preparation_reason = (
            ""
            if _blank(row.get("preparation_failure_reason"))
            else str(row.get("preparation_failure_reason")).strip().lower()
        )
        if preparation_reason and preparation_reason not in PREPARATION_FAILURE_REASONS:
            raise ValueError(
                f"Row {row_number} preparation_failure_reason must be one of "
                f"{sorted(PREPARATION_FAILURE_REASONS)}."
            )
        if preparation_reason:
            new_observations.append(
                _observation_row(
                    f"{observation_prefix}_preparation_reason_{preparation_reason}",
                    formulation_id,
                    batch_id,
                    replicate_id,
                    f"preparation_failure_reason:{preparation_reason}",
                    1.0,
                    "categorical_indicator",
                    "wetlab_feedback",
                    source_file,
                    notes=notes,
                )
            )
        preparation_failed = (
            any(value is False for value in preparation_values.values())
            or bool(preparation_reason)
        )

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
                    source_file,
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
                    source_file,
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
        if has_mechanical and preparation_failed:
            raise ValueError(
                f"Row {row_number} supplies mechanical data for preparation-failed formulation "
                f"{formulation_id}."
            )
        if has_mechanical and intact is not True:
            raise ValueError(
                f"Row {row_number} supplies mechanical data without a measured intact pass for formulation {formulation_id}."
            )

        needles = _safe_float(row.get("needles_compressed"))
        _require_range(needles, "needles_compressed", row_number, minimum=1.0)
        if needles is None and default_needles_compressed is not None:
            needles = float(default_needles_compressed)

        instron_file = row.get("instron_file")
        parsed_instron_file = False
        if terminal_policy and has_mechanical:
            import json, hashlib
            from .terminal_force import analyze_file, MODEL_FIELD
            if needles is not None and int(needles) != needles:
                raise ValueError(f"Row {row_number} loaded needle count must be an integer")
            if _blank(instron_file):
                raise ValueError(f"Row {row_number}: the frozen terminal endpoint requires instron_file; unlabeled manual maxima are not accepted")
            analysis = analyze_file(instron_file, frozen_endpoint, int(needles) if needles is not None else None)
            if analysis['source_file_hash'] in terminal_source_hashes:
                raise ValueError('A raw compression file cannot represent multiple specimens or formulations')
            terminal_source_hashes.add(analysis['source_file_hash'])
            if not _blank(row.get('initial_stiffness_N_per_mm_per_needle')):
                raise ValueError('Leave the legacy stiffness column blank under the terminal-force protocol')
            if not _blank(row.get("supplementary_analysis_file")):
                supplied = json.loads(Path(str(row['supplementary_analysis_file'])).read_text())
                for key in ('settings', 'source_file_hash', 'loaded_needle_count', 'status', 'endpoint_N_total', 'endpoint_N_per_needle'):
                    if supplied.get(key) != analysis.get(key):
                        raise ValueError('Terminal analysis does not reproduce from source: ' + key)
            for field, expected in [('critical_axial_load_N_per_needle', analysis['endpoint_N_per_needle']),
                                    ('critical_axial_load_N_total', analysis['endpoint_N_total'])]:
                if not _blank(row.get(field)):
                    supplied = _safe_float(row[field])
                    if expected is None or supplied is None or abs(supplied-expected) > 1e-6:
                        raise ValueError(f"Row {row_number} {field} conflicts with frozen terminal endpoint; leave it blank for automatic extraction")
            record = _observation_row(f'{observation_prefix}_terminal_status', formulation_id, batch_id, replicate_id,
                'terminal_force_08mm_status', 1.0 if analysis['status']=='complete' else 0.0,
                'status_flag', 'instron_5942_terminal_v1', str(instron_file), notes=analysis['status'])
            record.update(mechanical_definition_id=TERMINAL_DEFINITION, detector_version=analysis['detector_version'],
                          protocol_id=frozen_endpoint['protocol_id'], raw_file_hash=analysis['source_file_hash'],
                          analysis_provenance=json.dumps(analysis, sort_keys=True, allow_nan=False))
            new_observations.append(record)
            if analysis['status'] == 'complete':
                new_observations.append({**record, 'observation_id':f'{observation_prefix}_terminal_total',
                    'endpoint':'terminal_force_08mm_N_total', 'value':analysis['endpoint_N_total'], 'unit':'N'})
                if analysis['endpoint_N_per_needle'] is not None:
                    new_observations.append({**record, 'observation_id':f'{observation_prefix}_terminal_per_needle',
                        'endpoint':MODEL_FIELD, 'value':analysis['endpoint_N_per_needle'], 'unit':'N_per_needle'})
            parsed_instron_file = True
        elif not _blank(instron_file):
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
                        source_file,
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
                        source_file,
                        notes=notes,
                    )
                )

    for column in OBSERVATION_COLUMNS:
        if column not in observations.columns:
            observations[column] = ""
    for record in new_observations:
        prefix = record["observation_id"]
        matching = [k for k in metadata_by_prefix if prefix.startswith(k + "_")]
        if matching:
            provenance = {k: record[k] for k in ('mechanical_definition_id','protocol_id','raw_file_hash') if record.get(k)}
            record.update(metadata_by_prefix[max(matching, key=len)])
            record.update(provenance)
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
    extras = [c for c in combined_observations if c not in OBSERVATION_COLUMNS]
    return formulations, combined_observations[OBSERVATION_COLUMNS + extras]

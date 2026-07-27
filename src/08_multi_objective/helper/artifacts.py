"""Round-scoped proposal, completed-sheet, and report artifact management."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from .paths import RESULTS_V2_DIR


EDITABLE_WETLAB_COLUMNS = frozenset(
    {
        "replicate_id",
        "viability_percent",
        "intact_patch_formation_pass",
        "no_slurry",
        "no_collapse",
        "intact_tip_count",
        "total_tip_count",
        "instron_file",
        "needles_compressed",
        "critical_axial_load_N_per_needle",
        "critical_axial_load_N_total",
        "initial_stiffness_N_per_mm_per_needle",
        "preparation_feasibility_pass",
        "homogeneous_solution_pass",
        "fillability_pass",
        "preparation_failure_reason",
        "notes",
    }
)

_TRUE_VALUES = {"true", "yes", "y", "1", "pass", "passed"}
_FALSE_VALUES = {"false", "no", "n", "0", "fail", "failed"}


class ArtifactConflictError(ValueError):
    """Raised when a frozen artifact would be replaced with different bytes."""


class ProposalValidationError(ValueError):
    """Raised when a completed worksheet no longer matches its proposal."""


@dataclass(frozen=True)
class RoundArtifactPaths:
    round_dir: Path
    proposal_dir: Path
    proposal_csv: Path
    proposal_summary: Path
    proposal_metadata: Path
    proposal_plots_dir: Path
    completed_dir: Path
    completed_csv: Path
    reports_dir: Path
    report_tables_dir: Path
    report_plots_dir: Path


def _safe_batch_id(batch_id: str) -> str:
    value = str(batch_id).strip()
    if not value:
        raise ValueError("batch_id cannot be blank when resolving round artifacts.")
    if value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"batch_id is not safe for an artifact directory: {batch_id!r}")
    return value


def round_artifact_paths(
    batch_id: str,
    results_root: str | Path = RESULTS_V2_DIR,
) -> RoundArtifactPaths:
    """Resolve the canonical artifact paths for one wet-lab round."""
    safe_batch_id = _safe_batch_id(batch_id)
    round_dir = Path(results_root) / "rounds" / safe_batch_id
    proposal_dir = round_dir / "proposal"
    completed_dir = round_dir / "completed"
    reports_dir = round_dir / "reports"
    return RoundArtifactPaths(
        round_dir=round_dir,
        proposal_dir=proposal_dir,
        proposal_csv=proposal_dir / "proposal.csv",
        proposal_summary=proposal_dir / "summary.txt",
        proposal_metadata=proposal_dir / "selection_metadata.json",
        proposal_plots_dir=proposal_dir / "plots",
        completed_dir=completed_dir,
        completed_csv=completed_dir / "completed.csv",
        reports_dir=reports_dir,
        report_tables_dir=reports_dir / "tables",
        report_plots_dir=reports_dir / "plots",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_match(left: str | Path, right: str | Path) -> bool:
    left_path = Path(left)
    right_path = Path(right)
    if not left_path.exists() or not right_path.exists():
        return False
    if left_path.stat().st_size != right_path.stat().st_size:
        return False
    return sha256_file(left_path) == sha256_file(right_path)


def _atomic_copy(source: str | Path, destination: str | Path) -> Path:
    """Copy exact source bytes and atomically promote the destination."""
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.exists() or source_path.stat().st_size == 0:
        raise FileNotFoundError(f"Artifact source is missing or empty: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            dir=destination_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        shutil.copy2(source_path, temporary_path)
        os.replace(temporary_path, destination_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination_path


def copy_frozen(source: str | Path, destination: str | Path) -> Path:
    """Create an immutable-by-convention copy, allowing identical reruns."""
    source_path = Path(source)
    destination_path = Path(destination)
    if destination_path.exists():
        if files_match(source_path, destination_path):
            return destination_path
        raise ArtifactConflictError(
            f"Refusing to overwrite frozen artifact with different bytes: {destination_path}"
        )
    return _atomic_copy(source_path, destination_path)


def copy_working(source: str | Path, destination: str | Path) -> Path:
    """Atomically replace a mutable working artifact with exact source bytes."""
    return _atomic_copy(source, destination)


def freeze_proposal(
    batch_id: str,
    candidates_csv: str | Path,
    summary_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    results_root: str | Path = RESULTS_V2_DIR,
) -> RoundArtifactPaths:
    """Freeze the selection outputs for a round before wet-lab editing."""
    paths = round_artifact_paths(batch_id, results_root)
    copy_frozen(candidates_csv, paths.proposal_csv)
    if summary_path is not None and Path(summary_path).exists():
        copy_frozen(summary_path, paths.proposal_summary)
    if metadata_path is not None and Path(metadata_path).exists():
        copy_frozen(metadata_path, paths.proposal_metadata)
    return paths


def _is_blank(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _boolean_value(value: object) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def _values_equivalent(left: object, right: object) -> bool:
    if _is_blank(left) and _is_blank(right):
        return True
    if _is_blank(left) or _is_blank(right):
        return False

    left_boolean = _boolean_value(left)
    right_boolean = _boolean_value(right)
    if left_boolean is not None and right_boolean is not None:
        return left_boolean == right_boolean

    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    if not pd.isna(left_numeric) and not pd.isna(right_numeric):
        return bool(
            np.isclose(
                float(left_numeric),
                float(right_numeric),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
        )
    return str(left).strip() == str(right).strip()


def validate_completed_against_proposal(
    completed_csv: str | Path,
    proposal_csv: str | Path,
) -> dict[str, int]:
    """Validate editable wet-lab feedback without allowing proposal drift.

    Completed rows may be reordered or duplicated for technical replicates.
    Every completed row must still map to exactly one proposal candidate, and
    all non-editable values must match that proposal row.
    """
    completed_path = Path(completed_csv)
    proposal_path = Path(proposal_csv)
    if not proposal_path.exists():
        raise ProposalValidationError(
            f"Frozen proposal not found for completed worksheet: {proposal_path}"
        )

    proposal = pd.read_csv(proposal_path)
    completed = pd.read_csv(completed_path)
    if "candidate_id" not in proposal.columns or "candidate_id" not in completed.columns:
        raise ProposalValidationError("Proposal and completed worksheets require candidate_id.")
    if proposal["candidate_id"].isna().any() or proposal["candidate_id"].astype(str).str.strip().eq("").any():
        raise ProposalValidationError("Frozen proposal contains a blank candidate_id.")
    if proposal["candidate_id"].astype(str).duplicated().any():
        raise ProposalValidationError("Frozen proposal candidate_id values must be unique.")

    proposal_columns = list(proposal.columns)
    completed_columns = list(completed.columns)
    if len(proposal_columns) != len(completed_columns) or set(proposal_columns) != set(completed_columns):
        missing = sorted(set(proposal_columns) - set(completed_columns))
        added = sorted(set(completed_columns) - set(proposal_columns))
        raise ProposalValidationError(
            "Completed worksheet columns differ from the frozen proposal "
            f"(missing={missing}, added={added})."
        )

    proposal_ids = set(proposal["candidate_id"].astype(str))
    completed_ids = set(completed["candidate_id"].astype(str))
    missing_ids = sorted(proposal_ids - completed_ids)
    unknown_ids = sorted(completed_ids - proposal_ids)
    if missing_ids or unknown_ids:
        raise ProposalValidationError(
            "Completed worksheet candidate IDs differ from the frozen proposal "
            f"(missing={missing_ids}, unknown={unknown_ids})."
        )

    proposal_by_id = proposal.assign(
        candidate_id=proposal["candidate_id"].astype(str)
    ).set_index("candidate_id", drop=False)
    immutable_columns = [
        column
        for column in proposal_columns
        if column not in EDITABLE_WETLAB_COLUMNS
    ]
    for row_index, completed_row in completed.iterrows():
        candidate_id = str(completed_row["candidate_id"])
        proposal_row = proposal_by_id.loc[candidate_id]
        for column in immutable_columns:
            if not _values_equivalent(proposal_row[column], completed_row[column]):
                raise ProposalValidationError(
                    f"Completed row {row_index + 1} changes immutable column "
                    f"{column!r} for candidate_id={candidate_id!r}: "
                    f"proposal={proposal_row[column]!r}, completed={completed_row[column]!r}."
                )

    return {
        "proposal_candidate_count": int(len(proposal)),
        "completed_row_count": int(len(completed)),
        "technical_replicate_extra_rows": int(max(len(completed) - len(proposal), 0)),
    }


def assert_completed_archive_compatible(
    completed_source: str | Path,
    completed_destination: str | Path,
) -> None:
    """Fail before ingestion if a different completed artifact already exists."""
    destination = Path(completed_destination)
    if destination.exists() and not files_match(completed_source, destination):
        raise ArtifactConflictError(
            f"Completed round artifact already exists with different bytes: {destination}"
        )


def archive_completed(
    batch_id: str,
    completed_source: str | Path,
    results_root: str | Path = RESULTS_V2_DIR,
) -> Path:
    """Archive the exact successfully ingested worksheet bytes."""
    paths = round_artifact_paths(batch_id, results_root)
    return copy_frozen(completed_source, paths.completed_csv)


def reconstruct_proposal_csv(
    completed_source: str | Path,
    reconstructed_destination: str | Path,
) -> Path:
    """Create a clearly historical proposal by blanking editable result fields."""
    source = Path(completed_source)
    destination = Path(reconstructed_destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with source.open("r", encoding="utf-8", newline="") as input_handle:
            reader = csv.DictReader(input_handle)
            if reader.fieldnames is None:
                raise ValueError(f"Completed CSV has no header: {source}")
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as output_handle:
                temporary_path = Path(output_handle.name)
                writer = csv.DictWriter(
                    output_handle,
                    fieldnames=reader.fieldnames,
                    lineterminator="\n",
                )
                writer.writeheader()
                for row in reader:
                    for column in EDITABLE_WETLAB_COLUMNS:
                        if column in row:
                            row[column] = ""
                    writer.writerow(row)
        if temporary_path is None:
            raise RuntimeError("Failed to create reconstructed proposal temporary file.")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination

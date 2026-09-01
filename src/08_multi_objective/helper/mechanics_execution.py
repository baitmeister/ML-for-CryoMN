"""Validate actual-intact mechanical-test promotion and report its outcome."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import ProposalValidationError, copy_frozen
from .endpoints import aggregate_intact_patch_replicates, intact_patch_formation_pass


MECHANICAL_RESULT_COLUMNS = (
    "instron_file",
    "critical_axial_load_N_per_needle",
    "critical_axial_load_N_total",
    "initial_stiffness_N_per_mm_per_needle",
)


def _is_blank(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _row_has_mechanical_result(row: pd.Series) -> bool:
    return any(not _is_blank(row.get(column)) for column in MECHANICAL_RESULT_COLUMNS)


def _row_intact_measurement(row: pd.Series) -> bool | None:
    measured = any(
        not _is_blank(row.get(column))
        for column in (
            "intact_patch_formation_pass",
            "intact_tip_count",
            "no_slurry",
            "no_collapse",
        )
    )
    return intact_patch_formation_pass(row) if measured else None


def mechanics_execution_audit(
    completed: pd.DataFrame,
    proposal: pd.DataFrame,
    primary_capacity: int = 4,
    proposal_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive expected actual-intact promotions and any execution violations."""
    capacity = max(int(primary_capacity), 0)
    proposal_rows = proposal.copy()
    ranks = pd.to_numeric(
        proposal_rows.get(
            "mechanical_selection_rank",
            pd.Series(index=proposal_rows.index, dtype=float),
        ),
        errors="coerce",
    )
    mechanics_active = bool(ranks.notna().any())
    if mechanics_active:
        proposal_rows = proposal_rows.assign(_mechanical_rank=ranks)
        proposal_rows = proposal_rows.loc[
            proposal_rows["_mechanical_rank"].notna()
        ].sort_values(
            ["_mechanical_rank", "candidate_id"],
            ascending=[True, True],
            kind="mergesort",
        )

    intact_by_candidate: dict[str, float | None] = {}
    tested_ids: set[str] = set()
    for candidate_id, rows in completed.groupby(
        completed["candidate_id"].astype(str), sort=False
    ):
        measurements = [
            value
            for value in (_row_intact_measurement(row) for _, row in rows.iterrows())
            if value is not None
        ]
        intact_by_candidate[str(candidate_id)] = aggregate_intact_patch_replicates(
            measurements
        )
        if rows.apply(_row_has_mechanical_result, axis=1).any():
            tested_ids.add(str(candidate_id))

    actual_intact_passes = sorted(
        candidate_id
        for candidate_id, value in intact_by_candidate.items()
        if value == 1.0
    )
    actual_intact_failures = sorted(
        candidate_id
        for candidate_id, value in intact_by_candidate.items()
        if value == 0.0
    )
    ranked_actual_passes: list[str] = []
    if mechanics_active:
        ranked_actual_passes = [
            str(row["candidate_id"])
            for _, row in proposal_rows.iterrows()
            if intact_by_candidate.get(str(row["candidate_id"])) == 1.0
        ]
    expected_test_ids = ranked_actual_passes[:capacity] if mechanics_active else []
    primary_ids = set(
        proposal.loc[
            proposal.get(
                "mechanical_test_recommended",
                pd.Series(False, index=proposal.index),
            ).astype(bool),
            "candidate_id",
        ].astype(str)
    )
    ranked_ids = set(proposal_rows["candidate_id"].astype(str)) if mechanics_active else set()
    backup_ids = ranked_ids - primary_ids
    promoted_ids = [
        candidate_id for candidate_id in expected_test_ids if candidate_id in backup_ids
    ]

    failed_tested_ids = sorted(
        candidate_id
        for candidate_id in tested_ids
        if intact_by_candidate.get(candidate_id) == 0.0
    )
    unconfirmed_tested_ids = sorted(
        candidate_id
        for candidate_id in tested_ids
        if intact_by_candidate.get(candidate_id) is None
    )
    unexpected_ranked_test_ids = sorted(
        (tested_ids & ranked_ids) - set(expected_test_ids)
    ) if mechanics_active else []
    unranked_test_ids = sorted(tested_ids - ranked_ids)
    violations: list[str] = []
    if failed_tested_ids:
        violations.append(
            "mechanical data supplied for failed-intact candidates: "
            + ", ".join(failed_tested_ids)
        )
    if unconfirmed_tested_ids:
        violations.append(
            "mechanical data supplied without a measured intact pass: "
            + ", ".join(unconfirmed_tested_ids)
        )
    if len(tested_ids) > capacity:
        violations.append(
            f"mechanical test count {len(tested_ids)} exceeds capacity {capacity}"
        )
    if unexpected_ranked_test_ids:
        violations.append(
            "mechanical data do not follow actual-intact priority promotion: "
            + ", ".join(unexpected_ranked_test_ids)
        )
    if unranked_test_ids:
        violations.append(
            "mechanical data supplied for unranked candidates: "
            + ", ".join(unranked_test_ids)
        )

    requested_capacity = capacity if mechanics_active else 0
    ranked_assignments = []
    if mechanics_active:
        for _, row in proposal_rows.iterrows():
            ranked_assignments.append(
                {
                    "candidate_id": str(row.get("candidate_id", "")),
                    "formulation_id": str(row.get("formulation_id", "")),
                    "mechanical_selection_rank": int(row["_mechanical_rank"]),
                    "mechanical_transition_role": str(
                        row.get("mechanical_transition_role", "")
                    ),
                    "prior_mechanical_observation_count": int(
                        pd.to_numeric(
                            row.get("prior_mechanical_observation_count", 0),
                            errors="coerce",
                        )
                        if pd.notna(
                            pd.to_numeric(
                                row.get("prior_mechanical_observation_count", 0),
                                errors="coerce",
                            )
                        )
                        else 0
                    ),
                    "mechanical_repeat_status": str(
                        row.get("mechanical_repeat_status", "")
                    ),
                    "mechanical_repeat_allowed": bool(
                        row.get("mechanical_repeat_allowed", False)
                    ),
                }
            )
    proposal_policy = proposal_metadata or {}
    return {
        "manifest_version": 2,
        "mechanics_active_in_proposal": mechanics_active,
        "mechanics_execution_mode": (
            "actual_intact_priority_promotion"
            if mechanics_active
            else "screening_only_no_program_recommendations"
        ),
        "configured_mechanical_capacity": capacity,
        "program_requested_test_count": requested_capacity,
        "proposal_primary_count": len(primary_ids),
        "proposal_backup_count": len(backup_ids),
        "actual_intact_measured_count": len(actual_intact_passes) + len(actual_intact_failures),
        "actual_intact_pass_count": len(actual_intact_passes),
        "actual_intact_fail_count": len(actual_intact_failures),
        "ranked_actual_intact_pass_count": len(ranked_actual_passes),
        "expected_actual_intact_test_ids": expected_test_ids,
        "recorded_mechanical_test_ids": sorted(tested_ids),
        "promoted_backup_ids": promoted_ids,
        "promotion_count": len(promoted_ids),
        "actual_pass_shortfall": (
            max(capacity - len(ranked_actual_passes), 0)
            if mechanics_active
            else 0
        ),
        "failed_intact_mechanical_ids": failed_tested_ids,
        "unconfirmed_intact_mechanical_ids": unconfirmed_tested_ids,
        "unranked_mechanical_ids": unranked_test_ids,
        "mechanical_transition_assignments": ranked_assignments,
        "phase_resolution": proposal_policy.get("phase_resolution", {}),
        "mechanical_policy": proposal_policy.get("mechanical_policy", {}),
        "mechanics_transition": proposal_policy.get("mechanics_transition", {}),
        "continuous_qlognehvi": proposal_policy.get("continuous_qlognehvi", {}),
        "optimizer_mode": proposal_policy.get("optimizer_mode", ""),
        "optimizer_fallback_status": proposal_policy.get(
            "optimizer_fallback_status", ""
        ),
        "violations": violations,
    }


def validate_mechanics_execution(
    completed: pd.DataFrame,
    proposal: pd.DataFrame,
    primary_capacity: int = 4,
    proposal_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audit = mechanics_execution_audit(
        completed,
        proposal,
        primary_capacity,
        proposal_metadata=proposal_metadata,
    )
    if audit["violations"]:
        raise ProposalValidationError(
            "Mechanical execution violates the actual-intact priority policy: "
            + "; ".join(audit["violations"])
        )
    return audit


def freeze_mechanics_execution_manifest(
    audit: dict[str, Any],
    destination: str | Path,
) -> Path:
    destination_path = Path(destination)
    temporary_path = destination_path.with_name(
        f".{destination_path.name}.staged"
    )
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    try:
        return copy_frozen(temporary_path, destination_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

"""Derived round-status metadata for the v2 multi-objective workflow."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd


ROUND_ID_PATTERN = re.compile(r"^ROUND_(\d+)$")


def parse_round_number(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    match = ROUND_ID_PATTERN.fullmatch(str(value).strip())
    if match is None:
        return None
    return int(match.group(1))


def format_round_id(number: int) -> str:
    return f"ROUND_{int(number):03d}"


def derive_round_tracker(observations: pd.DataFrame) -> dict[str, Any]:
    round_numbers: set[int] = set()
    non_round_batch_ids: set[str] = set()

    if not observations.empty and "batch_id" in observations.columns:
        for raw_value in observations["batch_id"].dropna().astype(str):
            batch_id = raw_value.strip()
            if not batch_id:
                continue
            round_number = parse_round_number(batch_id)
            if round_number is None:
                non_round_batch_ids.add(batch_id)
                continue
            round_numbers.add(round_number)

    observed_round_numbers = sorted(round_numbers)
    observed_round_ids = [format_round_id(number) for number in observed_round_numbers]
    latest_observed_round_number = observed_round_numbers[-1] if observed_round_numbers else None
    latest_observed_round_id = (
        format_round_id(latest_observed_round_number)
        if latest_observed_round_number is not None
        else ""
    )
    next_round_id = format_round_id((latest_observed_round_number or 0) + 1)
    return {
        "latest_observed_round_id": latest_observed_round_id,
        "latest_observed_round_number": latest_observed_round_number,
        "next_round_id": next_round_id,
        "observed_round_count": len(observed_round_ids),
        "observed_round_ids": observed_round_ids,
        "non_round_batch_ids": sorted(non_round_batch_ids),
    }


def build_current_round_status(
    observations: pd.DataFrame,
    source_observations_path: str | Path,
    active_phase: str = "",
    phase_reason: str = "",
    proposed_batch_id: str = "",
    proposed_batch_override_used: bool = False,
) -> dict[str, Any]:
    tracker = derive_round_tracker(observations)
    proposed = str(proposed_batch_id).strip() if proposed_batch_id else ""
    return {
        "status_version": 1,
        "source_of_truth": "observations_csv",
        "source_observations_path": str(Path(source_observations_path)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **tracker,
        "active_phase": str(active_phase).strip(),
        "phase_reason": str(phase_reason).strip(),
        "proposed_batch_id": proposed,
        "proposed_batch_override_used": bool(proposed_batch_override_used),
        "proposed_batch_matches_next_round": (
            proposed == tracker["next_round_id"] if proposed else None
        ),
    }


def write_current_round_status(
    output_path: str | Path,
    observations: pd.DataFrame,
    source_observations_path: str | Path,
    active_phase: str = "",
    phase_reason: str = "",
    proposed_batch_id: str = "",
    proposed_batch_override_used: bool = False,
) -> Path:
    path = Path(output_path)
    payload = build_current_round_status(
        observations=observations,
        source_observations_path=source_observations_path,
        active_phase=active_phase,
        phase_reason=phase_reason,
        proposed_batch_id=proposed_batch_id,
        proposed_batch_override_used=proposed_batch_override_used,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path

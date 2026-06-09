"""Phase resolution for the v2 multi-objective workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .config import nested_get
from .models import build_training_frame
from .registry import IngredientRegistry


PHASE_AUTO = "auto"
PHASE_SCREENING = "screening_only"
PHASE_MECHANICS = "mechanics_enabled"
VALID_PHASE_MODES = {PHASE_AUTO, PHASE_SCREENING, PHASE_MECHANICS}


@dataclass(frozen=True)
class PhaseResolution:
    requested_phase_mode: str
    active_phase: str
    paired_observation_count: int
    distinct_formulation_count: int
    batch_count: int
    reason: str
    override_used: bool


def resolve_phase_mode(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    requested_phase_mode: str | None = None,
) -> PhaseResolution:
    configured = str(requested_phase_mode or optimization_config.get("phase_mode", PHASE_AUTO)).strip().lower()
    if configured not in VALID_PHASE_MODES:
        raise ValueError(
            f"phase mode must be one of {sorted(VALID_PHASE_MODES)}, got {configured!r}"
        )

    frame = build_training_frame(formulations, observations, registry)
    paired_mask = (
        frame.get("viability_percent", pd.Series(index=frame.index, dtype=float)).notna()
        & frame.get("critical_axial_load_N_per_needle", pd.Series(index=frame.index, dtype=float)).notna()
    )
    paired = frame.loc[paired_mask].copy()
    paired_count = int(len(paired))
    distinct_formulations = int(paired.get("formulation_id", pd.Series(dtype=str)).nunique()) if not paired.empty else 0
    batch_count = int(
        paired.get("batch_id", pd.Series(dtype=str)).astype(str).replace("", pd.NA).dropna().nunique()
    ) if not paired.empty else 0

    if configured == PHASE_SCREENING:
        return PhaseResolution(
            requested_phase_mode=configured,
            active_phase=PHASE_SCREENING,
            paired_observation_count=paired_count,
            distinct_formulation_count=distinct_formulations,
            batch_count=batch_count,
            reason="manual override requested screening-only phase",
            override_used=True,
        )
    if configured == PHASE_MECHANICS:
        return PhaseResolution(
            requested_phase_mode=configured,
            active_phase=PHASE_MECHANICS,
            paired_observation_count=paired_count,
            distinct_formulation_count=distinct_formulations,
            batch_count=batch_count,
            reason="manual override requested mechanics-enabled phase",
            override_used=True,
        )

    min_paired = int(nested_get(optimization_config, "phase_transition.mechanics_enable_min_paired_observations", 8))
    min_formulations = int(
        nested_get(optimization_config, "phase_transition.mechanics_enable_min_distinct_formulations", 6)
    )
    min_batches = int(nested_get(optimization_config, "phase_transition.mechanics_enable_min_batches", 2))

    enabled = (
        paired_count >= min_paired
        and distinct_formulations >= min_formulations
        and batch_count >= min_batches
    )
    active = PHASE_MECHANICS if enabled else PHASE_SCREENING
    reason = (
        f"auto-selected {active}: paired_observations={paired_count}/{min_paired}, "
        f"distinct_formulations={distinct_formulations}/{min_formulations}, "
        f"batches={batch_count}/{min_batches}"
    )
    return PhaseResolution(
        requested_phase_mode=configured,
        active_phase=active,
        paired_observation_count=paired_count,
        distinct_formulation_count=distinct_formulations,
        batch_count=batch_count,
        reason=reason,
        override_used=False,
    )

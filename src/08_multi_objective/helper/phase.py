"""Phase resolution for the v2 multi-objective workflow."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

import pandas as pd

from .config import nested_get
from .models import build_training_frame
from .registry import IngredientRegistry


PHASE_AUTO = "auto"
PHASE_SCREENING = "screening_only"
PHASE_BOOTSTRAP = "mechanics_bootstrap"
PHASE_HYBRID = "mechanics_hybrid"
PHASE_MECHANICS = "mechanics_enabled"
ACTIVE_PHASE_MODES = {
    PHASE_SCREENING,
    PHASE_BOOTSTRAP,
    PHASE_HYBRID,
    PHASE_MECHANICS,
}
VALID_PHASE_MODES = {PHASE_AUTO, *ACTIVE_PHASE_MODES}
ROUND_ID_PATTERN = re.compile(r"^ROUND_(\d+)$")


@dataclass(frozen=True)
class PhaseResolution:
    requested_phase_mode: str
    active_phase: str
    paired_observation_count: int
    distinct_formulation_count: int
    batch_count: int
    reason: str
    override_used: bool
    completed_screening_round_count: int = 0
    bootstrap_gate_met: bool = False
    full_gate_met: bool = False
    bootstrap_batch_index: int = 0
    transition_policy_version: str = ""
    minimum_completed_screening_rounds: int = 8
    bootstrap_min_paired_observations: int = 8
    bootstrap_min_distinct_formulations: int = 6
    bootstrap_min_batches: int = 2
    full_min_paired_observations: int = 16
    full_min_distinct_formulations: int = 12
    full_min_batches: int = 3
    target_proposal_round: int | None = None

    @property
    def hybrid_gate_met(self) -> bool:
        return self.bootstrap_gate_met

    @property
    def hybrid_min_paired_observations(self) -> int:
        return self.bootstrap_min_paired_observations

    @property
    def hybrid_min_distinct_formulations(self) -> int:
        return self.bootstrap_min_distinct_formulations

    @property
    def hybrid_min_batches(self) -> int:
        return self.bootstrap_min_batches

    @property
    def distinct_paired_formulation_count(self) -> int:
        return self.distinct_formulation_count

    @property
    def paired_mechanical_batch_count(self) -> int:
        return self.batch_count

    def to_metadata(self) -> dict:
        hybrid_remaining = {
            "paired_observations": max(
                self.bootstrap_min_paired_observations
                - self.paired_observation_count,
                0,
            ),
            "distinct_formulations": max(
                self.bootstrap_min_distinct_formulations
                - self.distinct_formulation_count,
                0,
            ),
            "batches": max(self.bootstrap_min_batches - self.batch_count, 0),
        }
        full_remaining = {
            "paired_observations": max(
                self.full_min_paired_observations - self.paired_observation_count,
                0,
            ),
            "distinct_formulations": max(
                self.full_min_distinct_formulations - self.distinct_formulation_count,
                0,
            ),
            "batches": max(self.full_min_batches - self.batch_count, 0),
        }
        hybrid_gate = {
            "min_paired_observations": self.bootstrap_min_paired_observations,
            "min_distinct_formulations": self.bootstrap_min_distinct_formulations,
            "min_batches": self.bootstrap_min_batches,
            "met": self.bootstrap_gate_met,
            "remaining": hybrid_remaining,
        }
        return {
            "requested_phase_mode": self.requested_phase_mode,
            "active_phase": self.active_phase,
            "paired_observation_count": self.paired_observation_count,
            "distinct_formulation_count": self.distinct_formulation_count,
            "distinct_paired_formulation_count": self.distinct_paired_formulation_count,
            "batch_count": self.batch_count,
            "paired_mechanical_batch_count": self.paired_mechanical_batch_count,
            "completed_screening_round_count": self.completed_screening_round_count,
            "bootstrap_gate_met": self.bootstrap_gate_met,
            "hybrid_gate_met": self.hybrid_gate_met,
            "full_gate_met": self.full_gate_met,
            "bootstrap_batch_index": self.bootstrap_batch_index,
            "transition_policy_version": self.transition_policy_version,
            "target_proposal_round": self.target_proposal_round,
            "minimum_completed_screening_rounds": self.minimum_completed_screening_rounds,
            "bootstrap_gate": dict(hybrid_gate),
            "hybrid_gate": dict(hybrid_gate),
            "full_gate": {
                "min_paired_observations": self.full_min_paired_observations,
                "min_distinct_formulations": self.full_min_distinct_formulations,
                "min_batches": self.full_min_batches,
                "met": self.full_gate_met,
                "remaining": full_remaining,
            },
            "reason": self.reason,
            "override_used": self.override_used,
        }


def _completed_screening_round_count(observations: pd.DataFrame) -> int:
    required = {"batch_id", "endpoint", "value"}
    if observations.empty or not required.issubset(observations.columns):
        return 0
    measured = observations.copy()
    measured["batch_id"] = measured["batch_id"].astype(str).str.strip()
    measured["value"] = pd.to_numeric(measured["value"], errors="coerce")
    measured = measured.loc[
        measured["batch_id"].map(lambda value: bool(ROUND_ID_PATTERN.fullmatch(value)))
        & measured["value"].notna()
        & measured["endpoint"].astype(str).isin(
            {"viability_percent", "intact_patch_formation_pass"}
        )
    ]
    if measured.empty:
        return 0
    endpoints_by_batch = measured.groupby("batch_id")["endpoint"].agg(
        lambda values: set(values.astype(str))
    )
    return int(
        sum(
            {"viability_percent", "intact_patch_formation_pass"}.issubset(endpoints)
            for endpoints in endpoints_by_batch
        )
    )


def resolve_phase_mode(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
    optimization_config: Mapping,
    requested_phase_mode: str | None = None,
    target_round_number: int | None = None,
    completed_screening_round_count: int | None = None,
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

    transition = nested_get(optimization_config, "mechanics_transition", {}) or {}
    policy_version = str(transition.get("policy_version", "mechanics_transition_v1"))
    minimum_screening_rounds = int(
        nested_get(
            optimization_config,
            "mechanics_transition.entry.minimum_completed_screening_rounds",
            8,
        )
    )
    bootstrap_min_paired = int(
        nested_get(
            optimization_config,
            "mechanics_transition.hybrid_gate.min_paired_observations",
            8,
        )
    )
    bootstrap_min_formulations = int(
        nested_get(
            optimization_config,
            "mechanics_transition.hybrid_gate.min_distinct_formulations",
            6,
        )
    )
    bootstrap_min_batches = int(
        nested_get(
            optimization_config,
            "mechanics_transition.hybrid_gate.min_batches",
            2,
        )
    )
    full_min_paired = int(
        nested_get(
            optimization_config,
            "mechanics_transition.full_gate.min_paired_observations",
            16,
        )
    )
    full_min_formulations = int(
        nested_get(
            optimization_config,
            "mechanics_transition.full_gate.min_distinct_formulations",
            12,
        )
    )
    full_min_batches = int(
        nested_get(
            optimization_config,
            "mechanics_transition.full_gate.min_batches",
            3,
        )
    )
    completed_rounds = (
        _completed_screening_round_count(observations)
        if completed_screening_round_count is None
        else max(int(completed_screening_round_count), 0)
    )
    bootstrap_gate_met = (
        paired_count >= bootstrap_min_paired
        and distinct_formulations >= bootstrap_min_formulations
        and batch_count >= bootstrap_min_batches
    )
    full_gate_met = (
        paired_count >= full_min_paired
        and distinct_formulations >= full_min_formulations
        and batch_count >= full_min_batches
    )
    bootstrap_batch_index = batch_count + 1

    common = {
        "paired_observation_count": paired_count,
        "distinct_formulation_count": distinct_formulations,
        "batch_count": batch_count,
        "completed_screening_round_count": completed_rounds,
        "bootstrap_gate_met": bootstrap_gate_met,
        "full_gate_met": full_gate_met,
        "bootstrap_batch_index": bootstrap_batch_index,
        "transition_policy_version": policy_version,
        "minimum_completed_screening_rounds": minimum_screening_rounds,
        "bootstrap_min_paired_observations": bootstrap_min_paired,
        "bootstrap_min_distinct_formulations": bootstrap_min_formulations,
        "bootstrap_min_batches": bootstrap_min_batches,
        "full_min_paired_observations": full_min_paired,
        "full_min_distinct_formulations": full_min_formulations,
        "full_min_batches": full_min_batches,
        "target_proposal_round": target_round_number,
    }

    if configured in ACTIVE_PHASE_MODES:
        return PhaseResolution(
            requested_phase_mode=configured,
            active_phase=configured,
            reason=f"manual override requested {configured} phase",
            override_used=True,
            **common,
        )

    if completed_rounds < minimum_screening_rounds:
        active = PHASE_SCREENING
        gate_description = (
            f"completed_screening_rounds={completed_rounds}/{minimum_screening_rounds}"
        )
    elif full_gate_met:
        active = PHASE_MECHANICS
        gate_description = (
            f"full_gate paired_observations={paired_count}/{full_min_paired}, "
            f"distinct_formulations={distinct_formulations}/{full_min_formulations}, "
            f"batches={batch_count}/{full_min_batches}"
        )
    elif bootstrap_gate_met:
        active = PHASE_HYBRID
        gate_description = (
            f"hybrid_gate met; full_gate paired_observations={paired_count}/{full_min_paired}, "
            f"distinct_formulations={distinct_formulations}/{full_min_formulations}, "
            f"batches={batch_count}/{full_min_batches}"
        )
    else:
        active = PHASE_BOOTSTRAP
        gate_description = (
            f"hybrid_gate paired_observations={paired_count}/{bootstrap_min_paired}, "
            f"distinct_formulations={distinct_formulations}/{bootstrap_min_formulations}, "
            f"batches={batch_count}/{bootstrap_min_batches}"
        )
    reason = f"auto-selected {active}: {gate_description}"
    return PhaseResolution(
        requested_phase_mode=configured,
        active_phase=active,
        reason=reason,
        override_used=False,
        **common,
    )

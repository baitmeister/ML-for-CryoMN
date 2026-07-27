"""History-aware formulation similarity policy for v2 candidate selection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import nested_get
from .registry import IngredientRegistry, presence_threshold


SIMILARITY_POLICY_VERSION = "round3_formulation_similarity_v1"
SIMILARITY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SimilarityPolicy:
    """Resolved similarity policy for one proposed round."""

    enabled: bool
    active: bool
    start_round: int
    metric: str
    distance_threshold: float
    single_ingredient_min_relative_difference: float
    compare_within_generated_pool: bool
    history_source_types: tuple[str, ...]
    version: str = SIMILARITY_POLICY_VERSION


@dataclass(frozen=True)
class SimilarityMatch:
    """One candidate/reference conflict."""

    reference_id: str
    reference_kind: str
    reference_origin: str
    normalized_distance: float
    reason: str
    single_ingredient_relative_difference: float | None = None


@dataclass
class SimilarityAudit:
    """Bounded, JSON-safe audit information for one selection run."""

    policy: SimilarityPolicy
    history_reference_count: int = 0
    example_limit: int = 20
    rejection_count: int = 0
    rejections_by_reference_kind: Counter = field(default_factory=Counter)
    rejections_by_origin: Counter = field(default_factory=Counter)
    rejections_by_origin_and_reference_kind: dict[str, Counter] = field(
        default_factory=dict
    )
    rejections_by_reason: Counter = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)
    _example_keys: set[tuple[str, str, str]] = field(
        default_factory=set,
        repr=False,
    )

    def record(self, row: Mapping[str, Any] | pd.Series, match: SimilarityMatch) -> None:
        self.rejection_count += 1
        origin = str(row.get("candidate_origin", "unknown") or "unknown")
        self.rejections_by_reference_kind[match.reference_kind] += 1
        self.rejections_by_origin[origin] += 1
        self.rejections_by_origin_and_reference_kind.setdefault(
            origin,
            Counter(),
        )[match.reference_kind] += 1
        self.rejections_by_reason[match.reason] += 1
        candidate_id = str(
            row.get("candidate_id", "")
            or row.get("formulation_id", "")
            or "unknown"
        )
        example_key = (candidate_id, match.reference_id, match.reason)
        if (
            len(self.examples) >= self.example_limit
            or example_key in self._example_keys
        ):
            return
        self._example_keys.add(example_key)
        example = {
            "candidate_id": candidate_id,
            "candidate_origin": origin,
            "reference_id": match.reference_id,
            "reference_kind": match.reference_kind,
            "reference_origin": match.reference_origin,
            "reason": match.reason,
            "normalized_distance": float(match.normalized_distance),
        }
        if match.single_ingredient_relative_difference is not None:
            example["single_ingredient_relative_difference"] = float(
                match.single_ingredient_relative_difference
            )
        self.examples.append(example)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy.version,
            "enabled": self.policy.enabled,
            "active": self.policy.active,
            "start_round": self.policy.start_round,
            "metric": self.policy.metric,
            "distance_threshold": self.policy.distance_threshold,
            "single_ingredient_min_relative_difference": (
                self.policy.single_ingredient_min_relative_difference
            ),
            "compare_within_generated_pool": self.policy.compare_within_generated_pool,
            "history_source_types": list(self.policy.history_source_types),
            "history_reference_count": int(self.history_reference_count),
            "rejection_count": int(self.rejection_count),
            "rejections_by_reference_kind": dict(
                sorted(self.rejections_by_reference_kind.items())
            ),
            "rejections_by_origin": dict(sorted(self.rejections_by_origin.items())),
            "rejections_by_origin_and_reference_kind": {
                origin: dict(sorted(counts.items()))
                for origin, counts in sorted(
                    self.rejections_by_origin_and_reference_kind.items()
                )
            },
            "rejections_by_reason": dict(sorted(self.rejections_by_reason.items())),
            "conflict_examples": list(self.examples),
        }


def resolve_similarity_policy(
    optimization_config: Mapping[str, Any],
    target_round_number: int | None,
) -> SimilarityPolicy:
    """Resolve and validate the similarity policy for one target round."""
    config = nested_get(optimization_config, "formulation_similarity", {}) or {}
    enabled = bool(config.get("enabled", False))
    start_round = int(config.get("start_round", 3))
    metric = str(config.get("metric", "registry_bounds_euclidean"))
    distance_threshold = float(config.get("distance_threshold", 0.05))
    single_relative = float(
        config.get("single_ingredient_min_relative_difference", 0.50)
    )
    source_types = tuple(
        str(value).strip()
        for value in config.get(
            "history_source_types",
            ["legacy_wetlab", "wetlab_feedback"],
        )
        if str(value).strip()
    )
    compare_within_pool = bool(config.get("compare_within_generated_pool", True))

    if metric != "registry_bounds_euclidean":
        raise ValueError(
            "formulation_similarity.metric must be "
            f"'registry_bounds_euclidean', got {metric!r}."
        )
    if distance_threshold < 0.0:
        raise ValueError(
            "formulation_similarity.distance_threshold must be non-negative."
        )
    if single_relative < 0.0:
        raise ValueError(
            "formulation_similarity.single_ingredient_min_relative_difference "
            "must be non-negative."
        )
    if enabled and not source_types:
        raise ValueError(
            "formulation_similarity.history_source_types cannot be empty when "
            "the policy is enabled."
        )

    active = bool(
        enabled
        and target_round_number is not None
        and int(target_round_number) >= start_round
    )
    return SimilarityPolicy(
        enabled=enabled,
        active=active,
        start_round=start_round,
        metric=metric,
        distance_threshold=distance_threshold,
        single_ingredient_min_relative_difference=single_relative,
        compare_within_generated_pool=compare_within_pool,
        history_source_types=source_types,
    )


class SimilarityIndex:
    """Mutable reference index using registry-scaled formulation vectors."""

    def __init__(self, registry: IngredientRegistry, policy: SimilarityPolicy):
        self.registry = registry
        self.policy = policy
        ingredients = registry.active_ingredients()
        self._lower = np.array(
            [ingredient.lower_bound for ingredient in ingredients],
            dtype=float,
        )
        upper = np.array(
            [ingredient.upper_bound for ingredient in ingredients],
            dtype=float,
        )
        self._ranges = np.maximum(upper - self._lower, 1e-12)
        self._capacity = 64
        feature_count = len(registry.feature_names)
        self._raw = np.empty((self._capacity, feature_count), dtype=float)
        self._scaled = np.empty((self._capacity, feature_count), dtype=float)
        self._single_feature = np.full(self._capacity, -1, dtype=int)
        self._reference_ids: list[str] = []
        self._reference_kinds: list[str] = []
        self._reference_origins: list[str] = []
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def _grow(self) -> None:
        new_capacity = self._capacity * 2
        raw = np.empty((new_capacity, self._raw.shape[1]), dtype=float)
        scaled = np.empty((new_capacity, self._scaled.shape[1]), dtype=float)
        single = np.full(new_capacity, -1, dtype=int)
        raw[: self._size] = self._raw[: self._size]
        scaled[: self._size] = self._scaled[: self._size]
        single[: self._size] = self._single_feature[: self._size]
        self._raw = raw
        self._scaled = scaled
        self._single_feature = single
        self._capacity = new_capacity

    def prepare(
        self,
        row: Mapping[str, Any] | pd.Series,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        raw = np.zeros(len(self.registry.feature_names), dtype=float)
        active_indices: list[int] = []
        for index, feature_name in enumerate(self.registry.feature_names):
            value = pd.to_numeric(row.get(feature_name, 0.0), errors="coerce")
            numeric = 0.0 if pd.isna(value) else float(value)
            if abs(numeric) < presence_threshold(feature_name):
                numeric = 0.0
            else:
                active_indices.append(index)
            raw[index] = numeric
        scaled = (raw - self._lower) / self._ranges
        single_feature = active_indices[0] if len(active_indices) == 1 else -1
        return raw, scaled, single_feature

    def add(
        self,
        row: Mapping[str, Any] | pd.Series,
        reference_id: str,
        reference_kind: str,
        reference_origin: str = "",
    ) -> None:
        raw, scaled, single_feature = self.prepare(row)
        if self._size >= self._capacity:
            self._grow()
        self._raw[self._size] = raw
        self._scaled[self._size] = scaled
        self._single_feature[self._size] = single_feature
        self._reference_ids.append(str(reference_id))
        self._reference_kinds.append(str(reference_kind))
        self._reference_origins.append(str(reference_origin))
        self._size += 1

    def _reference_mask(self, reference_kinds: set[str] | None) -> np.ndarray:
        if reference_kinds is None:
            return np.ones(self._size, dtype=bool)
        return np.array(
            [kind in reference_kinds for kind in self._reference_kinds[: self._size]],
            dtype=bool,
        )

    def nearest_distance(
        self,
        row: Mapping[str, Any] | pd.Series,
        reference_kinds: set[str] | None = None,
    ) -> float | None:
        if self._size == 0:
            return None
        _, scaled, _ = self.prepare(row)
        mask = self._reference_mask(reference_kinds)
        if not np.any(mask):
            return None
        distances = np.linalg.norm(self._scaled[: self._size][mask] - scaled, axis=1)
        return float(np.min(distances))

    def find_conflict(
        self,
        row: Mapping[str, Any] | pd.Series,
        reference_kinds: set[str] | None = None,
    ) -> SimilarityMatch | None:
        if not self.policy.active or self._size == 0:
            return None

        raw, scaled, single_feature = self.prepare(row)
        kind_mask = self._reference_mask(reference_kinds)
        if not np.any(kind_mask):
            return None

        distances = np.linalg.norm(self._scaled[: self._size] - scaled, axis=1)
        general_conflict = (
            distances
            <= self.policy.distance_threshold + SIMILARITY_TOLERANCE
        )
        single_relative = np.full(self._size, np.inf, dtype=float)
        single_conflict = np.zeros(self._size, dtype=bool)
        if single_feature >= 0:
            same_single = (
                self._single_feature[: self._size] == single_feature
            )
            if np.any(same_single):
                reference_values = self._raw[: self._size, single_feature]
                baseline = np.minimum(
                    np.abs(reference_values),
                    abs(raw[single_feature]),
                )
                valid = same_single & (baseline > 0.0)
                single_relative[valid] = (
                    np.abs(reference_values[valid] - raw[single_feature])
                    / baseline[valid]
                )
                single_conflict = valid & (
                    single_relative + SIMILARITY_TOLERANCE
                    < self.policy.single_ingredient_min_relative_difference
                )

        conflict = kind_mask & (general_conflict | single_conflict)
        if not np.any(conflict):
            return None
        conflict_indices = np.flatnonzero(conflict)
        chosen = int(conflict_indices[np.argmin(distances[conflict_indices])])
        relative = (
            float(single_relative[chosen])
            if np.isfinite(single_relative[chosen])
            else None
        )
        reason = (
            "normalized_distance"
            if bool(general_conflict[chosen])
            else "single_ingredient_relative_spacing"
        )
        return SimilarityMatch(
            reference_id=self._reference_ids[chosen],
            reference_kind=self._reference_kinds[chosen],
            reference_origin=self._reference_origins[chosen],
            normalized_distance=float(distances[chosen]),
            reason=reason,
            single_ingredient_relative_difference=relative,
        )


def build_history_similarity_index(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
    policy: SimilarityPolicy,
) -> SimilarityIndex:
    """Build a deterministic unique-formulation wet-lab reference index."""
    index = SimilarityIndex(registry, policy)
    required_observation_columns = {"formulation_id", "source_type"}
    if (
        not policy.active
        or formulations.empty
        or observations.empty
        or "formulation_id" not in formulations.columns
        or not required_observation_columns.issubset(observations.columns)
    ):
        return index

    observed_ids = (
        observations.loc[
            observations["source_type"].astype(str).isin(policy.history_source_types),
            "formulation_id",
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if not observed_ids:
        return index

    formulation_rows = (
        formulations.assign(
            formulation_id=formulations["formulation_id"].astype(str)
        )
        .drop_duplicates("formulation_id", keep="last")
        .set_index("formulation_id", drop=False)
    )
    for formulation_id in observed_ids:
        if formulation_id not in formulation_rows.index:
            continue
        row = formulation_rows.loc[formulation_id]
        index.add(
            row,
            reference_id=formulation_id,
            reference_kind="history",
            reference_origin=str(row.get("source", "wetlab_history")),
        )
    return index


def is_retest_row(row: Mapping[str, Any] | pd.Series) -> bool:
    return (
        str(row.get("recommendation_type", "")).strip() == "retest_priority"
        or str(row.get("candidate_origin", "")).strip() == "retest"
    )


def filter_frame_by_similarity(
    frame: pd.DataFrame,
    index: SimilarityIndex,
    audit: SimilarityAudit,
    accepted_reference_kind: str = "generated_pool",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter one frame in stable order, adding accepted non-retests to the index."""
    if frame.empty or not index.policy.active:
        return frame.copy().reset_index(drop=True), frame.head(0).copy()

    accepted_indices: list[Any] = []
    rejected_indices: list[Any] = []
    for row_index, row in frame.iterrows():
        if is_retest_row(row):
            accepted_indices.append(row_index)
            continue
        match = index.find_conflict(row)
        if match is not None:
            audit.record(row, match)
            rejected_indices.append(row_index)
            continue
        accepted_indices.append(row_index)
        if index.policy.compare_within_generated_pool:
            reference_id = str(
                row.get("candidate_id", "")
                or row.get("formulation_id", "")
                or row_index
            )
            index.add(
                row,
                reference_id=reference_id,
                reference_kind=accepted_reference_kind,
                reference_origin=str(row.get("candidate_origin", "")),
            )

    accepted = frame.loc[accepted_indices].reset_index(drop=True).copy()
    rejected = frame.loc[rejected_indices].reset_index(drop=True).copy()
    return accepted, rejected


def similarity_priority_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Order special candidates before finite candidates for conflict resolution."""
    if frame.empty or "candidate_origin" not in frame.columns:
        return frame.copy().reset_index(drop=True)
    priority = {
        "rescue_dilution": 0,
        "continuous_qlognehvi": 1,
    }
    ordered = frame.copy()
    ordered["_similarity_priority"] = (
        ordered["candidate_origin"].astype(str).map(priority).fillna(2).astype(int)
    )
    ordered["_similarity_input_order"] = np.arange(len(ordered))
    ordered = ordered.sort_values(
        ["_similarity_priority", "_similarity_input_order"],
        kind="mergesort",
    )
    return ordered.drop(
        columns=["_similarity_priority", "_similarity_input_order"]
    ).reset_index(drop=True)


def validate_selected_similarity(
    selected: pd.DataFrame,
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    registry: IngredientRegistry,
    policy: SimilarityPolicy,
) -> dict[str, Any]:
    """Fail if a non-retest final slate violates the active policy."""
    summary = {
        "selected_non_retest_count": 0,
        "minimum_history_distance": None,
        "minimum_within_slate_distance": None,
    }
    if not policy.active or selected.empty:
        return summary

    index = build_history_similarity_index(
        formulations,
        observations,
        registry,
        policy,
    )
    history_distances: list[float] = []
    slate_distances: list[float] = []
    non_retest_count = 0
    for row_index, row in selected.iterrows():
        if is_retest_row(row):
            continue
        non_retest_count += 1
        history_distance = index.nearest_distance(row, {"history"})
        if history_distance is not None:
            history_distances.append(history_distance)
        within_distance = index.nearest_distance(row, {"selected_slate"})
        if within_distance is not None:
            slate_distances.append(within_distance)
        match = index.find_conflict(row)
        if match is not None:
            candidate_id = str(
                row.get("candidate_id", "")
                or row.get("formulation_id", "")
                or row_index
            )
            raise ValueError(
                "Final non-retest slate violates formulation similarity policy: "
                f"candidate={candidate_id}, reference={match.reference_id}, "
                f"reference_kind={match.reference_kind}, "
                f"distance={match.normalized_distance:.6f}, reason={match.reason}."
            )
        reference_id = str(
            row.get("candidate_id", "")
            or row.get("formulation_id", "")
            or row_index
        )
        index.add(
            row,
            reference_id=reference_id,
            reference_kind="selected_slate",
            reference_origin=str(row.get("candidate_origin", "")),
        )

    summary["selected_non_retest_count"] = non_retest_count
    summary["minimum_history_distance"] = (
        float(min(history_distances)) if history_distances else None
    )
    summary["minimum_within_slate_distance"] = (
        float(min(slate_distances)) if slate_distances else None
    )
    return summary

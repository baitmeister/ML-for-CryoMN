"""Config loading helpers for the v2 optimizer."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import yaml

from .paths import (
    AVAILABILITY_CONFIG,
    ENDPOINTS_CONFIG,
    EVALUATION_CONFIG,
    INGREDIENTS_CONFIG,
    OPTIMIZATION_CONFIG,
)


class ConfigValidationError(ValueError):
    """Raised when a V2 configuration is internally inconsistent."""


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{key} must be a mapping.")
    return value


def _require_positive_integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigValidationError(f"{path} must be a positive integer; received {value!r}.")
    return value


def validate_ingredients_config(config: Mapping[str, Any]) -> None:
    ingredients = config.get("ingredients")
    if not isinstance(ingredients, list) or not ingredients:
        raise ConfigValidationError("ingredients must be a non-empty list.")
    feature_names: list[str] = []
    canonical_names: list[str] = []
    for index, ingredient in enumerate(ingredients):
        path = f"ingredients[{index}]"
        if not isinstance(ingredient, Mapping):
            raise ConfigValidationError(f"{path} must be a mapping.")
        for field in ("canonical_name", "feature_name", "display_name", "unit"):
            value = ingredient.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ConfigValidationError(f"{path}.{field} must be a non-empty string.")
        lower = ingredient.get("lower_bound")
        upper = ingredient.get("upper_bound")
        if not _is_number(lower) or not _is_number(upper):
            raise ConfigValidationError(f"{path} must define numeric lower_bound and upper_bound values.")
        if float(lower) > float(upper):
            raise ConfigValidationError(
                f"{path}.lower_bound must not exceed upper_bound; received {lower} > {upper}."
            )
        feature_names.append(str(ingredient["feature_name"]))
        canonical_names.append(str(ingredient["canonical_name"]))
    duplicate_features = sorted({name for name in feature_names if feature_names.count(name) > 1})
    duplicate_canonical = sorted({name for name in canonical_names if canonical_names.count(name) > 1})
    if duplicate_features:
        raise ConfigValidationError(f"Ingredient feature names must be unique; duplicates: {duplicate_features}.")
    if duplicate_canonical:
        raise ConfigValidationError(f"Ingredient canonical names must be unique; duplicates: {duplicate_canonical}.")


def validate_endpoints_config(config: Mapping[str, Any]) -> None:
    requirements = config.get("application_requirements", {})
    if not isinstance(requirements, Mapping):
        raise ConfigValidationError("application_requirements must be a mapping")
    for field, upper in (("minimum_viability_percent", 100.), ("minimum_fracture_force_N_per_needle", float("inf"))):
        value = requirements.get(field)
        if value is not None and (not _is_number(value) or not 0 <= value <= upper):
            raise ConfigValidationError(f"Invalid application_requirements.{field}")
    definition = requirements.get("fracture_force_definition")
    if definition is not None and (not isinstance(definition, str) or not definition.strip()):
        raise ConfigValidationError("fracture_force_definition must be null or nonempty")
    gate = _require_mapping(config, "screening_gate")
    if gate.get("name") != "intact_patch_formation_pass" or gate.get("type") != "binary":
        raise ConfigValidationError(
            "screening_gate must define intact_patch_formation_pass as a binary feasibility gate."
        )
    objectives = config.get("primary_objectives")
    if not isinstance(objectives, list):
        raise ConfigValidationError("primary_objectives must be a list.")
    by_name = {
        str(item.get("name")): item
        for item in objectives
        if isinstance(item, Mapping)
    }
    for endpoint in ("viability_percent", "critical_axial_load_N_per_needle"):
        item = by_name.get(endpoint)
        if item is None:
            raise ConfigValidationError(f"primary_objectives is missing required endpoint {endpoint!r}.")
        if item.get("direction") != "maximize":
            raise ConfigValidationError(f"primary_objectives.{endpoint}.direction must be 'maximize'.")
        if not isinstance(item.get("unit"), str) or not str(item.get("unit")).strip():
            raise ConfigValidationError(f"primary_objectives.{endpoint}.unit must be defined.")


def validate_optimization_config(config: Mapping[str, Any]) -> None:
    seed = config.get("random_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigValidationError(f"random_seed must be an integer; received {seed!r}.")
    phase_mode = config.get("phase_mode", "auto")
    valid_phases = {
        "auto",
        "screening_only",
        "mechanics_bootstrap",
        "mechanics_hybrid",
        "mechanics_enabled",
    }
    if phase_mode not in valid_phases:
        raise ConfigValidationError(
            f"phase_mode must be one of {sorted(valid_phases)}; received {phase_mode!r}."
        )
    pool_size = nested_get(config, "selection.generated_candidate_pool_size")
    slate_size = nested_get(config, "round_policy.viability_screens_per_round")
    capacity = nested_get(config, "mechanics_transition.bootstrap.mechanical_capacity")
    _require_positive_integer(pool_size, "selection.generated_candidate_pool_size")
    slate_size = _require_positive_integer(slate_size, "round_policy.viability_screens_per_round")
    capacity = _require_positive_integer(capacity, "mechanics_transition.bootstrap.mechanical_capacity")
    if capacity > slate_size:
        raise ConfigValidationError(
            "mechanics_transition.bootstrap.mechanical_capacity must not exceed "
            f"round_policy.viability_screens_per_round; received {capacity} > {slate_size}."
        )
    entry = nested_get(config, "mechanics_transition.entry.minimum_completed_screening_rounds")
    if not isinstance(entry, int) or isinstance(entry, bool) or entry < 0:
        raise ConfigValidationError(
            "mechanics_transition.entry.minimum_completed_screening_rounds must be a nonnegative integer."
        )
    for gate_name in ("hybrid_gate", "full_gate"):
        for field in ("min_paired_observations", "min_distinct_formulations", "min_batches"):
            value = nested_get(config, f"mechanics_transition.{gate_name}.{field}")
            _require_positive_integer(value, f"mechanics_transition.{gate_name}.{field}")
    for field in ("min_paired_observations", "min_distinct_formulations", "min_batches"):
        hybrid = int(nested_get(config, f"mechanics_transition.hybrid_gate.{field}"))
        full = int(nested_get(config, f"mechanics_transition.full_gate.{field}"))
        if full < hybrid:
            raise ConfigValidationError(
                f"optimization.mechanics_transition.full_gate.{field} must be greater than or equal "
                f"to the hybrid-gate value; received {full} < {hybrid}."
            )
    reference = _require_mapping(_require_mapping(config, "selection"), "reference_point")
    for endpoint in ("viability_percent", "critical_axial_load_N_per_needle"):
        if not _is_number(reference.get(endpoint)):
            raise ConfigValidationError(f"selection.reference_point.{endpoint} must be numeric.")


def validate_availability_config(
    config: Mapping[str, Any],
    ingredients_config: Mapping[str, Any],
) -> None:
    unavailable = config.get("temporarily_unavailable_feature_names", [])
    if not isinstance(unavailable, list) or any(not isinstance(value, str) for value in unavailable):
        raise ConfigValidationError("temporarily_unavailable_feature_names must be a list of strings.")
    known = {
        str(item.get("feature_name"))
        for item in ingredients_config.get("ingredients", [])
        if isinstance(item, Mapping)
    }
    unknown = sorted(set(unavailable) - known)
    if unknown:
        raise ConfigValidationError(
            f"Availability configuration references unknown ingredient feature names: {unknown}."
        )


def validate_evaluation_config(config: Mapping[str, Any]) -> None:
    formal_start = config.get("formal_start_round")
    if not isinstance(formal_start, int) or isinstance(formal_start, bool) or formal_start < 0:
        raise ConfigValidationError("formal_start_round must be a nonnegative integer.")
    aggregation = _require_mapping(config, "replicate_aggregation")
    if aggregation.get("continuous_endpoints") not in {"mean", "median"}:
        raise ConfigValidationError("replicate_aggregation.continuous_endpoints must be 'mean' or 'median'.")
    if aggregation.get("intact_patch_formation_pass") not in {"all_pass", "any_pass", "majority"}:
        raise ConfigValidationError(
            "replicate_aggregation.intact_patch_formation_pass has an unrecognized rule."
        )
    interval = _require_mapping(config, "prediction_interval")
    confidence = interval.get("confidence")
    z_value = interval.get("z_value")
    if not _is_number(confidence) or not 0 < float(confidence) < 1:
        raise ConfigValidationError("prediction_interval.confidence must lie strictly between 0 and 1.")
    if not _is_number(z_value) or float(z_value) <= 0:
        raise ConfigValidationError("prediction_interval.z_value must be positive.")
    recognized_provenance = {
        "reconstructed",
        "migration_frozen_supplementary",
        "formal_frozen",
    }
    provenance = _require_mapping(config, "round_provenance")
    invalid = sorted({str(value) for value in provenance.values()} - recognized_provenance)
    if invalid:
        raise ConfigValidationError(f"round_provenance contains unrecognized cohort classes: {invalid}.")
    endpoints = _require_mapping(config, "endpoints")
    required = {
        "viability_percent": "continuous",
        "intact_patch_formation_pass": "binary",
        "critical_axial_load_N_per_needle": "continuous",
    }
    for endpoint, metric_type in required.items():
        definition = endpoints.get(endpoint)
        if not isinstance(definition, Mapping):
            raise ConfigValidationError(f"evaluation.endpoints is missing required endpoint {endpoint!r}.")
        if definition.get("metric_type") != metric_type:
            raise ConfigValidationError(
                f"evaluation.endpoints.{endpoint}.metric_type must be {metric_type!r}."
            )
        if not isinstance(definition.get("prediction_mean_column"), str):
            raise ConfigValidationError(
                f"evaluation.endpoints.{endpoint}.prediction_mean_column must be a string."
            )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML file and return an empty dict for an empty document."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_ingredients_config(path: str | Path = INGREDIENTS_CONFIG) -> dict[str, Any]:
    config = load_yaml(path)
    validate_ingredients_config(config)
    return config


def load_endpoints_config(path: str | Path = ENDPOINTS_CONFIG) -> dict[str, Any]:
    config = load_yaml(path)
    validate_endpoints_config(config)
    return config


def load_optimization_config(path: str | Path = OPTIMIZATION_CONFIG) -> dict[str, Any]:
    config = load_yaml(path)
    validate_optimization_config(config)
    return config


def load_availability_config(path: str | Path = AVAILABILITY_CONFIG) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    config = load_yaml(path)
    ingredients_config = load_yaml(INGREDIENTS_CONFIG)
    validate_ingredients_config(ingredients_config)
    validate_availability_config(config, ingredients_config)
    return config


def load_evaluation_config(path: str | Path = EVALUATION_CONFIG) -> dict[str, Any]:
    config = load_yaml(path)
    validate_evaluation_config(config)
    return config


def nested_get(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Return a dotted-path value from nested mappings."""
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current

"""Candidate generation for the v2 optimizer."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .penalties import count_active_ingredients
from .registry import IngredientRegistry


def formulation_fingerprint(row: pd.Series | dict, registry: IngredientRegistry) -> str:
    """Return a stable short fingerprint from canonical feature values."""
    parts = []
    for feature_name in registry.feature_names:
        value = float(row.get(feature_name, 0.0) or 0.0)
        parts.append(f"{feature_name}={value:.9g}")
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


def stable_formulation_id(row: pd.Series | dict, registry: IngredientRegistry) -> str:
    return f"v2_{formulation_fingerprint(row, registry)}"


def unavailable_features_from_config(
    availability_config: Mapping | None,
    registry: IngredientRegistry,
) -> list[str]:
    """Resolve temporary selection restrictions to canonical feature names."""
    if not availability_config:
        return []
    raw_items = availability_config.get("temporarily_unavailable_feature_names", [])
    features: set[str] = set()
    for item in raw_items or []:
        name = str(item).strip()
        if not name:
            continue
        if name in registry.feature_names:
            features.add(name)
            continue
        resolved = registry.resolve_name(name)
        if resolved is not None and resolved.feature_name in registry.feature_names:
            features.add(resolved.feature_name)
    return sorted(features)


def filter_available_candidate_pool(
    candidates: pd.DataFrame,
    unavailable_feature_names: Iterable[str],
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Drop candidate rows containing ingredients unavailable for this campaign."""
    unavailable = [feature for feature in unavailable_feature_names if feature in candidates.columns]
    if not unavailable:
        return candidates.copy()
    numeric = candidates[unavailable].apply(pd.to_numeric, errors="coerce").fillna(0.0).abs()
    mask = (numeric <= tolerance).all(axis=1)
    return candidates.loc[mask].reset_index(drop=True).copy()


def generate_random_candidate_pool(
    registry: IngredientRegistry,
    n_candidates: int,
    random_seed: int = 42,
    max_sampled_active_ingredients: int = 10,
    unavailable_feature_names: Iterable[str] = (),
) -> pd.DataFrame:
    """Generate a sparse formulation pool from registry bounds."""
    rng = np.random.default_rng(random_seed)
    unavailable = set(unavailable_feature_names)
    ingredients = [
        ingredient
        for ingredient in registry.active_ingredients()
        if ingredient.feature_name not in unavailable
    ]
    if not ingredients:
        raise ValueError("No available active ingredients remain for candidate generation.")
    rows: list[dict] = []

    for index in range(n_candidates):
        row = {feature_name: 0.0 for feature_name in registry.feature_names}
        max_count = min(max_sampled_active_ingredients, len(ingredients))
        active_count = int(rng.integers(1, max_count + 1))
        chosen = rng.choice(len(ingredients), size=active_count, replace=False)
        for ingredient_index in chosen:
            ingredient = ingredients[int(ingredient_index)]
            if ingredient.upper_bound <= ingredient.lower_bound:
                value = ingredient.lower_bound
            else:
                value = float(rng.uniform(ingredient.lower_bound, ingredient.upper_bound))
            row[ingredient.feature_name] = value
        row["candidate_id"] = f"cand_{index + 1:06d}"
        row["formulation_id"] = stable_formulation_id(row, registry)
        row["active_ingredient_count"] = count_active_ingredients(row, registry)
        rows.append(row)

    return pd.DataFrame(rows)


def load_candidate_pool(path: str, registry: IngredientRegistry) -> pd.DataFrame:
    candidates = pd.read_csv(path)
    for feature_name in registry.feature_names:
        if feature_name not in candidates.columns:
            candidates[feature_name] = 0.0
    if "candidate_id" not in candidates.columns:
        candidates["candidate_id"] = [f"input_{index + 1:06d}" for index in range(len(candidates))]
    if "formulation_id" not in candidates.columns:
        candidates["formulation_id"] = candidates.apply(
            lambda row: stable_formulation_id(row, registry),
            axis=1,
        )
    candidates["active_ingredient_count"] = candidates.apply(
        lambda row: count_active_ingredients(row, registry),
        axis=1,
    )
    return candidates

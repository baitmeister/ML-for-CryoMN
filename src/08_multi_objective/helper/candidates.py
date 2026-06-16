"""Candidate generation for the v2 optimizer."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from .config import nested_get
from .feasibility import SupportContext, annotate_feasibility, annotate_support
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


def filter_candidate_pool_to_registry_bounds(
    candidates: pd.DataFrame,
    registry: IngredientRegistry,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Drop candidate rows that violate active ingredient lower/upper bounds."""
    if candidates.empty:
        return candidates.copy()
    filtered = candidates.copy()
    mask = pd.Series(True, index=filtered.index)
    for ingredient in registry.active_ingredients():
        if ingredient.feature_name not in filtered.columns:
            continue
        values = pd.to_numeric(filtered[ingredient.feature_name], errors="coerce").fillna(0.0)
        within_bounds = (
            (values >= (ingredient.lower_bound - tolerance))
            & (values <= (ingredient.upper_bound + tolerance))
        )
        mask &= within_bounds
    return filtered.loc[mask].reset_index(drop=True).copy()


def filter_nonzero_active_candidate_pool(
    candidates: pd.DataFrame,
    registry: IngredientRegistry,
) -> pd.DataFrame:
    """Drop candidate rows that contain no active ingredients at all."""
    if candidates.empty:
        return candidates.copy()
    filtered = candidates.copy()
    filtered["active_ingredient_count"] = filtered.apply(
        lambda row: count_active_ingredients(row, registry),
        axis=1,
    )
    mask = pd.to_numeric(filtered["active_ingredient_count"], errors="coerce").fillna(0).astype(int) > 0
    return filtered.loc[mask].reset_index(drop=True).copy()


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


def _finalize_generated_rows(
    rows: list[dict],
    registry: IngredientRegistry,
    candidate_id_prefix: str = "cand",
) -> pd.DataFrame:
    finalized: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        formulation_id = stable_formulation_id(row, registry)
        if formulation_id in seen:
            continue
        seen.add(formulation_id)
        payload = {feature: float(row.get(feature, 0.0) or 0.0) for feature in registry.feature_names}
        payload.update(
            {
                "candidate_id": f"{candidate_id_prefix}_{len(finalized) + 1:06d}",
                "formulation_id": formulation_id,
                "active_ingredient_count": count_active_ingredients(payload, registry),
                "candidate_origin": str(row.get("candidate_origin", "finite_pool_fallback")),
            }
        )
        for key, value in row.items():
            if key not in payload and key not in {"candidate_id", "formulation_id", "active_ingredient_count"}:
                payload[key] = value
        finalized.append(payload)
    return pd.DataFrame(finalized)


def generate_support_aware_candidate_pool(
    registry: IngredientRegistry,
    formulations: pd.DataFrame,
    optimization_config: Mapping,
    support: SupportContext,
    n_candidates: int,
    random_seed: int = 42,
    unavailable_feature_names: Iterable[str] = (),
) -> pd.DataFrame:
    """Generate the ROUND_002+ finite pool with a 40/35/25 policy."""
    rng = np.random.default_rng(random_seed)
    unavailable = set(unavailable_feature_names)
    ingredients = [
        ingredient
        for ingredient in registry.active_ingredients()
        if ingredient.feature_name not in unavailable
    ]
    if not ingredients:
        raise ValueError("No available active ingredients remain for candidate generation.")

    local_fraction = float(nested_get(optimization_config, "candidate_generation.local_fraction", 0.40))
    sparse_fraction = float(nested_get(optimization_config, "candidate_generation.sparse_fraction", 0.35))
    boundary_fraction = float(
        nested_get(optimization_config, "candidate_generation.boundary_fraction", 0.25)
    )
    if not np.isclose(
        local_fraction + sparse_fraction + boundary_fraction,
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "candidate_generation local_fraction, sparse_fraction, and "
            "boundary_fraction must sum to 1.0."
        )
    local_target = int(np.floor(n_candidates * local_fraction))
    sparse_target = int(np.floor(n_candidates * sparse_fraction))
    boundary_target = n_candidates - local_target - sparse_target
    attempt_multiplier = int(
        nested_get(optimization_config, "candidate_generation.max_attempt_multiplier", 100)
    )
    local_std = float(
        nested_get(optimization_config, "candidate_generation.local_relative_std", 0.20)
    )
    local_max = int(
        nested_get(optimization_config, "candidate_generation.local_max_active_ingredients", 6)
    )
    sparse_max = int(
        nested_get(optimization_config, "candidate_generation.sparse_max_active_ingredients", 2)
    )
    boundary_max = int(
        nested_get(optimization_config, "candidate_generation.boundary_max_active_ingredients", 4)
    )
    campaign_caps = (
        nested_get(optimization_config, "formulation_feasibility.ingredient_caps", {})
        or {}
    )
    bounds = {
        ingredient.feature_name: (
            ingredient.lower_bound,
            min(
                ingredient.upper_bound,
                float(campaign_caps.get(ingredient.feature_name, ingredient.upper_bound)),
            ),
        )
        for ingredient in ingredients
    }
    rejected_rows: list[dict] = []
    accepted_formulation_ids: set[str] = set()
    anchors = formulations.copy()
    for feature in registry.feature_names:
        if feature not in anchors.columns:
            anchors[feature] = 0.0
    anchors = anchors[registry.feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    def acceptable(row: dict, required_support: str | None = None) -> bool:
        frame = pd.DataFrame([row])
        for feature in registry.feature_names:
            if feature not in frame.columns:
                frame[feature] = 0.0
        frame = annotate_feasibility(frame, registry, optimization_config, policy_active=True)
        frame = annotate_support(frame, registry, support)
        if not bool(frame.iloc[0]["feasibility_pass"]):
            if len(rejected_rows) < n_candidates:
                rejected_rows.append(dict(row))
            return False
        if count_active_ingredients(frame.iloc[0], registry) <= 0:
            return False
        if required_support is not None and str(frame.iloc[0]["support_status"]) != required_support:
            return False
        formulation_id = stable_formulation_id(row, registry)
        if formulation_id in accepted_formulation_ids:
            return False
        accepted_formulation_ids.add(formulation_id)
        return True

    rows: list[dict] = []
    local_rows: list[dict] = []
    if not anchors.empty:
        attempts = 0
        while len(local_rows) < local_target and attempts < max(local_target * attempt_multiplier, 100):
            attempts += 1
            anchor = anchors.iloc[int(rng.integers(0, len(anchors)))]
            active = [
                feature
                for feature in registry.feature_names
                if feature in bounds and abs(float(anchor.get(feature, 0.0))) >= 1e-12
            ]
            if not active or len(active) > local_max:
                continue
            row = {feature: 0.0 for feature in registry.feature_names}
            for feature in active:
                low, high = bounds[feature]
                base = float(anchor[feature])
                perturbed = base * float(np.exp(rng.normal(0.0, local_std)))
                row[feature] = float(np.clip(perturbed, low, high))
            row["candidate_origin"] = "local_perturbation"
            if acceptable(row):
                local_rows.append(row)

    local_shortfall = max(local_target - len(local_rows), 0)
    sparse_target += local_shortfall
    rows.extend(local_rows)

    sparse_rows: list[dict] = []
    attempts = 0
    while len(sparse_rows) < sparse_target and attempts < max(sparse_target * attempt_multiplier, 100):
        attempts += 1
        active_count = int(rng.integers(1, min(sparse_max, len(ingredients)) + 1))
        chosen = rng.choice(len(ingredients), size=active_count, replace=False)
        row = {feature: 0.0 for feature in registry.feature_names}
        for index in chosen:
            ingredient = ingredients[int(index)]
            low, high = bounds[ingredient.feature_name]
            # Beta sampling covers the range while avoiding repeated concentration extremes.
            fraction = float(rng.beta(1.25, 1.25))
            row[ingredient.feature_name] = (
                low + fraction * (high - low)
            )
        row["candidate_origin"] = "sparse_exploration"
        if acceptable(row):
            sparse_rows.append(row)
    rows.extend(sparse_rows)

    boundary_rows: list[dict] = []
    attempts = 0
    boundary_attempt_limit = max(boundary_target * attempt_multiplier * 2, 200)
    while len(boundary_rows) < boundary_target and attempts < boundary_attempt_limit:
        attempts += 1
        maximum_active = min(boundary_max, len(ingredients))
        minimum_active = min(3, maximum_active)
        active_count = int(rng.integers(minimum_active, maximum_active + 1))
        chosen = rng.choice(len(ingredients), size=active_count, replace=False)
        row = {feature: 0.0 for feature in registry.feature_names}
        for index in chosen:
            ingredient = ingredients[int(index)]
            low, high = bounds[ingredient.feature_name]
            # Multi-axis, upper-half sampling reaches the observed support edge
            # without pinning concentrations at their configured maxima.
            fraction = float(0.50 + 0.45 * rng.beta(2.0, 1.5))
            row[ingredient.feature_name] = (
                low + fraction * (high - low)
            )
        row["candidate_origin"] = "boundary_probe"
        if acceptable(row):
            boundary_rows.append(row)
    if len(boundary_rows) < boundary_target:
        raise ValueError(
            "Unable to fill the configured support-boundary-style exploration quota "
            f"({len(boundary_rows)}/{boundary_target}) after "
            f"{boundary_attempt_limit} attempts. Review ingredient availability, "
            "hard feasibility limits, or the boundary-probe sampling policy."
        )
    rows.extend(boundary_rows)

    # Sparse generation fills only local-generation shortfall, never boundary quota.
    remaining = n_candidates - len(rows)
    attempts = 0
    while remaining > 0 and attempts < max(remaining * attempt_multiplier, 100):
        attempts += 1
        chosen = rng.choice(len(ingredients), size=int(rng.integers(1, min(sparse_max, len(ingredients)) + 1)), replace=False)
        row = {feature: 0.0 for feature in registry.feature_names}
        for index in chosen:
            ingredient = ingredients[int(index)]
            low, high = bounds[ingredient.feature_name]
            fraction = float(rng.beta(1.25, 1.25))
            row[ingredient.feature_name] = (
                low + fraction * (high - low)
            )
        row["candidate_origin"] = "sparse_exploration"
        if acceptable(row):
            rows.append(row)
            remaining -= 1

    pool = _finalize_generated_rows(rows, registry)
    pool = annotate_feasibility(pool, registry, optimization_config, policy_active=True)
    pool = annotate_support(pool, registry, support)
    accepted_pool = pool.head(n_candidates).reset_index(drop=True)
    rejected_pool = _finalize_generated_rows(
        rejected_rows,
        registry,
        candidate_id_prefix="rejected",
    )
    if not rejected_pool.empty:
        rejected_pool = annotate_feasibility(
            rejected_pool,
            registry,
            optimization_config,
            policy_active=True,
        )
        rejected_pool = annotate_support(rejected_pool, registry, support)
        rejected_pool["candidate_origin"] = "rejected_generation_attempt"
        return pd.concat([accepted_pool, rejected_pool], ignore_index=True, sort=False)
    return accepted_pool


def generate_rescue_candidate_pool(
    registry: IngredientRegistry,
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    optimization_config: Mapping,
    support: SupportContext,
    unavailable_feature_names: Iterable[str] = (),
) -> pd.DataFrame:
    """Generate dilution variants of high-viability formulations that failed formation."""
    if formulations.empty or observations.empty:
        return pd.DataFrame()
    required = {"formulation_id", "batch_id", "endpoint", "value"}
    if not required.issubset(observations.columns) or "formulation_id" not in formulations.columns:
        return pd.DataFrame()

    obs = observations.copy()
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    obs["endpoint"] = obs["endpoint"].fillna("").astype(str)
    obs["batch_id"] = obs["batch_id"].fillna("").astype(str)
    pivot = (
        obs[obs["endpoint"].isin(["viability_percent", "intact_patch_formation_pass"])]
        .pivot_table(
            index=["formulation_id", "batch_id"],
            columns="endpoint",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    if pivot.empty or "viability_percent" not in pivot.columns or "intact_patch_formation_pass" not in pivot.columns:
        return pd.DataFrame()

    min_viability = float(
        nested_get(
            optimization_config,
            "candidate_generation.rescue_min_viability_percent",
            50.0,
        )
    )
    scale_factors = list(
        nested_get(
            optimization_config,
            "candidate_generation.rescue_scale_factors",
            [0.25, 0.50, 0.75],
        )
        or []
    )
    scale_factors = sorted(
        {
            float(scale)
            for scale in scale_factors
            if pd.notna(scale) and 0.0 < float(scale) < 1.0
        }
    )
    if not scale_factors:
        return pd.DataFrame()

    failed_high_viability = pivot[
        (pivot["viability_percent"] >= min_viability)
        & (pivot["intact_patch_formation_pass"] < 0.5)
    ].copy()
    if failed_high_viability.empty:
        return pd.DataFrame()

    unavailable = set(unavailable_feature_names)
    candidate_formulations = formulations.copy()
    for feature in registry.feature_names:
        if feature not in candidate_formulations.columns:
            candidate_formulations[feature] = 0.0
    anchors = failed_high_viability.merge(
        candidate_formulations[["formulation_id", *registry.feature_names]],
        on="formulation_id",
        how="inner",
    )
    if anchors.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    seen: set[str] = set()
    for _, anchor in anchors.iterrows():
        active_features = [
            feature
            for feature in registry.feature_names
            if feature not in unavailable
            and float(pd.to_numeric(anchor.get(feature, 0.0), errors="coerce") or 0.0) > 1e-12
        ]
        if not active_features:
            continue
        for scale in scale_factors:
            row = {feature: 0.0 for feature in registry.feature_names}
            for feature in active_features:
                value = pd.to_numeric(anchor.get(feature, 0.0), errors="coerce")
                row[feature] = 0.0 if pd.isna(value) else float(value) * scale
            row["candidate_origin"] = "rescue_dilution"
            row["rescue_scale_factor"] = float(scale)
            row["rescue_anchor_formulation_id"] = str(anchor["formulation_id"])
            row["rescue_anchor_viability_percent"] = float(anchor["viability_percent"])
            formulation_id = stable_formulation_id(row, registry)
            if formulation_id in seen:
                continue
            seen.add(formulation_id)
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    rescue = _finalize_generated_rows(rows, registry, candidate_id_prefix="rescue")
    rescue = annotate_feasibility(rescue, registry, optimization_config, policy_active=True)
    rescue = annotate_support(rescue, registry, support)
    rescue = rescue.loc[
        rescue["feasibility_pass"].astype(bool)
        & (rescue["active_ingredient_count"].astype(int) > 0)
    ].reset_index(drop=True)
    if not rescue.empty:
        rescue["recommendation_type"] = "rescue_candidate"
        rescue["selection_explanation"] = (
            "rescue_dilution: scaled down a high-viability failed-patch formulation"
        )
    return rescue


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

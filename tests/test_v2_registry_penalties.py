from __future__ import annotations

import pandas as pd
import pytest

from helper.candidates import (
    filter_available_candidate_pool,
    filter_nonzero_active_candidate_pool,
    generate_random_candidate_pool,
    unavailable_features_from_config,
)
from helper.config import load_optimization_config
from helper.penalties import active_ingredient_excess, count_active_ingredients, single_molar_excesses
from helper.registry import load_registry


def test_registry_contains_new_active_ingredients() -> None:
    registry = load_registry()
    features = set(registry.feature_names)
    assert "taurine_M" in features
    assert "sericin_pct" in features
    assert "raffinose_M" in features
    assert "methylcellulose_pct" in features
    assert "myo_inositol_M" in features
    assert "methoxyphenyl_beta_d_glucopyranoside_M" in features
    assert "trehalose_M" in features


def test_unacquirable_literature_ingredients_are_inactive() -> None:
    registry = load_registry()
    features = set(registry.feature_names)
    assert "ficoll_pct" not in features
    assert "hes_pct" not in features


def test_registry_rejects_media_variables() -> None:
    registry = load_registry()
    with pytest.raises(ValueError, match="excluded"):
        registry.validate_no_excluded_variables(["dmso_M", "dmem_pct"])


def test_penalties_are_soft_and_per_single_molar_ingredient() -> None:
    registry = load_registry()
    optimization_config = load_optimization_config()
    row = {feature_name: 0.0 for feature_name in registry.feature_names}
    for feature_name in registry.feature_names[:9]:
        row[feature_name] = 0.2 if feature_name.endswith("_M") else 1.0
    row["trehalose_M"] = 0.49
    row["raffinose_M"] = 0.49
    row["ethylene_glycol_M"] = 0.51

    assert count_active_ingredients(row, registry) > 8
    assert active_ingredient_excess(row, registry, soft_limit=8) > 0

    excesses = single_molar_excesses(
        row,
        registry,
        limit_M=optimization_config["penalties"]["single_molar_ingredient_limit_M"],
    )
    assert "ethylene_glycol_M" in excesses
    assert "trehalose_M" not in excesses
    assert "raffinose_M" not in excesses


def test_temporary_availability_restrictions_only_affect_candidate_selection() -> None:
    registry = load_registry()
    unavailable = unavailable_features_from_config(
        {"temporarily_unavailable_feature_names": ["trehalose_M", "sericin"]},
        registry,
    )
    assert unavailable == ["sericin_pct", "trehalose_M"]

    pool = generate_random_candidate_pool(
        registry,
        n_candidates=50,
        random_seed=7,
        unavailable_feature_names=unavailable,
    )
    assert (pool["trehalose_M"] == 0.0).all()
    assert (pool["sericin_pct"] == 0.0).all()

    pool.loc[0, "trehalose_M"] = 0.2
    filtered = filter_available_candidate_pool(pool, unavailable)
    assert len(filtered) == len(pool) - 1


def test_zero_active_candidates_are_removed_at_candidate_pool_entry() -> None:
    registry = load_registry()
    pool = generate_random_candidate_pool(
        registry,
        n_candidates=5,
        random_seed=7,
        unavailable_feature_names=[],
    )
    zero_row = {feature_name: 0.0 for feature_name in registry.feature_names}
    zero_row.update({"candidate_id": "zero_row", "formulation_id": "v2_zero", "active_ingredient_count": 0})
    pool = pd.concat([pool, pd.DataFrame([zero_row])], ignore_index=True)

    filtered = filter_nonzero_active_candidate_pool(pool, registry)

    assert "zero_row" not in set(filtered["candidate_id"])
    assert (filtered["active_ingredient_count"] > 0).all()

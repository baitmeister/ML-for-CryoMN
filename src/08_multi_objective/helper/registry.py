"""Ingredient registry and canonical feature handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import load_ingredients_config
from .paths import INGREDIENTS_CONFIG


MOLAR_NEGLIGIBLE_THRESHOLD = 0.001
PERCENT_NEGLIGIBLE_THRESHOLD = 0.1
FEATURE_NEGLIGIBLE_THRESHOLD = 1e-6


@dataclass(frozen=True)
class Ingredient:
    """One canonical formulation ingredient."""

    canonical_name: str
    display_name: str
    feature_name: str
    unit: str
    active: bool
    molecular_weight_g_mol: float | None
    lower_bound: float
    upper_bound: float
    penalize_single_over_500mM: bool
    synonyms: tuple[str, ...] = field(default_factory=tuple)
    reagent_form: str = ""
    ingredient_class: str = ""
    aggregate_groups: tuple[str, ...] = field(default_factory=tuple)
    practical_upper_bound: float | None = None
    evidence_type: str = ""
    limit_basis: str = ""
    sources: tuple[str, ...] = field(default_factory=tuple)
    aqueous_solubility_M: float | None = None
    solubility_reference_temperature_C: float | None = None
    solubility_evidence_type: str = ""
    grade_or_molecular_weight: str = ""

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "Ingredient":
        synonyms = mapping.get("synonyms") or []
        aggregate_groups = mapping.get("aggregate_groups") or []
        sources = mapping.get("sources") or []
        if isinstance(sources, str):
            sources = [sources]
        return cls(
            canonical_name=str(mapping["canonical_name"]),
            display_name=str(mapping.get("display_name") or mapping["canonical_name"]),
            feature_name=str(mapping["feature_name"]),
            unit=str(mapping["unit"]),
            active=bool(mapping.get("active", True)),
            molecular_weight_g_mol=(
                None
                if mapping.get("molecular_weight_g_mol") in (None, "")
                else float(mapping["molecular_weight_g_mol"])
            ),
            lower_bound=float(mapping.get("lower_bound", 0.0)),
            upper_bound=float(mapping.get("upper_bound", 0.0)),
            penalize_single_over_500mM=bool(mapping.get("penalize_single_over_500mM", False)),
            synonyms=tuple(str(item) for item in synonyms),
            reagent_form=str(mapping.get("reagent_form") or ""),
            ingredient_class=str(mapping.get("ingredient_class") or ""),
            aggregate_groups=tuple(str(item) for item in aggregate_groups),
            practical_upper_bound=(
                None
                if mapping.get("practical_upper_bound") in (None, "")
                else float(mapping["practical_upper_bound"])
            ),
            evidence_type=str(mapping.get("evidence_type") or ""),
            limit_basis=str(mapping.get("limit_basis") or ""),
            sources=tuple(str(item) for item in sources),
            aqueous_solubility_M=(
                None
                if mapping.get("aqueous_solubility_M") in (None, "")
                else float(mapping["aqueous_solubility_M"])
            ),
            solubility_reference_temperature_C=(
                None
                if mapping.get("solubility_reference_temperature_C") in (None, "")
                else float(mapping["solubility_reference_temperature_C"])
            ),
            solubility_evidence_type=str(mapping.get("solubility_evidence_type") or ""),
            grade_or_molecular_weight=str(mapping.get("grade_or_molecular_weight") or ""),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "display_name": self.display_name,
            "feature_name": self.feature_name,
            "unit": self.unit,
            "active": self.active,
            "molecular_weight_g_mol": self.molecular_weight_g_mol,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "penalize_single_over_500mM": self.penalize_single_over_500mM,
            "synonyms": ";".join(self.synonyms),
            "reagent_form": self.reagent_form,
            "ingredient_class": self.ingredient_class,
            "aggregate_groups": ";".join(self.aggregate_groups),
            "practical_upper_bound": self.practical_upper_bound,
            "evidence_type": self.evidence_type,
            "limit_basis": self.limit_basis,
            "sources": ";".join(self.sources),
            "aqueous_solubility_M": self.aqueous_solubility_M,
            "solubility_reference_temperature_C": self.solubility_reference_temperature_C,
            "solubility_evidence_type": self.solubility_evidence_type,
            "grade_or_molecular_weight": self.grade_or_molecular_weight,
        }


class IngredientRegistry:
    """Lookup object for formulation feature metadata."""

    def __init__(self, ingredients: Iterable[Ingredient], excluded_variables: Iterable[str] = ()):
        self.ingredients = list(ingredients)
        self.excluded_variables = {self._normalize_name(name) for name in excluded_variables}
        self._by_feature = {ingredient.feature_name: ingredient for ingredient in self.ingredients}
        self._by_name: dict[str, Ingredient] = {}
        for ingredient in self.ingredients:
            names = [ingredient.canonical_name, ingredient.display_name, *ingredient.synonyms]
            for name in names:
                self._by_name[self._normalize_name(name)] = ingredient

    @classmethod
    def from_config(cls, path: str | Path = INGREDIENTS_CONFIG) -> "IngredientRegistry":
        config = load_ingredients_config(path)
        ingredients = [Ingredient.from_mapping(item) for item in config.get("ingredients", [])]
        return cls(ingredients, config.get("excluded_variables", []))

    @staticmethod
    def _normalize_name(name: str) -> str:
        return str(name).strip().lower().replace("-", "_").replace(" ", "_")

    @property
    def feature_names(self) -> list[str]:
        return [ingredient.feature_name for ingredient in self.ingredients if ingredient.active]

    @property
    def molar_feature_names(self) -> list[str]:
        return [
            ingredient.feature_name
            for ingredient in self.ingredients
            if ingredient.active and ingredient.unit == "M"
        ]

    def active_ingredients(self) -> list[Ingredient]:
        return [ingredient for ingredient in self.ingredients if ingredient.active]

    def get_by_feature(self, feature_name: str) -> Ingredient:
        return self._by_feature[feature_name]

    def resolve_name(self, name: str) -> Ingredient | None:
        return self._by_name.get(self._normalize_name(name))

    def validate_no_excluded_variables(self, feature_names: Iterable[str]) -> None:
        """Reject culture media or basal-buffer variables in formulation features."""
        excluded_hits = []
        for feature_name in feature_names:
            normalized = self._normalize_name(feature_name)
            root = normalized.removesuffix("_m").removesuffix("_pct")
            if normalized in self.excluded_variables or root in self.excluded_variables:
                excluded_hits.append(feature_name)
        if excluded_hits:
            raise ValueError(
                "Culture media and basal buffers are excluded as variables: "
                + ", ".join(sorted(excluded_hits))
            )

    def validate_feasibility_metadata(self) -> None:
        """Require Round-5 chemistry metadata for every selectable ingredient.

        Availability is deliberately absent from this validation. An ingredient
        remains active in the chemistry registry while temporary availability is
        handled by the sampling layer.
        """
        errors: list[str] = []
        for ingredient in self.active_ingredients():
            prefix = ingredient.feature_name
            for field_name, value in (
                ("reagent_form", ingredient.reagent_form),
                ("ingredient_class", ingredient.ingredient_class),
                ("evidence_type", ingredient.evidence_type),
                ("limit_basis", ingredient.limit_basis),
            ):
                if not value.strip():
                    errors.append(f"{prefix}: missing {field_name}")
            if not ingredient.sources:
                errors.append(f"{prefix}: missing sources")
            if ingredient.practical_upper_bound is None:
                errors.append(f"{prefix}: missing practical_upper_bound")
            elif not (
                ingredient.lower_bound - 1e-12
                <= ingredient.practical_upper_bound
                <= ingredient.upper_bound + 1e-12
            ):
                errors.append(
                    f"{prefix}: practical_upper_bound must be within registry bounds"
                )
            if ingredient.unit == "M" and ingredient.molecular_weight_g_mol is None:
                errors.append(f"{prefix}: missing molecular_weight_g_mol")
            if "crystalline_solute" in ingredient.aggregate_groups:
                if ingredient.unit != "M":
                    errors.append(f"{prefix}: crystalline solute must use M")
                if not ingredient.aqueous_solubility_M or ingredient.aqueous_solubility_M <= 0:
                    errors.append(f"{prefix}: missing positive aqueous_solubility_M")
                if ingredient.solubility_reference_temperature_C is None:
                    errors.append(
                        f"{prefix}: missing solubility_reference_temperature_C"
                    )
                if not ingredient.solubility_evidence_type.strip():
                    errors.append(f"{prefix}: missing solubility_evidence_type")
            if (
                "viscosity_active_polymer" in ingredient.aggregate_groups
                and not ingredient.grade_or_molecular_weight.strip()
            ):
                errors.append(f"{prefix}: missing grade_or_molecular_weight")
        if errors:
            raise ValueError(
                "Ingredient feasibility metadata validation failed:\n- "
                + "\n- ".join(errors)
            )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([ingredient.to_mapping() for ingredient in self.ingredients])

    def export_csv(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)


def presence_threshold(feature_name: str) -> float:
    if feature_name.endswith("_M"):
        return MOLAR_NEGLIGIBLE_THRESHOLD
    if feature_name.endswith("_pct"):
        return PERCENT_NEGLIGIBLE_THRESHOLD
    return FEATURE_NEGLIGIBLE_THRESHOLD


def load_registry(path: str | Path = INGREDIENTS_CONFIG) -> IngredientRegistry:
    registry = IngredientRegistry.from_config(path)
    registry.validate_no_excluded_variables(registry.feature_names)
    registry.validate_feasibility_metadata()
    return registry

#!/usr/bin/env python3
"""
CryoMN Formulation Data Parser

Parses cryoprotective solution formulation data from literature-derived CSV.
Extracts ingredients, normalizes concentrations to molar units, and handles
ingredient synonym merging.

Author: CryoMN ML Project
Date: 2026-01-24
"""

import pandas as pd
import numpy as np
import re
import os
import hashlib
import json
import warnings
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# =============================================================================
# INGREDIENT SYNONYM MAPPINGS
# =============================================================================

INGREDIENT_SYNONYMS = {
    # Permeating CPAs
    'propylene_glycol': ['1,2-propanediol', 'propylene glycol', 'proh', 'pg', '1,2-propane diol'],
    'dmso': ['dmso', 'me2so', 'dimethyl sulfoxide', 'dimethylsulfoxide'],
    'ethylene_glycol': ['eg', 'ethylene glycol', 'ethyleneglycol'],
    'glycerol': ['glycerol', 'glycerin'],
    
    # Sugars and sugar alcohols
    'trehalose': ['trehalose'],
    'sucrose': ['sucrose'],
    'glucose': ['glucose', 'dextrose'],
    'mannitol': ['mannitol'],
    'maltose': ['maltose'],
    'raffinose': ['raffinose'],
    
    # Polymers - PEG split by molecular weight (individual MWs as separate ingredients)
    # Low MW PEG (≤1000 Da): cell penetrating
    'peg_400': ['peg 400', 'peg-400', 'peg400'],
    'peg_600': ['peg 600', 'peg-600', 'peg600'],
    'peg_1k': ['peg 1k', 'peg-1k', 'peg1k', 'peg 1000', 'peg-1000', 'peg1000'],
    # Medium MW PEG (1.5K-10K Da): intermediate behavior
    'peg_1500': ['peg 1.5k', 'peg-1.5k', 'peg1.5k', 'peg 1500', 'peg-1500'],
    'peg_2k': ['peg 2k', 'peg-2k', 'peg2k', 'peg 2000', 'peg-2000'],
    'peg_3350': ['peg 3.35k', 'peg-3.35k', 'peg 3350', 'peg-3350'],
    'peg_4k': ['peg 4k', 'peg-4k', 'peg4k', 'peg 4000', 'peg-4000'],
    'peg_5k': ['peg 5k', 'peg-5k', 'peg5k', 'peg 5000', 'peg-5000'],
    'peg_6k': ['peg 6k', 'peg-6k', 'peg6k', 'peg 6000', 'peg-6000'],
    'peg_8k': ['peg 8k', 'peg-8k', 'peg8k', 'peg 8000', 'peg-8000'],
    'peg_10k': ['peg 10k', 'peg-10k', 'peg10k', 'peg 10000', 'peg-10000'],
    # High MW PEG (>10K Da): non-penetrating, extracellular CPA
    'peg_20k': ['peg 20k', 'peg-20k', 'peg20k', 'peg 20000', 'peg-20000'],
    'peg_35k': ['peg 35k', 'peg-35k', 'peg35k', 'peg 35000', 'peg-35000'],
    # Generic PEG (unspecified MW, default to common 3350 grade)
    'pvp': ['pvp', 'polyvinylpyrrolidone'],
    'pva': ['pva', 'polyvinyl alcohol'],
    'hes': ['hes', 'hydroxyethyl starch', 'hes450', 'hydroxychyl starch', 'hydroxyethylstarch'],
    'dextran': ['dextran', 'dextran-40', 'dextran40'],
    'ficoll': ['ficoll', 'ficoll 70', 'ficoll70'],
    'pentaisomaltose': ['pentaisomaltose'],
    
    # Proteins and sera
    'fbs': ['fbs', 'fcs', 'fetal bovine serum', 'fetal calf serum'],
    'human_serum': ['hs', 'human serum'],
    'hsa': ['hsa', 'human albumin', 'human serum albumin', 'albumin', 'bsa', 'bovine serum albumin'],
    
    # Amino acids and compatible solutes
    'proline': ['proline', 'l-proline'],
    'ectoin': ['ectoin', 'ectoine'],
    'isoleucine': ['isoleucine', 'l-isoleucine'],
    'taurine': ['taurine'],
    'betaine': ['betaine'],
    
    # Other additives
    'methylcellulose': ['mc', 'methylcellulose'],
    'hyaluronic_acid': ['hmw-ha', 'hyaluronic acid'],
    'creatine': ['creatine'],
    'acetamide': ['acetamide'],
    'sericin': ['sericin'],
    
    # Polyampholytes
    'cooh_pll': ['cooh-pll', 'polyampholyte'],
    'peg_pa': ['peg-pa', 'peg−pa'],
}

# Build reverse lookup
SYNONYM_TO_CANONICAL = {}
for canonical, synonyms in INGREDIENT_SYNONYMS.items():
    for syn in synonyms:
        SYNONYM_TO_CANONICAL[syn.lower()] = canonical

# =============================================================================
# MOLECULAR WEIGHTS (g/mol) for unit conversion
# =============================================================================

MOLECULAR_WEIGHTS = {
    'dmso': 78.13,
    'ethylene_glycol': 62.07,
    'propylene_glycol': 76.09,
    'glycerol': 92.09,
    'trehalose': 342.3,
    'sucrose': 342.3,
    'glucose': 180.16,
    'mannitol': 182.17,
    'maltose': 342.3,
    'raffinose': 504.42,
    'proline': 115.13,
    'ectoin': 142.16,
    'isoleucine': 131.17,
    'taurine': 125.15,
    'betaine': 117.15,
    'creatine': 131.13,
    'acetamide': 59.07,
}

# Density (g/mL) for % v/v conversions
DENSITIES = {
    'dmso': 1.10,
    'ethylene_glycol': 1.11,
    'propylene_glycol': 1.036,
    'glycerol': 1.26,
}

# =============================================================================
# PERCENTAGE-ONLY INGREDIENTS
# Ingredients that cannot/should not be converted to molar concentrations.
# These will retain percentage (%) units in the output with _pct suffix.
# =============================================================================

PERCENTAGE_ONLY_INGREDIENTS = {
    # Sera and proteins (complex mixtures, no defined MW)
    'fbs', 'human_serum', 'hsa', 'sericin',
    # Polymers - PEG by specific MW (individual ingredients)
    'peg_400', 'peg_600', 'peg_1k',  # Low MW
    'peg_1500', 'peg_2k', 'peg_3350', 'peg_4k', 'peg_5k', 'peg_6k', 'peg_8k', 'peg_10k',  # Medium MW
    'peg_20k', 'peg_35k',  # High MW
    # Other polymers
    'pvp', 'pva', 'hes', 'dextran', 'ficoll',
    'methylcellulose', 'hyaluronic_acid',
    # Polyampholytes and other complex molecules
    'cooh_pll', 'peg_pa', 'pentaisomaltose',
}

# =============================================================================
# CULTURE MEDIA TO EXCLUDE (not counted as separate ingredients)
# =============================================================================

CULTURE_MEDIA = [
    'dmem', 'dmem/f12', 'dmem/ham', "dulbecco's modified eagle medium",
    'α-mem', 'a-mem', 'alpha-mem', 'mem', 'minimum essential medium',
    'pbs', 'phosphate buffer saline', 'phosphate buffered saline',
    'dpbs', 'hbss', 'saline', 'nacl', '0.9% nacl',
    'mesencult', 'l-15', 'l15', 'medium 199', 'knockout dmem',
    'plasmalyte', 'plasmalyte a', 'lactated ringer', 'complete culture medium',
    'culture medium', 'basal medium', 'electroporation buffer',
    'e8 medium', 'cssc medium',
]


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def extract_concentration(text: str, ingredient: str) -> Tuple[Optional[float], Optional[str]]:
    """Extract the concentration using the unit written next to its value.

    Ingredient names are case-insensitive, but M (molar) and mM
    (millimolar) are distinct. Never infer units from the regex source or
    lowercase the input: either would turn molar values into millimolar.
    """
    name = rf'(?i:{re.escape(ingredient)})'
    # Do not accept the trailing "5M" in a malformed value such as "o.5M".
    value = r'(?<![\w.])(\d+(?:\.\d*)?|\.\d+)'

    percentage_patterns = [
        rf'{value}\s*%\s*(?i:v/v\s*)?{name}',
        rf'{name}\s*{value}\s*%',
        rf'{value}\s*%\s*\(?(?i:v/v)\)?\s*{name}',
        rf'{name}\s*\(\s*{value}\s*(?i:wt%|wt\s*%|w/v%)',
        rf'{name}\s*\(\s*{value}\s*%',
    ]
    for pattern in percentage_patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)), '%'

    unit = r'(?P<unit>mM|M|(?i:mmol/[lL]|mg/ml))'
    for pattern in [rf'{value}\s*{unit}\s*{name}',
                    rf'{name}\s*{value}\s*{unit}(?!\w)']:
        match = re.search(pattern, text)
        if match:
            matched_unit = match.group('unit')
            if matched_unit.lower().startswith('mmol/'):
                matched_unit = 'mM'
            elif matched_unit.lower() == 'mg/ml':
                matched_unit = 'mg/ml'
            return float(match.group(1)), matched_unit

    return None, None


def convert_to_molar(value: float, unit: str, ingredient: str) -> Optional[float]:
    """
    Convert concentration to molar (M).
    
    Args:
        value: Numeric concentration value
        unit: Unit string ('%', 'mM', 'M', 'mg/ml')
        ingredient: Canonical ingredient name
        
    Returns:
        Concentration in molar (M), or None if conversion not possible
    """
    if value is None or unit is None:
        return None
    
    canonical = SYNONYM_TO_CANONICAL.get(ingredient.lower(), ingredient.lower())
    
    if unit == 'M':
        return value
    
    if unit == 'mM':
        return value / 1000.0
    
    if unit == '%':
        # For % v/v with known density and MW
        if canonical in DENSITIES and canonical in MOLECULAR_WEIGHTS:
            density = DENSITIES[canonical]
            mw = MOLECULAR_WEIGHTS[canonical]
            # % v/v -> g/mL -> mol/L
            return (value / 100.0) * density * 1000 / mw
        # For % w/v with known MW
        elif canonical in MOLECULAR_WEIGHTS:
            mw = MOLECULAR_WEIGHTS[canonical]
            # % w/v = g/100mL -> g/L / MW = M
            return (value * 10) / mw
        else:
            # Can't convert, keep as percentage (will need manual review)
            return None
    
    if unit == 'mg/ml':
        if canonical in MOLECULAR_WEIGHTS:
            mw = MOLECULAR_WEIGHTS[canonical]
            # mg/mL = g/L -> M
            return value / mw
    
    return None


def classify_peg_mw(text: str) -> Optional[str]:
    """
    Classify PEG molecular weight from text.
    
    Args:
        text: Text containing PEG mention (e.g., "PEG 400", "PEG-20K", "polyethylene glycol")
        
    Returns:
        Specific PEG MW name (e.g., 'peg_400', 'peg_3350', 'peg_20k'), or None if no PEG found.
        For generic 'PEG' without MW, returns 'peg_3350' (most common lab grade).
    """
    text_lower = text.lower()
    
    # Check if PEG is mentioned at all
    if 'peg' not in text_lower and 'polyethylene glycol' not in text_lower:
        return None
    
    # Pattern to extract MW: "PEG 400", "PEG-1K", "PEG 20000", "PEG-1.5k"
    # Matches: peg[space/-]?[number][k/K]?
    mw_pattern = r'peg[\s\-]?(\d+\.?\d*)\s*([kK])?'
    match = re.search(mw_pattern, text, re.IGNORECASE)
    
    if match:
        mw_value = float(match.group(1))
        multiplier = match.group(2)
        
        # Convert to Da if 'k' or 'K' suffix
        if multiplier and multiplier.lower() == 'k':
            mw_da = mw_value * 1000
        else:
            mw_da = mw_value
        
        # Map to specific PEG ingredient name based on MW
        if mw_da <= 450:
            return 'peg_400'
        elif mw_da <= 800:
            return 'peg_600'
        elif mw_da <= 1200:
            return 'peg_1k'
        elif mw_da <= 1800:
            return 'peg_1500'
        elif mw_da <= 2500:
            return 'peg_2k'
        elif mw_da <= 4000:
            return 'peg_3350'
        elif mw_da <= 4500:
            return 'peg_4k'
        elif mw_da <= 7000:
            return 'peg_5k'
        elif mw_da <= 7500:
            return 'peg_6k'
        elif mw_da <= 9000:
            return 'peg_8k'
        elif mw_da <= 15000:
            return 'peg_10k'
        elif mw_da <= 30000:
            return 'peg_20k'
        else:
            return 'peg_35k'
    
    # Generic "PEG" or "polyethylene glycol" without MW -> default to peg_3350 (common grade)
    if 'peg' in text_lower or 'polyethylene glycol' in text_lower:
        return 'peg_3350'
    
    return None


def parse_formulation_text(text: str) -> Dict[str, Tuple[float, str]]:
    """
    Parse a formulation text to extract all ingredients and their concentrations.
    
    Args:
        text: Free-text formulation description
        
    Returns:
        Dictionary mapping canonical ingredient names to (value, unit_type) tuples
        where unit_type is 'M' for molar or '%' for percentage
    """
    if pd.isna(text) or not text.strip():
        return {}
    
    text_lower = text.lower()
    ingredients = {}
    
    # Check for each known ingredient
    for canonical, synonyms in INGREDIENT_SYNONYMS.items():
        for syn in synonyms:
            if syn.lower() in text_lower:
                value, unit = extract_concentration(text, syn)
                if value is not None:
                    # Check if this ingredient should remain as percentage
                    if canonical in PERCENTAGE_ONLY_INGREDIENTS:
                        # Keep as percentage, don't attempt molar conversion
                        ingredients[canonical] = (value, '%')
                    else:
                        molar = convert_to_molar(value, unit, canonical)
                        if molar is not None:
                            ingredients[canonical] = (molar, 'M')
                        else:
                            # Fallback to percentage if conversion fails
                            ingredients[canonical] = (value, '%')
                    break
                else:
                    # Ingredient mentioned but no concentration found
                    # Check if it's a qualitative mention (just presence)
                    pass
    
    # Special handling for generic "PEG" or "polyethylene glycol" mentions
    # that weren't caught by specific MW synonyms
    if not any(k.startswith('peg_') for k in ingredients.keys()):
        if 'peg' in text_lower or 'polyethylene glycol' in text_lower:
            # Try to classify by MW from text
            peg_category = classify_peg_mw(text)
            if peg_category:
                # Extract concentration for generic "peg" or "polyethylene glycol"
                value, unit = extract_concentration(text, 'peg')
                if value is None:
                    value, unit = extract_concentration(text, 'polyethylene glycol')
                if value is not None:
                    ingredients[peg_category] = (value, '%')
    
    return ingredients


def extract_viability(text: str) -> Optional[float]:
    """
    Extract viability percentage from viability column text.
    
    Args:
        text: Viability text (e.g., "82.5 ± 8.3%", "~80%", ">90%")
        
    Returns:
        Viability as float percentage, or None if not extractable
    """
    if pd.isna(text) or not text.strip():
        return None
    
    text = str(text).strip()
    
    # Skip non-mentioned values
    if 'not mentioned' in text.lower() or 'not reported' in text.lower():
        return None
    
    # Handle "compared to standard DMSO" cases - mark for review
    if 'compared to' in text.lower():
        # Extract the percentage if present
        match = re.search(r'(\d+\.?\d*)\s*%', text)
        if match:
            return float(match.group(1))
        return None
    
    # Handle ranges like "~50-65%"
    range_match = re.search(r'~?(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*%', text)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return (low + high) / 2
    
    # Handle "82.5 ± 8.3%" or "(82.5±8.3)%"
    match = re.search(r'\(?(\d+\.?\d*)\s*[±+/-]+\s*\d+\.?\d*\)?%?', text)
    if match:
        return float(match.group(1))
    
    # Handle ">80%" or "<10%"
    match = re.search(r'[<>≤≥~]?\s*(\d+\.?\d*)\s*%', text)
    if match:
        return float(match.group(1))
    
    # Handle plain percentage
    match = re.search(r'(\d+\.?\d*)\s*%', text)
    if match:
        return float(match.group(1))
    
    # Handle just a number that looks like percentage
    match = re.search(r'^(\d+\.?\d*)$', text.strip())
    if match:
        val = float(match.group(1))
        if 0 <= val <= 100:
            return val
    
    return None


def extract_dmso_percentage(dmso_col: str, formulation_text: str) -> float:
    """
    Extract DMSO percentage from the dedicated DMSO column or formulation text.
    
    Args:
        dmso_col: Value from "DMSO usage" column
        formulation_text: Full formulation text
        
    Returns:
        DMSO percentage as float, defaults to 0 if not found
    """
    # First try the dedicated DMSO column
    if pd.notna(dmso_col):
        text = str(dmso_col).strip()
        match = re.search(r'(\d+\.?\d*)\s*%?', text)
        if match:
            return float(match.group(1))
    
    # Fall back to parsing from formulation text
    ingredients = parse_formulation_text(formulation_text)
    if 'dmso' in ingredients:
        # Convert back to percentage if in molar
        molar = ingredients['dmso']
        # DMSO: MW = 78.13, density = 1.10
        # M -> % v/v
        percentage = molar * 78.13 / (1.10 * 10)
        return percentage
    
    return 0.0


def is_culture_media(text: str) -> bool:
    """Check if text refers to culture media only."""
    text_lower = text.lower().strip()
    for media in CULTURE_MEDIA:
        if media in text_lower:
            return True
    return False


def parse_csv(input_path: str) -> pd.DataFrame:
    """
    Parse the raw CSV file and extract structured formulation data.
    
    Args:
        input_path: Path to raw CSV file
        
    Returns:
        DataFrame with parsed formulation data
    """
    # Read CSV
    df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='skip')
    
    # Get column names
    formulation_col = df.columns[0]  # "All ingredients in cryoprotective solution"
    dmso_col = df.columns[1]  # "DMSO usage"
    viability_col = df.columns[3]  # "Viability"
    source_col = df.columns[8] if len(df.columns) > 8 else None  # "Source (DOI link)"
    
    # Collect all unique ingredients with their unit types
    # Dict[ingredient_name, unit_type] where unit_type is 'M' or '%'
    all_ingredients: Dict[str, str] = {}
    parsed_rows = []
    
    print("Parsing formulations...")
    
    for idx, row in df.iterrows():
        formulation_text = str(row[formulation_col]) if pd.notna(row[formulation_col]) else ''
        
        # Skip blank rows
        if not formulation_text.strip():
            continue
        
        # Extract viability
        viability = extract_viability(row[viability_col] if len(df.columns) > 3 else None)
        
        # Skip rows without viability data (can't train on them)
        if viability is None:
            continue
        
        # Parse ingredients into Dict[str, Tuple[float, str]]
        ingredients = parse_formulation_text(formulation_text)
        
        # Also get DMSO from dedicated column for accuracy
        dmso_pct = extract_dmso_percentage(
            row[dmso_col] if len(df.columns) > 1 else None,
            formulation_text
        )
        
        # Convert DMSO % to molar (DMSO is always molar)
        if dmso_pct > 0:
            dmso_molar = (dmso_pct / 100.0) * 1.10 * 1000 / 78.13
            ingredients['dmso'] = (dmso_molar, 'M')
        
        # Track all ingredients with their unit types
        for ingredient, (value, unit_type) in ingredients.items():
            if ingredient not in all_ingredients:
                all_ingredients[ingredient] = unit_type
        
        # Get source DOI
        source = row[source_col] if source_col and pd.notna(row[source_col]) else ''
        
        parsed_rows.append({
            'formulation_id': idx + 1,
            'original_text': formulation_text[:200],  # Truncate for readability
            'ingredients': ingredients,
            'viability_percent': viability,
            'dmso_percent': dmso_pct,
            'source_doi': source,
        })
    
    print(f"Parsed {len(parsed_rows)} formulations with viability data")
    print(f"Found {len(all_ingredients)} unique ingredients")
    
    # Create output DataFrame with one column per ingredient
    # Use correct suffix based on unit type: _M for molar, _pct for percentage
    output_data = []
    for row in parsed_rows:
        row_data = {
            'formulation_id': row['formulation_id'],
            'viability_percent': row['viability_percent'],
            'dmso_percent': row['dmso_percent'],
            'source_doi': row['source_doi'],
        }
        
        # Add each ingredient column with appropriate suffix
        for ingredient in sorted(all_ingredients.keys()):
            unit_type = all_ingredients[ingredient]
            suffix = '_pct' if unit_type == '%' else '_M'
            col_name = f'{ingredient}{suffix}'
            # Get value from ingredients dict, default to 0.0
            ing_data = row['ingredients'].get(ingredient)
            row_data[col_name] = ing_data[0] if ing_data else 0.0
        
        output_data.append(row_data)
    
    output_df = pd.DataFrame(output_data)
    
    return output_df


def apply_concentration_review(df: pd.DataFrame, input_path: str,
                               review_path: str) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """Apply explicit pending-review holds for the source-checked v2 release.

    These holds are not scientific corrections: their legacy values remain
    unverified. The raw-file digest prevents applying row IDs to another input.
    Verified concentrations are checked against the parser, never overridden.
    """
    with open(review_path, encoding='utf-8') as handle:
        review = json.load(handle)
    with open(input_path, 'rb') as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if digest != review['input_sha256']:
        raise ValueError('Raw CSV changed: update the concentration review before writing output.')

    reviewed = df.copy()
    for entry in review['confirmed'] + review['pending']:
        rows = reviewed['formulation_id'].isin(entry['raw_records'])
        if int(rows.sum()) != len(entry['raw_records']):
            raise ValueError(f"Missing reviewed raw records: {entry['raw_records']}")
        expected = entry.get('correct_M', entry.get('raw_text_M'))
        if not np.allclose(reviewed.loc[rows, entry['field']], expected,
                           rtol=1e-12, atol=1e-14):
            raise ValueError(f"Parsed concentration differs from review: {entry['field']}")

    for entry in review['pending']:
        rows = reviewed['formulation_id'].isin(entry['raw_records'])
        reviewed.loc[rows, entry['field']] = entry['retained_legacy_M']
    if review['pending']:
        warnings.warn(
            f"{len(review['pending'])} concentration fields remain UNVERIFIED; "
            "legacy values retained by the v2 review. See data/raw/concentration_review_v2.json. "
            "These holds must be resolved before treating the dataset as fully validated.",
            UserWarning,
            stacklevel=2,
        )

    protected_groups = {}
    for group in review['preserved_duplicate_groups']:
        for raw_id in group['raw_records']:
            if raw_id in protected_groups or raw_id not in set(df['formulation_id']):
                raise ValueError(f'Invalid preserved duplicate-group record: {raw_id}')
            protected_groups[raw_id] = group['formulation_id']
    return reviewed, protected_groups


def _formulation_key(row, ingredient_cols, protected_groups):
    # Preserve the baseline target groups for equal compositions across
    # studies. Concentration correction must not re-average targets.
    profile = tuple(round(row[col], 6) for col in ingredient_cols)
    group = (protected_groups or {}).get(int(row['formulation_id']))
    return profile, group


def find_duplicate_formulations(df: pd.DataFrame,
                                protected_groups: Optional[Dict[int, int]] = None
                                ) -> List[Tuple[int, int, float, float]]:
    """
    Find formulations with identical ingredient profiles but different viabilities.
    
    Returns:
        List of tuples: (row_idx1, row_idx2, viability1, viability2)
    """
    # Get ingredient columns (both molar and percentage)
    ingredient_cols = [c for c in df.columns if c.endswith('_M') or c.endswith('_pct')]
    
    duplicates = []
    seen = {}
    
    for idx, row in df.iterrows():
        # Create a hashable key from ingredient values
        key = _formulation_key(row, ingredient_cols, protected_groups)
        
        if key in seen:
            prev_idx, prev_viability = seen[key]
            curr_viability = row['viability_percent']
            if abs(prev_viability - curr_viability) > 1.0:  # More than 1% difference
                duplicates.append((prev_idx, idx, prev_viability, curr_viability))
        else:
            seen[key] = (idx, row['viability_percent'])
    
    return duplicates


def auto_resolve_duplicates(df: pd.DataFrame, duplicates: List[Tuple], interactive: bool = False,
                            protected_groups: Optional[Dict[int, int]] = None) -> pd.DataFrame:
    """
    Automatically resolve duplicate formulations by averaging viabilities.
    
    For identical formulations with different viabilities, this function:
    1. Groups all duplicates together
    2. Averages their viability values
    3. Keeps one representative row with the averaged viability
    
    Args:
        df: Parsed DataFrame
        duplicates: List of duplicate pairs
        interactive: If True, prompt for each duplicate (only for small numbers)
        protected_groups: Optional raw-record groups retained during source review
        
    Returns:
        DataFrame with duplicates resolved
    """
    if not duplicates:
        print("No duplicate formulations with different viabilities found.")
        return df
    
    print(f"\nFound {len(duplicates)} pairs of formulations with identical ingredients but different viabilities.")
    
    # Get ingredient columns (both molar and percentage)
    ingredient_cols = [c for c in df.columns if c.endswith('_M') or c.endswith('_pct')]
    
    # Group all rows by their ingredient profile
    profile_groups = defaultdict(list)
    for idx, row in df.iterrows():
        key = _formulation_key(row, ingredient_cols, protected_groups)
        profile_groups[key].append(idx)
    
    # Find groups with multiple entries and different viabilities
    rows_to_drop = []
    rows_to_update = {}
    
    for key, indices in profile_groups.items():
        if len(indices) > 1:
            viabilities = [df.loc[idx, 'viability_percent'] for idx in indices]
            sources = [df.loc[idx, 'source_doi'] for idx in indices if pd.notna(df.loc[idx, 'source_doi'])]
            
            # Calculate average viability
            avg_viability = np.mean(viabilities)
            
            # Keep first row, update with average, drop others
            keep_idx = indices[0]
            rows_to_update[keep_idx] = {
                'viability_percent': avg_viability,
                'source_doi': '; '.join(set(str(s) for s in sources if s))  # Combine sources
            }
            rows_to_drop.extend(indices[1:])
    
    # Apply updates
    for idx, updates in rows_to_update.items():
        for col, val in updates.items():
            df.loc[idx, col] = val
    
    # Drop duplicate rows
    if rows_to_drop:
        df = df.drop(rows_to_drop).reset_index(drop=True)
        print(f"Averaged {len(rows_to_update)} duplicate groups, dropped {len(rows_to_drop)} redundant rows.")
        print(f"{len(df)} unique formulations remaining.")
    
    return df




def main():
    """Main entry point for parsing script."""
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    input_path = os.path.join(project_root, 'data', 'raw', 'Cryopreservative Data 2026.csv')
    output_path = os.path.join(project_root, 'data', 'processed', 'parsed_formulations.csv')
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("=" * 80)
    print("CryoMN Formulation Data Parser")
    print("=" * 80)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print()
    
    # Parse CSV
    df = parse_csv(input_path)
    
    # The v2 review limits this release to confirmed concentration corrections.
    # Never silently replace pending source values or merge equal profiles.
    review_path = os.path.join(project_root, 'data', 'raw', 'concentration_review_v2.json')
    df, protected_groups = apply_concentration_review(df, input_path, review_path)

    # Retain baseline duplicate averaging within each reviewed target group.
    duplicates = find_duplicate_formulations(df, protected_groups)
    if duplicates:
        df = auto_resolve_duplicates(df, duplicates, protected_groups=protected_groups)
    
    # Save output
    df.to_csv(output_path, index=False)
    print(f"\nSaved parsed data to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total formulations: {len(df)}")
    print(f"Viability range: {df['viability_percent'].min():.1f}% - {df['viability_percent'].max():.1f}%")
    print(f"Mean viability: {df['viability_percent'].mean():.1f}%")
    print(f"Formulations with 0% DMSO: {(df['dmso_percent'] == 0).sum()}")
    print(f"Formulations with <5% DMSO: {(df['dmso_percent'] < 5).sum()}")
    
    # List ingredients
    ingredient_cols = [c for c in df.columns if c.endswith('_M') or c.endswith('_pct')]
    print(f"\nIngredients detected ({len(ingredient_cols)}):")
    for col in sorted(ingredient_cols):
        non_zero = (df[col] > 0).sum()
        if non_zero > 0:
            # Show unit type in the summary
            unit_label = '(%)' if col.endswith('_pct') else '(M)'
            ingredient_name = col.replace('_M', '').replace('_pct', '')
            print(f"  - {ingredient_name} {unit_label}: {non_zero} formulations")


if __name__ == '__main__':
    main()

# Step 1: Data Parsing

## Overview

This module parses cryoprotective solution formulation data from literature-derived CSV files. It extracts ingredients, normalizes concentrations, and handles ingredient synonym merging. Ingredients are categorized as either molar-convertible or percentage-only based on their molecular properties.

## Usage

```bash
cd "/path/to/project"
python src/01_data_parsing/parse_formulations.py
```

## Input

- **File**: `data/raw/Cryopreservative Data 2026.csv`
- **Format**: CSV with columns:
  1. `All ingredients in cryoprotective solution` - Free-text formulation
  2. `DMSO usage` - DMSO percentage
  3. `Cooling rate` - Freezing protocol
  4. `Viability` - Cell viability post-thaw

## Output

- **File**: `data/processed/parsed_formulations.csv`
- **Format**: Structured CSV with columns:
  - `formulation_id` - Unique identifier
  - `viability_percent` - Extracted viability value
  - `dmso_percent` - DMSO percentage
  - `source_doi` - Literature source
  - `{ingredient}_M` - Molar concentration (for molar-convertible ingredients)
  - `{ingredient}_pct` - Percentage concentration (for percentage-only ingredients)

## Features

### Dual Unit System

Ingredients are categorized by their molecular properties:

| Unit Type | Suffix | Examples |
|-----------|--------|----------|
| Molar | `_M` | DMSO, trehalose, glycerol, sucrose, ethylene glycol |
| Percentage | `_pct` | FBS, HSA, HES, PEG variants, PVP, hyaluronic acid |

### PEG Molecular Weight Handling

PEG is tracked as **individual MW-specific ingredients** to capture the different cryoprotective behaviors:

| Ingredient | MW Range | Behavior |
|------------|----------|----------|
| `peg_400` | 400 Da | Cell penetrating (low MW) |
| `peg_600` | 600 Da | Cell penetrating (low MW) |
| `peg_1k` | 1,000 Da | Cell penetrating (low MW) |
| `peg_1500` | 1,500 Da | Intermediate |
| `peg_3350` | 3,350 Da | Intermediate (default for generic "PEG") |
| `peg_5k` | 5,000 Da | Intermediate |
| `peg_10k` | 10,000 Da | Intermediate |
| `peg_20k` | 20,000 Da | Non-penetrating (high MW) |
| `peg_35k` | 35,000 Da | Non-penetrating (high MW) |

Generic "PEG" mentions without MW specification default to `peg_3350` (most common lab grade).

### Ingredient Synonym Mapping

The parser merges equivalent ingredient names:

| Canonical Name | Synonyms |
|----------------|----------|
| `dmso` | DMSO, Me2SO, dimethyl sulfoxide |
| `ethylene_glycol` | EG, ethylene glycol |
| `propylene_glycol` | 1,2-propanediol, PROH, PG |
| `fbs` | FBS, FCS, fetal bovine serum |
| `hsa` | HSA, human albumin, BSA, albumin |
| `hes` | HES, hydroxyethyl starch, HES450 |
| `hyaluronic_acid` | HMW-HA, hyaluronic acid |

### Unit Conversion

For molar-convertible ingredients:
- `%` → M (using molecular weight and density)
- `mM` → M (÷ 1000)
- `mg/mL` → M (using molecular weight)

For percentage-only ingredients:
- Values are kept as-is (percentage)

### Culture Media Exclusion

The following are excluded as separate variables:
- DMEM, α-MEM, PBS, HBSS, saline
- Culture media supplements

### Duplicate Detection

The script identifies identical formulations with different viabilities and averages their values automatically.

## Example Output

```csv
formulation_id,viability_percent,dmso_percent,source_doi,dmso_M,trehalose_M,fbs_pct,peg_3350_pct,peg_400_pct,...
1,82.5,10.0,10.1038/srep09596,1.409,0.0,0.0,0.0,0.0,...
2,55.0,0.0,10.1186/s40824-023-00356-z,0.0,0.0,0.0,0.0,10.0,...
```

## Dataset Statistics

- **Total formulations**: 198 unique (after duplicate averaging)
- **Unique ingredients**: 34
- **Molar ingredients**: 14 (dmso, trehalose, glycerol, sucrose, etc.)
- **Percentage ingredients**: 20 (fbs, hsa, hes, peg_400, peg_3350, peg_20k, pvp, etc.)


## v2 concentration correction

Unit matching distinguishes `M` from `mM`; ingredient matching is
case-insensitive. The active CSVs have matching `_v2.csv` snapshots. This release
corrects 42 verified concentration fields in 35 formulations, including raw
record 102's `o.5M` typo, and retains all 198 records and their viability targets.
Model retraining is outside the scope of this correction.

The default script reads `data/raw/concentration_review_v2.json` after parsing.
It warns about five unresolved concentration fields whose legacy values are
unverified, preserves the baseline target groups for two matching composition
pairs, and rejects a review whose input hash does not match the raw CSV.
Changes to the input or unresolved entries require a corresponding source-review
update. The pure `parse_csv()` function returns concentrations from the raw text.

See [the v2 audit](../../docs/concentration_corrections_v2.md) for the error
mechanism, commit provenance, unresolved fields and regeneration checks.

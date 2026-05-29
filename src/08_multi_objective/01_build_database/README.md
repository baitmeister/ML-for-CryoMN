# Stage 01: Build V2 Database

## Purpose

Build the v2 multi-objective database from legacy project evidence.

This stage transfers the old literature and wet-lab validation results as
viability-only observations. It does not infer mechanical labels from legacy
data.

## Command

```bash
python3 src/08_multi_objective/01_build_database/build_database.py
```

Optional inputs:

```bash
python3 src/08_multi_objective/01_build_database/build_database.py \
  --literature data/processed/parsed_formulations.csv \
  --validation data/validation/validation_results.csv \
  --output-dir data/processed_v2
```

## Inputs

- `data/processed/parsed_formulations.csv`
- `data/validation/validation_results.csv`
- `config_v2/ingredients.yaml`
- `config_v2/optimization.yaml`

## Outputs

- `data/processed_v2/formulations.csv`
- `data/processed_v2/observations.csv`

These two files are the persistent v2 model-state database.

## Transfer Rules

- Literature rows become `viability_percent` observations only.
- Legacy CryoMN wet-lab rows become `viability_percent` observations only.
- Literature viability noise defaults to `50.0`.
- Legacy wet-lab viability noise defaults to `5.0`.
- `cooh_pll`, `ficoll`, and `hes` are not active v2 features.
- Culture media and basal buffers are excluded as variables.

## When To Run

Run this once at the start of the v2 project, and rerun it only when you change
legacy-transfer configuration or want to rebuild `data/processed_v2` from
scratch.

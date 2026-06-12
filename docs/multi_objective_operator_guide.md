# Multi-Objective Operator Guide

This is the practical runbook for the multi-objective project in
`src/08_multi_objective`.

The workflow has:
- a one-time setup step
- a repeated wet-lab loop

The main persistent database files are:
- [data/processed_v2/formulations.csv](/Users/bait/Documents/ML-for-CryoMN/data/processed_v2/formulations.csv)
- [data/processed_v2/observations.csv](/Users/bait/Documents/ML-for-CryoMN/data/processed_v2/observations.csv)

The main working-round files are:
- [results/multi_objective_v2/current_round_status.json](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/current_round_status.json)
- [results/multi_objective_v2/next_round/next_round_candidates.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_candidates.csv)
- [results/multi_objective_v2/next_round/next_round_summary.txt](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_summary.txt)
- [results/multi_objective_v2/total_candidate_pool.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/total_candidate_pool.csv)

## Core Idea

- `formulations.csv` stores the canonical formulation definitions.
- `observations.csv` stores the experimental evidence.
- Stage 02 reads those files and proposes the next round.
- Stage 03 is the preferred day-to-day command because it:
  - captures one review snapshot of the exact state that produced the current slate
  - ingests new wet-lab results
  - generates the next slate

## Files To Know

### Config

- [config_v2/ingredients.yaml](/Users/bait/Documents/ML-for-CryoMN/config_v2/ingredients.yaml)
- [config_v2/endpoints.yaml](/Users/bait/Documents/ML-for-CryoMN/config_v2/endpoints.yaml)
- [config_v2/optimization.yaml](/Users/bait/Documents/ML-for-CryoMN/config_v2/optimization.yaml)
- [config_v2/availability.yaml](/Users/bait/Documents/ML-for-CryoMN/config_v2/availability.yaml)

### Persistent database

- [data/processed_v2/formulations.csv](/Users/bait/Documents/ML-for-CryoMN/data/processed_v2/formulations.csv)
- [data/processed_v2/observations.csv](/Users/bait/Documents/ML-for-CryoMN/data/processed_v2/observations.csv)

### Current round outputs

- [results/multi_objective_v2/current_round_status.json](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/current_round_status.json)
- [results/multi_objective_v2/total_candidate_pool.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/total_candidate_pool.csv)
- [results/multi_objective_v2/next_round/next_round_candidates.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_candidates.csv)
- [results/multi_objective_v2/next_round/next_round_summary.txt](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_summary.txt)

### Round review outputs

- [results/multi_objective_v2/round_review](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/round_review)
  This directory now contains one archived subfolder per round, such as `ROUND_001`.
  Inside each round folder, archived files are named with the same `ROUND_###_...` prefix.

## Programs And What They Do

### `01_build_database/build_database.py`

What it does:
- Builds the v2 database from old single-objective literature and old
  validation data.
- Transfers viability only.
- Does not create mechanical labels from legacy data.

Inputs:
- [data/processed/parsed_formulations.csv](/Users/bait/Documents/ML-for-CryoMN/data/processed/parsed_formulations.csv)
- [data/validation/validation_results.csv](/Users/bait/Documents/ML-for-CryoMN/data/validation/validation_results.csv)
- [config_v2/ingredients.yaml](/Users/bait/Documents/ML-for-CryoMN/config_v2/ingredients.yaml)
- [config_v2/optimization.yaml](/Users/bait/Documents/ML-for-CryoMN/config_v2/optimization.yaml)

Outputs:
- [data/processed_v2/formulations.csv](/Users/bait/Documents/ML-for-CryoMN/data/processed_v2/formulations.csv)
- [data/processed_v2/observations.csv](/Users/bait/Documents/ML-for-CryoMN/data/processed_v2/observations.csv)

When to run:
- Run once at the start of the v2 project.
- Rerun only if you want to rebuild the v2 database from scratch.

Command:
```bash
src/08_multi_objective/01_build_database/build_database.py
```

### `02_select_candidates/select_candidates.py`

What it does:
- Reads the persistent v2 database.
- Applies ingredient availability rules.
- Automatically resolves the active phase:
  - `screening_only`
  - `mechanics_enabled`
- Scores the full candidate pool.
- Builds the 12-row wet-lab slate.
- Marks mechanical follow-up rows only in `mechanics_enabled`.
- Can add `retest_priority` rows when existing formulations look unstable or
  off-trend.

Inputs:
- [data/processed_v2/formulations.csv](/Users/bait/Documents/ML-for-CryoMN/data/processed_v2/formulations.csv)
- [data/processed_v2/observations.csv](/Users/bait/Documents/ML-for-CryoMN/data/processed_v2/observations.csv)
- [config_v2/ingredients.yaml](/Users/bait/Documents/ML-for-CryoMN/config_v2/ingredients.yaml)
- [config_v2/optimization.yaml](/Users/bait/Documents/ML-for-CryoMN/config_v2/optimization.yaml)
- [config_v2/availability.yaml](/Users/bait/Documents/ML-for-CryoMN/config_v2/availability.yaml)

Outputs:
- [results/multi_objective_v2/current_round_status.json](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/current_round_status.json)
- [results/multi_objective_v2/total_candidate_pool.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/total_candidate_pool.csv)
- [results/multi_objective_v2/next_round/next_round_candidates.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_candidates.csv)
- [results/multi_objective_v2/next_round/next_round_summary.txt](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_summary.txt)

What the user needs to check before running:
- [config_v2/availability.yaml](/Users/bait/Documents/ML-for-CryoMN/config_v2/availability.yaml)
- [config_v2/optimization.yaml](/Users/bait/Documents/ML-for-CryoMN/config_v2/optimization.yaml)

What the user needs to check after running:
- [results/multi_objective_v2/current_round_status.json](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/current_round_status.json)
- [results/multi_objective_v2/next_round/next_round_summary.txt](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_summary.txt)
- [results/multi_objective_v2/next_round/next_round_candidates.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_candidates.csv)
- [results/multi_objective_v2/total_candidate_pool.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/total_candidate_pool.csv)

Command:
```bash
src/08_multi_objective/02_select_candidates/select_candidates.py
```

Optional phase override for debugging only:
```bash
src/08_multi_objective/02_select_candidates/select_candidates.py --phase-mode screening_only
```

### `helper/instron.py`

What it does:
- Parses one Bluehill/Instron CSV.
- Extracts:
  - `critical_axial_load_N_per_needle`
  - `critical_axial_load_N_total`
  - `initial_stiffness_N_per_mm_per_needle`
- Writes those values into the correct row of the current
  `next_round_candidates.csv`.

Input file you provide:
- one raw Instron CSV, ideally stored under a batch folder such as:
  - `data/raw/instron/ROUND_001/example.csv`

Output it updates:
- [results/multi_objective_v2/next_round/next_round_candidates.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_candidates.csv)

When to use:
- Use this if you want the program to fill the mechanical columns for you.
- Skip it if you plan to enter the mechanical values manually.

Command example:
```bash
src/08_multi_objective/helper/instron.py \
  data/raw/instron/ROUND_001/example.csv \
  --formulation-id v2_example \
  --batch-id ROUND_001 \
  --replicate-id rep_001 \
  --needles-compressed 100
```

### `03_run_round/run_round.py`

What it does:
- Advances one real experimental round in the intended order:
  1. generate one review snapshot of the state that produced the current slate
  2. ingest results
  3. generate the next slate
- The review logic and ingest both happen directly inside this program.
- New formulations are added to
  [data/processed_v2/formulations.csv](/Users/bait/Documents/ML-for-CryoMN/data/processed_v2/formulations.csv)
- New viability, intact-pass, and mechanical observations are added to
  [data/processed_v2/observations.csv](/Users/bait/Documents/ML-for-CryoMN/data/processed_v2/observations.csv)
- Separate validation batches remain separate model-visible observations so the
  model can learn batch-to-batch variance.

This is the preferred day-to-day command.

Input:
- the currently filled
  [results/multi_objective_v2/next_round/next_round_candidates.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_candidates.csv)

Outputs:
- updated persistent database files
- updated round review outputs under:
  - [results/multi_objective_v2/round_review/ROUND_001](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/round_review/ROUND_001)
  including:
  - `ROUND_001_next_round_candidates.csv`
  - `ROUND_001_next_round_summary.txt`
  - `ROUND_001_total_candidate_pool.csv`
  - `ROUND_001_model_evaluation_table.csv`
- a newly generated next round slate

Command:
```bash
src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv
```

Useful option:
```bash
src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --skip-generate
```

Use `--skip-generate` if you want to ingest and inspect the updated state
before allowing the next slate to be created.

## What The User Must Do, In Order

## One-Time Project Setup

### Step 1. Build the v2 database

Run:
```bash
src/08_multi_objective/01_build_database/build_database.py
```

Then check:
- [data/processed_v2/formulations.csv](/Users/bait/Documents/ML-for-CryoMN/data/processed_v2/formulations.csv)
- [data/processed_v2/observations.csv](/Users/bait/Documents/ML-for-CryoMN/data/processed_v2/observations.csv)

### Step 2. Generate the first slate

Run:
```bash
src/08_multi_objective/02_select_candidates/select_candidates.py
```

Then check:
- [results/multi_objective_v2/next_round/next_round_summary.txt](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_summary.txt)
- [results/multi_objective_v2/next_round/next_round_candidates.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_candidates.csv)

## Repeated Wet-Lab Round Workflow

### ROUND_002+ feasibility policy

Policy `round2_candidate_feasibility_v1` starts when Stage 02 proposes
`ROUND_002`. It does not change transferred data or the executed `ROUND_001`
slate.

For ROUND_002 and later, review these columns in
`total_candidate_pool.csv`:

- `feasibility_pass` and `feasibility_reasons`
- polymer, serum/protein, sugar, and osmolyte totals
- `estimated_small_solute_g_L`
- `nearest_support_distance` and `support_status`
- `candidate_origin`
- policy version and activation round
- `optimizer_mode` and `optimizer_fallback_status`

Rejected generation attempts remain in the audit pool but cannot enter the
wet-lab sheet. A final slate may contain at most one `boundary_probe`.
`next_round_summary.txt` and `next_round_metadata.json` identify whether
continuous qLogNEHVI or the constrained finite-pool fallback was used.
In the JSON, read `optimizer_mode`, `optimizer_fallback_status`, and
`continuous_qlognehvi.continuous_optimizer_reason` for the detailed status.

The complete rationale is in
[round2_candidate_failure_prevention.md](round2_candidate_failure_prevention.md).

### Step 1. Review the current proposed slate

Check:
- [results/multi_objective_v2/next_round/next_round_summary.txt](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_summary.txt)
- [results/multi_objective_v2/next_round/next_round_candidates.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_candidates.csv)

Look for:
- the active phase
- why that phase was chosen
- which 12 formulations to make
- which rows are mechanical follow-up, if the phase is `mechanics_enabled`
- which rows are `retest_priority`

### Step 2. Perform the wet-lab work

Make the listed formulations and measure:
- viability
- intact microneedle formation
- mechanical properties only after the selector has entered
  `mechanics_enabled`, or if you explicitly choose to test them anyway

### Step 3. Fill the candidate results sheet

Edit:
- [results/multi_objective_v2/next_round/next_round_candidates.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_candidates.csv)

Fill these fields as appropriate:
- `viability_percent`
- `intact_patch_formation_pass`
- optional:
  - `no_slurry`
  - `no_collapse`
  - `intact_tip_count`
  - `total_tip_count`
- if mechanical data is available:
  - `instron_file`
  - `needles_compressed`
  - or `critical_axial_load_N_per_needle`
  - or `critical_axial_load_N_total`
  - optionally `initial_stiffness_N_per_mm_per_needle`
- optionally:
  - `replicate_id`
  - `notes`

Starting with ROUND_002 sheets, these manually entered preparation fields are
also optional:

- `preparation_feasibility_pass`
- `homogeneous_solution_pass`
- `fillability_pass`
- `preparation_failure_reason`

Allowed reasons are:

- `insoluble_or_precipitated`
- `phase_separated`
- `excessive_viscosity`
- `incomplete_polymer_hydration`
- `other_preparation_failure`

Leave unknown fields blank. The importer does not infer preparation outcomes
from intact-patch, slurry, collapse, or notes. Explicit preparation failure may
be submitted without viability, but mechanical values must not be supplied for
that replicate.

Important:
- Do not enter wet-lab results into
  [results/multi_objective_v2/total_candidate_pool.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/total_candidate_pool.csv)

### Step 4. If needed, import Instron values into the sheet

Optional helper:
```bash
src/08_multi_objective/helper/instron.py \
  data/raw/instron/ROUND_001/example.csv \
  --formulation-id v2_example \
  --batch-id ROUND_001 \
  --replicate-id rep_001 \
  --needles-compressed 100
```

Then recheck:
- [results/multi_objective_v2/next_round/next_round_candidates.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_candidates.csv)

### Step 5. Advance the round

Preferred command:
```bash
src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv
```

This will:
- capture one review snapshot of the state that produced the current slate
- ingest the results
- update the persistent database
- generate the next slate

If you want to ingest and review but stop before creating the next slate, use:

```bash
src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --skip-generate
```

### Step 6. Review the updated state

Check:
- [data/processed_v2/observations.csv](/Users/bait/Documents/ML-for-CryoMN/data/processed_v2/observations.csv)
- [results/multi_objective_v2/round_review/ROUND_001/ROUND_001_visualization_summary.txt](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/round_review/ROUND_001/ROUND_001_visualization_summary.txt)
- [results/multi_objective_v2/round_review/ROUND_001/ROUND_001_best_performers_summary.txt](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/round_review/ROUND_001/ROUND_001_best_performers_summary.txt)
- `results/multi_objective_v2/round_review/ROUND_###/ROUND_###_next_round_candidates.csv`
- `results/multi_objective_v2/round_review/ROUND_###/ROUND_###_next_round_summary.txt`
- `results/multi_objective_v2/round_review/ROUND_###/ROUND_###_total_candidate_pool.csv`
- `results/multi_objective_v2/round_review/ROUND_###/ROUND_###_model_evaluation_table.csv`
- [results/multi_objective_v2/next_round/next_round_summary.txt](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_summary.txt)

Look for:
- whether the review matches the state that produced the just-tested slate
- whether the new batch was ingested correctly
- whether the project is still in `screening_only` or has entered
  `mechanics_enabled`
- whether any formulation has been flagged for `retest_priority`
- whether the next round makes scientific sense before you proceed

## Shortest Recommended Workflow

### First time only

```bash
src/08_multi_objective/01_build_database/build_database.py
src/08_multi_objective/02_select_candidates/select_candidates.py
```

### Every wet-lab cycle after that

1. Fill:
   - [results/multi_objective_v2/next_round/next_round_candidates.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_candidates.csv)
2. Run:
```bash
src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv
```

That is the best default operating procedure for the current implementation.

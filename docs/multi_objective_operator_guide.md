# Multi-Objective Operator Guide

This is the practical runbook for `src/08_multi_objective`.

## Operating Model

- `data/processed_v2/formulations.csv` stores canonical formulations.
- `data/processed_v2/observations.csv` stores experimental evidence.
- Stage 02 selects the next 12-row slate and freezes its pre-bench proposal.
- The operator edits only the active worksheet under `next_round/`.
- Stage 03 validates, ingests, archives the completed worksheet, creates the
  completed-round reports, and then selects the next slate.

This artifact reorganization does not alter candidate allocation, the CSV
schema, the early viability-plus-intact input, the mechanical phase transition,
or any Round 1/2 observations.

## Files To Know

### Persistent database

```text
data/processed_v2/formulations.csv
data/processed_v2/observations.csv
```

### Active operator workspace

```text
results/multi_objective_v2/next_round/next_round_candidates.csv
results/multi_objective_v2/next_round/next_round_summary.txt
results/multi_objective_v2/next_round/next_round_metadata.json
```

Only `next_round_candidates.csv` is edited. The summary and metadata are
read-only references.

### Round archive

```text
results/multi_objective_v2/rounds/ROUND_###/
├── proposal/
│   ├── proposal.csv
│   ├── summary.txt
│   ├── selection_metadata.json
│   └── plots/next_round_candidate_screen.png
├── completed/
│   └── completed.csv
└── reports/
    ├── report_summary.txt
    ├── best_performers_summary.txt
    ├── tables/
    └── plots/
```

- `proposal/` records the exact model output before bench work.
- `completed/completed.csv` records the exact successfully ingested worksheet.
- `reports/` records the post-ingest state for that round.
- Some report tables or plots are absent until enough endpoint data exist.

Never edit archived proposal or completed files.

### Campaign-level artifacts

```text
results/multi_objective_v2/reports/
```

This top-level directory is only for cumulative campaign reports generated on
demand. It is separate from each completed round's reports.

### Latest-only artifacts

```text
results/multi_objective_v2/total_candidate_pool.csv
results/multi_objective_v2/current_round_status.json
```

The pool is overwritten on each selection run and is for debugging and
auditing, not result entry.

## Stage 01: Build Database

Run once at the start:

```bash
python3 src/08_multi_objective/01_build_database/build_database.py
```

This transfers the legacy viability evidence into the v2 database. It does not
create mechanical labels from legacy data.

## Stage 02: Select Candidates

```bash
python3 src/08_multi_objective/02_select_candidates/select_candidates.py
```

Stage 02:

- reads the persistent database and configuration
- applies the current ingredient availability rules
- resolves `screening_only` or `mechanics_enabled` automatically
- retains the existing support-aware pool generation and 12-row allocation
- writes the active `next_round/` files
- freezes exact proposal copies under `rounds/ROUND_###/proposal/`
- generates the proposal candidate-screen plot
- overwrites the latest full candidate pool

An identical selector rerun is idempotent. Stage 02 refuses to replace a
different frozen proposal for the same round.

Review after selection:

```text
results/multi_objective_v2/current_round_status.json
results/multi_objective_v2/next_round/next_round_summary.txt
results/multi_objective_v2/next_round/next_round_candidates.csv
results/multi_objective_v2/rounds/ROUND_###/proposal/
```

## Stage 03: Complete and Advance a Round

### 1. Review the active slate

Use:

```text
results/multi_objective_v2/next_round/next_round_summary.txt
results/multi_objective_v2/next_round/next_round_candidates.csv
```

Review the active phase, the 12 formulations, any `retest_priority` rows and,
once mechanics is enabled, mechanical-test recommendations.

### 2. Perform the wet-lab work

During the current screening phase, measure:

- viability
- intact microneedle formation

Mechanical measurements remain optional until the existing automatic
paired-data threshold enables `mechanics_enabled`.

### 3. Fill the active worksheet

Edit only:

```text
results/multi_objective_v2/next_round/next_round_candidates.csv
```

Fill as applicable:

- `viability_percent`
- `intact_patch_formation_pass`
- optional intact detail: `no_slurry`, `no_collapse`, `intact_tip_count`,
  `total_tip_count`
- optional preparation fields already present in the sheet
- optional mechanical fields already present in the sheet
- `replicate_id`
- `notes`

Leave unmeasured fields blank. Duplicate a proposal row when technical
replicates are needed and give the rows distinct `replicate_id` values. Row
reordering is allowed.

Do not change:

- `batch_id`, `candidate_id` or `formulation_id`
- ingredient concentrations
- predictions or uncertainties
- recommendation, ranking or selection diagnostic fields
- the CSV columns

Do not enter results in `total_candidate_pool.csv` or anywhere under
`rounds/`.

### 4. Optionally import Instron data

Store raw files under a batch folder such as:

```text
data/raw/instron/ROUND_###/
```

Then use the existing helper:

```bash
python3 src/08_multi_objective/helper/instron.py \
  data/raw/instron/ROUND_###/example.csv \
  --formulation-id v2_example \
  --batch-id ROUND_### \
  --replicate-id rep_001 \
  --needles-compressed 100
```

The helper updates only the active `next_round_candidates.csv`.

### 5. Validate, ingest, report and roll over

```bash
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv
```

The order is deliberate:

1. validate the worksheet against the frozen proposal
2. ingest observations and formulations
3. archive the exact source bytes as `completed/completed.csv`
4. generate post-ingest reports in the completed round folder
5. generate and freeze the next proposal
6. replace the active workspace with the next editable slate

The next slate is generated only after reporting succeeds. If reporting fails,
the database update and completed worksheet remain preserved and the command
can be rerun safely.

To stop before next-slate generation:

```bash
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --skip-generate
```

### 6. Review the completed round and next proposal

For the round just ingested:

```text
results/multi_objective_v2/rounds/ROUND_###/completed/completed.csv
results/multi_objective_v2/rounds/ROUND_###/reports/report_summary.txt
results/multi_objective_v2/rounds/ROUND_###/reports/best_performers_summary.txt
results/multi_objective_v2/rounds/ROUND_###/reports/tables/
results/multi_objective_v2/rounds/ROUND_###/reports/plots/
```

For the new round:

```text
results/multi_objective_v2/next_round/
results/multi_objective_v2/rounds/ROUND_###/proposal/
```

## Round 2 Instructions

Round 2 was selected before this storage change. Its original active slate was
copied byte-for-byte into:

```text
results/multi_objective_v2/rounds/ROUND_002/proposal/proposal.csv
```

Enter Round 2 results in:

```text
results/multi_objective_v2/next_round/next_round_candidates.csv
```

Do not edit the archived proposal. After filling the working sheet, run:

```bash
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv
```

Successful ingestion creates `ROUND_002/completed/completed.csv`, creates the
Round 2 reports, freezes the Round 3 proposal, and leaves the editable Round 3
slate in `next_round/`.

## Historical Round 1

Round 1 remains valid historical evidence:

```text
results/multi_objective_v2/rounds/ROUND_001/completed/completed.csv
results/multi_objective_v2/rounds/ROUND_001/proposal/proposal_reconstructed.csv
results/multi_objective_v2/rounds/ROUND_001/reports/legacy_pre_ingest/
results/multi_objective_v2/rounds/ROUND_001/legacy/migration_manifest.json
```

The Round 1 proposal is explicitly labelled reconstructed because automatic
pre-bench freezing did not exist at that time. No Round 1 observations were
regenerated or invalidated.

# Multi-Objective Operator Guide

This runbook covers `src/08_multi_objective`.

## Operating Model

- `data/processed_v2/formulations.csv` stores canonical formulations.
- `data/processed_v2/observations.csv` stores endpoint evidence.
- Stage 02 selects a 12-row slate and freezes its pre-bench proposal.
- The operator edits only the worksheet under `next_round/`.
- Stage 03 validates, ingests, archives, reports and invokes Stage 02 for the
  following slate.

## Policy Schedule

| Round | Feasibility policy |
|---|---|
| `ROUND_001` | Stored proposal behavior |
| `ROUND_002`–`ROUND_004` | `round2_candidate_feasibility_v1` |
| `ROUND_005` and higher | `round5_solubility_viscosity_v2` |

Temporary availability restrictions affect sampling only. All registered
active and temporarily unavailable ingredients remain subject to individual,
aggregate, saturation-burden and preparation checks.

## Files

Persistent database:

```text
data/processed_v2/formulations.csv
data/processed_v2/observations.csv
```

Operator workspace:

```text
results/multi_objective_v2/next_round/next_round_candidates.csv
results/multi_objective_v2/next_round/next_round_summary.txt
results/multi_objective_v2/next_round/next_round_metadata.json
```

Only `next_round_candidates.csv` is editable. The summary and metadata are
read-only references.

Round archive:

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

Do not edit archived proposal or completed files.

Campaign reports:

```text
results/multi_objective_v2/reports/
```

Mutable audit artifacts:

```text
results/multi_objective_v2/total_candidate_pool.csv
results/multi_objective_v2/current_round_status.json
```

The candidate pool is overwritten by selection and is not a result-entry file.

## Stage 01: Build Database

```bash
python3 src/08_multi_objective/01_build_database/build_database.py
```

Stage 01 transfers viability evidence into the v2 database without creating
mechanical labels.

## Stage 02: Select Candidates

```bash
python3 src/08_multi_objective/02_select_candidates/select_candidates.py
```

Stage 02:

- reads the database and configuration
- applies ingredient availability to sampling
- applies the round-resolved feasibility policy to every candidate path
- resolves `screening_only`, `mechanics_bootstrap`, `mechanics_hybrid`, or
  `mechanics_enabled` from completed-screening and paired-mechanics evidence
- enforces support, similarity and slate-diversity controls
- writes the `next_round/` files
- freezes proposal copies under `rounds/ROUND_###/proposal/`
- writes the full candidate audit pool

An identical selector rerun is idempotent. Stage 02 refuses to replace a
different frozen proposal for the same round.

## Stage 03: Enter Results and Advance

### Result entry

Edit:

```text
results/multi_objective_v2/next_round/next_round_candidates.csv
```

Fill as applicable:

- `viability_percent`
- `intact_patch_formation_pass`
- `no_slurry`, `no_collapse`, `intact_tip_count`, `total_tip_count`
- optional preparation fields
- mechanical or Instron fields
- `replicate_id`
- `notes`

Leave unmeasured fields blank. Duplicate a proposal row for technical
replicates and use distinct `replicate_id` values. Row reordering is allowed.

Enter mechanical results only for rows with a numeric
`mechanical_selection_rank` that actually pass intact. Follow numeric rank
order until four passing rows have been tested. Leave unused capacity blank
when fewer than four ranked rows pass; do not substitute an unranked row.
Remeasure viability, intact formation, and mechanics for a screened-hit follow-up
or reference under the worksheet batch ID. Earlier frozen proposals may instead
use the one-time `mechanics_anchor` role.

Do not change:

- `batch_id`, `candidate_id` or `formulation_id`
- ingredient concentrations
- predictions or uncertainties
- recommendations, ranks or selection diagnostics
- CSV columns

### Instron input by frozen endpoint policy

For Group 10, enter each specimen's raw path in `instron_file` and its actual
`needles_compressed` when known. Leave the compatibility force and old stiffness
input columns blank. The normal round update extracts force at +0.8 mm after the
sustained 1 N trigger using the proposal's frozen settings. Missing loaded count
permits total-force reporting only; short traces supply no force training label.

The helper command below computes **legacy maximum-force metrics**. Use it only
for earlier frozen contracts such as Group 9, not for Group 10 terminal-force
worksheets. For an optional standalone Group 10 plot/JSON review, use the
[terminal-force analysis CLI](group10/mechanical_sop.md#round-csv-and-automatic-extraction).

Store raw files under a batch directory such as:

```text
data/raw/instron/ROUND_###/
```

Use:

```bash
python3 src/08_multi_objective/helper/instron.py \
  data/raw/instron/ROUND_###/example.csv \
  --formulation-id v2_example \
  --batch-id ROUND_### \
  --replicate-id rep_001 \
  --needles-compressed 100
```

The helper updates only `next_round_candidates.csv`.

### Ingest, report and select

```bash
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv
```

The command order is:

1. validate the worksheet against the frozen proposal
2. ingest observations and formulations
3. archive the source bytes as `completed/completed.csv`
4. generate replicate-aggregated descriptive and formulation-grouped
   cross-validation reports
5. evaluate frozen proposal-time predictions
6. refresh the cumulative prospective report
7. generate and freeze the following proposal
8. replace the operator workspace with the editable slate

Proposal generation occurs only if reporting succeeds. A reporting failure
retains the database update and completed worksheet so the command can be
rerun safely.

To omit proposal generation:

```bash
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --skip-generate
```

## Reports

`model_evaluation_*` artifacts are formulation-grouped cross-validation
diagnostics. All batches of one formulation stay in the same fold.

`prospective_*` artifacts compare frozen proposal-time predictions with
measurements without retraining. Round summaries report eligible, passed,
failed and missing measurements explicitly. Interval coverage is paired with
interval width.

To regenerate reports without ingestion or candidate selection:

```bash
python3 src/08_multi_objective/04_report_campaign/report_campaign.py \
  --all-rounds
```

## Archive Rules

- `proposal/proposal.csv` is the immutable pre-bench slate.
- `completed/completed.csv` is the exact ingested worksheet.
- `reports/` contains outputs derived from the database.
- `next_round/next_round_candidates.csv` is the only result-entry file.
- `total_candidate_pool.csv` is an audit pool and must not receive results.

The four-phase gates and ordinary mechanical scoring are defined in the
[canonical mechanics policy](../src/08_multi_objective/README.md#evidence-gated-mechanics-transition).
The Group 10 additions below supersede the earlier one-time anchor reservation.

## Group 10 workflow additions

See the [current Group 10 guide](group10/README.md), [settings register](group10/settings_register.md),
[terminal-force SOP](group10/mechanical_sop.md), and [Group 9 transition instructions](group10/transition_group9_to_10.md).

From Group 10, the endpoint is force at +0.8 mm after the sustained 1 N trigger. The existing mechanical output column is a compatibility name for nominal terminal force per loaded needle, not a fracture-strength claim. Raw whole-patch total force is also stored. Force drops do not select the endpoint.

The exact DMSO/sucrose reference is configured for Group 10 without preset replicate counts; actual counts come from completed CSV rows. Reference observations are monitoring-only and excluded from production GP training. Group 9 remains under its frozen original contract. Configured branch policy is distinct from a live frozen Group 10 proposal.

The [offline model audit](group10/audit/explained_comparison.md) does not promote a model. GP methodology changes remain deferred until a user decision before the first full-mechanics proposal. The mechanical endpoint revision is authorized separately and starts with Group 10.

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

For Group 10 and later, Stage 02 also hard-stops before writing unless the
reviewed Group 10 settings/protocol are complete, activation is fixed at Group
10, and the immediately preceding round has validated observations.

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
under the worksheet batch ID. Group 10 retains its already-planned reference
mechanical test. From Group 11 the reference is viability-only: it has no mechanical
rank or backup position, and all four mechanical positions are experimental candidates.
Earlier frozen proposals may instead use the one-time `mechanics_anchor` role.

Do not change:

- `batch_id`, `candidate_id` or `formulation_id`
- ingredient concentrations
- predictions or uncertainties
- recommendations, ranks or selection diagnostics
- CSV columns

### Instron input by frozen endpoint policy

For Group 9 under its endpoint addendum and for Group 10+, enter each specimen's raw path in `instron_file` and its actual
`needles_compressed` when known. Leave the compatibility force and old stiffness
input columns blank or enter values that reproduce from the raw trace. The normal round update extracts force at +0.8 mm after the
sustained 1 N trigger using the proposal's frozen settings. Missing loaded count
permits total-force reporting only; short traces supply no force training label.

The helper command below computes **legacy maximum-force metrics**. Use it only
to reproduce earlier historical contracts, not for Group 9+ terminal-force
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
retains the database update and completed worksheet. Regenerate reporting with
Stage 04; Stage 03 rejects a second set of colliding observation IDs rather
than silently replacing the first ingestion.

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

Group 9 validation/ingestion is the first use of force at +0.8 mm after the sustained 1 N trigger, bound by an addendum that hashes its frozen proposal artifacts. Group 10 keeps this endpoint under its configured activation. The existing mechanical output column is a compatibility name for nominal terminal force per loaded needle, not a fracture-strength claim. Raw whole-patch total force and apparent secant stiffness are also stored. Force drops do not select the endpoint.

The exact DMSO/sucrose reference is configured for Group 10 without preset replicate counts; actual counts come from completed CSV rows. Control viability is monitoring-only and excluded from viability GP training. From Group 11 the reference receives no mechanical rank; valid mechanical measurements already collected on that recipe remain ordinary endpoint evidence without control-based normalization. Stage 03 hard-stops if the Group 9 addendum hashes/settings fail or, from Group 10 onward, if the frozen effective configuration is missing, incomplete, or different from the reviewed configuration for that round (Group 10 v4; Group 11+ v5).

The [offline model audit](group10/audit/explained_comparison.md) does not promote a model. GP methodology changes remain deferred until a user decision before the first full-mechanics proposal. The mechanical endpoint revision is authorized separately and starts when Group 9 results are validated.


### Group 10 mechanical evidence clarification

The DMSO/sucrose row is a viability control only. Its valid Group 10 mechanical
result and measured same-batch viability/intact companions count as an ordinary
mechanical pair for phase progression, paired acquisition and observed trade-offs.
Control viability remains outside standalone viability-GP training/ordinary viability
metrics. Four valid Group 10 pairs plus four Group 9 pairs yield **8 pairs / 8
formulations / 2 batches**, meeting the unchanged hybrid gate for Group 11.
Group 10's planned tests stay unchanged; Group 11+ allocates no mechanical rank
to the recurring viability reference. Technical replicates still aggregate within
formulation/batch and do not inflate gate counts.

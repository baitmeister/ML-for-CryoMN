# Stage 03: Run Round

## Purpose

Advance one real wet-lab round with a provenance check and round-scoped
artifacts:

1. validate the filled working sheet against the frozen pre-bench proposal
2. ingest the wet-lab results
3. archive the exact filled worksheet bytes
4. generate replicate-aggregated descriptive reports and
   formulation-grouped cross-validation reports from the updated database
5. evaluate the archived proposal-time predictions for the completed round
6. refresh the cumulative prospective report
7. generate and freeze the next proposal

This is the supported entry point for normal round progression.

## Round 2 Result Entry

Enter Round 2 results only in:

```text
results/multi_objective_v2/next_round/next_round_candidates.csv
```

The corresponding frozen source record is:

```text
results/multi_objective_v2/rounds/ROUND_002/proposal/proposal.csv
```

Do not edit the frozen proposal. In the working file, fill the existing
viability and intact-patch fields, optional intact-patch detail fields,
optional mechanical fields when measured, and `replicate_id` or `notes` when
needed. The CSV schema and automatic endpoint phase progression are unchanged.

The migration-frozen Round 2 proposal has one `retest_priority` row with a
carried viability value of `26.53`. Replace it with the new measurement if the
retest was run. If a genuine new result is coincidentally also `26.53`, add a
`replicate_id`. Clear it if the retest was not run. Stage 03 rejects an
unchanged carried value with no replicate ID so it cannot be mistaken for new
evidence.

Then run:

```bash
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv
```

## Options

```bash
# Override the automatic selection phase for auditing/debugging
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --phase-mode mechanics_enabled

# Ingest, archive and report, but do not generate the next round
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --skip-generate
```

`--skip-review` is available for advanced debugging, but the normal workflow
always generates completed-round reports.

## Validation Rules

Before changing the database, Stage 03 requires the working sheet to match the
frozen proposal.

Allowed changes:

- existing wet-lab result fields
- existing optional preparation and mechanical fields
- `replicate_id` and `notes`
- row reordering
- row duplication for technical replicates

Rejected changes:

- unknown or missing proposal candidates
- changed formulation or candidate identity
- changed composition
- changed model predictions or uncertainties
- changed selection ranks, recommendation types or diagnostics
- added or removed CSV columns

## Outputs

After a successful `ROUND_###` ingestion:

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
    ├── prospective_evaluation_summary.txt
    ├── tables/
    │   ├── model_evaluation_table.csv
    │   ├── prospective_evaluation_table.csv
    │   └── prospective_metrics.csv
    └── plots/
        ├── endpoint_observation_counts.png
        ├── model_evaluation_overview.png
        ├── observed_performance_landscape.png
        ├── prospective_prediction_vs_observed.png
        └── prospective_gate_calibration.png
```

Reports that need more observations are omitted until the data support them.
`completed/completed.csv` is an exact byte-for-byte archive of the successfully
ingested working worksheet.

`best_performers_summary.txt` collapses technical replicates to one row per
candidate and selection rank, reporting viability mean, sample SD, replicate
count and one formulation-level intact outcome. Continuous endpoints use the
replicate mean; the intact gate passes only if every measured patch replicate
passes. Its main campaign rankings use feasible
`wetlab_feedback` formulations; literature leaders appear separately as
historical references.

`model_evaluation_*` uses formulation-grouped cross-validation, so every batch
of the same chemistry stays in one fold, and may retrain from the current
database. `prospective_*` uses only archived proposal-time predictions and
never retrains. Round 1 is reconstructed, Round 2 is supplementary, and Round
3+ supplies the formal pooled viability MAE. Coverage is reported with
interval width. The cumulative bundle is written under
`results/multi_objective_v2/reports/prospective/`.

Only after completed-round reporting succeeds does Stage 02 replace the active
`next_round/` files and freeze the following round proposal. If report
generation fails, the database update and completed archive are retained, the
next slate is not generated, and the same command can be rerun safely.

If no new result fields are filled, Stage 03 skips ingestion, completed
archival and completed-round reporting. Unless `--skip-generate` is supplied,
it still attempts to regenerate the current proposal; frozen-proposal conflict
protection remains active.

The command also refreshes
`results/multi_objective_v2/current_round_status.json`.

Read-only backfill or refresh commands are documented under
[`04_report_campaign`](../04_report_campaign/README.md).

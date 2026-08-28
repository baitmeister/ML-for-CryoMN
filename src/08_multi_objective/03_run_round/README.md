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

## Command

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

# Forward a guarded replacement request for a frozen but completely unstarted next proposal
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --supersede-unstarted-proposal
```

`--skip-review` is available for advanced debugging, but the normal workflow
always generates completed-round reports.

## Validation Rules

Before changing the database, Stage 03 requires the working sheet to match the
frozen proposal.

Allowed changes:

- wet-lab result fields
- optional preparation and mechanical fields
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
- mechanical data for any row without a measured intact pass
- more than four mechanical tests in a mechanics-enabled proposal
- mechanics results that skip an earlier actual-intact priority row in favor
  of a later backup

## Actual-Intact Mechanical Workflow

During `screening_only`, including Round 6, the program requests zero Instron
tests. `mechanical_test_recommended` is false and mechanical priority/backup
fields are blank. External mechanical observations can still bootstrap the
campaign, but they require an explicitly measured intact pass and remain
subject to the four-test capacity.

During `mechanics_enabled`, all 12 rows have a
`mechanical_selection_rank`. Ranks 1-4 are `primary`; ranks 5-12 are
`ordered_backup`.

1. Fabricate all 12 patches and fill the intact outcome.
2. Read rows in ascending `mechanical_selection_rank`.
3. Run Instron on the first four rows that actually passed intact.
4. Skip each failed primary and promote the lowest-ranked intact backup.
5. If fewer than four rows pass, test all intact rows and leave the remainder
   blank; the completion manifest records the shortfall.

Never enter an Instron file, critical load, or stiffness for a failed patch or
for a row whose intact gate is blank. The ingestion layer enforces this even if
the proposal is in compatibility mode. In active empirical mode, it also
enforces the ordered promotion sequence; it does not use the prospective
classifier probability to decide which fabricated patches are testable.

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
│   ├── completed.csv
│   └── mechanical_execution_manifest.json
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

Reports that need more observations are omitted when the data do not support
them.
`completed/completed.csv` is an exact byte-for-byte archive of the successfully
ingested working worksheet. Campaign observation provenance points to this
immutable round-scoped archive rather than the mutable active worksheet.

`mechanical_execution_manifest.json` schema v2 records whether mechanics was
active, configured capacity, proposal primary and backup counts, all measured
intact passes/failures, mechanically ranked intact passes, expected and recorded
test IDs, backup promotions, actual-pass shortfall, and any gate violations. It
is frozen alongside the completed worksheet.

`best_performers_summary.txt` collapses technical replicates to one row per
candidate and selection rank, reporting viability mean, sample SD, replicate
count and one formulation-level intact outcome. Continuous endpoints use the
replicate mean; the intact gate passes only if every measured patch replicate
passes. Its main campaign rankings use feasible
`wetlab_feedback` formulations; literature leaders appear separately as
literature references.

`model_evaluation_*` uses formulation-grouped cross-validation, so every batch
of the same chemistry stays in one fold, and may retrain from the
database. `prospective_*` uses only archived proposal-time predictions and
never retrains. Round 1 is reconstructed, Round 2 is supplementary, and Round
3+ supplies the formal pooled viability MAE. Coverage is reported with
interval width. The cumulative bundle is written under
`results/multi_objective_v2/reports/prospective/`.

Only after completed-round reporting succeeds does Stage 02 replace the active
`next_round/` files and freeze the following round proposal. If report
generation fails, the database update and completed archive are retained, the
next slate is not generated, and the same command can be rerun safely.

If no result fields are filled, Stage 03 skips ingestion, completed
archival and completed-round reporting. Unless `--skip-generate` is supplied,
it attempts to regenerate the proposal; frozen-proposal conflict
protection remains active.

## Supersession Safeguards

Normal proposal archives are immutable. The explicit
`--supersede-unstarted-proposal` option exists only for a policy migration such
as regenerating a blank Round 6. Stage 02 refuses supersession when any of the
following is true:

- `observations.csv` already contains the target batch;
- `rounds/ROUND_###/completed/completed.csv` exists; or
- the active worksheet contains any entered wet-lab/result value.

For an eligible blank proposal, the original proposal directory—including CSV,
summary, selection metadata, and plots—is moved to a timestamped
`rounds/ROUND_###/superseded-policy/` directory. The active worksheet and current
candidate pool are copied beside it. `supersession_manifest.json` records all
original SHA-256 hashes, replacement reason, UTC timestamp, safety checks, and
new policy versions. The archived proposal is recoverable; Rounds 1-5 are never
rewritten by this option.

The command also refreshes
`results/multi_objective_v2/current_round_status.json`.

Read-only backfill or refresh commands are documented under
[`04_report_campaign`](../04_report_campaign/README.md).

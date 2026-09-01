# Stage 03: Run Round

## Purpose

Stage 03 validates the filled worksheet against the frozen proposal, enforces
actual-intact mechanical execution, ingests same-batch measurements, archives
the exact completed sheet, generates reports, and freezes the following
proposal.

The canonical mechanics policy is documented in
[Evidence-gated mechanics transition](../README.md#evidence-gated-mechanics-transition).

## Command and phase controls

```bash
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv
```

Optional controls:

```bash
# Stop after validation, ingestion, archival, and reporting
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --skip-generate

# Override the phase used for the following proposal for audit/debugging
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --phase-mode mechanics_hybrid
```

`--phase-mode` accepts `auto`, `screening_only`, `mechanics_bootstrap`,
`mechanics_hybrid`, and `mechanics_enabled`. Explicit values are recorded as
manual overrides; `auto` evaluates the configured screening, hybrid, and full
evidence gates.

## Worksheet entry

Edit only:

```text
results/multi_objective_v2/next_round/next_round_candidates.csv
```

Fill the result fields that were measured:

| Column | Operator action |
|---|---|
| `replicate_id` | Give duplicated technical-replicate rows distinct IDs. |
| `viability_percent` | Enter post-thaw viability from `0` to `100`. |
| `intact_patch_formation_pass` | Enter `yes/no`, `pass/fail`, `true/false`, or `1/0`. |
| `no_slurry`, `no_collapse` | Optional formation details used by the intact gate. |
| `intact_tip_count`, `total_tip_count` | Optional quantitative formation details. |
| preparation fields | Record preparation, homogeneity, fillability, and failure reason when assessed. |
| `instron_file` | Enter the raw Bluehill CSV path only for an eligible actual-intact row. |
| `needles_compressed` | Required with an Instron file or total-load entry. |
| critical-load fields | Enter per-needle load, or total load with needle count. |
| initial-stiffness field | Optional diagnostic endpoint. |
| `notes` | Record handling, deviations, or test context. |

Leave unmeasured fields blank. Duplicate a proposal row for technical
replicates; do not replace the original candidate or create another formulation
identity. Row reordering is allowed.

Do not change candidate/formulation/batch IDs, ingredient concentrations,
recommendation type, screening or mechanical rank, transition role,
predictions, uncertainties, repeat status, diagnostics, or CSV columns.

## Same-batch paired measurements

A mechanics training pair requires `viability_percent` and
`critical_axial_load_N_per_needle` for the same `formulation_id` and
`batch_id`. An intact pass for that batch is required before the load can be
accepted. Measurements copied from another batch do not form a same-batch
pair.

Technical replicates may supply multiple worksheet rows. Continuous endpoints
aggregate by mean and intact replicates use the conservative all-pass rule.
The database therefore creates one formulation–batch training row, not one
paired observation per technical replicate.

A `mechanics_anchor` is deliberately measured in another batch. Its viability,
intact, and mechanical fields must all be filled from the repeated experiment;
never copy the source-batch values. Ingestion creates a second
formulation–batch pair and preserves the source pair.

## Actual-intact execution for partially ranked slates

Every proposal contains 12 viability/intact rows, but only rows with a numeric
`mechanical_selection_rank` belong to the mechanical priority list. A slate may
therefore be partially ranked when retests, mechanically observed formulations,
or other screening-only rows are present.

The execution sequence is:

1. Fabricate and record viability/intact results for all 12 rows.
2. Ignore every row whose mechanical rank is blank.
3. Read numeric mechanical ranks in ascending order.
4. Run Instron on the first four ranked rows that actually pass intact.
5. Skip a failed or blank-intact primary and promote the next ranked
   actual-intact backup.
6. If fewer than four ranked rows pass intact, test every ranked passing row and
   leave the unused capacity blank.

`mechanical_test_recommended=true` identifies the proposal-time primaries; it
does not authorize a failed patch. Actual intact outcome and numeric rank define
the executable set. The empirical or classifier pass probability is never a
substitute for the measured intact gate.

## Rejection and shortfall rules

Stage 03 rejects the worksheet before database mutation when it contains:

- unknown or missing proposal candidates;
- changed identity, composition, rank, role, prediction, or diagnostics;
- mechanical data on an unranked row;
- mechanical data on a failed-intact or blank-intact row;
- a later passing backup tested while an earlier passing rank is skipped;
- more than four formulations with mechanical results;
- carried mechanical values on an anchor instead of same-batch remeasurement;
- changed or missing CSV columns.

A lack of four passing ranked patches is not a validation error. It is an
execution shortfall: record the available ranked passing tests, leave all other
mechanical fields blank, and let the completion manifest report the unused
capacity. Do not fill a failed, unranked, or disallowed-repeat formulation to
reach the nominal capacity.

## Optional Instron import

Store raw files under a batch directory:

```text
data/raw/instron/ROUND_###/
```

The helper can parse one file into the active worksheet:

```bash
python3 src/08_multi_objective/helper/instron.py \
  data/raw/instron/ROUND_###/example.csv \
  --formulation-id v2_example \
  --batch-id ROUND_### \
  --replicate-id rep_001 \
  --needles-compressed 100
```

The worksheet is still subject to frozen-proposal and mechanical-execution
validation when Stage 03 runs.

## Validation and progression order

Stage 03 performs these operations in order:

1. compare the working worksheet with the frozen proposal;
2. validate actual-intact mechanics rank promotion and capacity;
3. ingest formulations and endpoint observations;
4. archive the exact worksheet bytes as `completed/completed.csv`;
5. freeze `completed/mechanical_execution_manifest.json`;
6. generate replicate-aggregated descriptive, formulation-grouped
   cross-validation, and frozen-proposal prospective reports;
7. refresh cumulative prospective reports;
8. run Stage 02 for the following proposal unless `--skip-generate` is set.

If reporting fails after ingestion, the database update and completed archive
remain available and proposal generation does not proceed. Re-running the
command is safe because archive compatibility is verified.

## Completion manifest

`mechanical_execution_manifest.json` records:

- resolved phase and proposal selection policy;
- configured capacity and proposal primary/backup counts;
- transition role, rank, prior-mechanics count, and repeat status for each
  mechanically ranked candidate;
- anchor decision, source batch, and selection score from proposal metadata;
- qLogNEHVI availability/errors and optimizer fallback mode;
- measured intact pass/fail counts;
- expected and recorded mechanical test IDs;
- promoted backup IDs and promotion count;
- unranked, failed-intact, and unconfirmed-intact mechanical IDs;
- actual-intact shortfall and validation violations.

The manifest should be interpreted with the frozen proposal. A nonzero
`actual_pass_shortfall` means the batch had fewer than four ranked actual-intact
patches; it does not authorize a replacement outside the numeric rank list.

## Outputs

```text
results/multi_objective_v2/rounds/ROUND_###/
├── proposal/
│   ├── proposal.csv
│   ├── summary.txt
│   ├── selection_metadata.json
│   └── plots/
├── completed/
│   ├── completed.csv
│   └── mechanical_execution_manifest.json
└── reports/
    ├── report_summary.txt
    ├── best_performers_summary.txt
    ├── prospective_evaluation_summary.txt
    ├── tables/
    └── plots/
```

`completed.csv` is the exact ingested worksheet. Proposal-time prospective
reports read frozen predictions and do not retrain. Model-evaluation reports
use formulation-grouped cross-validation so batches of the same chemistry stay
within one fold.

## Archive safeguards

Files under `rounds/ROUND_###/proposal/` and `completed/` are immutable campaign
records. The guarded `--supersede-unstarted-proposal` path applies only to a
proposal with no batch observations, no completion archive, and no entered
result values. It moves the superseded proposal and related working artifacts
to a recoverable directory with SHA-256 hashes; it does not alter completed or
historical artifacts.

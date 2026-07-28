# Multi-Objective Quick SOP

Use this for the shortest correct workflow.

## Round 2: Enter Results Here

Edit only:

```text
results/multi_objective_v2/next_round/next_round_candidates.csv
```

The Round 2 proposal is already frozen at:

```text
results/multi_objective_v2/rounds/ROUND_002/proposal/proposal.csv
```

Do not edit anything under `rounds/ROUND_002/proposal/`.

Fill the existing fields as applicable:

- `viability_percent`
- `intact_patch_formation_pass`
- optional intact-patch detail fields
- optional mechanical/Instron fields, if measured
- `replicate_id` and `notes`, if needed

For the existing Round 2 `retest_priority` row, replace the prefilled `26.53`
with the new result, add `replicate_id` if the new result is coincidentally
identical, or clear it if the retest was not performed.

The early-round input remains viability plus the intact gate. Mechanical fields
remain optional until the automatic paired-data phase transition enables
mechanics.

## Advance a Round

```bash
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv
```

The command:

1. validates the working sheet against the frozen proposal
2. validates that a carried Round 2 retest value is not silently re-ingested
3. updates `formulations.csv` and `observations.csv`
4. archives the exact filled sheet as
   `rounds/ROUND_###/completed/completed.csv`
5. generates descriptive and cross-validated reports
6. generates the completed round's prospective report
7. refreshes `reports/prospective/`
8. generates and freezes the next proposal
9. replaces `next_round/` with the next editable slate

Do not run Stage 01 or Stage 02 during a normal iteration. Stage 03 runs Stage
02 only after all reporting succeeds.

Use `--skip-generate` to stop after ingest, archive and reporting:

```bash
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --skip-generate
```

## Review After Ingestion

For the round just completed, review:

```text
results/multi_objective_v2/rounds/ROUND_###/completed/completed.csv
results/multi_objective_v2/rounds/ROUND_###/reports/report_summary.txt
results/multi_objective_v2/rounds/ROUND_###/reports/best_performers_summary.txt
results/multi_objective_v2/rounds/ROUND_###/reports/tables/
results/multi_objective_v2/rounds/ROUND_###/reports/plots/
results/multi_objective_v2/reports/prospective/
```

For the next round, review:

```text
results/multi_objective_v2/next_round/next_round_summary.txt
results/multi_objective_v2/next_round/next_round_candidates.csv
results/multi_objective_v2/rounds/ROUND_###/proposal/
```

## Rules

- Enter results only in `next_round/next_round_candidates.csv`.
- Never edit archived `proposal/` or `completed/` files.
- Do not enter results in `total_candidate_pool.csv`; it is a latest-only debug
  pool.
- Top-level `reports/` is for cumulative campaign reports.
- `model_evaluation_*` uses formulation-grouped cross-validation;
  `prospective_*` uses frozen proposal-time predictions. The formal
  prospective cohort begins at Round 3, and interval coverage is accompanied
  by interval width.
- Completed-round summaries aggregate technical replicates to 12 candidate
  results with mean, sample SD and replicate count.
- The selector stays at 12 rows and retains its existing origin allocation.
  From Round 3, ordinary and rescue candidates are similarity-gated against
  actual wet-lab history and candidates already accepted into the new pool;
  intentional retests remain exempt. Rejected generated candidates are
  resampled instead of reducing an origin quota.
- No non-retest shared ingredient pair may appear in more than five selected
  rows, even when those rows contain additional ingredients. Rescue rows
  count; retests are exempt.
- Each individual registry ingredient may appear in at most five of all 12
  Round 3+ rows. Retests and rescues count toward this limit but are protected
  from removal; ordinary replacements retain their candidate origin.
- Automatic retests are based on campaign batch disagreement, latest-batch
  replicate SD, or one single-batch neighbour-anomaly confirmation. Model
  uncertainty alone never requests a retest.
- The phase remains automatic: early `screening_only`, later
  `mechanics_enabled`.

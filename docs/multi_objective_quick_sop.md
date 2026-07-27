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
2. updates `formulations.csv` and `observations.csv`
3. archives the exact filled sheet as
   `rounds/ROUND_###/completed/completed.csv`
4. generates post-ingest reports under `rounds/ROUND_###/reports/`
5. generates and freezes the next proposal
6. replaces `next_round/` with the next editable slate

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
- The selector stays at 12 rows and retains its existing allocation policy.
- The phase remains automatic: early `screening_only`, later
  `mechanics_enabled`.

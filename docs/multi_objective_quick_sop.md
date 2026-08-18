# Multi-Objective Quick SOP

## Enter Results

Edit only:

```text
results/multi_objective_v2/next_round/next_round_candidates.csv
```

Fill the applicable fields:

- `viability_percent`
- `intact_patch_formation_pass`
- optional intact-patch detail fields
- optional preparation fields
- optional mechanical or Instron fields
- `replicate_id` and `notes`

Leave unmeasured fields blank. Duplicate a proposal row for technical
replicates and assign a distinct `replicate_id` to each copy.

Do not change candidate identity, formulation concentrations, predictions,
uncertainties, ranks, selection diagnostics or CSV columns. Do not edit files
under `results/multi_objective_v2/rounds/`.

## Advance a Round

```bash
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv
```

The command:

1. validates the worksheet against the frozen proposal
2. ingests formulations and observations
3. archives the filled worksheet as `completed/completed.csv`
4. generates descriptive, cross-validated and prospective reports
5. generates and freezes the following proposal
6. replaces `next_round/` with its editable worksheet

Use `--skip-generate` to stop after ingestion, archival and reporting:

```bash
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --skip-generate
```

Stage 01 is database initialization. Stage 03 invokes Stage 02 as part of the
round workflow, so a separate Stage 02 run is unnecessary for routine result
ingestion.

## Review Outputs

Completed-round artifacts:

```text
results/multi_objective_v2/rounds/ROUND_###/completed/completed.csv
results/multi_objective_v2/rounds/ROUND_###/reports/report_summary.txt
results/multi_objective_v2/rounds/ROUND_###/reports/best_performers_summary.txt
results/multi_objective_v2/rounds/ROUND_###/reports/tables/
results/multi_objective_v2/rounds/ROUND_###/reports/plots/
results/multi_objective_v2/reports/prospective/
```

Editable proposal artifacts:

```text
results/multi_objective_v2/next_round/next_round_summary.txt
results/multi_objective_v2/next_round/next_round_candidates.csv
results/multi_objective_v2/rounds/ROUND_###/proposal/
```

## Policy Rules

- `ROUND_001` retains its stored proposal behavior.
- `ROUND_002` through `ROUND_004` use
  `round2_candidate_feasibility_v1`.
- `ROUND_005` and higher use `round5_solubility_viscosity_v2`.
- Availability restricts sampling but never disables chemistry checks.
- Every non-retest shared ingredient pair may appear in at most five slate
  rows.
- Every registry ingredient may appear in at most five slate rows under the
  `ROUND_003` similarity/diversity policy.
- Automatic retests require campaign disagreement, replicate variability or
  the configured neighbour-anomaly evidence; model uncertainty only breaks
  ties among eligible formulations.
- `screening_only` uses viability selection and the intact-patch gate.
- `mechanics_enabled` adds critical axial load per needle to selection.

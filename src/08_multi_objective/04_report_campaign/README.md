# Stage 04 — Prospective Campaign Reports

This read-only reporting stage evaluates archived proposal-time predictions
against normalized measurements in `observations.csv`. It never retrains a
model and does not update formulations, observations, proposals, or the active
candidate worksheet.

Round 1 is labelled `reconstructed`, Round 2 is
`migration_frozen_supplementary`, and Round 3 onward is the locked formal
prospective cohort. The primary metric is pooled formal Round 3+ viability MAE.
Cross-validated model diagnostics remain separate from these prospective
results.

Continuous technical replicates are averaged within formulation and round.
The intact-patch gate uses a conservative all-pass rule: any measured failed
patch replicate makes the formulation-level gate fail. Round summaries report
formulations observed, passed and failed explicitly; they do not list every
individual patch observation.

Refresh only the cumulative campaign report:

```bash
python3 src/08_multi_objective/04_report_campaign/report_campaign.py
```

Regenerate one completed round and then the cumulative report:

```bash
python3 src/08_multi_objective/04_report_campaign/report_campaign.py \
  --round ROUND_002
```

Backfill all completed rounds and then the cumulative report:

```bash
python3 src/08_multi_objective/04_report_campaign/report_campaign.py \
  --all-rounds
```

Round outputs are written below
`results/multi_objective_v2/rounds/ROUND_N/reports/`. Cumulative prospective
outputs are written below
`results/multi_objective_v2/reports/prospective/`.

During a normal experimental iteration, do not call this script separately.
Stage 03 generates the completed round report and cumulative report before it
creates the next proposal. Use this CLI only to refresh or backfill reports.

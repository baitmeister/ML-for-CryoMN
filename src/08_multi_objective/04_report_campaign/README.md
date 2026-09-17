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

Generate the read-only Round 3-6 counterfactual audit for the Round-6 intact
and cold-start policies:

```bash
python3 src/08_multi_objective/04_report_campaign/counterfactual_policy_audit.py
```

This writes a row-level CSV, structured JSON summary, and human-readable README
under `results/multi_objective_v2/reports/round6_policy_audit/`. It reconstructs
the evidence that would have been available before each target round and never
changes a historical proposal. Identified freed positions have unknown
counterfactual outcomes; the report does not claim their replacements would
have performed better.

Round outputs are written below
`results/multi_objective_v2/rounds/ROUND_N/reports/`. Cumulative prospective
outputs are written below
`results/multi_objective_v2/reports/prospective/`.

During a normal experimental iteration, do not call this script separately.
Stage 03 generates the completed round report and cumulative report before it
creates the next proposal. Use this CLI only to refresh or backfill reports.

## Selected plotting suite

See [plotting documentation](PLOTTING.md) for A/B1/C3/D, the two diagnostic sheets, optional E, transparent PNG exports, and safe rendering-only backfill.


## Dynamic GP-strategy audit (separate from staged prospective reports)

`audit_group10.py` refits alternative strategies at preceding training cutoffs;
`report_campaign.py` evaluates frozen production predictions without refitting.
The audit discovers observed rounds and available frozen proposals automatically,
starting at Group 3 by default. Viability and mechanical definitions are evaluated
separately; insufficient training is recorded as skipped evidence.

```sh
python3 src/08_multi_objective/04_report_campaign/audit_group10.py \
  --output-dir /private/tmp/cryomn-gp-audit-new
```

Use a new empty directory. Optional `--start-round`, `--end-round`, and `--endpoints`
restrict the comparison. This never promotes a strategy or rewrites production data.
It is a retrospective forward refit, not a frozen prospective challenger trial.
See [decision guidance](../../../docs/group10/readiness_and_gp_decisions.md).


The default `--suite expanded` now includes one-factor GP sensitivities and explicit
latent/future interval columns; `--suite historical` keeps the original comparison set.
`sensitivity_settings.json` records all settings, and `ucb_sensitivity.csv` explores
uncertainty bonuses on already measured slates. See
[scope and limitations](../../../docs/group10/gp_sensitivity_and_fallbacks.md).

## Batch-aware GP comparisons

`compare_gp_strategies.py` provides `audit`, `freeze`, `evaluate` and saved-evidence-only
`render` commands. See `docs/group10/gp_strategy_implementation.md` for the full workflow.
Group 10 forecasts are frozen under its `gp_comparison/` directory, separate from the
original proposal. Ingestion evaluates frozen comparisons before next-round generation.
Group 11 generation requires an explicit GP strategy decision; no strategy is promoted
by an audit or report.

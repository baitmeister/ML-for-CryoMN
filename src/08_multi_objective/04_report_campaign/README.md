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

## Ingredient influence and retrospective SHAP

`explain_models.py` creates mean absolute SHAP rankings and directional SHAP
summary plots for each endpoint model. Viability and mechanical resistance are
separate objectives; intact and preparation probabilities are diagnostic gate
models, and initial stiffness is a diagnostic endpoint. An intact classifier
explanation does not describe the separate empirical feasibility selection policy.

```sh
# Current model cutoff: all completed rounds, before the next round.
python3 src/08_multi_objective/04_report_campaign/explain_models.py

# Models before Round 9: strictly preceding rounds and baseline legacy evidence.
python3 src/08_multi_objective/04_report_campaign/explain_models.py --round ROUND_009

# All completed rounds plus the current cutoff and a signed history heatmap.
python3 src/08_multi_objective/04_report_campaign/explain_models.py --all-rounds

# Redraw only the historical heatmap from stored SHAP summaries; no refitting.
python3 src/08_multi_objective/04_report_campaign/explain_models.py --history-only

# Compare SHAP only where each ingredient is present, with cell sample counts.
python3 src/08_multi_objective/04_report_campaign/explain_models.py --presence-history-only
```

The default output directory is
`results/multi_objective_v2/reports/explainability/`. Each `ROUND_NNN/` directory
explains the model **before** that round; `current/` explains the model cutoff
for the next round. Mechanical models use the comparable endpoint definition
for that target round. Missing measurements or single-class gates are reported
as unavailable, rather than as zero ingredient importance.

Existing saved viability comparison GPs matching the recorded production
strategy are reused after checking their hashes, features and training cutoff.
They remain labelled as frozen *comparison* evidence, separate from production
proposal forecasts. Other endpoint models, and earlier viability models without
snapshots, are retrospective refits using today's implementation and only
preceding outcomes. The baseline legacy cohort is assumed to have been available
before campaign Round 1, consistent with existing retrospective audits.
This command writes only its explanation outputs and does not change campaign
observations, proposals, model decisions or future selection.

Outputs include `feature_importance.png`, endpoint `*_shap_summary.png` files,
`feature_importance.csv`, row-level `shap_values.csv`, background/explained rows,
and a `manifest.json` with model provenance, settings and additivity checks.
The all-round history shows **signed mean SHAP**, normalized within endpoint and
round as `100 * mean(SHAP_j) / sum_k mean(abs(SHAP_k))`. Blue means a net reduction
in prediction, orange a net increase relative to that model's background, and
white zero net contribution. Grey cells mean unavailable. Features remain
selected by their absolute influence so important features with cancelling
contributions remain visible. Signed values do not sum to 100%; opposing
contributions can cancel, so a near-zero signed mean does not imply low overall
importance or a monotonic dose effect. Absolute rankings and directional row
plots remain available separately. Raw importance is in each endpoint's units,
so magnitudes cannot be compared directly between viability, force and probability.

The presence-conditioned historical comparison is saved as
`feature_presence_shap_history.png`, with separate
`viability_percent_presence_shap_history.png` and
`critical_axial_load_N_per_needle_presence_shap_history.png` exports for closer
inspection. Every active ingredient is displayed. Each cell shows
`mean(SHAP_j | ingredient_j > 1e-6)` in the endpoint's native units and `n`, the
number of containing formulation/round rows in the stored explanation sample.
Column labels give `N`, the full explained sample size. Grey cells and dashes
mean no containing rows in that sample; they are not zero effects. Mechanical
columns begin when an eligible model is available.

The `1e-6` presence cutoff is a numerical zero tolerance in each ingredient's
stored unit (M or percent). Counts concern the explained samples (default cap
48), not all training observations or independent biological batches. The
companion CSV also includes distinct formulation counts, training sizes, dose
ranges, and separate statuses for unavailable models and zero present rows.
Original SHAP backgrounds and values are retained; conditioning selects rows
and does not estimate adding or removing an ingredient. Co-occurring ingredients,
concentration, and round effects can still be confounded. The metadata verifies
all row-level source hashes. `--all-rounds` also generates these comparisons;
`--presence-history-only` redraws them without refitting or recomputing SHAP.

The method is SHAP's model-agnostic `PermutationExplainer` with an independent
empirical training background, a fixed seed, and an identity link. Defaults are
48 explained formulation/batch rows, 16 background rows and 16 forward/reverse
permutation cycles. Increase `--max-explain`, `--max-background` and
`--permutations` for a larger analysis. These are sampled model explanations,
not causal ingredient effects, predictive-performance tests, or explanations
of the acquisition function and its penalties. Masking can create untested
mixtures when ingredients are correlated. Historical backgrounds and viability
strategies change, so the ranking history is descriptive.

This separate command deliberately permits retrospective refitting;
`report_campaign.py` retains its saved-prediction-only prospective contract.

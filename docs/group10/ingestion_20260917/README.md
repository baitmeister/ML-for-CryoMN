# Group 10 ingestion and GP decision brief

## Readiness

Group 10 was validated and ingested using the user-confirmed 100 loaded needles for all seven runs. All 12 formulations passed intact. There are 48 viability observations, seven attempted mechanical runs and six valid terminal-force measurements, aggregating to four mechanical formulation–batch endpoints.

The campaign now meets the hybrid gate: eight eligible pairs, eight distinct formulations and two biological batches. The full phase gate remains unmet (requires 16 pairs, 12 formulations and three batches). No Group 11 proposal was generated and no model strategy was promoted.

## Mechanical results

The endpoint is force at +0.8 mm after sustained 1 N, divided by 100 loaded needles; it is not fracture force.

| Formulation number | Valid replicates | Mean force (N/needle) |
|---|---:|---:|
| 1 (viability reference recipe) | 1 | 0.635854 |
| 2 | 2 | 0.178708 |
| 6 | 2 | 0.272010 |
| 7 | 1 | 0.109429 |

Formulation 1 replicate 1 reached only +0.6237 mm after the trigger; its attempt and QC status are retained without an imputed force. Replicate 2 supplies its valid mechanical evidence. Formulation 7 was explicitly a sole sample. Neither single-sample endpoint has an estimable sample standard deviation. Reference-recipe mechanics are ordinary evidence with no control normalization.

## Frozen GP evidence

Forecasts were frozen on 2026-09-16 using data through Group 9, after Group 10 slate selection and before these outcomes were supplied. All frozen artifact hashes and reproduced prediction means passed verification. No challenger was refitted for this evaluation. Full-slate scores below evaluate the 11 ordinary formulations; the viability control is excluded.

| Strategy | Mean absolute error (percentage points) | Future coverage | Mean future width (percentage points) |
|---|---:|---:|---:|
| Current GP | 31.45 | 54.5% | 53.63 |
| Proposed GP — campaign-only | 19.52 | 72.7% | 44.43 |
| Proposed GP — campaign-only + fitted residual noise | 17.80 | 90.9% | 62.02 |
| Proposed GP — campaign-only + batch effect | 18.33 | 81.8% | 68.38 |
| Proposed GP — campaign-only + batch effect + control calibration | 18.33 | 81.8% | 68.38 |

All strategies overpredicted every ordinary formulation in this batch. All full-slate R² values are negative; none establishes accurate absolute prediction in new batches. These are nominal 95% intervals for formulation–batch means, with fitted-parameter uncertainty omitted. One batch cannot establish long-run calibration.

On the same eight non-reference outcomes, the batch-effect Proposed GP improved from 15.60 to 5.56 percentage-point error after revealing only the three predesignated retests. Adding the control gave 5.48 points. This is supporting information-value evidence, not a pre-experiment selection improvement. The small additional benefit from the control does not establish reliable control calibration; only one historical control batch is now available.

## Decision before Group 11

The leading choices to discuss are Proposed GP — campaign-only + fitted residual noise (lowest before-batch error here) and Proposed GP — campaign-only + batch effect (stronger within-batch ordering here and useful reference adjustment). Group 10 alone does not establish a decisive winner. Review alongside the separate historical forward-validation report; do not pool provenance into a winner score.

Record the chosen viability strategy, comparable mechanical configuration, supported qLogNEHVI acquisition and any selection acceptance thresholds before generating Group 11. Control-calibrated production remains unsupported with fewer than two historical control batches. The user makes this choice.

## Evidence

- [GP decision report and figures](../../../results/multi_objective_v2/rounds/ROUND_010/reports/gp_comparison/README.md)
- [Historical comparison](../audit/batch_gp_20260916/README.md)
- [Completed worksheet](../../../results/multi_objective_v2/rounds/ROUND_010/completed/completed.csv)
- [Input provenance](input_manifest.json)

Validation: frozen worksheet identity/rank checks passed for 12 candidates and 67 endpoint-specific rows; terminal traces were reproduced from archived raw files; canonical counts and hybrid status were reconciled. Raw source files and the frozen Group 10 proposal/forecasts remain unchanged.

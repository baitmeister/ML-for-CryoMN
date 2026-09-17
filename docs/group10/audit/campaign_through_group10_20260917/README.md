# Whole-campaign GP comparison through Group 10

## Why Groups 1–2 were missing

The previous audit used a default start of Group 3. Group 2 was evaluable and is now included. Group 1 has a reconstructed proposal filename, which the prior audit also skipped. The audit now accepts that file for explicitly retrospective evaluation and records its path, hash and reconstructed status.

There are no preceding campaign observations before Group 1, so the campaign-only Proposed GPs cannot be trained for that evaluation. Those four unavailable fits are recorded in the audit manifest rather than assigned invented scores. The Current GP can use legacy evidence. Group 2 Proposed GPs train on Group 1 only; its batch-variance estimate is especially weak with just one training batch.

Groups 1–2 were already included as training evidence for later historical evaluations; their absence was from evaluation rows, not from that training history.

## How to read this report

Each historical evaluation trains on preceding Groups only, retaining eligible legacy evidence for Current GP. These are retrospective refits of today's configurations, not the original models that chose each historical slate. Group 1's reconstructed slate does not establish original forecast provenance. Available non-round legacy evidence is assumed to predate the campaign; this audit does not independently establish its historical availability.

Group 10 is evaluated from the saved pre-outcome challenger models trained through Group 9. Its original frozen evidence is preserved. We keep this evidence separate from retrospective refits; the combined CSV files retain an `evidence` column and do not imply a pooled winner score.

Error, bias and interval width are viability percentage points. Positive bias means overprediction. Coverage is for nominal 95% future-observation intervals. Rank correlation measures within-Group ordering. Summary values give each Group equal weight. Counts are formulation–batch means, not technical replicates or independent biological batches.

## Comparable historical window: Groups 2–9

All five strategies evaluated the same 96 outcomes across eight Groups.

| Strategy | Groups | Formulation–batch outcomes | Mean absolute error | Signed bias | Rank correlation | Future coverage | Future width |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current GP | 8 | 96 | 25.30 | 22.55 | 0.30 | 66.7% | 65.78 |
| Proposed GP — campaign-only | 8 | 96 | 15.23 | 7.79 | 0.53 | 80.2% | 53.43 |
| Proposed GP — campaign-only + fitted residual noise (matched comparison) | 8 | 96 | 14.98 | 7.72 | 0.51 | 89.6% | 58.49 |
| Proposed GP — campaign-only + batch effect | 8 | 96 | 15.63 | 8.89 | 0.55 | 89.6% | 61.16 |
| Proposed GP — campaign-only + batch effect + control calibration | 8 | 96 | 15.63 | 8.89 | 0.55 | 89.6% | 61.16 |

## Group 1: Current GP only

| Strategy | Groups | Formulation–batch outcomes | Mean absolute error | Signed bias | Rank correlation | Future coverage | Future width |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current GP | 1 | 12 | 21.78 | 21.58 | 0.44 | 100.0% | 216.35 |

## Group 10: frozen pre-outcome challengers

These scores cover 11 ordinary formulations; the viability control is excluded.

| Strategy | Groups | Formulation–batch outcomes | Mean absolute error | Signed bias | Rank correlation | Future coverage | Future width |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current GP | 1 | 11 | 31.45 | 31.45 | 0.28 | 54.5% | 53.63 |
| Proposed GP — campaign-only | 1 | 11 | 19.52 | 19.52 | 0.48 | 72.7% | 44.43 |
| Proposed GP — campaign-only + fitted residual noise (matched comparison) | 1 | 11 | 17.80 | 17.80 | 0.42 | 90.9% | 62.02 |
| Proposed GP — campaign-only + batch effect | 1 | 11 | 18.33 | 18.33 | 0.55 | 81.8% | 68.38 |
| Proposed GP — campaign-only + batch effect + control calibration | 1 | 11 | 18.33 | 18.33 | 0.55 | 81.8% | 68.38 |

## Interpretation and decision

The campaign-only Proposed GPs improve substantially over Current GP on the matched historical window and Group 10. Proposed GP — campaign-only + fitted residual noise has the lowest mean before-batch absolute error among these choices in both windows. Proposed GP — campaign-only + batch effect remains a plausible alternative, particularly for representing shared batch variation and reference conditioning; these results do not establish superior before-batch absolute accuracy for it.

Control-calibrated and non-calibrated batch models have identical before-batch scores here because the evaluated training histories contain no preceding control batches. That equality is not evidence that controls never help. With only Group 10 now supplying control history, control-calibrated production remains insufficiently supported.

On Group 10's same eight non-reference outcomes, reference conditioning reduces batch-model error from 15.60 to 5.56 points (5.48 with the additional control). Keep this separate from full-slate before-batch performance: references were revealed and cannot count as unseen cases. All Group 10 full-slate R² scores remain negative. No strategy is automatically promoted.

## Figures and source evidence

- [Historical Groups 1–9 report and decomposed figures](historical_groups1_9/README.md)
- [Frozen Group 10 report and decomposed figures](../../../../results/multi_objective_v2/rounds/ROUND_010/reports/gp_comparison/README.md)
- [All per-Group metrics, with evidence stages](all_campaign_metrics.csv)
- [All prediction rows, with evidence stages](all_campaign_predictions.csv)

Group 1 appears only for Current GP; its absence from Proposed GP figures is an explicit training-data limitation. Mechanical evidence spans Groups 9–10 only and is not evaluated by this viability GP comparison.

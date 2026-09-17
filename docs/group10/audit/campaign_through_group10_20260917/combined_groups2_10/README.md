# Combined GP comparison: Groups 2–10

One aggregate across nine Groups and 107 ordinary formulation–batch outcomes per strategy. Each Group receives equal weight. Only before-batch, full-slate predictions enter this summary; reference-conditioned rows are excluded.

| Strategy | Mean absolute error | Signed bias | Within-Group ranking | Future coverage | Future width |
|---|---:|---:|---:|---:|---:|
| Current GP | 25.99 | 23.54 | 0.30 | 65.3% | 64.43 |
| Proposed GP — campaign-only | 15.71 | 9.10 | 0.52 | 79.4% | 52.43 |
| Proposed GP — campaign-only + fitted residual noise (matched comparison) | 15.29 | 8.84 | 0.50 | 89.7% | 58.88 |
| Proposed GP — campaign-only + batch effect | 15.93 | 9.94 | 0.55 | 88.7% | 61.96 |
| Proposed GP — campaign-only + batch effect + control calibration | 15.93 | 9.94 | 0.55 | 88.7% | 61.96 |

Error, bias and width are viability percentage points. Ranking is mean Spearman correlation. Coverage refers to nominal 95% future-observation intervals. Positive bias means overprediction.

Provenance: Groups 2–9 are retrospective forward refits; Group 10 uses frozen pre-outcome challenger forecasts. The aggregate is a descriptive campaign summary with mixed provenance, not nine prospectively evaluated Groups. Original provenance is retained in the source rows. Group 1 is excluded from every strategy.

Proposed GP — campaign-only + fitted residual noise has the lowest aggregate absolute error. Proposed GP — campaign-only + batch effect has slightly better within-Group ranking. These metrics do not automatically select a production strategy. The control-calibrated model had no preceding control evidence in these evaluations; identical scores do not validate control calibration.

## Graphs and data

- [Combined summary graph](gp_strategy_summary.png)
- [Comparison by Group](gp_strategy_by_batch.png)
- [Aggregate metrics](aggregate_metrics.csv)
- [Per-Group metrics](metrics.csv)
- [Prediction records](predictions.csv)

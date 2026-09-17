# GP strategy decision report

pre_outcome_challenger_after_slate_selection | percentage points; not an automatic model decision

No model is automatically promoted. Current GP and Proposed GP labels identify data and modifications.
Error and bias are percentage points; rank correlation measures within-Group ordering. Coverage and width refer to future-observation intervals.
Historical results are retrospective. Pre-outcome challenger forecasts were made after the official slate was selected.
Control calibration with fewer than two historical reference batches is insufficiently supported; do not infer validated correction from it.
Hyperparameters use bounded maximum likelihood. Intervals do not integrate hyperparameter uncertainty.
Reference-conditioned results exclude the conditioning observations and use matched test rows. They do not retrospectively change selection.

## Reading future coverage and width

**Future coverage** is the percentage of later measured formulation–batch outcomes inside the nominal 95% future-observation intervals. About 95% is the intended long-run target, not a guarantee for one small Group. Summary values average the per-Group coverage.
**Future width** is the average upper minus lower interval bound, in viability percentage points. An interval from 40% to 80% has width 40 points. It describes uncertainty, not prediction error. Summary values average the per-Group widths.
These intervals include residual noise and, for batch-effect models, new-batch variation. They concern the modeled formulation–batch mean, not every technical replicate. Wider intervals can improve coverage without improving the central prediction; assess coverage and width together.

## Summary

| Strategy | Groups | Observations | MAE | Bias | Mean within-Group rank | Future coverage | Future width |
|---|---:|---:|---:|---:|---:|---:|---:|
| Proposed GP — campaign-only + batch effect | 1 | 11 | 18.33 | 18.33 | 0.55 | 81.8% | 68.38 |
| Proposed GP — campaign-only | 1 | 11 | 19.52 | 19.52 | 0.48 | 72.7% | 44.43 |
| Proposed GP — campaign-only + fitted residual noise (matched comparison) | 1 | 11 | 17.80 | 17.80 | 0.42 | 90.9% | 62.02 |
| Proposed GP — campaign-only + batch effect + control calibration | 1 | 11 | 18.33 | 18.33 | 0.55 | 81.8% | 68.38 |
| Current GP | 1 | 11 | 31.45 | 31.45 | 0.28 | 54.5% | 53.63 |

## Figures

- [gp_strategy_summary](gp_strategy_summary.png)
- [gp_strategy_by_batch](gp_strategy_by_batch.png)
- [reference_conditioning_comparison](reference_conditioning_comparison.png)
- [batch_effect_estimates](batch_effect_estimates.png)
- [gp_predictions_round_010_current](gp_predictions_round_010_current.png)
- [gp_predictions_round_010_campaign](gp_predictions_round_010_campaign.png)
- [gp_predictions_round_010_campaign_noise](gp_predictions_round_010_campaign_noise.png)
- [gp_predictions_round_010_batch](gp_predictions_round_010_batch.png)
- [gp_predictions_round_010_control](gp_predictions_round_010_control.png)

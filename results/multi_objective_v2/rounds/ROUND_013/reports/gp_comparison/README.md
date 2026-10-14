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
| Proposed GP — campaign-only + batch effect | 1 | 11 | 11.75 | -3.49 | 0.52 | 100.0% | 72.28 |
| Proposed GP — campaign-only | 1 | 11 | 11.56 | -2.64 | 0.53 | 72.7% | 45.25 |
| Proposed GP — campaign-only + fitted residual noise (matched comparison) | 1 | 11 | 12.66 | -5.61 | 0.48 | 100.0% | 68.84 |
| Proposed GP — campaign-only + batch effect, informed by viability controls | 1 | 11 | 11.81 | -3.54 | 0.52 | 100.0% | 72.05 |
| Current GP | 1 | 11 | 18.54 | -4.00 | 0.26 | 36.4% | 45.39 |

## Figures

- [gp_strategy_summary](gp_strategy_summary.png)
- [gp_strategy_by_batch](gp_strategy_by_batch.png)
- [reference_conditioning_comparison](reference_conditioning_comparison.png)
- [batch_effect_estimates](batch_effect_estimates.png)
- [gp_predictions_round_013_current](gp_predictions_round_013_current.png)
- [gp_predictions_round_013_campaign](gp_predictions_round_013_campaign.png)
- [gp_predictions_round_013_campaign_noise](gp_predictions_round_013_campaign_noise.png)
- [gp_predictions_round_013_batch](gp_predictions_round_013_batch.png)
- [gp_predictions_round_013_control](gp_predictions_round_013_control.png)

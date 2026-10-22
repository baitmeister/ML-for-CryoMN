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
| Proposed GP — campaign-only + batch effect | 1 | 11 | 11.24 | 6.23 | 0.69 | 90.9% | 72.72 |
| Proposed GP — campaign-only | 1 | 11 | 11.01 | 7.36 | 0.78 | 63.6% | 39.77 |
| Proposed GP — campaign-only + fitted residual noise (matched comparison) | 1 | 11 | 10.97 | 3.36 | 0.69 | 90.9% | 70.10 |
| Proposed GP — campaign-only + batch effect, informed by viability controls | 1 | 11 | 11.31 | 6.31 | 0.69 | 90.9% | 71.90 |
| Current GP | 1 | 11 | 17.13 | 6.88 | 0.58 | 36.4% | 34.05 |

## Figures

- [gp_strategy_summary](gp_strategy_summary.png)
- [gp_strategy_by_batch](gp_strategy_by_batch.png)
- [reference_conditioning_comparison](reference_conditioning_comparison.png)
- [batch_effect_estimates](batch_effect_estimates.png)
- [gp_predictions_round_015_current](gp_predictions_round_015_current.png)
- [gp_predictions_round_015_campaign](gp_predictions_round_015_campaign.png)
- [gp_predictions_round_015_campaign_noise](gp_predictions_round_015_campaign_noise.png)
- [gp_predictions_round_015_batch](gp_predictions_round_015_batch.png)
- [gp_predictions_round_015_control](gp_predictions_round_015_control.png)

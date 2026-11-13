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
| Proposed GP — campaign-only + batch effect | 1 | 11 | 9.11 | 7.03 | 0.74 | 90.9% | 75.37 |
| Proposed GP — campaign-only | 1 | 11 | 13.35 | 5.56 | 0.60 | 63.6% | 39.86 |
| Proposed GP — campaign-only + fitted residual noise (matched comparison) | 1 | 11 | 7.91 | 3.40 | 0.66 | 90.9% | 72.93 |
| Proposed GP — campaign-only + batch effect, informed by viability controls | 1 | 11 | 9.09 | 7.01 | 0.73 | 90.9% | 74.09 |
| Current GP | 1 | 11 | 17.84 | -2.57 | 0.73 | 36.4% | 27.12 |

## Figures

- [gp_strategy_summary](gp_strategy_summary.png)
- [gp_strategy_by_batch](gp_strategy_by_batch.png)
- [reference_conditioning_comparison](reference_conditioning_comparison.png)
- [batch_effect_estimates](batch_effect_estimates.png)
- [gp_predictions_round_017_current](gp_predictions_round_017_current.png)
- [gp_predictions_round_017_campaign](gp_predictions_round_017_campaign.png)
- [gp_predictions_round_017_campaign_noise](gp_predictions_round_017_campaign_noise.png)
- [gp_predictions_round_017_batch](gp_predictions_round_017_batch.png)
- [gp_predictions_round_017_control](gp_predictions_round_017_control.png)

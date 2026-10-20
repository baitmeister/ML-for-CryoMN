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
| Proposed GP — campaign-only + batch effect | 1 | 11 | 21.13 | 9.91 | 0.28 | 81.8% | 70.71 |
| Proposed GP — campaign-only | 1 | 11 | 21.49 | 13.44 | 0.28 | 63.6% | 42.85 |
| Proposed GP — campaign-only + fitted residual noise (matched comparison) | 1 | 11 | 20.47 | 9.06 | 0.47 | 90.9% | 66.80 |
| Proposed GP — campaign-only + batch effect, informed by viability controls | 1 | 11 | 21.18 | 10.13 | 0.28 | 81.8% | 70.05 |
| Current GP | 1 | 11 | 20.13 | 12.36 | 0.31 | 45.5% | 31.20 |

## Figures

- [gp_strategy_summary](gp_strategy_summary.png)
- [gp_strategy_by_batch](gp_strategy_by_batch.png)
- [reference_conditioning_comparison](reference_conditioning_comparison.png)
- [batch_effect_estimates](batch_effect_estimates.png)
- [gp_predictions_round_014_current](gp_predictions_round_014_current.png)
- [gp_predictions_round_014_campaign](gp_predictions_round_014_campaign.png)
- [gp_predictions_round_014_campaign_noise](gp_predictions_round_014_campaign_noise.png)
- [gp_predictions_round_014_batch](gp_predictions_round_014_batch.png)
- [gp_predictions_round_014_control](gp_predictions_round_014_control.png)

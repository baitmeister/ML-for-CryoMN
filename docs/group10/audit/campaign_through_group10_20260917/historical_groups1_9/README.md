# GP strategy decision report

historical_forward_refit | percentage points; not an automatic model decision

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
| Proposed GP — campaign-only + batch effect | 8 | 96 | 15.63 | 8.89 | 0.55 | 89.6% | 61.16 |
| Proposed GP — campaign-only | 8 | 96 | 15.23 | 7.79 | 0.53 | 80.2% | 53.43 |
| Proposed GP — campaign-only + fitted residual noise (matched comparison) | 8 | 96 | 14.98 | 7.72 | 0.51 | 89.6% | 58.49 |
| Proposed GP — campaign-only + batch effect + control calibration | 8 | 96 | 15.63 | 8.89 | 0.55 | 89.6% | 61.16 |
| Current GP | 9 | 108 | 24.91 | 22.44 | 0.31 | 70.4% | 82.51 |

## Figures

- [gp_strategy_summary](gp_strategy_summary.png)
- [gp_strategy_by_batch](gp_strategy_by_batch.png)
- [reference_conditioning_comparison](reference_conditioning_comparison.png)
- [batch_effect_estimates](batch_effect_estimates.png)
- [gp_predictions_round_001_current](gp_predictions_round_001_current.png)
- [gp_predictions_round_002_current](gp_predictions_round_002_current.png)
- [gp_predictions_round_002_campaign](gp_predictions_round_002_campaign.png)
- [gp_predictions_round_002_campaign_noise](gp_predictions_round_002_campaign_noise.png)
- [gp_predictions_round_002_batch](gp_predictions_round_002_batch.png)
- [gp_predictions_round_002_control](gp_predictions_round_002_control.png)
- [gp_predictions_round_003_current](gp_predictions_round_003_current.png)
- [gp_predictions_round_003_campaign](gp_predictions_round_003_campaign.png)
- [gp_predictions_round_003_campaign_noise](gp_predictions_round_003_campaign_noise.png)
- [gp_predictions_round_003_batch](gp_predictions_round_003_batch.png)
- [gp_predictions_round_003_control](gp_predictions_round_003_control.png)
- [gp_predictions_round_004_current](gp_predictions_round_004_current.png)
- [gp_predictions_round_004_campaign](gp_predictions_round_004_campaign.png)
- [gp_predictions_round_004_campaign_noise](gp_predictions_round_004_campaign_noise.png)
- [gp_predictions_round_004_batch](gp_predictions_round_004_batch.png)
- [gp_predictions_round_004_control](gp_predictions_round_004_control.png)
- [gp_predictions_round_005_current](gp_predictions_round_005_current.png)
- [gp_predictions_round_005_campaign](gp_predictions_round_005_campaign.png)
- [gp_predictions_round_005_campaign_noise](gp_predictions_round_005_campaign_noise.png)
- [gp_predictions_round_005_batch](gp_predictions_round_005_batch.png)
- [gp_predictions_round_005_control](gp_predictions_round_005_control.png)
- [gp_predictions_round_006_current](gp_predictions_round_006_current.png)
- [gp_predictions_round_006_campaign](gp_predictions_round_006_campaign.png)
- [gp_predictions_round_006_campaign_noise](gp_predictions_round_006_campaign_noise.png)
- [gp_predictions_round_006_batch](gp_predictions_round_006_batch.png)
- [gp_predictions_round_006_control](gp_predictions_round_006_control.png)
- [gp_predictions_round_007_current](gp_predictions_round_007_current.png)
- [gp_predictions_round_007_campaign](gp_predictions_round_007_campaign.png)
- [gp_predictions_round_007_campaign_noise](gp_predictions_round_007_campaign_noise.png)
- [gp_predictions_round_007_batch](gp_predictions_round_007_batch.png)
- [gp_predictions_round_007_control](gp_predictions_round_007_control.png)
- [gp_predictions_round_008_current](gp_predictions_round_008_current.png)
- [gp_predictions_round_008_campaign](gp_predictions_round_008_campaign.png)
- [gp_predictions_round_008_campaign_noise](gp_predictions_round_008_campaign_noise.png)
- [gp_predictions_round_008_batch](gp_predictions_round_008_batch.png)
- [gp_predictions_round_008_control](gp_predictions_round_008_control.png)
- [gp_predictions_round_009_current](gp_predictions_round_009_current.png)
- [gp_predictions_round_009_campaign](gp_predictions_round_009_campaign.png)
- [gp_predictions_round_009_campaign_noise](gp_predictions_round_009_campaign_noise.png)
- [gp_predictions_round_009_batch](gp_predictions_round_009_batch.png)
- [gp_predictions_round_009_control](gp_predictions_round_009_control.png)

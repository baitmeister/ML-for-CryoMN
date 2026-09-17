# Dynamic offline GP audit

[Plain-language guide to columns, model names and conclusions](../../audit_reading_guide.md)

**Interval coverage below is original/latent coverage. Future-observation coverage is a separate column in metrics.csv.**

Observed rounds discovered: 3, 4, 5, 6, 7, 8, 9.
Each model is refitted using preceding outcomes only. Current-round outcomes are used only for evaluation.
No strategy is promoted. Original production predictions remain in the staged prospective reports.
Mechanical definitions are evaluated separately. Unavailable training is recorded in skipped.json, not replaced by fabricated forecasts.

| Endpoint / definition | Model | n | MAE | Bias | R² | Interval coverage |
|---|---|---:|---:|---:|---:|---:|
| viability_percent /  | campaign_bounds_learned | 84 | 14.47 | 9.19 | -0.040 | 0.726 |
| viability_percent /  | campaign_existing | 84 | 15.18 | 10.41 | -0.107 | 0.786 |
| viability_percent /  | campaign_existing_adaptive | 84 | 14.9 | 10.44 | -0.089 | 0.833 |
| viability_percent /  | campaign_mean | 84 | 16.25 | 8.842 | -0.224 | 0.940 |
| viability_percent /  | campaign_median | 84 | 16 | 7.659 | -0.180 | 0.940 |
| viability_percent /  | current_mixed | 84 | 26.72 | 24.29 | -2.119 | 0.655 |
| viability_percent /  | legacy_correction | 84 | 15.17 | 9.917 | -0.121 | 0.988 |
| viability_percent /  | sens_amplitude_learned | 84 | 15.18 | 10.41 | -0.107 | 0.774 |
| viability_percent /  | sens_ard_learned | 84 | 14.81 | 6.036 | -0.036 | 0.631 |
| viability_percent /  | sens_baseline | 84 | 15.18 | 10.41 | -0.107 | 0.786 |
| viability_percent /  | sens_bounds | 84 | 13.24 | 5.635 | 0.094 | 0.488 |
| viability_percent /  | sens_bounds_learned | 84 | 14.47 | 9.19 | -0.040 | 0.726 |
| viability_percent /  | sens_bounds_learned_restarts | 84 | 14.47 | 9.19 | -0.040 | 0.726 |
| viability_percent /  | sens_bounds_learned_wide | 84 | 14.47 | 9.19 | -0.040 | 0.726 |
| viability_percent /  | sens_length_learned | 84 | 14.78 | 9.573 | -0.044 | 0.738 |
| viability_percent /  | sens_length_long | 84 | 13.65 | 7.365 | 0.102 | 0.571 |
| viability_percent /  | sens_length_short | 84 | 15.69 | 8.496 | -0.165 | 0.833 |
| viability_percent /  | sens_logit | 84 | 14.87 | 10.08 | -0.062 | 0.798 |
| viability_percent /  | sens_matern15 | 84 | 15.03 | 10.3 | -0.093 | 0.798 |
| viability_percent /  | sens_mean_median | 84 | 14.89 | 9.639 | -0.065 | 0.786 |
| viability_percent /  | sens_mixed_sources | 84 | 26.72 | 24.29 | -2.119 | 0.655 |
| viability_percent /  | sens_noise_10pp | 84 | 14.41 | 10.49 | -0.064 | 0.905 |
| viability_percent /  | sens_noise_1pp | 84 | 15.18 | 10.41 | -0.107 | 0.786 |
| viability_percent /  | sens_noise_3pp | 84 | 15.08 | 10.42 | -0.100 | 0.821 |
| viability_percent /  | sens_noise_5pp | 84 | 14.9 | 10.44 | -0.089 | 0.833 |
| viability_percent /  | sens_noise_learned | 84 | 14.89 | 10.32 | -0.089 | 0.833 |
| viability_percent /  | sens_rbf | 84 | 15.69 | 10.67 | -0.152 | 0.702 |
| viability_percent /  | sens_recent3 | 84 | 13.85 | 5.92 | 0.069 | 0.810 |

Viability errors are percentage points; mechanical errors are N per loaded needle.
Historical intervals use mean ± 1.96 × returned SD. Expanded cases also store explicit latent and future intervals, including inverse-transformed intervals. Neither is automatically calibrated; future coverage and width have separate metric columns.
Select using bias, within-round ranking, uncertainty and simple baselines together. Pooled R² alone does not establish useful selection.
The acquisition ModelListGP remains offline-only. This audit compares prediction strategies, not the experimental benefit of alternative candidate slates.
Expanded sensitivity settings and paired one-factor comparisons are in sensitivity_settings.json and sensitivity_comparisons.csv. Negative MAE change favors the challenger on matched outcomes.
Expanded cases distinguish latent uncertainty from future intervals under their stated noise assumptions. These intervals require empirical validation; unknown-noise fallback and transformed-noise approximations are recorded.
This is a one-factor study plus selected combined configurations, not an exhaustive factorial or a nested model-selection validation. Source selection also changes empirical centering/scaling when those are data-dependent.
ucb_sensitivity.csv varies the uncertainty bonus on the measured historical slate only; it is not a replay of the full constrained selector or unknown alternative formulations.
Acquisition, correlated-output transfer, batch effects and future-outcome benefit are not inferred from this prediction audit; see the decision guide for the required complementary checks.
Freeze challenger predictions before future outcomes for a prospective strategy comparison.
Failed fits: 0; skipped cohorts/comparisons: 14.

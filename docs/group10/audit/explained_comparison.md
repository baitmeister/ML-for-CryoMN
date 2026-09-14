# Reading the Group 3–8 viability audit

## What was compared

The audit contains 72 formulation–round viability means: 12 from each of Groups 3–8. For Group 3, it fits using preceding campaign groups (and historical sources for variants that allow them); it predicts the actual Group 3 slate. It repeats with each subsequent cutoff. It does not use Group 9 outcomes. It refits the current methodology at historical cutoffs; this is not a table of the original archived predictions.

All variants predict the same historically selected formulations. These data cannot establish what a different optimizer would have discovered by proposing a different slate. Repeated formulations and shared experimental groups also mean 72 rows are not 72 independent experimental campaigns.

## Models and principal results

All errors below are viability percentage points. Bias means predicted minus measured.

| Model | What changes | MAE | Bias | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Current mixed | Existing GP with campaign and legacy observations | 27.46 | +25.22 | 31.63 | −2.303 |
| Campaign mean | Predict preceding campaign mean for every candidate | 16.61 | +10.79 | 19.69 | −0.280 |
| Campaign median | Predict preceding campaign median for every candidate | 16.30 | +9.53 | 19.27 | −0.227 |
| Campaign existing | Existing scaling/kernel/noise rules, campaign observations only | 15.85 | +11.15 | 19.01 | −0.193 |
| Campaign bounds learned | Physical-range scaling, learned shared Matérn length and amplitude | 14.99 | +9.74 | 18.34 | −0.111 |
| Campaign existing adaptive | Campaign existing with offline noise assignments | 15.59 | +11.25 | 18.89 | −0.178 |
| Legacy correction | Legacy GP plus campaign residual-correction GP | 15.74 | +10.54 | 19.09 | −0.204 |

The largest improvement is current mixed → campaign existing: MAE falls by 11.60 points (about 42%). This comparison supports a source-treatment mismatch as an important issue in this replay. It does not prove that legacy observations are intrinsically bad or isolate a biological cause; removing them also changes fitted scaling and the distribution to which the GP reverts.

Campaign existing → bounds learned gains only another 0.86 points (about 5%). The bounded variant changes both scaling and kernel fitting, so this comparison cannot identify which of these changes caused the gain. The gain is also small relative to remaining error. It should not be presented as a decisive GP replacement result.

The bounded variant beats campaign existing in five of six groups. It beats the historical campaign mean in only four of six. In Group 8 the campaign mean achieves MAE 15.53, versus 17.30 for bounds learned and 19.34 for campaign existing. More elaborate modeling is not consistently superior.

## What the error numbers mean

MAE of 14.99 means an average absolute error of about 15 viability percentage points. It does not mean every prediction is within 15 points. RMSE gives larger errors greater weight. Positive bias indicates systematic overprediction: the current model averages 54.39% predictions against 29.17% measured viability across these 72 rows. The bounded model still overpredicts by 9.74 points on average.

R² = 1 − squared prediction error / squared deviation from the evaluated outcomes’ pooled mean. Negative R² means the model has greater squared error than that retrospective constant. It does not mean there is no ranking information. Nor does it contradict beating the forward historical-mean baseline: that baseline knows only earlier groups; the R² denominator uses the mean of the held-out evaluated outcomes themselves.

Do not exclude predictions that reverted to a prior mean to improve the headline R². A supported-candidate subgroup can be a secondary diagnostic if support was frozen before outcomes, with the complete-slate result retained. Here 48 of 72 rows lack historical support labels, so support conclusions are limited.

## Ranking and uncertainty

The pooled Spearman rank correlation is 0.60 for campaign existing versus 0.50 for bounds learned. This cautions against declaring the lower-MAE model the best candidate selector. Pooled correlations also combine between-group differences; within-group rankings are more relevant to choosing candidates. The bounded model improves within-group Spearman over campaign existing in only three of six groups. Both have very weak rankings in Group 4.

The report’s nominal 95% intervals cover 79.2% of outcomes for campaign existing and 72.2% for bounds learned, with average total widths of 53.30 and 46.75 points respectively. The bounded model trades lower mean error for narrower intervals with worse observed coverage.

These are mean ± 1.96 × returned model SD. With sklearn’s training-noise alpha, GP posterior SD does not automatically include new-batch observation noise. Baseline intervals use historical spread, while the legacy correction adds component variances under an independence approximation. Thus these are uncertainty diagnostics with different constructions, not validated, directly interchangeable 95% future-batch prediction intervals. Future comparison should report latent-mean and future-observation uncertainty separately.

The legacy correction covers 98.6%, but its average interval width is 98.21 points—almost the entire viability scale. High coverage alone is not useful precision.

## Adaptive noise and what remains unknown

Every historical noise estimate used the conservative fallback because preparation independence was not recorded. The adaptive row therefore demonstrates sensitivity to fallback noise assignment, not the benefit of learning preparation variability from replicates. Its MAE gain over campaign existing is 0.26 points and its bias slightly worsens; there is no strong adoption case here.

The Groups 3–8 audit contains no repeated reference-control groups or comparable terminal-force outcomes. Batch correction and real shared-GP mechanical acquisition performance cannot be assessed from that snapshot. The endpoint is already defined for Group 10; this historical viability audit does not defer that endpoint decision. The ModelListGP tests establish software operation on synthetic unequal datasets, not experimental superiority.

## Decision implication

Keep the production method unchanged under the agreed transition policy. Before full mechanics begins, compare the simple campaign-only GP first against the bounded variant using fresh, frozen evaluation. Review MAE, bias, within-group ranking, and properly specified uncertainty together. The present audit identifies source treatment as the biggest modeling concern; it does not yet establish that any replacement reliably predicts absolute viability or improves the viability–load trade-off.

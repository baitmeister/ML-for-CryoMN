# How to read the GP audit

## What was tested

GP means Gaussian process: a model returning both a predicted response and uncertainty.
The audit refits models using preceding rounds, then checks their predictions against the
next observed round. This snapshot evaluates 84 viability observations from Groups 3–9.
These are retrospective comparisons: strategies were chosen after outcomes existed.
They are not independent confirmation of a winning strategy. No mechanical winner can
be established yet from forward comparisons of the comparable terminal-force endpoint.

## Column dictionary

| Column | Plain meaning | Interpretation |
|---|---|---|
| endpoint | What is predicted | Viability percentage or force in N per loaded needle |
| variant / Model | Settings used for that comparison | See model names below |
| cohort | Subset evaluated | pooled = all evaluated rows; ROUND = one round; role = one experimental role |
| n | Number of evaluated observations | Not number of training samples or independent batches |
| MAE / mae | Mean absolute error | Average prediction miss, ignoring direction; smaller is better |
| Bias / bias | Mean prediction minus observation | Positive = overprediction; negative = underprediction; zero is desirable |
| RMSE / rmse | Root mean squared error | Like MAE, but penalizes large misses more strongly |
| R² / r2 | Fraction of squared variation explained | 1 is perfect; 0 matches the evaluation-set mean; negative is worse than that constant benchmark |
| spearman | Rank correlation | +1 perfectly orders results, 0 has no monotonic association, −1 reverses order |
| interval_95_coverage | Fraction inside the reported nominal 95% interval | 0.786 means 78.6%; see interval distinction below |
| interval_95_width | Average full interval width | Wider is easier to cover with, but less informative |
| future_interval_95_coverage / width | Coverage / width after adding assumed measurement noise | More appropriate for comparing an interval with a future measured outcome |

Viability errors and widths are **percentage points**: predicting 70% when measured
viability is 55% gives a +15-point error, not a 15% relative error.
R² uses the evaluation mean as a descriptive benchmark; that mean was not available
before the round. Pooled rank correlation can hide poor within-round ordering, so inspect
ROUND rows when deciding whether a model can select the best recipe in a batch.

### Two kinds of uncertainty

**Latent uncertainty** describes the unknown underlying average response.
**Future-observation uncertainty** additionally includes measurement/preparation noise.
A future measurement can legitimately fall outside a narrow latent interval. Therefore,
do not interpret latent coverage against measured outcomes as a complete calibration test.
The compact report's `Interval coverage` is the original/latent column, not the separate
future column. Historical variants lack separately computed future intervals; blank does
not mean zero. Expanded `sens_` variants provide both. A 95% interval ideally captures
about 95% of matching future outcomes, but observed coverage is uncertain in a small,
batch-dependent sample. Always assess width alongside coverage.

## Model-name dictionary

`sens_` means sensitivity experiment, not a different kind of GP. `learned` means parameters
were fitted from preceding observations, using bounded maximum likelihood in this audit.

| Name or suffix | Meaning |
|---|---|
| current_mixed / sens_mixed_sources | Current-style fit mixing eligible literature and campaign evidence |
| campaign_existing / sens_baseline / sens_noise_1pp | Campaign-only fixed-kernel baseline with the recorded 1-point campaign noise in this snapshot |
| campaign_mean / campaign_median | Simple constant campaign-response benchmarks |
| legacy_correction | Historical literature-to-campaign correction comparison |
| campaign_existing_adaptive | Historical adaptive-noise comparison; distinct from a fixed-noise sensitivity |
| bounds | Scale ingredient concentrations by configured concentration spans |
| bounds_learned | Bounds scaling with fitted shared length scale and signal amplitude |
| length_short / length_long | Fixed length scale 0.3 / 3 instead of 1 |
| length_learned | Fit a shared length scale and amplitude |
| ard_learned | Automatic relevance determination: fit separate ingredient length scales |
| amplitude_learned | Fit response signal amplitude |
| matern15 | Matérn smoothness 1.5 instead of baseline 2.5; permits rougher response functions |
| rbf | Radial basis function kernel; assumes a very smooth response function |
| mean_median | Center the GP on the campaign median instead of mean |
| recent3 | Train on the most recent three eligible batches |
| paired_only | Use only endpoint observations that have the other objective available |
| noise_3pp / 5pp / 10pp | Assume measurement SD of 3 / 5 / 10 percentage points |
| noise_learned | Fit a single shared observation-noise level |
| logit | Transform bounded viability before fitting, then transform predictions back |
| log | Log-transform positive mechanical force |
| wide / restarts | Test wider fitting bounds / multiple fitting initializations |

SD means standard deviation; variance is SD squared. UCB means upper confidence bound,
the predicted mean plus kappa times SD. Kappa is the exploration weight. qLogNEHVI means
log noisy expected hypervolume improvement for batch selection: an acquisition criterion
that values improvements to the joint viability/force trade-off. It is not a GP model.

## What these results say

1. **Source treatment has the largest demonstrated effect.** Mixed-source MAE is 26.72
   points versus 15.18 for the campaign baseline, with overprediction bias falling from
   24.29 to 10.41 points. The comparison also changes data-dependent centering/scaling;
   it does not isolate one biological cause.
2. **Bounds-only has the smallest MAE here (13.24), but is too confident for future
   measurements under the tested assumptions:** future coverage is only about 51.2%.
   Its compact-table latent coverage is 48.8%. These are different columns.
3. **Noise assumptions matter more for intervals than for average error.** Changing
   campaign SD from 1 to 5 points changes MAE from 15.18 to 14.90 and future coverage
   from 78.6% to 89.3%; SD 10 gives MAE 14.41 and future coverage 95.2%, with wider
   intervals. This does not establish that the laboratory noise really is 10 points.
4. **More flexible is not automatically better.** Fitted per-ingredient length scales
   give MAE 14.81; fitted bounds models give 14.47. Neither dominates all criteria.
   Wider bounds and restarts produced essentially the same results in this snapshot.
5. **Prediction quality is still limited.** Most pooled R² values remain negative and
   most models overpredict. The audit supports shortlisting, not declaring readiness
   from a high-accuracy model or proving experimental acceptance criteria are met.

Recommendation: retain a simple campaign baseline; compare a constrained fitted model
and defensible noise assumptions prospectively. Do not promote a strategy solely because
it wins one column on these already-seen outcomes. Group 10 provides additional evidence;
freeze subsequent challenger predictions before seeing their outcomes.

## Files

`metrics.csv` contains numerical summaries. `predictions.csv` contains individual predictions.
`sensitivity_settings.json` records exact settings. `sensitivity_comparisons.csv` compares
matched observations (negative MAE change favors the challenger). `ucb_sensitivity.csv`
reranks measured historical slates only; it does not simulate unknown recipe outcomes.
`failures.json` records failed fits; `skipped.json` records comparisons lacking sufficient
eligible data. A skipped comparison is not a failed model.

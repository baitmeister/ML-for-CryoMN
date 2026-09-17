# GP sensitivity, selection history and fallback decisions

Updated 2026-09-15. This is offline decision evidence; no production GP or acquisition
strategy is promoted. The user's decision boundary is now **before generating the first
hybrid proposal**, so ingest Group 10 with `--skip-generate` while that decision is pending.

## Correction: previous viability selection did not require mechanical pairs

The frozen metadata identifies Rounds 2–8 as `screening_only` with
`support_aware_finite_pool_screening`, and Rounds 9–10 as `mechanics_bootstrap`
with `bootstrap_utility_diversity`. Their continuous qLogNEHVI path was disabled.
Round 1 has a preserved legacy scoring branch in code; it has no equivalent original
selection-metadata file in this checkout.

The viability screening score starts with `mean + 0.35 * SD`, using the endpoint GP's
eligible viability data, including viability-only screens. Support-dependent uncertainty
caps, chemistry/support/intact deductions and slate-level constraints then affect selection.
Bootstrap mechanics combines screening utility, empirical intact evidence and diversity,
plus the frozen Group 10 protected roles. These stages do not depend on having eight pairs.

The paired-only limitation described earlier belongs to the separate BoTorch model
fitted in the hybrid/full-mechanics acquisition path. It does not describe how the
campaign selected all its earlier viability candidates.

## What the feasible pool is

A set of proposed recipes that pass implemented ingredient availability, concentration,
chemistry and other screening constraints. It is not a set of wet-lab-validated candidates.
Some recipes are new; retest or follow-up recipes may have prior observations. A rule-pass
is not proof of solubility, successful fabrication, viability or mechanical performance.
The final selector also applies support, diversity, repeat and experimental-role policies.

## Required BoTorch acquisition; no heuristic fallback

Updated policy: production mechanics acquisition must use BoTorch. Missing torch,
gpytorch or botorch raises an error and stops selection. Failed or non-finite finite-pool
qLogNEHVI scores also stop selection; the UCB-product proxy has been removed.
Continuous optimization may still retry genuine BoTorch scoring on a finite pool.
The offline acquisition prototype is research code, not a production fallback backend.
Screening/bootstrap scoring remains its explicit phase strategy, not a substitute for
failed mechanics acquisition. Group 10's frozen slate remains unchanged.

For a plain-language explanation of every audit column and model label, see
[audit_reading_guide.md](audit_reading_guide.md).

## Expanded audit

The default `--suite expanded` keeps historical comparisons and adds 22 viability
sensitivity configurations / 21 mechanical configurations. The catalogue itself is the
source of truth for counts and exact settings. Most cases change one factor relative to
`sens_baseline`; the two fitting-stability cases compare against `sens_bounds_learned`.

Audited factors: campaign versus mixed sources; empirical-standard versus physical-bound
scaling; Matérn 1.5/2.5 versus RBF; short/long/fitted shared length scale; fitted per-ingredient
length scales (ARD); signal amplitude; mean versus median prior center; recent three
batches versus all prior batches; paired-only training versus all endpoint labels;
recorded, fixed and fitted shared observation noise; logit viability/log force targets;
combined bounds/learned models; wider fit bounds and optimizer restarts.

Noise sensitivities are 1, 3, 5 and 10 **viability percentage points**, or endpoint-relative
force sensitivities. The force model never receives a viability noise floor in Newtons.
The fitted-noise sensitivity uses bounded maximum likelihood, not a fully Bayesian noise
prior. Transform cases use a delta-method observation-noise approximation, monotone
transformed intervals and quadrature for original-unit predictive moments. Logit inputs
are clipped to 0.5–99.5%, log-force inputs to at least 1e-6 N/needle, explicitly as audit
assumptions. They are not production endpoint changes.

Each fit uses only preceding outcomes. The original proposal identities determine the
comparison slate. Input hashes, exact settings, fitted kernels, training-row counts and
warnings are retained. Future intervals add the stated observation-noise assumption;
for mixed-source fits, future noise comes from preceding campaign evidence, not literature
noise. These intervals are assumptions to evaluate, not automatically calibrated intervals.

Outputs include:

- `sensitivity_settings.json`: exact predeclared scenarios.
- `sensitivity_comparisons.csv`: matched-row MAE and prediction changes.
- `predictions.csv`: means, latent SDs, latent/future intervals and fit details.
- `metrics.csv`: round/cohort errors, ranking, interval coverage and widths.
- `ucb_sensitivity.csv`: kappa 0, 0.35, 1 and 2 on measured historical slates only.
- `skipped.json` and `failures.json`: insufficient evidence versus genuine fit failures.

This is not an exhaustive parameter grid or a nested, unbiased selection of a winning
model. Testing many alternatives on the same outcomes can overfit strategy selection.
Freeze the shortlisted strategies before a future round for confirmation.

### Major questions not answered by this prediction audit

Correlated-output transfer and batch-effect normalization need more appropriate paired/
repeated-batch evidence. Full qLogNEHVI versus qLogNParEGO, candidate-pool generation,
feasibility and diversity policy comparisons need a common prospective candidate pool;
the outcomes of alternative untested recipes cannot be inferred. Fully Bayesian priors
are also not benchmarked here. These gaps are listed in `audit_scope.json`. The UCB file
reranks the already measured slate without replaying production constraints; it does not
show what an alternative complete optimizer would have discovered.

## Noise interpretation and provisional recommendation

An SD of 1 percentage point supplies variance 1; an SD of 5 supplies variance 25. In
normalized target coordinates, that variance must be divided by the target scaling
variance. BoTorch `train_Yvar` takes variance, not SD. Larger assumed observation noise
makes a GP trust a measured fluctuation less and generally widens future-observation
intervals. It cannot remove systematic source bias or manufacture biological independence.

If a mean is based on n independent preparations with preparation SD s, its sampling
variance is s²/n under that independence assumption. Four technical readouts from one
preparation are not four independent preparations; they do not justify dividing the full
preparation/batch variance by four. Missing independence metadata must stay unknown.

For the next fixed-noise campaign-only challenger, use **5 percentage points as a
provisional effective observation SD**, keeping 3 and 10 as sensitivity cases. This is
an operational assumption, not an experimentally measured error estimate. Compare it
with a fitted single noise level. Do not retain 1 point as unquestioned precision or
choose 10 solely because it reaches nominal retrospective coverage. No live noise value
has been changed by the audit.

## Actual Group 10 recipe example (offline refits, not frozen predictions)

Formulation #2 is ectoin 0.240084 M + ethylene glycol 1.065174 M. Formulation #6 is
0.411098 M + 1.091449 M respectively. The changes are 0.171014 M ectoin and 0.026275 M EG.

Using preceding campaign data through Group 9, their differences in standardized input
coordinates are about 1.230 and 0.051. Dividing by registry spans (0.5 M ectoin, 2.5 M EG)
instead gives 0.342 and 0.0105. At length scale 1 the bounds-scaled model therefore treats
these two recipes as closer. A learned scale can reverse that: the fitted shared bounds
length is about 0.149 in this snapshot. All training recipes influence a prediction; this
pairwise distance is only an illustration of the changed geometry.

| Offline fit | Predicted viability #2 | Predicted viability #6 |
|---|---:|---:|
| Current mixed-source assumptions | 65.32% | 82.43% |
| Campaign-only, current standard scaling, noise 1 point | 65.54% | 74.94% |
| Same campaign data/noise, bounds scaling only | 61.13% | 65.74% |
| Bounds scaling + fitted shared length/amplitude | 65.52% | 62.31% |
| Standard scaling, noise changed only to 5 points | 63.83% | 62.64% |

These are raw model calculations using Group 9 and earlier outcomes. They are not Group
10 experimental results, not promises of accuracy, and not replacements for its frozen
proposal values. They show that source, input geometry and noise can change both magnitude
and ranking. A narrower interval after scaling is not automatically a better interval.

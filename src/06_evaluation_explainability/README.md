# Step 6: Evaluation and Explainability

## Overview

This module groups the post-update analysis tools:

| Script | Purpose | Main outputs |
|--------|---------|--------------|
| `evaluate_iterations.py` | Score each frozen stage against the wet-lab batch it actually produced | `results/evaluation/` |
| `explainability.py` | Visualize what the active model is learning | `results/explainability/<iteration_tag>/` |
| `stage_r2_predicted_vs_actual.py` | Generate stage-indexed prospective/cumulative wet-lab R² scatter plots | `results/explainability/stage_r2/` |

Both scripts use the same iteration-aware model resolution used by
`03_optimization` and `05_bo_optimization`.

## Usage

```bash
cd "/path/to/project"

# Stage-based evaluation
python src/06_evaluation_explainability/evaluate_iterations.py

# Model explainability
python src/06_evaluation_explainability/explainability.py

# Model explainability (alternate palette profile)
python src/06_evaluation_explainability/explainability.py --palette-profile legacy

# Stage-indexed wet-lab R² plots (default: prospective cumulative)
python src/06_evaluation_explainability/stage_r2_predicted_vs_actual.py
```

## Shared Inputs

- `models/model_metadata.json`
- `data/validation/iteration_history.json`
- `models/iteration_*`

For the active iteration, `06_evaluation_explainability` resolves the exact
checkpoint from metadata plus iteration history. If the active metadata is
missing or inconsistent, `explainability.py` can prompt for an iteration number
and repair `models/model_metadata.json` with an explicit overwrite notice.

When the canonical observed-context artifact is missing, `explainability.py`
reconstructs it from literature plus wet-lab inputs. It can also read the
`data/processed/evaluation_data.csv` mirror when needed.

## Stage Evaluation

`evaluate_iterations.py` compares each frozen stage against the later wet-lab
rows that stage actually generated:

- result files without an iteration suffix map to the literature-only stage
- `iteration_1_*` outputs map to the first post-validation wet-lab batch
- `iteration_2_*`, `iteration_3_*`, and later outputs follow the same stage ID rule
- when available, the literature baseline is loaded from `models/literature_only/`

Outputs:

- `results/evaluation/iteration_prospective_summary.json`
- `results/evaluation/iteration_prospective_metrics.csv`
- `results/evaluation/stage_performance.png`
- `results/evaluation/single_objective_progress.png`
- `results/evaluation/single_objective_progress_metrics.csv`

The evaluator reports:

- batch-level predictive metrics such as RMSE, MAE, Spearman, Kendall, coverage, and hit rates
- candidate-rank cross references showing which frozen candidate rows were later tested in wet lab
- recommendation-slate evaluation for `results/next_formulations/<iteration_tag>/next_formulations.csv`, including exploit/explore and origin-level summaries when those files exist
- calibration-aware prediction scoring using active model metadata (`bias_shift_percent`, `uncertainty_scale`) so `06` matches `05` and `07` uncertainty conventions

`stage_performance.png` is a categorized small-multiples dashboard rather than
a 3-metric summary chart. It groups stage metrics into:

- error: RMSE and MAE
- ranking: Spearman rho and Kendall tau
- calibration: mean uncertainty and coverage @ 1σ
- threshold decision: hit rate @ 50% and hit rate @ 70%

Each metric gets its own raw-value bar-chart subplot, with the category
groupings preserved in the overall layout. Missing stages render as `N/A`
annotations instead of bars.

Candidate-hit matching uses the same practical concentration floor as `05` and
`07`:

- `_pct` values `<0.1%` are treated as absent
- `_M` values `<0.001 M` (`<1.0 mM`) are treated as absent

This means a frozen candidate row can count as a wet-lab hit in subsequent stages when
the only difference is a trace ingredient that should effectively be zero.

Additional evaluation outputs:

- `results/evaluation/next_formulations_performance.png`

The recommendation-slate audit rescales the saved `07` rows with the frozen
stage model inside `06`, then compares them with subsequent wet-lab measurements.
It reports:

- overall `07` slate performance
- `exploit` versus `explore` summaries
- origin-level summaries such as `bo_candidate`, `local_rank_probe`, `blindspot_probe`, and `explore_fallback`

## Explainability

`explainability.py` generates iteration-specific artifacts under:

- `results/explainability/<iteration_tag>/`

`<iteration_tag>` comes from the resolved active model identity, for example:

- `iteration_1`
- `iteration_3_weighted_simple`
- `iteration_8_prior_mean`

If no explicit iteration metadata exists, the fallback directory is:

- `results/explainability/active_model/`

The explainability suite is intentionally support-aware:

- slice and contour axes default to observed quantile bounds instead of raw extrema
- observed literature and wet-lab rows are overlaid wherever support matters
- stronger-support regions are marked with dashed boundaries or line-style changes instead of masking the surface
- the BO landscape keeps the contour aesthetic, but documents which production penalties are included
- default colormaps use a warm, color-blind-friendly profile (`magma` / `cividis` / `viridis`);
  use `--palette-profile legacy` to apply the alternate palette profile (`RdYlGn` / `YlOrRd` / `viridis`)

Support cues:

- In `partial_dependence_plots.png`, dashed curve segments indicate the same empirical slice continued outside stronger local 1D support.
- In `interaction_contours.png` and `acquisition_landscape.png`, the contrast-adaptive dashed boundary marks the stronger pairwise support envelope estimated from observed formulations.
- Inside that dashed boundary, the surface is better grounded in observed data. Outside it, the surface is shown for continuity, but should be interpreted as more extrapolative.
- In `feature_importance.png`, the vertical dashed line is only a visual dominance cutoff to separate the strongest features from the long tail; it is not a hard statistical threshold.
- In `shap_summary.png`, only the top features are shown. Color encodes feature value and horizontal spread shows directional contribution magnitude across observed formulations.

Artifacts:

| File | Description |
|------|-------------|
| `feature_importance.csv` | Recomputed permutation importance (weighted for composite model) |
| `feature_importance.png` | Publication-style overview of weighted permutation importance with dominant-feature emphasis |
| `shap_summary.png` | SHAP beeswarm focused on the top features and their directional impact on observed rows |
| `shap_importance.png` | SHAP-based feature importance ranking |
| `partial_dependence_plots.png` | Support-aware empirical marginal slices over observed rows, with dashed segments outside stronger local support |
| `interaction_contours.png` | Support-aware pairwise contour maps with observed-point overlays and dashed support boundaries |
| `acquisition_landscape.png` | Static BO score landscape using the `05` visual language, with support and sparsity penalties but no sequential batch-diversity term |
| `uncertainty_analysis.png` | Decision-focused uncertainty dashboard covering calibration, residual growth, and uncertainty by viability band |
| `support_diagnostics.png` | Compact support-envelope view for the top features and top pair, split by literature vs wet lab |

Feature importance is always recomputed at runtime against the resolved active
model. When using the composite model, weighting follows `context_weight` so
wet-lab rows influence importance consistently with the active iteration.

`acquisition_landscape.png` should be interpreted as a static approximation of
the production BO objective. It reuses the `05_bo_optimization` acquisition
settings and static penalties, but it does not include sequential
batch-diversity effects that depend on already-selected candidates.

## Stage-Indexed Wet-Lab R² Plots

`stage_r2_predicted_vs_actual.py` writes stage-indexed wet-lab R² plots to:

- `results/explainability/stage_r2/`

Default behavior is `--evaluation-mode prospective_batch`, implemented as
prospective cumulative staging:

- `iteration_0_wetlab_r2_predicted_vs_actual.png`: literature-only model on wet-lab stage `<=0`
- `iteration_1_wetlab_r2_predicted_vs_actual.png`: iteration-1 model on wet-lab stages `<=1`
- ...
- `iteration_7_wetlab_r2_predicted_vs_actual.png`: iteration-7 model on wet-lab stages `<=7`

The same script also writes:

- `literature_only_r2_predicted_vs_actual.png` (reported-vs-predicted on literature rows)

Plot styling details:

- x-axis: measured/reported viability; y-axis: predicted mean viability
- points are colored by prediction uncertainty (`std`) using `plasma` + colorbar
- annotation box includes `n` and `R²`

Alternative mode:

- `--evaluation-mode post_update_cutoff` uses each iteration model with wet-lab rows up to that model's `updated_at` date.

> `shap` is optional. If it is not installed, the SHAP plots are skipped while
> the other explainability outputs run.

## Feature Name Handling

Display labels are cleaned automatically:

- `dmso_M` -> `Dmso`
- `fbs_pct` -> `Fbs`
- `hyaluronic_acid_pct` -> `Hyaluronic Acid`

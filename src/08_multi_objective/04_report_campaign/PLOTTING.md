# Production plotting suite

All new exports are transparent-background PNGs at 300 dpi, intended for light backgrounds. Each PNG has a source CSV and metadata containing its source hash. No PDF export or alternate-design generator is retained.

| Artifact | Question answered | Evidence |
|---|---|---|
| `observed_tradeoff.png` (A) | What viability/load trade-off has been observed? | Feasible paired measurements define the frontier; failed or unknown intact status is separate. Predictions never define the observed frontier. Untrained mechanical fallback values are not reportable predictions. |
| `campaign_timeline.png` (B1) | How have tested outcomes, prediction error, interval coverage, and intact success changed? | Round-specific evidence with reconstructed/supplementary/formal provenance and explicit missing-round gaps. Error and coverage use separately labelled scales; crossings do not have scientific meaning. |
| `surrogate_trust.png` (C3) | How wrong, precise, and reliable were frozen predictions? | Formal eligible frozen predictions only. Pink signed-error violins retain every dot; blue interval widths show reported precision, green calibration shows coverage, and orange symbols show intact outcomes. |
| `candidate_decisions.png` (D) | Which candidates are proposed, and how are mechanical tests assigned? | Stored selection order, public predictions, empirical combination intact probabilities, primary/backup flags and ranks. Unknown predictions have no numerical position. Full identifiers and ingredient concentrations are in the CSV. |
| `diagnostics_1.png` | How do cross-validated models perform? | All-data viability/mechanics parity and intact classifier evaluations; the R² trend separately uses cumulative paired cohorts. |
| `diagnostics_2.png` | How do paired models and observed objective-space progress compare? | Paired-cohort parity; existing normalized hypervolume and IGD relative to the final observed frontier in the report. These are retrospective diagnostics, not proposal-time benchmarks or a known optimum. |
| `publication_summary.png` (E, optional) | What compact figure summarizes current campaign evidence? | Actual feasible pairs, formal prospective parity and tested batch outcomes. |
| `mechanical_trace_formulation_*.png` | Where were the sustained trigger, terminal point and apparent secant extracted on each raw curve? | Recorded trajectory with the fixed trigger-to-+0.8 mm window and source hash. |
| `mechanical_summary.png` | How do terminal force and apparent secant stiffness compare across tested formulations? | Every replicate, means, sample SD only for `n>=2`, and explicit lost-replicate annotations. |

The diagnostic intact classifier is distinct from the empirical combination probability that drives selection. Cross-validation is distinct from prospective evaluation. Negative R² values and extreme finite errors/interval widths remain visible. Missing mechanical evidence receives an explicit unavailable panel, not a fabricated frontier.

## Uncertainty and color

Signed error is prediction minus observation. Interval width is the full nominal 95% model interval in percentage points: wider means less precise. Coverage tests whether those intervals actually contain later results. These answer different questions; a wide interval can cover an inaccurate central prediction. Coverage shading uses descriptive Wilson intervals, not a dependence-adjusted guarantee for an adaptive campaign.

The selected violin renderer uses Scott bandwidth, stops at observed extrema, and falls back to dots/summaries with fewer than five observations or three distinct values. This is a rendering guard, not a statistical adequacy claim. Dark rose marks support the pastel pink fill. Blue circles/solid lines, vermilion squares/dashed lines, and green triangles/dash-dot lines distinguish B's metrics without depending on color alone.

Sources retained from the subsequent visualization revision (not attributed to Sol's original research):

- [Okabe and Ito, Color Universal Design](https://jfly.uni-koeln.de/color/): distinguishable colors supplemented by shape, line style, position and text.
- [Matplotlib violin plot documentation](https://matplotlib.org/stable/gallery/statistics/violinplot.html): configurable density summaries; observed points remain the evidence.
- [Seaborn categorical visualization guidance](https://seaborn.pydata.org/tutorial/categorical.html): distinguish raw observations from estimated distribution shapes. Seaborn is not a runtime dependency.
- [Machado, Oliveira and Fernandes (2009)](https://doi.org/10.1109/TVCG.2009.113): simulated color-vision previews used for visual QA, alongside grayscale checks.

## Generate and backfill

Normal proposal and completed-round entrypoints generate the selected suite automatically. Existing reporting arguments retain their meanings. The optional publication summary is enabled explicitly:

```sh
python3 src/08_multi_objective/04_report_campaign/report_campaign.py --include-publication-summary
```

Safely redraw existing reports from stored evidence, with no model fitting or canonical-table rewriting:

```sh
python3 src/08_multi_objective/04_report_campaign/report_campaign.py --restyle-existing
python3 src/08_multi_objective/04_report_campaign/report_campaign.py --restyle-existing --round ROUND_008
```

Backfill uses archived prospective aggregates and metrics, frozen proposal fields, and stored cross-validation rows. Single-round plots are labelled as such; the campaign report pools the campaign. Unsupported historical plots remain in their original style. In particular, absence of archived paired CV/R² inputs does not justify refitting past models. The backfill manifest records skips and source hashes. Legacy and superseded-policy archives remain untouched.

The corrected fixed-reference hypervolume helper retains its API: absent finite paired evidence returns `not_estimable` and a missing value; valid measurements contributing no area beyond the reference return `estimated` and zero. Its positive-area calculation is unchanged. The separate legacy normalized HV/IGD diagnostic calculation is preserved.

No optimizer policy, canonical observations, frozen proposals, or compatibility selection bridges are changed by this plotting integration.

Mechanical figures use the same selected publication theme and bundle contract:
transparent 300 dpi PNG, source CSV and metadata JSON. Generate these during
completed-round reporting after ingestion. Pre-ingestion staging uses
`run_round.py --validate-only` without an output directory and does not generate
mechanical graph bundles.

## GP strategy comparison bundles

`gp_strategy_summary`, `gp_strategy_by_batch`, `reference_conditioning_comparison`,
`batch_effect_estimates` and `gp_prediction_comparison` use the same theme and bundle
contract. Source CSVs retain model, stage, evidence and conditioning provenance.
Reference-conditioned accuracy uses matched non-reference outcomes only. Batch offsets
are model estimates, not measured cell health. Sparse calibration and absent outcomes
are explicit; no winner is selected automatically. `compare_gp_strategies.py render`
redraws saved evidence without model fitting.

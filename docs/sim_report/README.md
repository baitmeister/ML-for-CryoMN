# Simulated Multi-Objective Runs

This directory contains the high-level reporting outputs from the fake
multi-objective scenario study. Raw simulation artifacts are excluded from the
worktree.

## Included reports

- [simulation_evaluation.md](simulation_evaluation.md)
  Overall summary of the simulated campaign behavior, phase switching, ingredient tendencies, and interpretation notes.
- [scenario_comparison_report.md](scenario_comparison_report.md)
  Cleaner A/B/C comparison focused on the scenario parameters that were changed and the likely ramifications.

## Included figures

- [scenario_metric_progression.png](scenario_metric_progression.png)
  Cross-scenario progression of predicted and observed viability/load across the 8 simulated rounds.
- [phase_transition_diagnostics.png](phase_transition_diagnostics.png)
  Paired-label accumulation and the round at which each scenario switched into mechanics-enabled mode.
- [ingredient_selection_heatmap.png](ingredient_selection_heatmap.png)
  Relative ingredient selection frequency across scenarios.

## Excluded artifacts

The fake-run generator scripts and raw scenario artifacts under
`results/multi_objective_v2/simulations/` are excluded. This directory retains
only report-ready Markdown and comparison figures.

## Main takeaways

- The simulated scenarios use shared round-wise candidate pools, so their divergence reflects different fake wet-lab assumptions rather than different candidate-generation draws.
- The balanced scenario remained the most viability-favorable.
- The stricter or more mechanics-favoring scenarios activated mechanics-enabled selection at different round IDs or with different endpoint tradeoffs.
- These outputs are simulation audits only. They are not the real experimental multi-objective figures for the v2 workflow.

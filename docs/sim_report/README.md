# Simulated Multi-Objective Runs

This directory preserves the high-level reporting outputs from the fake multi-objective scenario study after the raw simulation artifacts were removed from the worktree.

## Included reports

- [simulation_evaluation.md](/Users/bait/.codex/worktrees/518c/ML-for-CryoMN/docs/sim_report/simulation_evaluation.md)
  Overall summary of the simulated campaign behavior, phase switching, ingredient tendencies, and interpretation notes.
- [scenario_comparison_report.md](/Users/bait/.codex/worktrees/518c/ML-for-CryoMN/docs/sim_report/scenario_comparison_report.md)
  Cleaner A/B/C comparison focused on the scenario parameters that were changed and the likely ramifications.

## Included figures

- [scenario_metric_progression.png](/Users/bait/.codex/worktrees/518c/ML-for-CryoMN/docs/sim_report/scenario_metric_progression.png)
  Cross-scenario progression of predicted and observed viability/load across the 8 simulated rounds.
- [phase_transition_diagnostics.png](/Users/bait/.codex/worktrees/518c/ML-for-CryoMN/docs/sim_report/phase_transition_diagnostics.png)
  Paired-label accumulation and the round at which each scenario switched into mechanics-enabled mode.
- [ingredient_selection_heatmap.png](/Users/bait/.codex/worktrees/518c/ML-for-CryoMN/docs/sim_report/ingredient_selection_heatmap.png)
  Relative ingredient selection frequency across scenarios.

## What was removed

The fake-run generator scripts and the raw scenario artifacts under `results/multi_objective_v2/simulations/` were intentionally cleared from the worktree. Only the report-ready markdown and comparison figures were preserved here.

## Main takeaways

- The simulated scenarios now use shared round-wise candidate pools, so their divergence reflects different fake wet-lab assumptions rather than different candidate-generation draws.
- The balanced scenario remained the most viability-favorable.
- The stricter or more mechanics-favoring scenarios switched into mechanics-enabled selection earlier or with different endpoint tradeoffs.
- These outputs are simulation audits only. They are not the real experimental multi-objective figures for the v2 workflow.

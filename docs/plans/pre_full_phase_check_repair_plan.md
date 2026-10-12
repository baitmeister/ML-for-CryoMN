# Full-phase search repair: must-dos

Status: implemented and validated on `pre-full-phase-check`, 12 October 2026. Campaign activation remains separate.

Objective: connect continuous concentration search to the approved models, then compare it fairly with the current finite-pool search.

1. **Keep the work isolated.** Use the existing `pre-full-phase-check` branch and saved post-Group-12 baseline at `results/multi_objective_v2/comparisons/pre_full_phase_check/baseline/`. Preserve the ingested results and official ROUND_013 proposal. The saved baseline includes uncommitted data that are absent from `main`.

2. **Use the same approved models and score.** Retain the campaign-only, control-informed batch-effect Matérn-5/2 viability GP (`control`, `batch_gp_v1`) and the existing comparable-endpoint mechanical GP (`current_comparable_endpoint`). Reuse their fitted posteriors for both searches, with the same qLogNEHVI, intact weighting and chemistry penalties. Do not fit a replacement GP inside the continuous path.

3. **Connect continuous search and preserve existing rules.** Start from feasible pool recipes, vary concentrations within fixed ingredient combinations, and validate the outputs using existing feasibility, availability, support and similarity rules. Retain the configured search budget initially: up to 16 combinations, at most four active ingredients per combination, and a target of 24 accepted continuous candidates. Merge valid refinements with the finite pool, rescore consistently, and apply the existing slate selection, including four mechanical primaries. Additional ingredient-combination ranking, pipetting constraints and concentration rounding are outside this repair.

4. **Make execution and failure explicit.** Keep finite-only replay available and introduce an explicit continuous-required comparison mode without rewriting historical decisions. If continuous optimization fails or produces no valid candidate, stop that run instead of silently falling back. Record what actually ran and label candidate origins correctly in metadata, summaries and plots. Continuous candidates need not win final slots.

5. **Verify the repair.** Check posterior and score agreement between paths, valid gradients, constraint enforcement and failure handling. Exercise a real continuous run with the approved GP, and run the relevant existing regression tests. Verify that historical finite-only behavior remains available and campaign files remain unchanged.

6. **Run and report a matched ROUND_013 comparison.** First reproduce the frozen finite baseline. Compare it with finite-pool plus continuous refinement using identical post-Group-12 data, fitted models, starting pool, seed and scoring settings. Check reproducibility; report recipe changes, mechanical assignments, acquisition scores, feasibility, diversity and runtime. Save comparison tables, plots and run records under `results/multi_objective_v2/comparisons/pre_full_phase_check/`. Predicted gains are not measured wet-lab gains. Keep campaign activation separate from this comparison.

Done when the repaired search executes with the approved models, passes validation, reports its behavior accurately and has a reviewable comparison against the preserved finite-pool baseline.

Completed comparison: refinement produced 24 candidates, 20 survived existing filters and none displaced the frozen 12-row slate. Repeated refinement was reproducible. All 752 protected campaign files were unchanged. The retained mask ordering chose 16 single-ingredient sets, so this result does not assess continuous refinement of mixtures. The final report and receipts are in `results/multi_objective_v2/comparisons/pre_full_phase_check/matched_comparison_final/`.

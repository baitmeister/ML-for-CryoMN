# Group 11 control-informed strategy

Decision `group11-control-informed-user-v1`, effective Round 11, selects the
campaign-only, control-informed batch-effect Matérn-5/2 viability GP (`control`,
`batch_gp_v1`), the comparable terminal-force mechanical GP
(`current_comparable_endpoint`), independent-output finite-pool qLogNEHVI and
no additional performance thresholds. See the
[round progression register](../multi_objective_round_progression.md) for the
GP, search and allocation policies by round.

The recorded decision is `results/multi_objective_v2/gp_strategy_decision.json`. Its explicit exception permits one historical control batch only when the actual target round is Group 11. Zero-control history is rejected. For later rounds the normal two-control-batch requirement remains; two batches are an eligibility minimum, not proof of scientific calibration.

Controls contribute through a separate uncertain reference baseline and shared viability batch covariance. They are not ordinary formulation-response labels. The prior standard deviation remains 50 percentage points. No new-batch measurements are required for proposal generation. The mechanical GP remains independent, and the reference recipe's valid force measurement remains ordinary evidence.

For the Round 11 sparse-control exception, the workflow refits the model at reference-prior SDs 25, 50 and 100 points on the same training data and evaluates the same ordinary feasible candidate pool using identical qLogNEHVI settings (128 Monte Carlo samples, seed 42). It exports candidate scores and Group 10 shift estimates to `results/multi_objective_v2/group11_control_sensitivity/`. A maximum latent prediction change over 5 points or fewer than 15 shared top-20 candidates stops generation. These are review flags, not performance acceptance thresholds. No alternative official proposal is generated.

The acquisition posterior excludes fresh biological batch and assay variance; these remain part of future-measurement forecasts. A passed sensitivity check does not validate the control baseline or remove its uncertainty. Review methodology after Groups 11–12; do not automatically switch models.

Frozen proposals retain their decision and predictions. The `control` identifier
denotes the control-informed viability GP independently of its display name.
Version-1 acquisition `shared_qlognehvi_finite_pool` remains finite-only.
Continuous refinement requires a separate version-2 decision with acquisition
`shared_qlognehvi` and search mode `continuous_required`; a phase transition
does not alter the search mode.

# Group 11 control-informed strategy

The user selected Proposed GP — campaign-only + batch effect, informed by viability controls; Current GP mechanics on comparable terminal-force labels; independent-output qLogNEHVI over the feasible pool; and no new performance thresholds.

The recorded decision is `results/multi_objective_v2/gp_strategy_decision.json`. Its explicit exception permits one historical control batch only when the actual target round is Group 11. Zero-control history is rejected. For later rounds the normal two-control-batch requirement remains; two batches are an eligibility minimum, not proof of scientific calibration.

Controls contribute through a separate uncertain reference baseline and shared viability batch covariance. They are not ordinary formulation-response labels. The prior standard deviation remains 50 percentage points. No new-batch measurements are required for proposal generation. The mechanical GP remains independent, and the reference recipe's valid force measurement remains ordinary evidence.

Before selection, the workflow refits the model at reference-prior SDs 25, 50 and 100 points on the same training data and evaluates the same ordinary feasible candidate pool using identical qLogNEHVI settings (128 Monte Carlo samples, seed 42). It exports candidate scores and Group 10 shift estimates to `results/multi_objective_v2/group11_control_sensitivity/`. A maximum latent prediction change over 5 points or fewer than 15 shared top-20 candidates stops generation. These are review flags, not performance acceptance thresholds. No alternative official proposal is generated.

The acquisition posterior excludes fresh biological batch and assay variance; these remain part of future-measurement forecasts. A passed sensitivity check does not validate the control baseline or remove its uncertainty. Review methodology after Groups 11–12; do not automatically switch models.

Historical frozen evidence remains unchanged. New reports use the clearer model display name while retaining the existing `control` strategy identifier.

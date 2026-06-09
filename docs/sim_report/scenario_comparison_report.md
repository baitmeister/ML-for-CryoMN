# Scenario Comparison Report

## Setup

- All scenarios now use the same shared candidate pool for each round. That removes candidate-generation differences as a confounder and makes the comparison a cleaner A/B/C test of the fake wet-lab response assumptions.
- The scenario differences therefore come from the response-model parameters and each scenario's wet-lab noise seed, not from different proposed candidate pools.

## Tweaked Parameters

- `balanced_learning`: viability_bias=5.0, viability_noise=4.0, intact_bias=0.05, intact_round_gain=0.015, intact_threshold=0.58, mechanical_round_gain=0.02, load_bias=0.12, load_noise=0.03, stiffness_bias=0.2, stiffness_noise=0.08.
- `fragile_process`: viability_bias=-6.0, viability_noise=6.0, intact_bias=-0.06, intact_round_gain=0.02, intact_threshold=0.66, mechanical_round_gain=0.035, load_bias=0.16, load_noise=0.05, stiffness_bias=0.25, stiffness_noise=0.12.
- `mechanics_tradeoff`: viability_bias=-2.0, viability_noise=5.5, intact_bias=0.02, intact_round_gain=0.01, intact_threshold=0.6, mechanical_round_gain=0.03, load_bias=0.18, load_noise=0.04, stiffness_bias=0.28, stiffness_noise=0.1.

## Parameter Intent

- `viability_bias` shifts the whole viability landscape up or down.
- `viability_noise` broadens within-round scatter and makes viability look less uniformly optimistic.
- `intact_bias` and `intact_round_gain` control how easily formulations pass the screening gate and how fast that process improves with rounds.
- `intact_threshold` is now part of the intact-pass rule, making some scenarios stricter about when a formulation is considered fabrication-ready.
- `load_bias`, `load_noise`, and `mechanical_round_gain` control mechanical baseline quality, spread, and late-round improvement.
- `stiffness_bias` and `stiffness_noise` control the secondary mechanical endpoint scale and dispersion.

## Comparison Findings

- `balanced_learning` reached mechanics-enabled selection at round `7` and finished with predicted viability `77.354` vs predicted load `0.7830`.
- `fragile_process` reached mechanics-enabled selection at round `6` and finished with predicted viability `60.454` vs predicted load `1.5150`.
- `mechanics_tradeoff` reached mechanics-enabled selection at round `6` and finished with predicted viability `56.147` vs predicted load `0.7118`.

## Selection Overlap

- `balanced_learning` vs `fragile_process` mean selected-formulation Jaccard overlap: `0.153`.
- `balanced_learning` vs `mechanics_tradeoff` mean selected-formulation Jaccard overlap: `0.130`.
- `fragile_process` vs `mechanics_tradeoff` mean selected-formulation Jaccard overlap: `0.130`.

## Ingredient Preference Signals

- `balanced_learning` favored `pvp_pct` (0.69), `dmso_M` (0.36), `fbs_pct` (0.34), `trehalose_M` (0.31), `dextran_pct` (0.25).
- `fragile_process` favored `pvp_pct` (0.67), `fbs_pct` (0.57), `dextran_pct` (0.55), `ectoin_M` (0.50), `sucrose_M` (0.45).
- `mechanics_tradeoff` favored `dmso_M` (0.72), `fbs_pct` (0.58), `trehalose_M` (0.57), `hsa_pct` (0.51), `proline_M` (0.51).

## Potential Ramifications

- Shared pools make cross-scenario conclusions much easier to trust, because any divergence in selected formulations now comes from the simulated wet-lab evolution instead of different candidate generation draws.
- Higher viability noise and round-level batch shifts produce a more realistic spread, but they also make short-run conclusions more sensitive to seed choice.
- Using `intact_threshold` as an active gate makes stricter scenarios slower to accumulate paired mechanics data, which is closer to a real fabrication bottleneck.
- Relaxing the earlier hard pass-for-mechanics shortcut reduces optimism in the phase switch; if a scenario still reaches mechanics-enabled mode, that is stronger evidence the switch logic is behaving sensibly.
- A higher mechanical ceiling avoids artificial saturation, so later-round mechanical gains can remain interpretable instead of flattening at the clamp.

# Fake Campaign Evaluation

## Phase transition

- `balanced_learning` switched to mechanics-enabled selection at round `7` after fabricated mechanics started at round `4`.
- `fragile_process` switched to mechanics-enabled selection at round `6` after fabricated mechanics started at round `4`.
- `mechanics_tradeoff` switched to mechanics-enabled selection at round `6` after fabricated mechanics started at round `4`.

## Scenario parameter differences

- `balanced_learning`: viability_bias=5.0, viability_noise=4.0, intact_bias=0.05, intact_round_gain=0.015, mechanical_round_gain=0.02, load_bias=0.12, load_noise=0.03, stiffness_bias=0.2, stiffness_noise=0.08, intact_threshold=0.58.
- `fragile_process`: viability_bias=-6.0, viability_noise=6.0, intact_bias=-0.06, intact_round_gain=0.02, mechanical_round_gain=0.035, load_bias=0.16, load_noise=0.05, stiffness_bias=0.25, stiffness_noise=0.12, intact_threshold=0.66.
- `mechanics_tradeoff`: viability_bias=-2.0, viability_noise=5.5, intact_bias=0.02, intact_round_gain=0.01, mechanical_round_gain=0.03, load_bias=0.18, load_noise=0.04, stiffness_bias=0.28, stiffness_noise=0.1, intact_threshold=0.6.

## Final round outcomes

- `balanced_learning` finished with predicted viability `77.354`, observed viability `83.475`, predicted load `0.7830`, and observed load `0.9513`.
- `fragile_process` finished with predicted viability `60.454`, observed viability `52.957`, predicted load `1.5150`, and observed load `1.4973`.
- `mechanics_tradeoff` finished with predicted viability `56.147`, observed viability `47.053`, predicted load `0.7118`, and observed load `0.9582`.

## Top selected ingredients by scenario

- `balanced_learning`: pvp_pct (0.69), dmso_M (0.36), fbs_pct (0.34), trehalose_M (0.31), dextran_pct (0.25)
- `fragile_process`: pvp_pct (0.67), fbs_pct (0.57), dextran_pct (0.55), ectoin_M (0.50), sucrose_M (0.45)
- `mechanics_tradeoff`: dmso_M (0.72), fbs_pct (0.58), trehalose_M (0.57), hsa_pct (0.51), proline_M (0.51)

## Findings

- The fake campaign harness originally generated CSV and JSON outputs only; it did not call any plotting/evaluation step, so the lack of graphs was an omission in the harness rather than a core optimizer failure.
- Mechanical loads are still capped in the fake generator, but the ceiling was raised to reduce the strong flattening seen in the earlier draft.
- Mechanical follow-up rows now get a probability boost instead of an automatic intact pass, which makes the paired-data accumulation less optimistic than the earlier draft.

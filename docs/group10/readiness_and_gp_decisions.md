# Group 10–11 roles and GP decision guide

**Group 10 uses frozen workflow v4/schema 4.** The viability-only reference
allocation applies from **Group 11**, workflow `group10_workflow_v5`, schema 5.
Decision `group11-control-informed-user-v1` selects the control-informed viability
GP effective Group 11. The [round progression register](../multi_objective_round_progression.md)
separates the model, search and allocation changes.

## Group 10 mechanical checklist

Mechanical rank is different from worksheet selection rank (the formulation number).
These are proposal-time primaries; actual-intact failures still promote the existing
ranked backups. Recipes below are rounded for identification; use the frozen worksheet
for preparation precision.

| Mechanical rank | Worksheet formulation number | Formulation ID | Recipe |
|---:|---:|---|---|
| 1 | 2 | `v2_3e8ef993abba` | Ectoin 0.240084 M + ethylene glycol 1.065174 M |
| 2 | 1 | `reference_b5a5b7b1176f` | DMSO 2.5% v/v + sucrose 0.100 M |
| 3 | 6 | `v2_211d2693ca45` | Ectoin 0.411098 M + ethylene glycol 1.091449 M |
| 4 | 7 | `v2_55f87bb2c839` | Ectoin 0.358521 M + ethylene glycol 1.533069 M + glycerol 0.123099 M |

## Reference scope from Group 11

- The 2.5% v/v DMSO + 100 mM sucrose formulation is **strictly a viability control**.
- It remains one of the twelve screening rows, with viability monitoring kept separate
  from ordinary viability-GP training and prospective viability scores.
- It has no mechanical rank, no primary mechanical recommendation and no backup rank.
  It receives no mechanical repeat exemption. There is no recurring mechanical control
  or reserved replacement anchor in Group 11+.
- All four mechanical formulation positions are experimental candidates. With an eligible
  screened-hit follow-up, the usual allocation is one screened hit plus three fresh
  candidates; insufficient eligible/intact candidates still produce an explicit shortfall.
- Any valid comparable mechanics already measured on the reference recipe, including
  the planned Group 10 test, are retained as ordinary mechanical endpoint evidence.
  They are not a mechanical baseline, calibration measurement, normalization factor or
  a reason to retest that recipe every round.
- Its viability is excluded from ordinary formulation-response labels and ordinary
  viability metrics. The Group 11 `control` GP uses it through a separate reference
  intercept and shared batch covariance. It is also the measured companion to its same-batch
  mechanical result. That pair counts normally for phase progression, paired acquisition
  and observed trade-off evidence. There is no mechanical-control exclusion.
- After all four valid Group 10 formulation pairs are ingested, the expected count is
  **8 pairs, 8 distinct formulations, 2 batches** (four from Group 9 plus four from
  Group 10). This satisfies the existing hybrid gate, so Group 11 can be hybrid.
  No threshold is lowered. Raw-trace validity, measured loaded count, same-batch
  viability and actual-intact execution requirements still apply.
- No raw measurements or frozen historical records are relabelled or erased.

## GP and acquisition by round

A Gaussian process (GP) estimates an endpoint's mean and uncertainty for a recipe.
An acquisition function uses a GP's predicted distribution to score the value of
running an experiment. **An acquisition function is not itself another physical endpoint.**

| Path | Group 10 production | Group 11 approved decision |
|---|---|---|
| Viability prediction | Pooled legacy/campaign sklearn Matérn-5/2 GP | Campaign-only, control-informed batch-effect Matérn-5/2 GP (`control`, `batch_gp_v1`) |
| Noise | Supplied per-row observation noise | Fitted total residual SD plus fitted batch SD; control baseline has SD 50 percentage points |
| Mechanics | Comparable terminal-force Matérn-5/2 GP | Same endpoint-model specification, fitted to eligible evidence |
| Acquisition used by the frozen proposal | Bootstrap utility/diversity; no qLogNEHVI | Genuine qLogNEHVI using the approved independent endpoint posteriors |
| Search used by the frozen proposal | Finite pool | Finite pool under the version-1 decision |
| Continuous search activation | Not used in bootstrap | Version-2 `continuous_required` enables shared-posterior refinement in hybrid/full mechanics |

The approved bundle contains **two endpoint-specific GPs with unequal training sets,
shared by reporting and acquisition**. It retains viability-only screening evidence.
Independent objective models do not establish learned biological correlation between
viability and mechanics. The separate fitted-BoTorch `ModelListGP` audit prototype
is not the implementation selected by `group11-control-informed-user-v1`.

Source selection (mixed legacy/campaign versus campaign-only), kernel/scaling, observation
noise, and acquisition architecture are separate decisions. `campaign_existing` and
`campaign_bounds_learned` in the strategy audit are prediction challengers; selecting either
one does not automatically install the separate `ModelListGP` acquisition prototype.

## What the decision boundary means

Generation for Group 11 and higher requires a recorded, supported GP strategy
decision. Use `--skip-generate` when ingestion is authorized without candidate
generation. The recorded choice is `group11-control-informed-user-v1`;
phase entry does not replace it. See the
[implementation and operating guide](gp_strategy_implementation.md) for supported
models, acquisition consistency, commands and the decision record.

The full gate requires 16 paired formulation–batch observations, 12 distinct
formulations and three batches; the hybrid gate requires 8, 6 and 2 respectively.
Technical replicates improve precision but collapse into a single formulation–batch row.
These are numerical phase-entry rules, not evidence of application success or model accuracy.

## Dynamic audit versus staged reports

Staged reports evaluate **frozen production proposal predictions** against measurements
from the corresponding experimental round.

`audit_group10.py` discovers observed rounds from canonical observations and their
available frozen proposals. The default starts at Group 3 and ends at the latest observed
round, with optional explicit bounds. It compares viability strategies and mechanical
strategies separately, using only preceding data at each fit cutoff. Mechanical definition
cohorts are kept separate. It records missing proposals, missing targets and insufficient
prior training in `skipped.json`, rather than scoring dummy forecasts.

Example from the project root (choose a new empty output directory each time):

```sh
python3 src/08_multi_objective/04_report_campaign/audit_group10.py \
  --output-dir /private/tmp/cryomn-gp-audit-after-group10
```

Use `--start-round 9 --end-round 10` to restrict a comparison, or
`--endpoints viability_percent` for viability only. Output contains prediction rows,
metrics by endpoint/definition/round/role, input hashes, skipped comparisons and failed fits.

This is **offline decision evidence**: it does not promote a GP, change acquisition,
modify observations or regenerate candidates. It cannot show what untested candidates
from another optimizer would have achieved. An audit rerun after outcomes are known is
not a prospective challenger trial, even though each refit excludes its test round.

For a prospective strategy comparison, freeze each challenger's predictions,
training cutoff, input/model settings and uncertainty definition without access to
that round's outcomes. Round 10 has a separate frozen challenger archive; Group 11+
generation freezes challenger predictions alongside its official slate. Compare bias, absolute
error, within-round ranking and uncertainty with simple campaign-mean/median baselines.
Returned model SD and historical spread are labelled diagnostics, not calibrated future-batch
prediction intervals. The mechanical audit does not apply viability noise floors to force.

## Verification and decision prerequisites

Workflow tests use isolated campaign states and identify the inserted experimental
batch explicitly. They cover the prerequisites for Group 10 generation and its
endpoint extraction. A blank proposal worksheet is a legitimate input state.

| Decision | Prerequisite |
|---|---|
| Force measurement definition, units and protocol | Freeze the definition as a condition of collecting measurements for a common endpoint. Group 9's addendum and Group 10+ use +0.8 mm from the sustained 1 N trigger. |
| GP comparison criteria and challenger predictions | Freeze them without access to the test round's outcomes. |
| Production GP/noise/acquisition/search choice | Record a supported decision as a condition of Group 11+ proposal generation. The approved version-1 decision selects finite search; version 2 can request continuous refinement. |
| Minimum acceptable viability and mechanics | Specify the requirements independently of the outcomes used to validate them. |
| Replicate/batch plan and pass/fail analysis | Predeclare the design for the independent confirmation experiment. |

Exploratory data collection can proceed with application thresholds unset. If thresholds
are chosen using these outcomes, these outcomes are exploratory for
that decision; use a subsequent independent experiment to validate the claim. Do not call
terminal compression force “fracture strength.” The legacy acceptance configuration names
must be bound to the actual endpoint definition. Requirements affect reporting
only, not candidate ranking or the hypervolume reference point.


## Expanded audit and fallback clarification

See [GP sensitivity and fallbacks](gp_sensitivity_and_fallbacks.md) for the actual historical
selection paths, expanded parameter audit, observation-noise examples and Group 10
recipe comparison. GP strategy is established by the recorded Group 11 decision;
the presence of an audit report or a completed phase gate does not deploy a strategy.
Search activation and the full-phase 1/1/1/9 allocation are documented in the
[round progression register](../multi_objective_round_progression.md).

# Group 10 plan and decisions before full mechanics

Updated 2026-09-15. **Group 10's frozen proposal and working worksheet are unchanged.**
The user is already planning its mechanical tests. The viability-only reference
allocation begins with **Group 11**, workflow `group10_workflow_v5`, schema 5.
The round-specific loader retains workflow v4/schema 4 for Group 10.

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
- Its viability remains excluded from standalone viability-GP training and ordinary
  viability metrics, but is retained as the measured companion to its same-batch
  mechanical result. That pair counts normally for phase progression, paired acquisition
  and observed trade-off evidence. There is no mechanical-control exclusion.
- After all four valid Group 10 formulation pairs are ingested, the expected count is
  **8 pairs, 8 distinct formulations, 2 batches** (four from Group 9 plus four from
  Group 10). This satisfies the existing hybrid gate, so Group 11 can be hybrid.
  No threshold is lowered. Raw-trace validity, measured loaded count, same-batch
  viability and actual-intact execution requirements still apply.
- No raw measurements or frozen historical records are relabelled or erased.

## What the two GP alternatives mean

A Gaussian process (GP) estimates an endpoint's mean and uncertainty for a recipe.
An acquisition function uses a GP's predicted distribution to score the value of
running an experiment. **An acquisition function is not itself another physical endpoint.**

| Path | Current production | Offline alternative |
|---|---|---|
| Endpoint predictions | Separate sklearn GPs for viability and mechanics; a further stiffness model is diagnostic | One BoTorch GP per objective, bundled in `ModelListGP` |
| Training rows | Each endpoint uses its available eligible labels; viability includes legacy and campaign evidence; mechanics uses one comparable endpoint definition | Viability and mechanics can use different input sets, retaining viability-only screens |
| GP assumptions | Input standardization, fixed Matérn-2.5 length scales, target normalization, supplied per-row observation noise | Registry-bound scaling and fitted BoTorch GPs; known endpoint noise variances may be supplied |
| Acquisition | Fits an additional two-output `SingleTaskGP` on paired rows only; it does not reuse the displayed prediction GPs | qLogNEHVI uses the same `ModelListGP` bundle that supplies predictions |
| Continuous versus finite-pool search | Separate fits; finite-pool scaling uses the train/candidate range, continuous search uses registry bounds | Both search paths reuse the same fitted bundle and scaling |
| Status | Live, unchanged by this update | Offline prototype; not automatically promoted |

Thus the proposed improvement is not merely “two GPs” (the prediction path already has
that). It is **two endpoint-specific GPs with unequal training sets, shared consistently
by reporting and acquisition**. This bundles independent objective models; it does not
establish learned biological correlation between viability and mechanics.

Source selection (mixed legacy/campaign versus campaign-only), kernel/scaling, observation
noise, and acquisition architecture are separate decisions. `campaign_existing` and
`campaign_bounds_learned` in the strategy audit are prediction challengers; selecting either
one does not automatically install the separate `ModelListGP` acquisition prototype.

## What the decision boundary means

Updated 2026-09-16: Group 11 and later generation requires a recorded, supported GP
strategy decision. Ingestion and reports remain available; use `--skip-generate` when
completing Group 10. No Proposed GP is promoted automatically. See the
[implementation and operating guide](gp_strategy_implementation.md) for supported
models, acquisition consistency, commands and the decision record.

The current full gate requires 16 paired formulation–batch observations, 12 distinct
formulations and three batches; the hybrid gate requires 8, 6 and 2 respectively.
Technical replicates improve precision but collapse into a single formulation–batch row.
These are numerical phase-entry rules, not evidence of application success or model accuracy.

## Dynamic audit versus staged reports

The existing staged reports already evaluate the **frozen production proposal predictions**
against subsequent measurements. Those reports should continue unchanged in purpose.

`audit_group10.py` now discovers observed rounds from canonical observations and their
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

For a genuinely prospective strategy comparison, freeze every challenger's predictions,
training cutoff, input/model settings and uncertainty definition before viewing the next
round's outcomes. Current staged reports freeze/evaluate production predictions; they do
not already provide that archive for all alternative strategies. Compare bias, absolute
error, within-round ranking and uncertainty with simple campaign-mean/median baselines.
Returned model SD and historical spread are labelled diagnostics, not calibrated future-batch
prediction intervals. The mechanical audit does not apply viability noise floors to force.

## Tests and latest decision dates

The two previously reported failures were stale test fixtures, not missing Group 10 results
or evidence of failed force extraction. One assumed that Group 9 was still un-ingested; the
other accidentally selected an old Group 9 result while checking a newly synthesized trace.
They now use isolated campaign states / isolate the inserted batch. Both pre-Group-10 and
ready-for-Group-10 states are tested. A blank Group 10 worksheet remains a legitimate state.

| Decision | Latest defensible point |
|---|---|
| Force measurement definition, units and protocol | Before collecting measurements to be pooled; the +0.8 mm after sustained 1 N terminal protocol is already frozen |
| Criteria for choosing GP strategies and frozen challenger predictions | Before seeing the outcomes of the round intended as their prospective comparison |
| Production GP/noise/acquisition choice | Before generating the first full-mechanics proposal; an explicit choice to retain the current model is also a decision |
| Minimum acceptable viability and mechanical performance | Before starting the independent experiment intended to establish that the final formulation meets requirements; preferably before designing full-scale tests |
| Replicate/batch plan and pass/fail analysis for that final claim | Before running that confirmation experiment |

You may continue exploratory Group 10 data collection while application thresholds are
unset. If thresholds are chosen using these outcomes, these outcomes are exploratory for
that decision; use a subsequent independent experiment to validate the claim. Do not call
terminal compression force “fracture strength.” The legacy acceptance configuration names
must be bound to the actual endpoint definition. Requirements currently affect reporting
only, not candidate ranking or the hypervolume reference point.


## Expanded audit and fallback clarification

See [GP sensitivity and fallbacks](gp_sensitivity_and_fallbacks.md) for the actual historical
selection paths, expanded parameter audit, observation-noise examples and Group 10
recipe comparison. The current operator decision is due **before the first hybrid
proposal**; use `--skip-generate` during ingestion while it is pending. This earlier
operator review does not rewrite frozen configuration fields or deploy a strategy.

# CryoMN round decisions

The supervisor reads each frozen proposal, records its priorities, then opens the validated results for that round. It starts with an empty decision history at the first full-mechanics round in this replay. The archived model retains its original training history.

## What the project learned from R13 through R19

The frozen viability top three averaged **63.72%**, compared with **42.63%** for three uniform picks from the same rankable nonreference slate (+21.08 percentage points). The shortlist recovered **17/21** observed top-three places. The comparison weights each of 7 completely observed rounds equally and uses the exact random expectation.

| Round | Intact / measured, all screens | Primary intact / measured | Viability top 3 / same-slate mean (%) | Viability ρ (n) | Force ρ (n) | Cumulative pairs | Frontier area |
|---|---:|---:|---:|---:|---:|---:|---:|
| R13 | 9/12 | 3/4 | 59.84 / 49.99 | 0.518 (11) | 0.800 (4) | 4 | 77.30 |
| R14 | 9/12 | 4/4 | 53.30 / 39.77 | 0.282 (11) | 0.800 (5) | 9 | 79.10 |
| R15 | 10/12 | 3/4 | 74.56 / 44.23 | 0.691 (11) | 0.200 (4) | 13 | 82.34 |
| R16 | 10/12 | 3/4 | 68.44 / 31.71 | 0.855 (11) | 0.800 (4) | 17 | 82.75 |
| R17 | 9/12 | 3/4 | 53.97 / 40.82 | 0.727 (11) | 1.000 (4) | 21 | 82.75 |
| R18 | 9/12 | 3/4 | 61.72 / 45.63 | 0.809 (11) | 0.200 (4) | 25 | 82.75 |
| R19 | 12/12 | 4/4 | 74.19 / 46.29 | 0.864 (11) | 0.600 (5) | 30 | 85.43 |

Each pair is an intact-passing, nonreference formulation with both endpoints measured in the same round. The frontier area is the area dominated by those observations above a fixed origin of 0% viability and 0 N/loaded needle. Its units are percentage points × N/needle. Historical compression measurements cover a selected subset.

| Frozen forecasts, pooled | n | Spearman ρ | Mean of within-round ρ | MAE | Nominal 95% interval hits | Mean width | Mean interval score |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSC viability (pp) | 77 | 0.542 | 0.678 | 13.10 | 65/77 | 52.51 | 132.61 |
| Terminal force (N/loaded needle) | 30 | 0.569 | 0.629 | 0.196 | 28/30 | 1.099 | 1.416 |

Intervals use the archived proposal mean ± 1.96 standard deviations. The interval score equals width plus 40 times the distance outside the bounds. Lower scores are better. Within-round correlations receive equal weight. The shortlist ranks viability alone.

## Which scientific questions could the archive answer?

The policy first requests paired evidence for the highest-viability intact formulation lacking force in the latest round. It also identifies the highest-viability point on the cumulative frontier with paired evidence in only one replayed round. It reserves a repeat slot when all previous primary patches passed formation. A missing result stays open.

| Round revealed | Prior screened hit on this slate | Prior frontier target on this slate | First-choice mechanical nominations with pairs / failed gate / unknown |
|---|---|---|---:|
| R13 | initial round | initial round | 3 / 1 / 0 |
| R14 | off slate | off slate | 3 / 0 / 1 |
| R15 | off slate | off slate | 1 / 1 / 2 |
| R16 | off slate | off slate | 2 / 1 / 1 |
| R17 | off slate | off slate | 2 / 1 / 1 |
| R18 | off slate | off slate | 3 / 0 / 1 |
| R19 | paired, extended prior frontier | off slate | 4 / 0 / 0 |

Of 28 first-choice mechanical nominations, the archive contains 18 matching paired results and 4 observed formation failures before any backup promotion. It leaves 6 unresolved. The other choices could have changed subsequent learning, so this replay cannot estimate the yield of an alternative campaign.

The question ledger contains 11 distinct target questions, including those raised at the latest close. Later matched measurements or formation failures resolved 1. Open requests are checked again when their exact formulation returns.

**R18 → R19.** 0.091205 M ectoin, 0.63738 M ethylene glycol, 8.0415% FBS passed formation at 64.63% viability. The frozen next-round forecasts were 65.08% and 0.478 N/needle. The researcher measured 82.66% and 0.500 N/needle. Individual force readings were 0.684, 0.316 N/needle. The decision record reads *paired, extended prior frontier*.

## Measured trade-offs still on the frontier

| Round | Formulation at recorded concentrations | Viability (%) | Terminal force (N/loaded needle) |
|---|---|---:|---:|
| R19 | 0.091205 M ectoin, 0.63738 M ethylene glycol, 8.0415% FBS | 82.66 | 0.500 |
| R15 | 0.12313 M ectoin, 0.78246 M ethylene glycol, 8.9547% FBS | 71.31 | 0.580 |
| R13 | 0.49001 M glycerol, 0.17661 M acetamide | 58.91 | 0.824 |
| R13 | 0.26462 M glycerol, 0.11542 M glucose, 0.01526 M creatine | 46.45 | 1.341 |

These are point estimates from selected batch means. Their order can change with independent preparations.

## R20 briefing before results

**Round priority** — Seek an independent batch for the leading measured trade-off. The archived slate has 12 screens and 4 primary mechanical slots, conditional on formation.

**Agent decision** — Keep the default allocation (accepted).

**Research question** — On fresh preparations from a new cell batch, do both recipes form, and does the ectoin repeat match or exceed the screened hit on viability (%) and terminal force (N/loaded needle)?

**Competing explanation** — Releasing the slot tests another optimizer choice. The ectoin result could reflect batch variation.

**If supported** — Keep the ectoin recipe as the comparator for the next round.

**If refuted** — Prioritize the hit if it matches or exceeds ectoin on both endpoints and leads on one. Retain both if each leads on one. Record failed formation for that attempt. Missing force leaves the joint comparison open.

**Critic** — Testing both formulations from the same new cell batch allows a direct comparison if each passes formation and has both endpoints measured. Keeping the repeat uses one slot to resolve uncertainty in the R19 force estimate. Record preparation and cell-batch IDs and both force readings. Any observed dominance applies to this batch. The missing R19 identifiers limit verification of independence, and no margin defines reproducibility or application success.

**Pair the screened hit.** Does this intact screened hit extend the prior measured frontier when both endpoints are measured together? The target is 0.81162 M propylene glycol, 0.24209 M raffinose. The prior round recorded 62.90% viability and left force unmeasured. Its position on the archived slate is *primary*. If a prior point matches or exceeds both means, retain the prior frontier. Otherwise add the new trade-off.

The archived forecast for selection rank 2 is 54.25% and 0.483 N/needle.

**Repeat the frontier formulation.** How much do both endpoints change for this exact frontier formulation in a separately prepared batch? The target is 0.091205 M ectoin, 0.63738 M ethylene glycol, 8.0415% FBS. The prior round recorded 82.66% viability and 0.500 N/needle. Its position on the archived slate is *off slate*. Record both shifts and preparation and batch identities. Apply a reproducibility margin only if the project has fixed one.

### How the new project would spend its mechanical budget

| Priority | Purpose | Formulation |
|---:|---|---|
| 1 | Pair the screened hit | 0.81162 M propylene glycol, 0.24209 M raffinose |
| 2 | Repeat the frontier formulation | 0.091205 M ectoin, 0.63738 M ethylene glycol, 8.0415% FBS |
| 3 | Optimizer selection | 0.2918 M glycerol, 0.13357 M glucose, 0.021204 M creatine |
| 4 | Optimizer selection | 0.84392 M propylene glycol, 0.27457 M raffinose |

The provisional portfolio has 12 distinct screens and 6 ordered mechanical backups. It substitutes 1 screen. Bench outcomes for this briefing are pending at the chosen cutoff.

The cost of reserving a slot is the original mechanical primary at screening rank 6 (0.13073 M ectoin, 0.5256 M ethylene glycol, 10% FBS). Its frozen forecasts were 64.20% and 0.502 N/needle.

# What did the agents add to the CryoMN project?

The [9-minute, 15-second recording](findings_read_aloud.m4a) has seven chapters. The direct comparison of the 28 mechanical nominations begins at 2:41. The Round 20 decision begins at 6:32.

## The answer in one minute

Your existing optimizer found the promising formulations. The agents helped decide what a follow-up experiment should establish, and what evidence would change the next decision. We have measured the first capability. The second is demonstrated in a planning decision, and its effect on experimental outcomes is still open.

There are three versions to keep in mind. The original optimizer generated the historical slates. A new rule-based supervisor replayed seven completed rounds, carrying unanswered questions forward. A scientist agent and a separate critic reviewed the proposed Round 20 plan. I ran that last comparison with and without their review. Both versions chose exactly the same four mechanical tests. What the agents added was a direct comparison, a competing explanation, and an explicit account of what each result would mean.

## What the original optimizer achieved

The existing software already ranks recipes, applies feasibility rules, proposes controls and follow-ups, validates the returned worksheet, scores its forecasts, and generates the next proposal. In Rounds 13 through 19, the lab screened 84 formulation–round entries. Seven were fixed references. All 77 other entries had a viability result. Thirty also had an intact patch and a mechanical result. Another 31 formed intact but lacked force data, and 16 failed formation.

Within each round, the original model’s three highest viability forecasts averaged 63.72% measured viability. Three random picks from the same tested slate would average 42.63%. That is a 21.08 percentage point advantage. Its first pick finished first in all seven rounds. Its three picks recovered 17 of the 21 available top-three places. The original model earned those results.

The model could identify a frontrunner even when its numerical forecast missed. Across all seven rounds, the Spearman rank correlations were 0.54 for viability and 0.57 for force. Average absolute error was 13.10 percentage points for viability and 0.196 newtons per loaded needle for terminal force. The model came within ten percentage points of viability for 40 of 77 cases. A forecast based on similar earlier recipes did so for 36. The broader future-batch intervals contained 70 of 77 viability results, and averaged 73 percentage points wide. That width helps explain why a high forecast still warrants bench confirmation.

## What changed when the supervisor chose the tests

The comparison uses the first four mechanical nominations in each round, before backups or extra tests. Across 28 nominations, the original optimizer produced 23 paired measurements and five formation failures. The replayed supervisor kept 22 of the same choices. Eighteen of those produced pairs, and four failed formation. It substituted six choices. All six lack a measured joint result in the corresponding round.

Five of the six original choices that would have been displaced produced pairs. Holding the shared results and historical rounds fixed, five of the six substitutions would therefore need to produce pairs just to tie the original yield. All six would need to do so to beat it. This calculation measures the chance to collect both endpoints. A repeat can still answer a valuable question even if it adds no new formulation to the frontier.

Round 15 shows what the supervisor might give up. The optimizer prioritized a recipe with about 0.123 molar ectoin, 0.782 molar ethylene glycol, and 8.95% fetal bovine serum. The lab measured 71.31% viability and 0.580 newtons per loaded needle. It remained one of the four best observed trade-offs through Round 19. The supervisor’s draft moved it to backup. A backup could still have been tested, but the original optimizer gave this useful candidate higher priority.

## What the lab learned about the two objectives

The highest-viability measured formulation reached 82.66%, with a terminal force of 0.500 newtons per loaded needle. At the other end of the trade-off, the strongest measured formulation reached 1.341 newtons per loaded needle, with 46.45% viability. The lab had four observed choices on the frontier. None of the four was surpassed on both endpoints by another measured formulation. Between Rounds 16 and 18, the number of paired measurements rose from 17 to 25 while the frontier stayed in place. Round 19 expanded it at the high-viability end.

The same reference formulation also varied. Its mean viability was 16.71% in Round 17 and 42.31% in Round 19. That makes batch variation a plausible explanation to investigate when a promising recipe improves on a later run. The archive cannot assign a cause to that change.

## How the agents reasoned about Rounds 18 and 19

Consider the formulation with about 0.0912 molar ectoin, 0.6374 molar ethylene glycol, and 8.04% fetal bovine serum. In Round 18 it formed intact and reached 64.63% viability. Its force was unknown. A critic who saw only the evidence through Round 18 recommended measuring both endpoints for this exact recipe. The original optimizer independently put it on the Round 19 mechanical slate.

The frozen forecasts were 65.08% viability and 0.478 newtons per loaded needle. The lab measured 82.66% and 0.500. The force mean hides a substantial difference between two patches. One gave 0.316 and the other 0.684 newtons per loaded needle. The preparation and cell-batch identifiers were blank. That evidence supports asking whether the result repeats on a separately documented preparation. The two systems agreed on the experiment that produced the discovery.

Across the full replay, the supervisor tracked 11 distinct follow-up questions. One was resolved by Round 19. The others remained visible, with missing measurements recorded as open questions.

## What the Round 20 review added

The original Round 20 plan already prioritized a propylene glycol and raffinose formulation that had reached 62.90% viability with force still unmeasured. The rule-based supervisor kept that choice and reserved a second slot to repeat the Round 19 ectoin leader. It kept the budget at 12 screens and four first-choice mechanical tests. Three of the four original mechanical choices stayed. The candidate moved to backup had forecasts of 64.20% viability and 0.502 newtons per loaded needle. That was the price of investigating repeatability.

The scientist proposed comparing the two formulations in the same new cell batch, with separately identified preparations. The critic accepted that design and explained its limits. If the ectoin repeat matches or exceeds the other recipe on both endpoints, retain it as a comparator. If the other recipe matches or exceeds ectoin on both and leads on at least one, prioritize it for confirmation. If each leads on one, retain both trade-offs. A missing force result leaves the question open. The project has yet to define how close a repeat must be to count as reproducible.

Round 20 is still awaiting validation. The rule-based supervisor changed one first-choice mechanical slot relative to the original optimizer. Adding the scientist and critic changed zero further slots. Their recorded contribution was a testable question and a decision rule the researcher could inspect before seeing the results.

## What would establish that the agents earn their place?

My conclusion is that the existing optimizer has earned credit for finding promising formulations. The clearest use for the agents so far is deciding how to distinguish a promising one-batch result from a reproducible one. The current prototype demonstrates that planning step. Automated intake and worksheet delivery for a new project still need to be connected. We have no measured improvement in viability, force, discovery rate, or researcher time attributable to the language agents.

To measure that contribution, compare the original optimizer, the optimizer with the rule-based supervisor, and the same supervisor with scientist–critic review. Keep the experimental budget, controls, and confirmation criteria the same. Freeze all three choices before validation. When choices differ, randomly assign the discretionary test across comparable batches. Count independently confirmed improvements in the viability–force trade-off per mechanical test, and record researcher time if time savings are part of the claim. That comparison would tell you whether the review changes the experiment and whether the change pays off.

---

### Numbers behind the recording

| R13–R19 frozen model forecasts | Viability | Terminal force per loaded needle |
|---|---:|---:|
| Nonreference cases with an eligible measurement | 77 | 30 |
| Pooled Spearman rank correlation | 0.542 | 0.569 |
| Mean of the seven within-round correlations | 0.678 | 0.629 |
| Mean absolute error | 13.10 percentage points | 0.196 N |
| Proposal-time 95% interval coverage | 65/77 | 28/30 |
| Mean proposal interval width | 52.51 percentage points | 1.099 N |

The 95% future-batch viability intervals covered 70/77 with mean width 73.15 percentage points. Their mean interval score was 103.41, compared with 132.61 for the proposal bands. Lower scores reward narrower intervals and penalize missed outcomes. At a tolerance of ±10 percentage points, the frozen model, similar-recipe baseline, and earlier-round mean achieved 40/77, 36/77, and 26/77 hits, respectively. At ±20 points the counts were 61/77, 56/77, and 47/77. The terminal force was measured at 0.8 mm beyond the sustained 1 N trigger and normalized by loaded-needle count.

| First-choice mechanical nominations across R13–R19 | Paired | Failed formation | Unknown |
|---|---:|---:|---:|
| Original optimizer, all 28 | 23 | 5 | 0 |
| Shared by both policies, 22 | 18 | 4 | 0 |
| Only in the original first choices, 6 | 5 | 1 | 0 |
| Only in the replayed supervisor’s first choices, 6 | 0 observed | 0 observed | 6 |

Under a fixed-round comparison holding the shared outcomes constant, the supervisor’s total could range from 18 to 24 pairs, versus 23 observed for the original optimizer. A campaign that actually ran the alternative choices could learn differently in subsequent rounds.

| Observed frontier through R19 | Viability | Terminal force (N/loaded needle) |
|---|---:|---:|
| R19 · 0.091205 M ectoin, 0.63738 M ethylene glycol, 8.0415% fetal bovine serum | 82.66% | 0.500 |
| R15 · 0.12313 M ectoin, 0.78246 M ethylene glycol, 8.9547% fetal bovine serum | 71.31% | 0.580 |
| R13 · 0.49001 M glycerol, 0.17661 M acetamide | 58.91% | 0.824 |
| R13 · 0.26462 M glycerol, 0.11542 M glucose, 0.01526 M creatine | 46.45% | 1.341 |

The planned R20 screened hit contains 0.81162 M propylene glycol and 0.24209 M raffinose. The original fourth mechanical choice moved to backup contains 0.13073 M ectoin, 0.5256 M ethylene glycol, and 10% fetal bovine serum.

The design borrows from experimental literature on adaptive allocation, independent confirmation, and critique of hypotheses. The numerical comparison above uses this campaign’s own frozen choices and measurements. [Project design and publications](README.md) · [23-figure guide](visuals/index.html) · [Round-by-round evidence](replay.md) · [Recorded scientist and critic decision](round20_agent_review.json) · [Original selector](../../src/08_multi_objective/02_select_candidates/README.md) · [Frozen evaluation table](../../results/multi_objective_v2/reports/prospective/tables/prospective_evaluation_table.csv)

# Stage 02: Select Candidates

## Purpose

Stage 02 generates and scores the candidate pool, resolves the evidence-gated
mechanics phase, exports a 12-row viability/intact slate, and freezes the exact
pre-bench proposal. Mechanical eligibility and rank are assigned only after the
screening slate has satisfied the common chemistry and diversity policies.

The canonical transition policy is documented in
[Evidence-gated mechanics transition](../README.md#evidence-gated-mechanics-transition).
The [round progression register](../../../docs/multi_objective_round_progression.md)
identifies the GP, search and policy versions used by each frozen proposal.

## Command and phase controls

```bash
python3 src/08_multi_objective/02_select_candidates/select_candidates.py
```

Useful options:

```bash
# Generate a larger finite candidate pool
python3 src/08_multi_objective/02_select_candidates/select_candidates.py \
  --pool-size 5000

# Score an externally prepared candidate pool
python3 src/08_multi_objective/02_select_candidates/select_candidates.py \
  --candidate-pool path/to/candidate_pool.csv

# Audit one phase without changing the automatic gate rules
python3 src/08_multi_objective/02_select_candidates/select_candidates.py \
  --phase-mode mechanics_hybrid
```

`--phase-mode` accepts `auto`, `screening_only`, `mechanics_bootstrap`,
`mechanics_hybrid`, and `mechanics_enabled`. `auto` is the operating mode.
Every explicit phase is an audit/debug override; proposal metadata records the
requested value and `override_used=true`.

`--batch-id` overrides the derived `ROUND_###` label. It does not satisfy the
screening-entry or mechanics-evidence gates. `--supersede-unstarted-proposal`
can replace a different frozen proposal only when the batch has no observations,
no completed archive, and no entered result values. The superseded proposal is
preserved with hashes and a reason manifest.

## Isolated search comparison

The Group 11 decision selects the campaign-only, control-informed batch-effect
Matérn-5/2 GP (`control`, `batch_gp_v1`), paired with the existing comparable-endpoint
mechanical GP. The historical version-1 decision remains finite-only.

With Round 12 ingested and an official Round 13 proposal available, run the matched
comparison in an unused output directory:

```bash
python3 src/08_multi_objective/02_select_candidates/compare_search.py \
  --output-dir results/multi_objective_v2/comparisons/pre_full_phase_check/my_comparison
```

This records hashes of the campaign inputs and official outputs at invocation, fits the
approved models once and reuses one starting pool. It requires the preserved
post-Group-12 data and official ROUND_013 proposal, but no previous comparison
outputs or baseline archive. It replays the source-quota allocator from commit `6b759c7`, then
compares it with the 1/1/1/9 allocator for finite and refined search using identical scored
pools. The source-quota finite slate must match the official proposal; its refined
slate is reconstructed from code rather than checked against a saved comparison.
It independently repeats refinement, checks screening/bootstrap/hybrid allocation
parity and verifies protected campaign hashes. The data baseline at commit
`0a0a7328` does not meet the Round 12 ingestion/Round 13 proposal prerequisites.
The runner exports recipes, decision plots, model
snapshots, a comparison report and execution receipts. It does not publish an
official round or change the active campaign decision.

For normal Stage 02 execution, a version-2 GP decision separates acquisition
(`shared_qlognehvi`) from search (`finite_pool` or `continuous_required`). Pass
that file through `--gp-strategy-decision`. Continuous-required needs active
hybrid/full mechanics and enabled continuous settings; optimizer failure or
zero surviving candidates stops selection. `continuous_search_failure.json`
retains optimizer diagnostics when available.

Continuous search uses feasible pool recipes as starting points. The configured
ingredient-set budget, chemistry, preparation, support, availability, similarity,
special-row and mechanical allocation policies remain in force. It adds no
pipetting or rounding constraints. `candidate_origin` retains selection groups;
`generation_origin` preserves the original source and is shown in decision plots.
All eligible ingredient sets are scored by their best feasible recipe using the
shared acquisition and penalties. The best set per ingredient count is reserved,
then remaining slots are filled by score, within the configured budget. This
prevents single-ingredient sets from exhausting the budget before mixtures are
considered. Comparison runs verify actual search coverage and export
`ingredient_set_ranking.csv`, including scores, choices and optimizer attempts.

## Inputs and outputs

Inputs:

- `data/processed_v2/formulations.csv`
- `data/processed_v2/observations.csv`
- `config_v2/ingredients.yaml`
- `config_v2/optimization.yaml`
- `config_v2/availability.yaml`

Mutable working outputs:

- `results/multi_objective_v2/total_candidate_pool.csv`
- `results/multi_objective_v2/current_round_status.json`
- `results/multi_objective_v2/next_round/next_round_candidates.csv`
- `results/multi_objective_v2/next_round/next_round_summary.txt`
- `results/multi_objective_v2/next_round/next_round_metadata.json`

Frozen proposal outputs:

- `results/multi_objective_v2/rounds/ROUND_###/proposal/proposal.csv`
- `results/multi_objective_v2/rounds/ROUND_###/proposal/summary.txt`
- `results/multi_objective_v2/rounds/ROUND_###/proposal/selection_metadata.json`
- `results/multi_objective_v2/rounds/ROUND_###/proposal/plots/`

`total_candidate_pool.csv` is a scored audit pool, not a wet-lab worksheet.
Only `next_round_candidates.csv` is filled by the operator. Editable result
fields are blank for every proposal row, including retests and mechanics
anchors. Source measurements stay in the observation database.

## Phase resolution

The resolver receives the target proposal round for provenance and derives the
completed-screening count and paired mechanics evidence from the observation
database. A screening round counts as complete when one `ROUND_###` batch has
measured viability and intact endpoints. A mechanical pair is one
formulation–batch training row with both viability and critical load under the
same batch ID. Technical replicates aggregate before counting.

| Phase | Automatic condition |
|---|---|
| `screening_only` | Completed screening rounds `< 8` |
| `mechanics_bootstrap` | Screening entry met and the hybrid gate is incomplete |
| `mechanics_hybrid` | Paired observations `>= 8`, distinct paired formulations `>= 6`, paired batches `>= 2`; full gate incomplete |
| `mechanics_enabled` | Paired observations `>= 16`, distinct paired formulations `>= 12`, paired batches `>= 3` |

The phase metadata contains the target proposal round, completed-screening
count, paired count, distinct formulation count, batch count, both gate
requirements and results, bootstrap batch index, policy version, resolution
reason, and override flag.

There is no special-case `ROUND_009` branch in Stage 02. Under the default
entry requirement, eight completed screening rounds cause the following
proposal to resolve as bootstrap when neither mechanics evidence gate is met.
To change that behavior, edit `mechanics_transition.entry`, `hybrid_gate`, or
`full_gate` in `config_v2/optimization.yaml` and keep `phase_mode: auto`.
Detailed safe-change guidance is in the
[canonical transition policy](../README.md#changing-the-transition-policy).

## Common screening selection

Every phase exports 12 viability/intact rows. Candidate generation and slate
selection preserve:

- registry bounds and formulation feasibility;
- temporary ingredient availability;
- observation-supported distance controls;
- unified history and within-pool formulation similarity;
- phase-specific role reservations and source allocation; full-phase ordinary
  positions have no origin quotas;
- exact ingredient-combination, shared-pair, and marginal-frequency caps;
- cold-start cap, with graduation allocation governed by the phase-specific selector;
- rescue dilution and evidence-based retest reservations.

The screening acquisition uses viability upper confidence bound and the
configured chemistry, support, and exact-combination intact deductions:

```text
viability UCB = predicted viability mean + 0.35 × predicted viability SD

screening_phase_score =
  minmax(viability UCB)
  - chemistry penalties
  - support penalty
  - empirical intact-combination deduction
```

For exact active-ingredient-set evidence with weighted passes `P` and failures
`F`:

```text
p_combo = (1 + P) / (2 + P + F)
intact deduction = 0.20 × max(0, (0.50 - p_combo) / 0.50)
```

The additive intact classifier remains a diagnostic. It does not filter the
active screening or mechanical ranking.

## Mechanical eligibility and repeat filtering

The [automatic repeat and rescue rules](../README.md#automatic-repeat-and-rescue-rules)
apply before allocation. Two cumulative failed intact batches retire an exact
recipe, and one completed automatic viability confirmation exhausts its retest
allowance. Completed confirmations require a frozen `retest_priority` role plus
finite measured viability, including historical confirmations. Mechanics follow-up
and shared retest/rescue slots are optional; unfilled capacity returns to ordinary
acquisition ranking. Counts and source batch IDs are recorded in proposal metadata.

Mechanical ranking considers only rows inside the finalized 12-row slate.
Ordinary eligibility requires all of the following:

- numeric, feasible, supported composition under the common slate rules;
- no `retest_priority` recommendation;
- no prior critical-load observation for the formulation;
- no disallowed exact mechanical repeat.
- fewer than two cumulative failed intact batches.

The one exception is an eligible second-bootstrap-batch anchor. Screening rows
that fail mechanical eligibility retain blank mechanical rank, role, mode, and
backup status. They cannot provide Instron/load data and cannot be promoted as
backups.

The proposal and total-pool audit expose:

- `mechanical_transition_role`
- `prior_mechanical_observation_count`
- `mechanical_repeat_status`
- `mechanical_repeat_allowed`

Allowed transition roles are `anchor`, `bootstrap_utility`,
`bootstrap_coverage`, `hybrid_qlognehvi`, `hybrid_local`,
`hybrid_coverage`, `ordered_backup`, and blank. A full-mechanics primary has a
numeric rank and primary status but a blank transition role.

## Bootstrap algorithm

Bootstrap never attempts qLogNEHVI. For every mechanically eligible row:

```text
bootstrap utility =
  0.70 × minmax(screening_phase_score)
  + 0.30 × empirical_combination_pass_probability
```

The first fresh row maximizes bootstrap utility. Further rows are chosen
greedily:

```text
bootstrap rank score =
  0.60 × minmax(bootstrap utility)
  + 0.40 × minmax(distance to nearest selected formulation)
```

Distance uses the complete composition vector scaled by registry bounds. The
first fresh row receives role `bootstrap_utility`; subsequent fresh rows
receive `bootstrap_coverage`. Numeric ranks beyond the four primaries are
actual-intact backups.

The first mechanically active bootstrap batch uses four fresh primaries. The
second may use one anchor and three fresh primaries. Anchor candidates come
from the immediately preceding mechanical batch and require same-batch
viability, critical load, and actual intact pass. Replicate means are scored by:

```text
anchor score =
  0.50 × minmax(viability)
  + 0.50 × minmax(critical load)
```

Ties use lower viability replicate SD and formulation ID. Availability,
present-day feasibility, and absence of any recorded preparation failure are
mandatory. The chosen anchor receives rank 1 and role `anchor`; all editable
wet-lab fields remain blank for remeasurement. If no anchor is eligible, four
fresh rows are used and the reason is recorded. The automatic policy cannot
repeat an anchor in another batch.

## Hybrid algorithm

Hybrid mode trains the mechanics surrogate and uses the recorded search mode.
The version-1 `control` decision uses finite-pool qLogNEHVI; a version-2
`continuous_required` decision requests refinement. It preserves original
origins such as `local_perturbation` and
combines screening and mechanics for the 12-row slate:

```text
hybrid_phase_score =
  0.50 × minmax(screening_phase_score)
  + 0.50 × minmax(mechanics_phase_score)
```

The four primary roles are allocated deterministically:

1. highest empirical-feasibility-weighted mechanics score;
2. a diversity-aware second high mechanics score;
3. highest unused `local_perturbation` mechanics score;
4. maximum registry-scaled coverage distance from prior mechanical evidence and
   the first three selections.

The coverage row must be in the upper half of slate screening scores, have
`p_combo >= 0.50`, remain mechanically untested, and satisfy every common slate
constraint. If the local or coverage role is empty, the highest unused hybrid
score fills the capacity. Metadata records the unfilled role, fallback reason,
and replacement. Remaining eligible rows are ordered by mechanics score and
labeled `ordered_backup`.

## Full mechanics algorithm

Full mode uses the recorded finite-only or continuous-required search mode.
Missing dependencies, failed required optimization or invalid acquisition scores
stop selection. Both search paths use the approved shared posterior and the
same qLogNEHVI score. Viability and critical load per needle are the Pareto
objectives; initial stiffness is diagnostic.

```text
mechanics score = qLogNEHVI + log(max(p_combo, 1e-9)) - screening_acquisition_penalty
```

The final 12-row allocation is one control, up to one eligible mechanics follow-up,
up to one shared retest/rescue slot and 9–11 ordinary recipes. Eligible evidence-ranked retests
take precedence for the shared slot; an acquisition-ranked rescue fills it only
if no retest can be selected. Unfilled special slots return to ordinary ranking.
Extra retests/rescues are excluded. The ordinary
ordinary rows compete across origins by mechanics score, with all existing
feasibility and diversity constraints. Origin counts are descriptive outcomes;
`group10.ordinary_origin_quotas` is empty. A missing campaign control or total-capacity
shortfall stops the run. Screening, bootstrap and hybrid retain their phase-specific allocation.

An eligible follow-up has first mechanical priority and the best remaining eligible ordinary
recipes complete the four primaries; without a follow-up, all four are acquisition-ranked. Further eligible ranks
are ordered actual-intact backups. Retests and exact mechanically observed
formulations remain unranked.

## Metadata and fallback interpretation

`next_round_metadata.json`, `current_round_status.json`, and the generated
summary report:

- resolved phase, reason, transition-policy version, and override status;
- completed-screening and paired evidence counts;
- hybrid and full gate requirements, results, and remaining deficits;
- transition role and candidate ID assignments;
- anchor enablement, source batch, eligibility decision, score, and no-anchor
  reason;
- prior-mechanics and retest exclusion counts;
- hybrid quota fulfillment and deterministic fallbacks;
- requested and executed search mode, BoTorch availability and optimizer diagnostics;
- mechanical capacity, primary count, backup count, and actual-intact rule.

The generated summary uses phase-specific operator instructions. It does not
describe policy changes or rely on a named campaign round.

Frozen Rounds 11–12 use the label `hybrid_finite_pool_fallback`, but their
continuous-search reason explicitly records the finite-only GP decision.
That label does not establish an attempted continuous solve. A repaired
`continuous_required` run stops on failure rather than substituting finite-only search.

## Batch and artifact rules

The batch ID is the next `ROUND_###` implied by `observations.csv` unless
explicitly overridden. Re-running selection before ingesting results resolves
the same batch ID. An identical frozen proposal is accepted; a different byte
sequence for the same batch is rejected unless the guarded unstarted-proposal
supersession flow is explicitly requested.

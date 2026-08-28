# Stage 02: Select Candidates

## Purpose

Generate and score a candidate pool, export the next wet-lab slate, and freeze
an exact pre-bench proposal for that round.

This stage performs optimization in the v2 workflow. The candidate
pool is generated randomly or loaded from a CSV, scored by the surrogate models,
and reduced to 12 wet-lab candidates.

## Command

```bash
python3 src/08_multi_objective/02_select_candidates/select_candidates.py
```

Useful options:

```bash
# Use a different generated pool size
python3 src/08_multi_objective/02_select_candidates/select_candidates.py --pool-size 5000

# Use a fixed batch ID
python3 src/08_multi_objective/02_select_candidates/select_candidates.py --batch-id ROUND_002

# Score an external candidate pool instead of generating one
python3 src/08_multi_objective/02_select_candidates/select_candidates.py \
  --candidate-pool path/to/candidate_pool.csv

# Explicitly replace a different frozen proposal only if the round is unstarted
python3 src/08_multi_objective/02_select_candidates/select_candidates.py \
  --supersede-unstarted-proposal
```

## Inputs

- `data/processed_v2/formulations.csv`
- `data/processed_v2/observations.csv`
- `config_v2/ingredients.yaml`
- `config_v2/optimization.yaml`
- `config_v2/availability.yaml`

## Outputs

- `results/multi_objective_v2/total_candidate_pool.csv`
- `results/multi_objective_v2/next_round/next_round_candidates.csv`
- `results/multi_objective_v2/next_round/next_round_summary.txt`
- `results/multi_objective_v2/next_round/next_round_metadata.json` for
  `ROUND_002+`
- `results/multi_objective_v2/rounds/ROUND_###/proposal/proposal.csv`
- `results/multi_objective_v2/rounds/ROUND_###/proposal/summary.txt`
- `results/multi_objective_v2/rounds/ROUND_###/proposal/selection_metadata.json`
  when metadata is available
- `results/multi_objective_v2/rounds/ROUND_###/proposal/plots/next_round_candidate_screen.png`

`total_candidate_pool.csv` is the full generated/scored audit pool. It is not
wet-lab input.

`next_round_candidates.csv` is the file to fill after validation. It contains
the 12 selected wet-lab formulations and blank result columns. Round 3 and
higher proposals leave every editable result field blank, including for
`retest_priority` rows; source measurements remain in the
database rather than being carried into the operator worksheet.

The files under `rounds/ROUND_###/proposal/` are the frozen record of what the
model proposed before bench work. Do not edit them. An identical selector rerun
is allowed; a rerun that would replace the same round with different proposal
bytes is rejected. The active `next_round/` files are promoted only after the
proposal has been frozen successfully.

`total_candidate_pool.csv` is a mutable debug artifact and is
overwritten. It is not copied into every round.

`--supersede-unstarted-proposal` is a narrow policy-migration escape hatch. It
rejects supersession if the batch has observations, a completed artifact, or
entered wet-lab values in the active worksheet. Otherwise it archives the old
proposal directory (CSV, summary, metadata, and plots), active worksheet, and
current total pool under `rounds/ROUND_###/superseded-policy/`. Its manifest records
the original SHA-256 hashes, UTC timestamp, replacement reason, and policy
versions. Normal proposal immutability remains the default.

## Selection Logic

1. Resolve the proposed `ROUND_###`. ROUND_001 retains the original generator;
   Rounds 2-4 use `round2_candidate_feasibility_v1`, and Round 5+ uses
   `round5_solubility_viscosity_v2`.
2. Generate `selection.generated_candidate_pool_size` candidates. The default
   is `2000`. ROUND_002+ uses 40% local perturbation, 35% sparse exploration,
   and 25% boundary-style exploration. Local shortfall is reassigned to
   sparse exploration; an unfillable boundary-style quota stops generation with
   a diagnostic rather than silently changing the policy.
   Support is measured against the evidence subset implied by
   `observations.csv`, not against every formulation row ever written into
   `formulations.csv`. Legacy literature, legacy wet-lab, and campaign wet-lab
   observations all remain support evidence. Boundary-style generation samples
   chemically feasible upper-range probes; those probes may be classified
   as `in_support` when the observed support radius is broad.
   ROUND_002+ also adds capped `rescue_dilution` candidates by scaling down
   high-viability formulations that failed intact-patch formation. These rows
   test whether concentration reduction can preserve viability while restoring
   patch formation, without letting rescue hypotheses dominate the slate.
3. Exclude temporary unavailable ingredients listed in
   `config_v2/availability.yaml`.
4. Apply the round-selected hard formulation guardrails and retain rejected
   attempts in the audit pool with explicit reasons. Round 5 checks practical
   ingredient ceilings, permeating-CPA/sugar/nonpermeating totals, crystalline
   saturation burden and polymer/protein loads regardless of availability.
5. For ROUND_003 and higher, apply one unified formulation-similarity policy
   before model scoring. Concentrations below the configured practical-presence
   thresholds are treated as zero, then every ingredient is scaled by its
   registry range. Ordinary candidates must have bounds-normalized Euclidean
   distance greater than `0.05` from every actual wet-lab formulation and every
   candidate accepted into the generated pool. Two formulations containing
   the same single ingredient must also differ by at least 50% relative to the
   lower concentration. Rescue dilutions receive no exception; intentional
   `retest_priority` rows are exempt. Literature-only formulations are not
   similarity references. Rejected generated rows are replaced by continued
   sampling so the configured 40/35/25 pool targets remain intact where
   feasible.
6. Train v2 surrogate models from `formulations.csv` and `observations.csv`,
   preserving separate validation batches instead of collapsing everything to
   one formulation-wide mean. For Round 3 and higher, each regression uses a
   fixed Matérn 2.5 kernel and per-observation `alpha` values as its only noise
   mechanism; a fixed `WhiteKernel(5.0)` is absent.
7. Resolve the active selection phase automatically:
   - `screening_only` when the paired viability + mechanical evidence threshold is unsatisfied
   - `mechanics_enabled` once the configured evidence thresholds are met
8. Score the feasible pool with the active phase policy. From Round 6,
   `screening_phase_score` is normalized viability UCB minus existing chemistry
   penalties, the support penalty, and the bounded empirical exact-combination
   intact deduction. The additive intact classifier is reported but cannot
   filter or score active screening.
9. Build the 12-row wet-lab slate from that full-pool ranking, then apply four
   diversity controls before finalizing:
   - **Origin quota** — each candidate-origin bucket (`local_perturbation`,
     `sparse_exploration`, `boundary_probe`, `rescue_dilution`, `retest`,
     `continuous_qlognehvi`, `finite_pool_fallback`) contributes at most a
     bounded share of the slate, so one bucket's high scores can't crowd out
     the others.
   - **Ingredient-combination cap** — caps how many selected candidates may
     share the exact same active-ingredient set (using the registry's
     authoritative feature list, not a column-suffix heuristic). Exact pairs
     use the looser `selection.max_candidates_per_ingredient_combination`
     cap (default `2`); any exact combination of 3+ ingredients (trio,
     four-a-kind, etc.) is far more specific and is capped at `1` per round
     by default via `selection.max_candidates_per_larger_ingredient_combination`.
     Both caps are enforced by swapping the lowest-scoring offender for the
     best-scoring eligible pool candidate below its own cap; the
     slate is never shrunk, and an over-cap combination is left in place if
     no eligible replacement exists.
   - **Shared-core-pair cap** — counts every unordered ingredient pair within
     each non-retest row, including pairs inside three-or-more-ingredient
     formulations. No pair may occur in more than five non-retest rows.
     Rescue rows count and are preserved; retests are exempt. Ordinary
     offenders are replaced from the same origin bucket, and Stage 02 refuses
     to freeze a violating slate.
   - **Marginal ingredient-frequency cap** — for Round 3 and higher, each
     registry ingredient may occur above its practical presence threshold in
     at most five of all 12 rows. Retests and rescues count but are protected
     from removal. The lowest-scoring ordinary offender is replaced by the
     highest-scoring eligible candidate from the same origin, preserving the
     slate size, origin allocation and all other diversity constraints.
     Stage 02 refuses to freeze a violating slate.
   - **Independent cold-start cap** — from Round 6, every available ingredient
     with fewer than three distinct prior campaign formulations with measured
     viability is independently capped at two ordinary rows. A multi-cold row
     consumes one slot for every cold ingredient it contains. Retest and rescue
     origins are exempt from this cap, but remain subject to the universal
     five-row marginal cap. Origin allocation and all other constraints must be
     preserved; a violation stops freezing rather than being silently relaxed.
10. Add at most two feasible `retest_priority` formulations using campaign
    `wetlab_feedback` evidence only. Eligibility requires a feedback
    batch-mean range of at least 15 points, a highest-numbered-batch sample SD of at
    least 8 points from at least three replicates, or one anomaly-confirmation
    opportunity for a single-batch formulation whose viability is at least 20
    points off its bounds-normalized chemical neighbours. Two agreeing
    batches suppress further neighbour-driven retesting. Model uncertainty
    only breaks ties between eligible rows.
11. Reserve up to four ordinary Round-6+ positions for cold-start graduation.
    Offer at most one initial slot per cold ingredient in closest-to-graduation,
    least-recently-tested, registry order. Use any remaining slots for second
    distinct-active-set tests closest to graduation, without exceeding the
    independent cap. A row must contain exactly one cold ingredient and retain
    its candidate origin. Failed allocations are recorded and reassigned; any
    unused capacity returns to ordinary score selection.
12. If `mechanics_enabled`, attempt continuous constrained qLogNEHVI and fall
    back to the constrained finite pool when unavailable or unsuccessful.
    Final mechanics ranking applies the empirical exact-combination feasibility
    weight and ranks all 12 slate rows. Ranks 1-4 are primaries and 5-12 are
    ordered actual-intact backups.

ROUND_001 pool generation is random. ROUND_002+ generation is support-aware and
chemically constrained. The final 12-candidate selection remains model-scored
and diversity-aware, subject to the origin-quota and ingredient-combination
controls in step 9 and the ROUND_003+ similarity gate in step 5. During
`screening_only`, the selector does not emit any
mechanical-test recommendations. The mechanical recommender and continuous
qLogNEHVI path turn on after the phase transitions to `mechanics_enabled`.

`support_policy.max_boundary_candidates_per_slate` applies only to selected
rows classified as `support_status=boundary`. It does not limit the number of
selected rows whose generation label is `candidate_origin=boundary_probe`;
proposal metadata records both counts explicitly.

The similarity rule is configured under `formulation_similarity` in
`config_v2/optimization.yaml`. Selection metadata records its activation round,
thresholds, wet-lab history count, rejection counts by origin/reference type,
bounded conflict examples, and the final slate's minimum history and pairwise
distances. It also records retest evidence, candidate-pool uncertainty,
selected shared-pair counts, marginal ingredient counts and any
frequency-driven replacements. Round-6 diagnostics are intentionally forwarded
to the operator worksheet as immutable context; only wet-lab result fields are
editable.

## Round 6 Screening Formula

The viability surrogate is unchanged. It is a `StandardScaler` plus Gaussian
process regression with fixed Matérn 2.5 kernel, heteroscedastic
per-observation `alpha`, `normalize_y=True`, and `optimizer=None`.

```text
viability UCB = predicted viability mean + 0.35 × predicted viability SD
screening score = minmax(viability UCB)
                  - existing chemistry penalties
                  - support penalty
                  - intact screening penalty
```

Viability UCB is a simple screening acquisition heuristic. It is not BoTorch
qLogNEHVI. A formulation far from support generally receives the GP global mean
with higher uncertainty. For the pre-Round-6 taurine examples, `56.30 + 0.35 ×
24.72 = 64.95%`. The cold-start policy limits how many such uncertain rows can
enter a slate; it does not change scaling, regression fitting, means, or SDs.

Starting in Round 7, evidence-aware prediction labeling prevents that global
mean from being presented as a reliable viability estimate. The selector keeps
`raw_surrogate_viability_mean` and `raw_surrogate_viability_std` for acquisition
and frozen prospective evaluation, but blanks the public
`predicted_viability_percent` and `viability_std` fields when an unobserved
formulation contains a cold-start ingredient, lies outside formulation support,
or has mean/uncertainty consistent with prior reversion. The corresponding
`viability_prediction_status`, label and reason fields explain the decision.
Exact observed formulations take precedence and remain labeled
`observed_supported` or `observed_retest`.

### Exact-combination intact evidence

Evidence is limited to completed prior-round `wetlab_feedback`. Technical
replicates are aggregated per formulation and batch; every measured replicate
must pass. Active sets use the standard `0.001 M` and `0.1%` presence floors.
Only identical complete active sets match: an `{A,B}` failure is not evidence
against `A`, `B`, `{A}`, `{B}`, or `{A,B,C}`.

For a matching set, each ingredient concentration is normalized by its full
registry range. The distance is RMS over the matched ingredients. A record at
distance `d ≤ 0.35` receives weight `w=max(0,1-d/0.35)`; farther records have
zero weight. If `P` and `F` are weighted passes and failures:

```text
p_combo = (1 + P) / (2 + P + F)
intact screening penalty = 0.20 × max(0, (0.50 - p_combo) / 0.50)
```

| Evidence at distance zero | `p_combo` | Screening deduction |
|---|---:|---:|
| none | 0.50 | 0.000 |
| one pass | 0.67 | 0.000 |
| one failure | 0.33 | 0.067 |
| two failures | 0.25 | 0.100 |
| one pass + one failure | 0.50 | 0.000 |

Retest and rescue priority is protected. Their empirical evidence and deduction
are displayed, but the deduction does not remove their reserved status. The
additive logistic classifier continues writing
`intact_patch_pass_probability`; it has no active screening selection role.

## Mechanics Acquisition and Actual-Intact Backups

Mechanics keeps exactly two objectives: viability and critical axial load per
needle. Intact formation is feasibility, not a third objective and not a fixed
subtraction from a scale-dependent multi-objective acquisition.

For BoTorch log acquisition:

```text
constrained log acquisition = qLogNEHVI + log(max(p_combo, 1e-9))
```

For the executable finite-pool fallback:

```text
constrained proxy = log1p(raw hypervolume-like improvement × p_combo)
```

An unseen combination uses `p_combo=0.50`; it remains selectable when its
objective improvement justifies the uncertainty. In active mode there is no
`0.80 × (1-classifier probability)` deduction, no classifier `p ≥ 0.50`
subset, and no classifier-based fallback-to-all. Changing classifier
predictions therefore cannot change active mechanics ranking.

The selector ranks all 12 slate rows by the constrained mechanics acquisition.
It writes `mechanical_selection_rank=1..12`, marks 1-4 `primary`, and marks
5-12 `ordered_backup`. Fabrication determines the executable subset: Instron
receives the first four priority rows that actually pass intact. Failed
primaries are skipped; the lowest-numbered intact backup is promoted. If fewer
than four form intact patches, every intact row is tested and the shortfall is
recorded. Failed patches can never supply mechanical data.

The compatibility mode `classifier_threshold_penalty` preserves the previous
classifier threshold, `0.80` classifier-failure deduction, and fallback-to-all
behavior for reproducibility only. Its threshold key is
`round_policy.intact_probability_threshold=0.50`; metadata labels it
`compatibility_only_not_applied` under the active empirical mode.

## Cold-Start Cap and Graduation Algorithm

Cold-start count uses distinct prior campaign formulation IDs with measured
`viability_percent` and the ingredient above its presence floor. Replicates and
repeat batches do not add counts; a measured-viability failed patch does.
Unavailable ingredients are not scheduled. Before Round 6 the available cold
counts are taurine `0`, myo-inositol `1`, methylcellulose `2`, and propylene
glycol `2`.

The cold cap applies to ordinary rows only and independently to each cold
ingredient. A `{taurine,myo-inositol}` row consumes one of both allowances.
`candidate_origin=retest` and `rescue_dilution` are exempt; completed exempt
rows can still graduate an ingredient later. Replacements preserve origin,
slate size, feasibility, similarity, support, boundary count, exact-combination
limits, shared pairs, and the universal cap of five. No eligible replacement
means selection fails before freezing.

Graduation ordering is deterministic:

1. Sort by highest prior distinct-formulation count (closest to three).
2. Break ties by least-recent campaign round, then registry order.
3. Offer one initial slot per ingredient.
4. Use remaining capacity for ingredients still below three, considering the
   evidence count plus already allocated rows and never exceeding two.
5. Require exactly one cold ingredient and, for two rows targeting the same
   ingredient, different complete active sets.
6. Reassign an unfillable slot to the next eligible ingredient; return any
   unused capacity to ordinary selection.

Selected rows keep their original `candidate_origin` and receive
`recommendation_type=cold_start_graduation`.

## Configuration and Diagnostics

The active policies start in Round 6 and are versioned as
`round6_empirical_combination_intact_v1` and
`round6_cold_start_graduation_v1`.

| Configuration key | Default | Meaning |
|---|---:|---|
| `intact_combination_policy.policy_version` | `round6_empirical_combination_intact_v1` | Frozen behavior label in proposals. |
| `intact_combination_policy.start_round` | `6` | Forward activation round. |
| `intact_combination_policy.evidence_source_types` | `wetlab_feedback` | Allowed intact evidence source. |
| `intact_combination_policy.evidence_radius` | `0.35` | Largest matching normalized RMS distance. |
| `intact_combination_policy.beta_prior_pass`, `beta_prior_fail` | `1.0`, `1.0` | Beta prior giving unseen probability `0.50`. |
| `intact_combination_policy.screening_neutral_probability` | `0.50` | Probability below which screening is deducted. |
| `intact_combination_policy.screening_max_penalty` | `0.20` | Largest screening-score deduction. |
| `intact_combination_policy.mechanics_mode` | `empirical_feasibility_weighted` | Active mechanics feasibility policy. |
| `intact_combination_policy.compatibility_mode` | `classifier_threshold_penalty` | Deprecated behavior label only. |
| `intact_combination_policy.numerical_probability_floor` | `1e-9` | Protects `log(p_combo)`. |
| `intact_combination_policy.mechanics_primary_test_count` | `4` | Initial mechanics capacity. |
| `intact_combination_policy.mechanics_backup_behavior` | `actual_intact_priority_promotion` | Promotion policy. |
| `cold_start_policy.policy_version` | `round6_cold_start_graduation_v1` | Frozen cold-policy label. |
| `cold_start_policy.start_round` | `6` | Forward activation round. |
| `cold_start_policy.evidence_source_types` | `wetlab_feedback` | Evidence sources allowed to graduate an ingredient. |
| `cold_start_policy.evidence_endpoint` | `viability_percent` | Measurement required for a distinct formulation to count. |
| `cold_start_policy.minimum_distinct_formulations` | `3` | Graduation threshold. |
| `cold_start_policy.max_ordinary_rows_per_ingredient` | `2` | Independent cold cap. |
| `cold_start_policy.graduation_slots_per_round` | `4` | Maximum graduation reserve. |
| `cold_start_policy.exempt_candidate_origins` | `retest`, `rescue_dilution` | Cold-cap exceptions. |
| `cold_start_policy.graduation_max_cold_ingredients_per_row` | `1` | Graduation-row cold content. |
| `cold_start_policy.graduation_require_distinct_active_sets` | `true` | Diversity for repeated graduation target. |
| `cold_start_policy.same_origin_replacement_preferred` | `true` | Search ordering favors same-origin replacements. |
| `cold_start_policy.preserve_origin_allocation` | `true` | Forbids cross-origin replacements. |

Proposal and pool diagnostics include `empirical_combination_pass_probability`,
weighted nearby passes/failures, nearest pass/failure distances,
`intact_combination_screening_penalty`, policy version,
`mechanical_feasibility_weight`, cold ingredient list and prior counts,
graduation eligibility/priority/reason, and mechanics primary/backup status when
active. Metadata records classifier role, counts before/after, replacements and
score changes, allocations/skips/reassignments, primary/backup counts, and the
actual-pass rule. Completed-round mechanics promotions and shortfalls live in
`completed/mechanical_execution_manifest.json`.

## Round 6 Rollout

Round 6 is `screening_only`: exactly 12 viability/intact rows, blank mechanics
priorities, and zero requested Instron tests. The same observation snapshot,
availability file, random seed (`42`), pool size (`2000`), and 40/35/25
generation mix are used. Rounds 1-5 remain immutable. If an already-frozen but
blank Round 6 must be migrated, use the guarded supersession option; it refuses
any started round and leaves a hash manifest and recoverable archive.

The phase thresholds are unchanged: eight paired objective observations, six
distinct formulations, and two batches. There is no mechanical-bootstrap phase.
Unless mechanical data are collected externally or the phase is explicitly
overridden, automatic selection remains screening-only and the weighted
mechanics policy stays dormant.

## Batch ID

The batch ID is generated as `ROUND_###` from `observations.csv`. After `03_run_round/run_round.py`
ingests `ROUND_001`, the next Stage 02 run emits `ROUND_002`. If you rerun Stage
02 before ingesting results, it will emit the same next unused round ID and
must match the frozen proposal.

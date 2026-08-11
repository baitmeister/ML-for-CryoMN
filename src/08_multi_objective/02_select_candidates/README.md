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
8. Score the feasible pool with the active phase policy. During
   `screening_only`, `screening_phase_score` is purely viability-based;
   predicted intact-formation probability does not gate or score screening
   candidates. Intact-formation risk is instead handled by the
   `rescue_dilution` candidates from step 2 and, once `mechanics_enabled`,
   by mechanics-phase scoring (`penalties.intact_failure_weight`,
   `round_policy.intact_probability_threshold`).
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
10. Add at most two feasible `retest_priority` formulations using campaign
    `wetlab_feedback` evidence only. Eligibility requires a feedback
    batch-mean range of at least 15 points, a highest-numbered-batch sample SD of at
    least 8 points from at least three replicates, or one anomaly-confirmation
    opportunity for a single-batch formulation whose viability is at least 20
    points off its bounds-normalized chemical neighbours. Two agreeing
    batches suppress further neighbour-driven retesting. Model uncertainty
    only breaks ties between eligible rows.
11. If `mechanics_enabled`, attempt continuous constrained qLogNEHVI and fall
    back to the constrained finite pool when unavailable or unsuccessful.

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
frequency-driven replacements. None of these diagnostics adds columns to the
operator worksheet.

## Batch ID

The batch ID is generated as `ROUND_###` from `observations.csv`. After `03_run_round/run_round.py`
ingests `ROUND_001`, the next Stage 02 run emits `ROUND_002`. If you rerun Stage
02 before ingesting results, it will emit the same next unused round ID and
must match the frozen proposal.

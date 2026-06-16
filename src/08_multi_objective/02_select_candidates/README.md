# Stage 02: Select Candidates

## Purpose

Generate and score a candidate pool, then export the next wet-lab slate.

This is where optimization happens in the current v2 workflow. The candidate
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

`total_candidate_pool.csv` is the full generated/scored audit pool. It is not
wet-lab input.

`next_round_candidates.csv` is the file to fill after validation. It contains
the 12 selected wet-lab formulations and blank result columns.

## Selection Logic

1. Resolve the proposed `ROUND_###`. ROUND_001 retains the original generator;
   ROUND_002+ activates `round2_candidate_feasibility_v1`.
2. Generate `selection.generated_candidate_pool_size` candidates. The default
   is `2000`. ROUND_002+ uses 40% local perturbation, 35% sparse exploration,
   and 25% boundary-style exploration. Local shortfall is reassigned to
   sparse exploration; an unfillable boundary-style quota stops generation with
   a diagnostic rather than silently changing the policy.
   Support is measured against the evidence subset implied by
   `observations.csv`, not against every formulation row ever written into
   `formulations.csv`. Legacy literature, legacy wet-lab, and new wet-lab
   observations all remain support evidence. Boundary-style generation samples
   chemically feasible upper-range probes; those probes may still be classified
   as `in_support` when the observed support radius is broad.
   ROUND_002+ also adds capped `rescue_dilution` candidates by scaling down
   high-viability formulations that failed intact-patch formation. These rows
   test whether concentration reduction can preserve viability while restoring
   patch formation, without letting rescue hypotheses dominate the slate.
3. Exclude temporary unavailable ingredients listed in
   `config_v2/availability.yaml`.
4. Apply ROUND_002+ hard formulation guardrails and retain rejected attempts in
   the audit pool with explicit reasons.
5. Train v2 surrogate models from `formulations.csv` and `observations.csv`,
   preserving separate validation batches instead of collapsing everything to
   one formulation-wide mean.
6. Resolve the active selection phase automatically:
   - `screening_only` while real paired viability + mechanical data are still sparse
   - `mechanics_enabled` once the configured evidence thresholds are met
7. Score the feasible pool with the active phase policy. During
   `screening_only`, `screening_phase_score` is purely viability-based;
   predicted intact-formation probability does not gate or score screening
   candidates. Intact-formation risk is instead handled by the
   `rescue_dilution` candidates from step 2 and, once `mechanics_enabled`,
   by mechanics-phase scoring (`penalties.intact_failure_weight`,
   `round_policy.intact_probability_threshold`).
8. Build the 12-row wet-lab slate from that full-pool ranking, then apply two
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
     cap (default `3`); any exact combination of 3+ ingredients (trio,
     four-a-kind, etc.) is far more specific and is capped at `1` per round
     by default via `selection.max_candidates_per_larger_ingredient_combination`.
     Both caps are enforced by swapping the lowest-scoring offender for the
     best-scoring eligible pool candidate not already at its own cap; the
     slate is never shrunk, and an over-cap combination is left in place if
     no eligible replacement exists.
9. Add any `retest_priority` formulations separately when the latest batch for
   an existing formulation appears off-trend or unstable.
10. If `mechanics_enabled`, attempt continuous constrained qLogNEHVI and fall
    back to the constrained finite pool when unavailable or unsuccessful.

ROUND_001 pool generation is random. ROUND_002+ generation is support-aware and
chemically constrained. The final 12-candidate selection remains model-scored
and diversity-aware, subject to the origin-quota and ingredient-combination
controls in step 8. During `screening_only`, the selector does not emit any
mechanical-test recommendations. The mechanical recommender and continuous
qLogNEHVI path turn on after the phase transitions to `mechanics_enabled`.

## Batch ID

The batch ID is generated as `ROUND_###` from `observations.csv`. After `03_run_round/run_round.py`
ingests `ROUND_001`, the next Stage 02 run emits `ROUND_002`. If you rerun Stage
02 before ingesting results, it will emit the same next unused round ID.

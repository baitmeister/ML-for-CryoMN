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
   and 25% support-boundary exploration. Local shortfall is reassigned to
   sparse exploration; an unfillable boundary quota stops generation with a
   diagnostic rather than silently changing the policy.
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
7. Score the feasible pool with the active phase policy, then build the
   12-row wet-lab slate directly from that full-pool ranking.
8. Add any `retest_priority` formulations separately when the latest batch for
   an existing formulation appears off-trend or unstable.
9. If `mechanics_enabled`, attempt continuous constrained qLogNEHVI and fall
   back to the constrained finite pool when unavailable or unsuccessful.

ROUND_001 pool generation is random. ROUND_002+ generation is support-aware and
chemically constrained. The final 12-candidate selection remains model-scored
and diversity-aware. During `screening_only`, the selector does not emit any
mechanical-test recommendations. The mechanical recommender and continuous
qLogNEHVI path turn on after the phase transitions to `mechanics_enabled`.

## Batch ID

The batch ID is generated as `ROUND_###` from `observations.csv`. After `03_run_round/run_round.py`
ingests `ROUND_001`, the next Stage 02 run emits `ROUND_002`. If you rerun Stage
02 before ingesting results, it will emit the same next unused round ID.

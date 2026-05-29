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

1. Generate `selection.generated_candidate_pool_size` candidates. The default is
   `2000`.
2. Exclude temporary unavailable ingredients listed in
   `config_v2/availability.yaml`.
3. Train v2 surrogate models from `formulations.csv` and `observations.csv`.
4. Score every candidate with predicted viability, uncertainty, intact-patch
   probability, predicted mechanical metrics, and soft penalties.
5. Select 12 viability-screen candidates using a greedy diversity-aware ranking.
6. Select 3-4 mechanical-test recommendations from those 12.

The pool generation step is random. The final 12-candidate selection is model
scored and diversity-aware, not random.

## Batch ID

The batch ID is generated as `ROUND_###` from `observations.csv`. After Stage 03
ingests `ROUND_001`, the next Stage 02 run emits `ROUND_002`. If you rerun Stage
02 before ingesting results, it will emit the same next unused round ID.

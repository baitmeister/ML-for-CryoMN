# Stage 04: Visualization

## Purpose

Generate diagnostic plots and a short text summary for the v2 database and
current next-round slate.

This stage is analogous to the legacy explainability/reporting modules, but it
is scoped to the v2 multi-objective workflow.

## Command

```bash
python3 src/08_multi_objective/04_visualization/visualize.py
```

Optional inputs:

```bash
python3 src/08_multi_objective/04_visualization/visualize.py \
  --formulations data/processed_v2/formulations.csv \
  --observations data/processed_v2/observations.csv \
  --candidates results/multi_objective_v2/next_round/next_round_candidates.csv \
  --output-dir results/multi_objective_v2/visualizations
```

## Inputs

- `data/processed_v2/formulations.csv`
- `data/processed_v2/observations.csv`
- `results/multi_objective_v2/next_round/next_round_candidates.csv`

## Outputs

- `results/multi_objective_v2/visualizations/endpoint_observation_counts.png`
- `results/multi_objective_v2/visualizations/next_round_candidate_screen.png`
- `results/multi_objective_v2/visualizations/visualization_summary.txt`

## When To Run

Run this after Stage 01 to inspect the transferred database, after Stage 02 to
inspect candidate selection, and after Stage 03 to confirm that new wet-lab
observations were ingested.

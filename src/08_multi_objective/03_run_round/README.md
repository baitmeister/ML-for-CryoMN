# Stage 03: Run Round

## Purpose

Advance one real wet-lab round while preserving the one review state that
matters for slate provenance:

1. generate a pre-update review snapshot of the exact database state that
   produced the current slate
2. ingest the filled wet-lab results
3. generate the next candidate slate

This is the supported entry point for normal round progression.

## Command

```bash
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv
```

Useful options:

```bash
# Override the automatic selection phase for auditing/debugging
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --phase-mode mechanics_enabled

# Ingest only, without generating the next round
python3 src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --skip-generate
```

## Notes

- Default behavior uses the automatic phase selector.
- The command ingests wet-lab results directly; there is no separate updater
  stage anymore.
- The command creates one pre-update review snapshot before any new wet-lab
  observations are appended.
- The command also refreshes `results/multi_objective_v2/current_round_status.json`
  so the current/latest round and next round remain visible outside the CSVs.
- That review output is archived per batch under:
  - `results/multi_objective_v2/round_review/ROUND_###`
- Each round archive also stores:
  - `ROUND_###_next_round_candidates.csv`
  - `ROUND_###_next_round_summary.txt`
  - `ROUND_###_total_candidate_pool.csv`
  - `ROUND_###_model_evaluation_table.csv`
- The command fails if ingestion or review generation fails.
- Candidate generation only runs after ingestion succeeds.

# Multi-Objective Quick SOP

Use this when you just need the shortest correct workflow.

## First-Time Setup

```bash
src/08_multi_objective/01_build_database/build_database.py
src/08_multi_objective/02_select_candidates/select_candidates.py
```

Then review:
- [results/multi_objective_v2/next_round/next_round_summary.txt](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_summary.txt)
- [results/multi_objective_v2/next_round/next_round_candidates.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_candidates.csv)

## Every Wet-Lab Round

1. Open and fill:
   - [results/multi_objective_v2/next_round/next_round_candidates.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_candidates.csv)
2. Enter:
   - `viability_percent`
   - `intact_patch_formation_pass`
   - mechanical values if available
3. If using Instron CSVs, optionally import them:

```bash
src/08_multi_objective/helper/instron.py \
  data/raw/instron/ROUND_001/example.csv \
  --formulation-id v2_example \
  --batch-id ROUND_001 \
  --replicate-id rep_001 \
  --needles-compressed 100
```

4. Advance the round:

Preferred one-command path:

```bash
src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv
```

What this command does internally, in order:
- writes one review snapshot for the state that produced the current slate
- ingests the filled `next_round_candidates.csv`
- runs `02_select_candidates/select_candidates.py`

If you want to inspect the updated state before allowing the next slate to be
created, run:

```bash
src/08_multi_objective/03_run_round/run_round.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --skip-generate
```

5. Review:
   - [results/multi_objective_v2/next_round/next_round_summary.txt](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_summary.txt)
   - [results/multi_objective_v2/round_review/ROUND_001/ROUND_001_visualization_summary.txt](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/round_review/ROUND_001/ROUND_001_visualization_summary.txt)
   - [results/multi_objective_v2/round_review/ROUND_001/ROUND_001_best_performers_summary.txt](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/round_review/ROUND_001/ROUND_001_best_performers_summary.txt)
   - `results/multi_objective_v2/round_review/ROUND_###/ROUND_###_next_round_candidates.csv`
   - `results/multi_objective_v2/round_review/ROUND_###/ROUND_###_next_round_summary.txt`
   - `results/multi_objective_v2/round_review/ROUND_###/ROUND_###_total_candidate_pool.csv`
   - `results/multi_objective_v2/round_review/ROUND_###/ROUND_###_model_evaluation_table.csv`

## Important Reminders

- Only fill wet-lab results in:
  - [results/multi_objective_v2/next_round/next_round_candidates.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/next_round/next_round_candidates.csv)
- Do not type wet-lab results into:
  - [results/multi_objective_v2/total_candidate_pool.csv](/Users/bait/Documents/ML-for-CryoMN/results/multi_objective_v2/total_candidate_pool.csv)
- The program now switches phase automatically:
  - early rounds: `screening_only`
  - later rounds: `mechanics_enabled`
- Watch for rows labeled `retest_priority`

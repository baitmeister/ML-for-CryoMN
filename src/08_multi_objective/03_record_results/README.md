# Stage 03: Record Results

## Purpose

Convert wet-lab feedback into the persistent v2 database.

You fill the 12-row `next_round_candidates.csv` produced by Stage 02. Stage 03
then writes those measured endpoints into `data/processed_v2/observations.csv`
and adds any new formulation identities to `data/processed_v2/formulations.csv`.

Do not fill wet-lab results into `total_candidate_pool.csv`.

## Main Command

```bash
python3 src/08_multi_objective/03_record_results/update_from_results.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv
```

Useful options:

```bash
# Override the noise assigned to new viability rows for this import
python3 src/08_multi_objective/03_record_results/update_from_results.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --viability-noise 0.8

# Supply a default number of compressed needles for total-load entries
python3 src/08_multi_objective/03_record_results/update_from_results.py \
  results/multi_objective_v2/next_round/next_round_candidates.csv \
  --default-needles-compressed 100
```

## Columns To Fill

Required or common fields:

- `viability_percent`: number from `0` to `100`.
- `intact_patch_formation_pass`: `yes/no`, `true/false`, `pass/fail`, or `1/0`.
- `replicate_id`: optional; duplicate a row and use `rep_001`, `rep_002`, etc.
  for technical replicates.

Optional intact-patch details:

- `no_slurry`
- `no_collapse`
- `intact_tip_count`
- `total_tip_count`

Mechanical fields for intact mechanical-tested rows:

- `instron_file` and `needles_compressed`, or
- `critical_axial_load_N_per_needle`, or
- `critical_axial_load_N_total` plus `needles_compressed`
- `initial_stiffness_N_per_mm_per_needle` if measured or parsed

## Noise Settings

Default noise values live in `config_v2/optimization.yaml`:

- `transfer.literature_viability_noise_percent: 50.0`
- `transfer.legacy_wetlab_viability_noise_percent: 5.0`
- `feedback.new_viability_noise_percent: 1.0`

New validation viability therefore defaults to one fifth of legacy wet-lab
noise.

## Instron Helper

Use `import_instron.py` to parse one Bluehill/Instron CSV into the current
candidate sheet:

```bash
python3 src/08_multi_objective/03_record_results/import_instron.py \
  data/raw/instron/ROUND_001/example.csv \
  --formulation-id v2_example \
  --batch-id ROUND_001 \
  --replicate-id rep_001 \
  --needles-compressed 100
```

The normal output is an updated `next_round_candidates.csv`. The database is
updated only after `update_from_results.py` runs.

## Outputs

- updated `data/processed_v2/formulations.csv`
- updated `data/processed_v2/observations.csv`

The next Stage 02 run retrains from these updated tables and writes a new
candidate slate.

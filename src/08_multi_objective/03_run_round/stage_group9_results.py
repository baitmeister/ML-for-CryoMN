#!/usr/bin/env python3
"""Stage the reviewed Group 9 viability, intact and Instron results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROPOSAL = ROOT / 'results/multi_objective_v2/rounds/ROUND_009/proposal/proposal.csv'
DEFAULT_OUTPUT = ROOT / 'results/multi_objective_v2/next_round/next_round_candidates.csv'

MECHANICAL = {
    5: [(3, 1, 21.508950, 0.215090, 0.256356, 'replicate 2 lost in testing; no value imputed')],
    8: [(4, 1, 15.135425, 0.151354, 0.176638, 'replicate 2 lost in testing; no value imputed')],
    9: [(1, 1, 30.522467, 0.305225, 0.369017, ''),
        (5, 2, 17.716533, 0.177165, 0.208915, '')],
    12: [(2, 1, 21.067167, 0.210672, 0.250792, ''),
         (6, 2, 12.144467, 0.121445, 0.139262, '')],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument('--viability', type=Path, required=True)
    parser.add_argument('--instron-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def blank_result_fields(row: pd.Series) -> pd.Series:
    for column in [
        'replicate_id', 'viability_percent', 'intact_patch_formation_pass',
        'no_slurry', 'no_collapse', 'intact_tip_count', 'total_tip_count',
        'instron_file', 'needles_compressed',
        'critical_axial_load_N_per_needle', 'critical_axial_load_N_total',
        'initial_stiffness_N_per_mm_per_needle', 'preparation_feasibility_pass',
        'homogeneous_solution_pass', 'fillability_pass',
        'preparation_failure_reason', 'notes',
    ]:
        if column in row.index:
            row[column] = ''
    return row


def main() -> None:
    args = parse_args()
    proposal = pd.read_csv(args.proposal, dtype=object, keep_default_na=False)
    viability = pd.read_csv(args.viability, dtype=object, keep_default_na=False)
    if sorted(pd.to_numeric(proposal.selection_rank).astype(int)) != list(range(1, 13)):
        raise ValueError('Group 9 proposal must contain selection ranks 1 through 12 exactly once')
    group_to_rank = {str(number): number for number in range(1, 10)}
    group_to_rank.update({'X': 10, 'XI': 11, 'XII': 12})
    viability = viability.assign(
        selection_rank=viability.group.astype(str).map(group_to_rank),
        replicate_number=pd.to_numeric(viability.replicate, errors='raise').astype(int),
    )
    if viability.selection_rank.isna().any():
        raise ValueError('Unrecognized viability group label')
    counts = viability.groupby('selection_rank').size().to_dict()
    if counts != {rank: 4 for rank in range(1, 13)}:
        raise ValueError(f'Expected four viability samples per formulation; found {counts}')

    by_rank = proposal.copy()
    by_rank.index = pd.to_numeric(proposal.selection_rank).astype(int)
    rows = []
    for rank in range(1, 13):
        base = blank_result_fields(by_rank.loc[rank].copy())
        for measurement in viability.loc[viability.selection_rank.eq(rank)].sort_values('replicate_number').itertuples():
            row = base.copy()
            row['replicate_id'] = f'viability_{measurement.replicate_number:03d}'
            row['viability_percent'] = measurement.viability_pct
            row['notes'] = (
                f'source_image={measurement.file}; total_cells={measurement.total_cells}; '
                f'alive_cells={measurement.alive_cells}'
            )
            rows.append(row)
        intact = base.copy()
        intact['replicate_id'] = 'intact_final'
        intact['intact_patch_formation_pass'] = 'fail' if rank == 4 else 'pass'
        intact['notes'] = 'formulation-level final intact-patch gate'
        rows.append(intact)
        for file_number, replicate_number, total_force, per_needle_force, stiffness, note in MECHANICAL.get(rank, []):
            source = (args.instron_dir / f'group_9-1_{file_number}.csv').resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            mechanical = base.copy()
            mechanical['replicate_id'] = f'mechanical_{replicate_number:03d}'
            mechanical['instron_file'] = str(source)
            mechanical['needles_compressed'] = 100
            mechanical['critical_axial_load_N_total'] = total_force
            mechanical['critical_axial_load_N_per_needle'] = per_needle_force
            mechanical['initial_stiffness_N_per_mm_per_needle'] = stiffness
            mechanical['notes'] = note
            rows.append(mechanical)

    staged = pd.DataFrame(rows, columns=proposal.columns)
    if len(staged) != 66:
        raise AssertionError(f'Expected 66 staged rows, found {len(staged)}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staged.to_csv(args.output, index=False)
    print(f'Staged {len(staged)} Group 9 rows in {args.output}')


if __name__ == '__main__':
    main()

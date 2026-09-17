from __future__ import annotations

import importlib.util
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT / "src" / "08_multi_objective"
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.artifacts import validate_completed_against_proposal  # noqa: E402
from helper.mechanics_execution import validate_mechanics_execution  # noqa: E402
from helper.group10_config import (  # noqa: E402
    Group10HardStop,
    assert_frozen_group10_protocol,
    load_group10_config,
    load_round_mechanical_endpoint,
)


def _load_run_round_module():
    path = V2_ROOT / "03_run_round" / "run_round.py"
    spec = importlib.util.spec_from_file_location("v2_run_round_characterization", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V2RoundWorkflowCharacterizationTests(unittest.TestCase):
    def test_archived_round_eight_completed_sheet_matches_frozen_proposal(self) -> None:
        round_dir = (
            PROJECT_ROOT
            / "results"
            / "multi_objective_v2"
            / "rounds"
            / "ROUND_008"
        )
        validate_completed_against_proposal(
            round_dir / "completed" / "completed.csv",
            round_dir / "proposal" / "proposal.csv",
        )
        manifest = json.loads(
            (round_dir / "completed" / "mechanical_execution_manifest.json").read_text()
        )
        self.assertFalse(manifest["mechanics_active_in_proposal"])
        self.assertEqual(manifest["actual_intact_measured_count"], 12)

    def test_blank_proposal_is_not_a_completed_experiment(self) -> None:
        module = _load_run_round_module()
        proposal = pd.read_csv(
            PROJECT_ROOT
            / "results"
            / "multi_objective_v2"
            / "rounds"
            / "ROUND_009"
            / "proposal"
            / "proposal.csv"
        )
        with tempfile.TemporaryDirectory() as temporary_name:
            path = Path(temporary_name) / "blank_proposal.csv"
            proposal.to_csv(path, index=False)
            self.assertFalse(module._round_has_new_results(path))

    def test_mechanical_data_on_unranked_row_is_rejected(self) -> None:
        proposal = pd.read_csv(
            PROJECT_ROOT
            / "results"
            / "multi_objective_v2"
            / "rounds"
            / "ROUND_009"
            / "proposal"
            / "proposal.csv"
        )
        completed = proposal.copy()
        unranked = completed["mechanical_selection_rank"].isna().idxmax()
        completed.loc[unranked, "intact_patch_formation_pass"] = 1.0
        completed.loc[unranked, "critical_axial_load_N_per_needle"] = 1.0
        with self.assertRaisesRegex(ValueError, "unranked"):
            validate_mechanics_execution(completed, proposal, primary_capacity=4)

    def test_mechanical_data_without_confirmed_intact_pass_is_rejected(self) -> None:
        proposal = pd.read_csv(
            PROJECT_ROOT
            / "results"
            / "multi_objective_v2"
            / "rounds"
            / "ROUND_009"
            / "proposal"
            / "proposal.csv"
        )
        completed = proposal.copy()
        ranked = completed["mechanical_selection_rank"].notna().idxmax()
        completed.loc[ranked, "critical_axial_load_N_per_needle"] = 1.0
        with self.assertRaisesRegex(ValueError, "without a measured intact pass"):
            validate_mechanics_execution(completed, proposal, primary_capacity=4)

    def test_group10_ingestion_requires_complete_matching_frozen_protocol(self) -> None:
        from helper.group10_config import load_group10_config_for_round
        config = load_group10_config_for_round(10)
        with self.assertRaisesRegex(Group10HardStop, "no complete Group 10 effective_config"):
            assert_frozen_group10_protocol(config, 10, {})
        metadata = {"group10": {"effective_config": json.loads(json.dumps(config))}}
        metadata["group10"]["effective_config"]["mechanical_endpoint"].pop("protocol_id")
        with self.assertRaisesRegex(Group10HardStop, "settings/protocol are invalid"):
            assert_frozen_group10_protocol(config, 10, metadata)

    def test_round9_endpoint_addendum_is_hash_bound(self) -> None:
        source = PROJECT_ROOT / 'results/multi_objective_v2/rounds/ROUND_009/proposal'
        endpoint = load_round_mechanical_endpoint(source, 'ROUND_009')
        self.assertEqual(endpoint['definition_id'], 'terminal_force_08mm_after_1N_v1')
        with tempfile.TemporaryDirectory() as temporary_name:
            destination = Path(temporary_name)
            for name in ['proposal.csv', 'selection_metadata.json', 'mechanical_endpoint_addendum.json']:
                (destination / name).write_bytes((source / name).read_bytes())
            (destination / 'proposal.csv').write_text('tampered\n')
            with self.assertRaisesRegex(Group10HardStop, 'hash mismatch'):
                load_round_mechanical_endpoint(destination, 'ROUND_009')

    def test_validate_only_parses_all_endpoints_without_writes_even_on_duplicates(self) -> None:
        module = _load_run_round_module()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            results = root / 'results'
            next_round = results / 'next_round'
            proposal_dir = results / 'rounds/ROUND_009/proposal'
            proposal_dir.mkdir(parents=True)
            next_round.mkdir(parents=True)
            source = PROJECT_ROOT / 'results/multi_objective_v2/rounds/ROUND_009/proposal'
            for filename in ('proposal.csv', 'selection_metadata.json', 'mechanical_endpoint_addendum.json'):
                shutil.copyfile(source / filename, proposal_dir / filename)
            for filename in ('formulations.csv', 'observations.csv'):
                shutil.copyfile(PROJECT_ROOT / 'data/processed_v2' / filename, root / filename)
            historical = pd.read_csv(root / 'observations.csv')
            historical.loc[historical.batch_id.ne('ROUND_009')].to_csv(root / 'observations.csv', index=False)
            (results / 'current_round_status.json').write_text('{"unchanged": true}\n')
            reports = results / 'reports'
            reports.mkdir()
            (reports / 'existing_report.txt').write_text('Previously published report\n')
            (next_round / 'next_round_summary.txt').write_text('Frozen proposal summary\n')

            # Use real frozen candidate identities with synthetic measurements;
            # the regression must not depend on external wet-lab source files.
            proposal = pd.read_csv(proposal_dir / 'proposal.csv')
            rows = []
            mechanical_counts = {5: 1, 8: 1, 9: 2, 12: 2}
            time = np.arange(205) * .025
            displacement = np.arange(205) * .005 + 2
            force = 1 + (displacement - displacement[4]) * 10
            force[:4] = [.1, .4, .6, .9]
            for _, candidate in proposal.iterrows():
                base = candidate.to_dict()
                for column in (*module.RESULT_COLUMNS, 'no_slurry', 'no_collapse', 'total_tip_count', 'needles_compressed'):
                    if column in base:
                        base[column] = np.nan
                rank = int(candidate.selection_rank)
                for replicate in range(1, 5):
                    rows.append({**base, 'replicate_id': f'viability_{replicate:03d}', 'viability_percent': 60 + replicate})
                rows.append({**base, 'replicate_id': 'intact_final', 'intact_patch_formation_pass': 'fail' if rank == 4 else 'pass'})
                for replicate in range(1, mechanical_counts.get(rank, 0) + 1):
                    raw = root / f'raw_{rank}_{replicate}.csv'
                    pd.DataFrame({
                        'Time': time, 'Displacement': displacement,
                        'Force': force * (1 + rank / 100 + replicate / 1000),
                    }).to_csv(raw, index=False)
                    rows.append({**base, 'replicate_id': f'mechanical_{replicate:03d}', 'instron_file': str(raw), 'needles_compressed': 100})
            completed = pd.DataFrame(rows, columns=proposal.columns)
            self.assertEqual(len(completed), 66)
            worksheet = next_round / 'next_round_candidates.csv'
            argv = [
                str(V2_ROOT / '03_run_round/run_round.py'), str(worksheet), '--validate-only',
                '--formulations', str(root / 'formulations.csv'),
                '--observations', str(root / 'observations.csv'),
                '--output-dir', str(next_round), '--total-candidate-pool', str(root / 'pool.csv'),
            ]

            def snapshot():
                return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in root.rglob('*') if p.is_file()}

            for duplicate in (False, True):
                with self.subTest(duplicate=duplicate):
                    feedback = pd.concat([completed, completed.iloc[[0]]], ignore_index=True) if duplicate else completed
                    feedback.to_csv(worksheet, index=False)
                    before = snapshot()
                    output = io.StringIO()
                    with patch.object(sys, 'argv', argv), redirect_stdout(output), patch.object(module, '_run') as generate:
                        if duplicate:
                            with self.assertRaisesRegex(ValueError, 'Duplicate observation keys'):
                                module.main()
                        else:
                            module.main()
                            for endpoint, count in (
                                ('viability_percent', 48), ('intact_patch_formation_pass', 12),
                                ('critical_axial_load_N_per_needle', 6), ('initial_stiffness_N_per_mm_per_needle', 6),
                            ):
                                self.assertIn(f'  {endpoint}: {count}\n', output.getvalue())
                            for rank, count in mechanical_counts.items():
                                fid = proposal.loc[proposal.selection_rank.eq(rank), 'formulation_id'].item()
                                self.assertIn(f'  {fid}: {count}\n', output.getvalue())
                        generate.assert_not_called()
                    self.assertEqual(snapshot(), before)
                    self.assertFalse((results / 'rounds/ROUND_009/completed').exists())
                    self.assertFalse((results / 'rounds/ROUND_009/reports').exists())
                    self.assertFalse((results / 'rounds/ROUND_010').exists())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = PROJECT_ROOT / 'src' / '08_multi_objective'
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.feedback import ingest_feedback  # noqa: E402
from helper.group10_config import load_group10_config  # noqa: E402
from helper.plot_reporting import mechanical_report_table, write_mechanical, write_plot  # noqa: E402
from helper.models import build_training_frame  # noqa: E402
from helper.registry import load_registry  # noqa: E402
from helper.terminal_force import MODEL_FIELD, STIFFNESS_MODEL_FIELD  # noqa: E402


class Group9TerminalIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_registry()
        self.config = load_group10_config()
        self.forms = pd.read_csv(PROJECT_ROOT / 'data/processed_v2/formulations.csv')
        self.candidate = self.forms.iloc[[0]].copy()
        self.candidate['candidate_id'] = 'candidate_one'
        self.candidate['selection_rank'] = 1

    def _raw(self, path: Path) -> None:
        time = np.arange(205) * .025
        displacement = np.arange(205) * .005 + 2
        force = np.maximum(0., 1 + (displacement-displacement[4]) * 10)
        force[:4] = [.1, .4, .6, .9]
        pd.DataFrame({'Time': time, 'Force': force, 'Displacement': displacement}).to_csv(path, index=False)

    def _ingest(self, sheet: Path, raw: Path):
        rows = [
            {
                'formulation_id': self.candidate.iloc[0].formulation_id,
                'replicate_id': 'mechanical_001',
                'instron_file': str(raw),
                'needles_compressed': 100,
            },
            {
                'formulation_id': self.candidate.iloc[0].formulation_id,
                'replicate_id': 'intact_final',
                'intact_patch_formation_pass': 'pass',
            },
        ]
        pd.DataFrame(rows).to_csv(sheet, index=False)
        return ingest_feedback(
            sheet, [self.candidate_path], self.forms, pd.DataFrame(), self.registry,
            'ROUND_010', proposal_metadata={'group10': {'effective_config': self.config}},
        )[1]

    def test_candidate_level_gate_and_terminal_stiffness(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.candidate_path = root / 'candidate.csv'
            self.candidate.to_csv(self.candidate_path, index=False)
            raw, sheet = root / 'raw.csv', root / 'sheet.csv'
            self._raw(raw)
            observations = self._ingest(sheet, raw)
            self.assertEqual(observations.endpoint.eq(MODEL_FIELD).sum(), 1)
            self.assertEqual(observations.endpoint.eq(STIFFNESS_MODEL_FIELD).sum(), 1)
            stiffness = observations.loc[observations.endpoint.eq(STIFFNESS_MODEL_FIELD)].iloc[0]
            self.assertAlmostEqual(stiffness.value, .1)
            provenance = json.loads(stiffness.analysis_provenance)
            self.assertEqual(provenance['stiffness_formula'], '(terminal_force_N-trigger_force_N)/0.8_mm')

    def test_failed_formulation_gate_rejects_separate_mechanical_row(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.candidate_path = root / 'candidate.csv'
            self.candidate.to_csv(self.candidate_path, index=False)
            raw, sheet = root / 'raw.csv', root / 'sheet.csv'
            self._raw(raw)
            pd.DataFrame([
                {'formulation_id': self.candidate.iloc[0].formulation_id,
                 'replicate_id': 'mechanical_001', 'instron_file': str(raw), 'needles_compressed': 100},
                {'formulation_id': self.candidate.iloc[0].formulation_id,
                 'replicate_id': 'intact_final', 'intact_patch_formation_pass': 'fail'},
            ]).to_csv(sheet, index=False)
            with self.assertRaisesRegex(ValueError, 'formulation-level all-pass'):
                ingest_feedback(
                    sheet, [self.candidate_path], self.forms, pd.DataFrame(), self.registry,
                    'ROUND_010', proposal_metadata={'group10': {'effective_config': self.config}},
                )

    def test_negative_secant_keeps_force_in_observations_training_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.candidate_path = root / 'candidate.csv'
            self.candidate.to_csv(self.candidate_path, index=False)
            raw, sheet = root / 'raw.csv', root / 'sheet.csv'
            self._raw(raw)
            trace = pd.read_csv(raw)
            trace.loc[80:, 'Force'] = .5
            trace.to_csv(raw, index=False)
            observations = self._ingest(sheet, raw)
            self.assertAlmostEqual(observations.loc[observations.endpoint.eq(MODEL_FIELD), 'value'].item(), .005)
            self.assertAlmostEqual(observations.loc[observations.endpoint.eq(STIFFNESS_MODEL_FIELD), 'value'].item(), -.00625)
            training = build_training_frame(self.forms, observations, self.registry)
            self.assertAlmostEqual(training[MODEL_FIELD].dropna().item(), .005)
            table = mechanical_report_table(observations, self.candidate, batch_id='ROUND_010')
            self.assertEqual(table.stiffness_qc.tolist(), ['negative_secant'])
            self.assertAlmostEqual(table.terminal_force_N_per_needle.item(), .005)
            self.assertAlmostEqual(table.apparent_secant_stiffness_N_per_mm_per_needle.item(), -.00625)

    def test_duplicate_keys_fail_and_automatic_ids_reserve_explicit_ids(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            candidate_path, sheet = root / 'candidate.csv', root / 'sheet.csv'
            self.candidate.to_csv(candidate_path, index=False)
            fid = self.candidate.iloc[0].formulation_id
            pd.DataFrame([
                {'formulation_id': fid, 'replicate_id': 'same', 'viability_percent': 50},
                {'formulation_id': fid, 'replicate_id': 'same', 'viability_percent': 51},
            ]).to_csv(sheet, index=False)
            with self.assertRaisesRegex(ValueError, 'Duplicate observation keys'):
                ingest_feedback(sheet, [candidate_path], self.forms, pd.DataFrame(), self.registry, 'UNIT_BATCH')
            pd.DataFrame([
                {'formulation_id': fid, 'replicate_id': '', 'viability_percent': 50},
                {'formulation_id': fid, 'replicate_id': 'rep_001', 'viability_percent': 51},
            ]).to_csv(sheet, index=False)
            observations = ingest_feedback(
                sheet, [candidate_path], self.forms, pd.DataFrame(), self.registry, 'UNIT_BATCH'
            )[1]
            self.assertEqual(set(observations.replicate_id), {'rep_001', 'rep_002'})
            with self.assertRaisesRegex(ValueError, 'already exist'):
                ingest_feedback(
                    sheet, [candidate_path], self.forms, observations, self.registry, 'UNIT_BATCH'
                )

    def test_mechanical_graph_bundle_has_source_metadata_and_300_dpi_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.candidate_path = root / 'candidate.csv'
            self.candidate.to_csv(self.candidate_path, index=False)
            raw, sheet, output = root / 'raw.csv', root / 'sheet.csv', root / 'plots'
            self._raw(raw)
            observations = self._ingest(sheet, raw)
            paths = write_mechanical(observations, self.candidate, output, batch_id='ROUND_010')
            self.assertEqual(len(paths), 6)
            pngs = sorted(output.glob('*.png'))
            self.assertEqual(len(pngs), 2)
            for png in pngs:
                with Image.open(png) as image:
                    self.assertEqual(image.mode, 'RGBA')
                    self.assertAlmostEqual(image.info['dpi'][0], 300, delta=1)
                self.assertTrue(png.with_name(png.stem + '_source.csv').is_file())
                metadata = json.loads(png.with_name(png.stem + '_metadata.json').read_text())
                self.assertEqual(metadata['dpi'], 300)
                self.assertEqual(len(metadata['source_sha256']), 64)

    def test_existing_logical_key_rejected_across_instron_and_manual_input(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            candidate, sheet, raw = root / 'candidate.csv', root / 'sheet.csv', root / 'raw.csv'
            self.candidate.to_csv(candidate, index=False)
            self._raw(raw)
            fid = self.candidate.iloc[0].formulation_id
            pd.DataFrame([{
                'formulation_id': fid, 'replicate_id': 'mechanical_001',
                'intact_patch_formation_pass': 'pass', 'instron_file': str(raw),
                'needles_compressed': 100,
            }]).to_csv(sheet, index=False)
            existing = ingest_feedback(
                sheet, [candidate], self.forms, pd.DataFrame(), self.registry, 'UNIT_BATCH'
            )[1]
            pd.DataFrame([
                {'formulation_id': fid, 'replicate_id': 'mechanical_001', MODEL_FIELD: .09},
                {'formulation_id': fid, 'replicate_id': 'intact_final', 'intact_patch_formation_pass': 'pass'},
            ]).to_csv(sheet, index=False)
            before = existing.copy(deep=True)
            with self.assertRaisesRegex(ValueError, 'observation keys already exist'):
                ingest_feedback(sheet, [candidate], self.forms, existing, self.registry, 'UNIT_BATCH')
            pd.testing.assert_frame_equal(existing, before)
            # A new batch or a different endpoint is still a distinct observation.
            updated = ingest_feedback(sheet, [candidate], self.forms, existing, self.registry, 'OTHER_BATCH')[1]
            self.assertEqual(len(updated), len(existing) + 2)
            pd.DataFrame([{'formulation_id': fid, 'replicate_id': 'mechanical_001', 'viability_percent': 50}]).to_csv(sheet, index=False)
            updated = ingest_feedback(sheet, [candidate], self.forms, existing, self.registry, 'UNIT_BATCH')[1]
            self.assertEqual(len(updated), len(existing) + 1)

    def test_changed_raw_source_rejected_before_any_report_is_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.candidate_path = root / 'candidate.csv'
            self.candidate.to_csv(self.candidate_path, index=False)
            raw, sheet, output = root / 'raw.csv', root / 'sheet.csv', root / 'plots'
            self._raw(raw)
            observations = self._ingest(sheet, raw)
            second_raw = root / 'second_raw.csv'
            self._raw(second_raw)
            trace = pd.read_csv(second_raw)
            trace['Force'] *= 1.1
            second_raw.write_text('Specimen metadata\n\n' + trace.to_csv(index=False))
            second = self._ingest(sheet, second_raw)
            second['replicate_id'] = 'mechanical_002'
            observations = pd.concat([observations, second], ignore_index=True)
            write_mechanical(observations, self.candidate, output, batch_id='ROUND_010')
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            trace['Force'] += 10
            second_raw.write_text('Specimen metadata\n\n' + trace.to_csv(index=False))
            with patch('helper.plot_reporting.write_plot', wraps=write_plot) as render:
                with self.assertRaisesRegex(ValueError, 'Raw source hash mismatch'):
                    write_mechanical(observations, self.candidate, output, batch_id='ROUND_010')
                render.assert_not_called()
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)


if __name__ == '__main__':
    unittest.main()

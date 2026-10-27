"""Regressions for molar/millimolar parsing and the source-reviewed v2 release."""
from pathlib import Path
import contextlib
import io
import json
import runpy
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARSER = runpy.run_path(str(ROOT / 'src/01_data_parsing/parse_formulations.py'))
RAW = ROOT / 'data/raw/Cryopreservative Data 2026_v2.csv'
REVIEW = ROOT / 'data/raw/concentration_review_v2.json'


class ConcentrationUnitTests(unittest.TestCase):
    def test_case_sensitive_units_and_case_insensitive_ingredients(self):
        examples = [
            ('0.25 M trehalose', 'trehalose', 0.25, 'M'),
            ('250 mM TREHALOSE', 'trehalose', 250.0, 'mM'),
            ('Trehalose 0.25M', 'TREHALOSE', 0.25, 'M'),
            ('trehalose 250mM', 'TREHALOSE', 250.0, 'mM'),
            ('30 mmol/L trehalose', 'trehalose', 30.0, 'mM'),
            ('TREHALOSE 30 mmol/l', 'trehalose', 30.0, 'mM'),
            ('1.5M EG', 'eg', 1.5, 'M'),
            ('.5 M propylene glycol', 'propylene glycol', 0.5, 'M'),
            ('0.5M PG', 'pg', 0.5, 'M'),
        ]
        for text, ingredient, value, unit in examples:
            with self.subTest(text=text):
                self.assertEqual(PARSER['extract_concentration'](text, ingredient), (value, unit))

    def test_molar_and_millimolar_equivalence(self):
        for text in ['0.25 M trehalose', '250 mM trehalose', '250 mmol/L trehalose']:
            with self.subTest(text=text):
                self.assertEqual(PARSER['parse_formulation_text'](text)['trehalose'], (0.25, 'M'))

    def test_multicomponent_units_are_local_to_each_ingredient(self):
        parsed = PARSER['parse_formulation_text'](
            'PEG 400 (10 wt% in DMEM), 0.25 M trehalose, 50 mM glucose')
        self.assertEqual(parsed['peg_400'], (10.0, '%'))
        self.assertEqual(parsed['trehalose'], (0.25, 'M'))
        self.assertEqual(parsed['glucose'], (0.05, 'M'))

    def test_malformed_or_unsupported_units_do_not_become_molar(self):
        for text in ['o.5M PG', 'o.5mM PG', '0.5m PG', '5 uM PG', '5 µM PG']:
            with self.subTest(text=text):
                self.assertEqual(PARSER['extract_concentration'](text, 'pg'), (None, None))

    def test_percentage_and_mass_concentrations(self):
        for text, name, expected in [
            ('10% DMSO', 'dmso', (10.0, '%')),
            ('10 % (v/v) DMSO', 'dmso', (10.0, '%')),
            ('DMSO 10%', 'dmso', (10.0, '%')),
            ('PEG 400 (10 wt% in DMEM)', 'peg 400', (10.0, '%')),
            ('3.423 mg/mL trehalose', 'trehalose', (3.423, 'mg/ml')),
        ]:
            with self.subTest(text=text):
                self.assertEqual(PARSER['extract_concentration'](text, name), expected)
        self.assertAlmostEqual(PARSER['parse_formulation_text']('10% DMSO')['dmso'][0], 1.407909893766799)
        self.assertAlmostEqual(PARSER['parse_formulation_text']('3.423 mg/mL trehalose')['trehalose'][0], 0.01)

    def test_source_references_are_not_truncated(self):
        frame = pd.DataFrame([
            {'formulation_id': 1, 'viability_percent': 20.0, 'trehalose_M': 0.25, 'source_doi': 'a'*600},
            {'formulation_id': 2, 'viability_percent': 80.0, 'trehalose_M': 0.25, 'source_doi': 'b'*700},
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            resolved = PARSER['auto_resolve_duplicates'](frame, PARSER['find_duplicate_formulations'](frame))
        self.assertEqual(set(resolved.iloc[0].source_doi.split('; ')), {'a'*600, 'b'*700})


class ReviewedV2DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = json.loads(REVIEW.read_text())
        cls.saved = pd.read_csv(ROOT / 'data/processed/parsed_formulations_v2.csv')
        with contextlib.redirect_stdout(io.StringIO()):
            cls.parsed = PARSER['parse_csv'](str(RAW))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            cls.reviewed, cls.groups = PARSER['apply_concentration_review'](cls.parsed, str(RAW), str(REVIEW))
        cls.review_warnings = [str(w.message) for w in caught]
        with contextlib.redirect_stdout(io.StringIO()):
            duplicates = PARSER['find_duplicate_formulations'](cls.reviewed, cls.groups)
            cls.regenerated = PARSER['auto_resolve_duplicates'](
                cls.reviewed, duplicates, protected_groups=cls.groups)

    def test_all_saved_values_reproduce_from_raw_with_documented_review(self):
        self.assertEqual(len(self.parsed), 320)
        self.assertEqual(len(self.saved), 198)
        self.assertEqual(list(self.saved.columns), list(self.regenerated.columns))
        numeric = self.saved.select_dtypes('number').columns
        np.testing.assert_allclose(self.saved[numeric], self.regenerated[numeric], rtol=1e-12, atol=1e-14)
        for expected, actual in zip(self.saved.source_doi, self.regenerated.source_doi):
            self.assertEqual(set(str(expected).split('; ')), set(str(actual).split('; ')))

    def test_all_42_confirmed_concentrations_and_raw_typo(self):
        self.assertEqual(len(self.review['confirmed']), 42)
        self.assertEqual(len({e['formulation_id'] for e in self.review['confirmed']}), 35)
        by_id = self.saved.set_index('formulation_id')
        for entry in self.review['confirmed']:
            with self.subTest(id=entry['formulation_id'], field=entry['field']):
                self.assertEqual(by_id.loc[entry['formulation_id'], entry['field']], entry['correct_M'])
        self.assertEqual(by_id.loc[2, 'trehalose_M'], 0.25)
        self.assertEqual(by_id.loc[102, 'propylene_glycol_M'], 0.5)
        raw = pd.read_csv(RAW)
        self.assertIn('0.5M', raw.iloc[101, 0])
        self.assertNotIn('o.5M', raw.iloc[101, 0])

    def test_five_pending_values_are_explicitly_unverified_and_preserved(self):
        expected = {(237, 'sucrose_M'): 0.0005, (326, 'trehalose_M'): 0.0002,
                    (332, 'ethylene_glycol_M'): 0.0015, (332, 'sucrose_M'): 0.0001,
                    (339, 'sucrose_M'): 0.0003}
        self.assertEqual({(e['formulation_id'], e['field']): e['retained_legacy_M']
                          for e in self.review['pending']}, expected)
        by_id = self.saved.set_index('formulation_id')
        for key, value in expected.items():
            self.assertEqual(by_id.loc[key[0], key[1]], value)
        self.assertTrue(any('5 concentration fields remain UNVERIFIED' in m for m in self.review_warnings))
        # Pure parsing remains correct; the unresolved source hold is a separate step.
        self.assertEqual(self.parsed.set_index('formulation_id').loc[237, 'sucrose_M'], 0.5)

    def test_equal_compositions_keep_baseline_viability_targets(self):
        by_id = self.regenerated.set_index('formulation_id')
        for record_id, expected in [(2, 12.5), (386, 77.0), (41, 51.5), (372, 22.5)]:
            self.assertEqual(by_id.loc[record_id, 'viability_percent'], expected)
        features = [c for c in by_id if c.endswith(('_M', '_pct'))]
        for first, second in [(2, 386), (41, 372)]:
            np.testing.assert_allclose(by_id.loc[first, features].astype(float),
                                       by_id.loc[second, features].astype(float))

    def test_stale_input_review_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / 'changed.csv'
            changed.write_bytes(RAW.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'Raw CSV changed'):
                PARSER['apply_concentration_review'](self.parsed, str(changed), str(REVIEW))

    def test_unexpected_parsed_value_cannot_be_silently_overridden(self):
        changed = self.parsed.copy()
        changed.loc[changed.formulation_id == 2, 'trehalose_M'] = 0.00025
        with self.assertRaisesRegex(ValueError, 'differs from review'):
            PARSER['apply_concentration_review'](changed, str(RAW), str(REVIEW))


if __name__ == '__main__':
    unittest.main()

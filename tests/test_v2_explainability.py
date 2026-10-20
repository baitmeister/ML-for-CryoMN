"""Scientific and provenance checks for report-only ingredient explanations."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/08_multi_objective"))

from helper.batch_gp import BatchGP
from helper.explainability import (
    ExplanationSettings, batch_gp_mean, compute_shap, endpoint_predictors,
    explain_round, load_round_evidence, preceding_observations,
    load_saved_reports, signed_history_table, write_history,
    presence_history_table, presence_history_figure,
)
from helper.registry import IngredientRegistry, load_registry


class ShapNumericsTests(unittest.TestCase):
    def test_linear_contributions_additivity_constant_feature_and_rng(self):
        background = np.array([[0., 2., 7.], [2., 4., 7.]])
        explained = np.array([[4., 5., 7.], [1., 1., 7.]])
        coefficients = np.array([3., -2., 0.])
        def predict(x):
            return 5 + x @ coefficients
        settings = ExplanationSettings(permutations=2)
        np.random.seed(127)
        state = np.random.get_state()
        values, base, predicted = compute_shap(predict, explained, background, settings)
        restored = np.random.get_state()
        np.testing.assert_array_equal(state[1], restored[1])
        self.assertEqual(state[2:], restored[2:])
        np.testing.assert_allclose(values, (explained - background.mean(0)) * coefficients, atol=1e-10)
        np.testing.assert_allclose(base + values.sum(1), predicted, atol=1e-10)
        np.testing.assert_array_equal(values[:, 2], 0)

    def test_mean_only_predictor_matches_full_batch_gp_with_controls(self):
        x = np.array([[0.], [1.], [2.], [.5]])
        model = BatchGP.fit(x, [20., 50., 70., 40.], ['ROUND_001'] * 4,
                            controls=[False, False, False, True], strategy='control')
        test = np.array([[.2], [.7], [1.5]])
        np.testing.assert_allclose(batch_gp_mean(model)(test), model.predict(test), atol=1e-10)

    def test_interaction_attributions_conserve_prediction(self):
        background = np.array([[0., 0., 5.]])
        x = np.array([[2., 3., 5.]])
        values, base, prediction = compute_shap(lambda a: a[:, 0] * a[:, 1], x, background,
                                                 ExplanationSettings(permutations=2))
        np.testing.assert_allclose(values, [[3., 3., 0.]], atol=1e-10)
        np.testing.assert_allclose(base + values.sum(1), prediction)


class HistoricalEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forms = pd.read_csv(ROOT / 'data/processed_v2/formulations.csv')
        cls.obs = pd.read_csv(ROOT / 'data/processed_v2/observations.csv')
        cls.registry = load_registry(ROOT / 'config_v2/ingredients.yaml')

    def test_target_and_future_measurements_do_not_enter_cutoff(self):
        obs = pd.DataFrame({'batch_id': ['legacy_literature', 'ROUND_001', 'ROUND_002', 'ROUND_003'],
                            'value': [10., 20., 30., 40.]})
        prior = preceding_observations(obs, 2)
        self.assertEqual(prior.value.tolist(), [10., 20.])
        changed = obs.copy()
        changed.loc[changed.batch_id.isin(['ROUND_002', 'ROUND_003']), 'value'] = 9999.
        pd.testing.assert_frame_equal(prior, preceding_observations(changed, 2))

    def test_snapshot_strategy_and_rows_are_reused(self):
        prior, snapshot, info = load_round_evidence(ROOT, 14, self.obs, self.registry.feature_names)
        self.assertIsNotNone(snapshot)
        self.assertEqual(info['viability_strategy'], 'control')
        models = endpoint_predictors(self.forms, prior, self.registry, 14, snapshot, 'control', {})
        viability = models['viability_percent']
        self.assertEqual(viability['model_evidence'], 'frozen_gp_comparison_snapshot')
        self.assertTrue(viability['rows'].formulation_id.str.startswith('v2_').all())
        x = viability['rows'][self.registry.feature_names].to_numpy(float)[:4]
        np.testing.assert_allclose(viability['predict'](x), snapshot.predict(x), atol=1e-10)
        self.assertEqual(models['critical_axial_load_N_per_needle']['mechanical_definition_id'], 'terminal_force_08mm_after_1N_v1')

    def test_no_early_mechanics_or_gate_importance_is_fabricated(self):
        prior = preceding_observations(self.obs, 1)
        models = endpoint_predictors(self.forms, prior, self.registry, 1, None, 'current', {})
        self.assertEqual(models['critical_axial_load_N_per_needle']['status'], 'unavailable')
        self.assertEqual(models['intact_patch_formation_pass']['status'], 'unavailable')
        self.assertEqual(models['viability_percent']['status'], 'available')

    def test_older_frozen_archive_with_empty_roles_can_be_explained(self):
        from helper.gp_comparison import attach_roles
        prior, snapshot, info = load_round_evidence(ROOT, 10, self.obs, self.registry.feature_names)
        prior = attach_roles(prior, ROOT)
        models = endpoint_predictors(self.forms, prior, self.registry, 10, snapshot, info['viability_strategy'], {})
        self.assertEqual(info['viability_strategy'], 'current')
        self.assertEqual(models['viability_percent']['model_evidence'], 'frozen_gp_comparison_snapshot')

    def test_corrupt_snapshot_is_rejected(self):
        source = ROOT / 'results/multi_objective_v2/rounds/ROUND_014'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root / 'results/multi_objective_v2/rounds/ROUND_014'
            (dest / 'proposal').mkdir(parents=True)
            (dest / 'proposal/gp_strategy_decision.json').write_bytes((source / 'proposal/gp_strategy_decision.json').read_bytes())
            (dest / 'gp_comparison').mkdir()
            for name in ('manifest.json', 'training_observations.csv', 'control.json'):
                (dest / 'gp_comparison' / name).write_bytes((source / 'gp_comparison' / name).read_bytes())
            (dest / 'gp_comparison/control.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'model hash mismatch'):
                load_round_evidence(root, 14, self.obs, self.registry.feature_names)

    def test_actual_report_additivity_provenance_and_source_immutability(self):
        from PIL import Image
        paths = [ROOT / 'data/processed_v2/formulations.csv', ROOT / 'data/processed_v2/observations.csv',
                 ROOT / 'results/multi_objective_v2/rounds/ROUND_014/proposal/proposal.csv',
                 ROOT / 'results/multi_objective_v2/rounds/ROUND_014/gp_comparison/control.json']
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            report = explain_round(ROOT, self.forms, self.obs, self.registry, 14, output,
                                   ExplanationSettings(max_explain=3, max_background=3, permutations=2))
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['training_cutoff'], 13)
            for item in manifest['endpoints'].values():
                if item['status'] == 'available':
                    self.assertLess(item['additivity_max_absolute_error'], 1e-7)
            self.assertEqual(manifest['endpoints']['preparation_feasibility_pass']['status'], 'unavailable')
            self.assertFalse(report['importance'].empty)
            with Image.open(output / 'feature_importance.png') as image:
                self.assertEqual(image.mode, 'RGBA')
                self.assertEqual(image.getchannel('A').getextrema()[0], 0)
                self.assertAlmostEqual(image.info['dpi'][0], 300, delta=1)
        self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


class SignedHistoryTests(unittest.TestCase):
    def test_signed_means_preserve_direction_and_cancellation(self):
        table = pd.DataFrame({
            'target_round': [1, 1, 1, 2], 'endpoint': ['viability_percent'] * 4,
            'feature': ['dmso_M', 'ectoin_M', 'glycerol_M', 'dmso_M'],
            'mean_abs_shap': [4., 3., 3., 0.], 'mean_signed_shap': [-2., 1., 0., 0.],
            'relative_importance_percent': [40., 30., 30., 0.],
        })
        result = signed_history_table([{'importance': table}])
        np.testing.assert_allclose(result.signed_relative_contribution_percent, [-20., 10., 0., 0.])
        self.assertNotIn('signed_relative_contribution_percent', table)
        self.assertEqual(result.iloc[2].mean_abs_shap, 3.)

    def test_history_render_is_symmetric_and_keeps_unavailable_rounds_missing(self):
        import matplotlib.pyplot as plt
        reports = []
        for target in (1, 2):
            rows = pd.DataFrame({'target_round': [target, target], 'endpoint': ['viability_percent'] * 2,
                                 'feature': ['dmso_M', 'ectoin_M'], 'mean_abs_shap': [4., 6.],
                                 'mean_signed_shap': [-2., 1.], 'relative_importance_percent': [40., 60.]})
            if target == 2:
                rows = pd.concat([rows, pd.DataFrame({'target_round': [2],
                    'endpoint': ['critical_axial_load_N_per_needle'], 'feature': ['dmso_M'],
                    'mean_abs_shap': [1.], 'mean_signed_shap': [-.5], 'relative_importance_percent': [100.]})], ignore_index=True)
            reports.append({'importance': rows, 'manifest': {'target_round': target, 'training_cutoff': target - 1,
                            'viability_strategy': 'current', 'training_evidence': 'test_fixture'}})
        with tempfile.TemporaryDirectory() as tmp, patch('helper.explainability.save_png') as save:
            directory = Path(tmp)
            write_history(reports, directory, load_registry(), ExplanationSettings())
            figure = save.call_args.args[0]
            viability, mechanics = figure.axes[:2]
            self.assertEqual(viability.images[0].get_clim(), (-50., 50.))
            np.testing.assert_allclose(viability.images[0].get_array(), [[10., 10.], [-20., -20.]])
            matrix = mechanics.images[0].get_array()
            self.assertTrue(matrix.mask[0, 0])
            self.assertEqual(matrix[0, 1], -50.)
            metadata = json.loads((directory / 'feature_importance_history_metadata.json').read_text())
            self.assertEqual(metadata['color_scale_percent'], [-50., 50.])
            plt.close(figure)

    def test_saved_history_reuses_verified_signed_values(self):
        reports = load_saved_reports(ROOT / 'results/multi_objective_v2/reports/explainability')
        self.assertEqual([r['manifest']['target_round'] for r in reports], list(range(1, 15)))
        result = signed_history_table(reports)
        self.assertTrue(result.signed_relative_contribution_percent.lt(0).any())
        self.assertTrue(result.signed_relative_contribution_percent.gt(0).any())


class PresenceHistoryTests(unittest.TestCase):
    def fixture(self):
        registry = load_registry()
        registry = IngredientRegistry([registry.get_by_feature(feature) for feature in ['dmso_M', 'ectoin_M']])
        values = []
        for index, (concentration, contribution) in enumerate([(1., 2.), (2., 4.), (0., -12.)]):
            for feature in registry.feature_names:
                values.append({'endpoint': 'viability_percent', 'row_index': index,
                    'formulation_id': 'repeated' if index < 2 else 'other', 'feature': feature,
                    'feature_value': concentration if feature == 'dmso_M' else 0.,
                    'shap_value': contribution if feature == 'dmso_M' else 0.})
        manifest = {'target_round': 1, 'training_cutoff': 0, 'viability_strategy': 'current',
                    'endpoints': {'viability_percent': {'status': 'available', 'training_rows': 3,
                    'explained_rows': 3, 'model_evidence': 'test_fixture'},
                    'critical_axial_load_N_per_needle': {'status': 'unavailable', 'training_rows': 0,
                    'model_evidence': 'test_fixture'}}}
        return registry, {'manifest': manifest, 'shap_values': pd.DataFrame(values)}

    def test_conditioning_excludes_absent_rows_and_counts_recurring_formulations(self):
        registry, report = self.fixture()
        table = presence_history_table([report], registry)
        dmso = table.loc[table.endpoint.eq('viability_percent') & table.feature.eq('dmso_M')].iloc[0]
        self.assertEqual(dmso.mean_shap_present, 3.)
        self.assertEqual(dmso.n_explained, 3)
        self.assertEqual(dmso.n_present, 2)
        self.assertEqual(dmso.n_distinct_formulations_present, 1)
        self.assertEqual(dmso.positive_shap_fraction_present, 1.)

    def test_no_present_rows_and_unavailable_models_are_not_zero_effects(self):
        registry, report = self.fixture()
        table = presence_history_table([report], registry)
        absent = table.loc[table.endpoint.eq('viability_percent') & table.feature.eq('ectoin_M')].iloc[0]
        unavailable = table.loc[table.endpoint.eq('critical_axial_load_N_per_needle')].iloc[0]
        self.assertEqual(absent.status, 'no_present_explained_rows')
        self.assertEqual(absent.n_present, 0)
        self.assertTrue(pd.isna(absent.mean_shap_present))
        self.assertEqual(unavailable.status, 'model_unavailable')
        self.assertTrue(pd.isna(unavailable.mean_shap_present))

    def test_duplicate_shap_rows_cannot_inflate_sample_counts(self):
        registry, report = self.fixture()
        report['shap_values'] = pd.concat([report['shap_values'], report['shap_values'].iloc[[0]]])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            presence_history_table([report], registry)

    def test_actual_saved_samples_have_expected_counts_and_conditional_signs(self):
        registry = load_registry()
        reports = load_saved_reports(ROOT / 'results/multi_objective_v2/reports/explainability', include_shap_values=True)
        table = presence_history_table(reports, registry)
        current = table.loc[table.target_round.eq(14)]
        glycerol = current.loc[current.feature.eq('glycerol_M')].set_index('endpoint')
        self.assertEqual(glycerol.loc['viability_percent', 'n_present'], 6)
        self.assertEqual(glycerol.loc['critical_axial_load_N_per_needle', 'n_present'], 8)
        self.assertTrue(glycerol.mean_shap_present.gt(0).all())
        ectoin = current.loc[current.feature.eq('ectoin_M')].set_index('endpoint')
        self.assertGreater(ectoin.loc['viability_percent', 'mean_shap_present'], 0)
        self.assertLess(ectoin.loc['critical_axial_load_N_per_needle', 'mean_shap_present'], 0)
        fig, scales = presence_history_figure(table, registry, ['critical_axial_load_N_per_needle'])
        self.assertIn('n=8', [text.get_text().split('\n')[-1] for text in fig.axes[0].texts])
        self.assertEqual(scales['critical_axial_load_N_per_needle'][0], -scales['critical_axial_load_N_per_needle'][1])
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_corrupted_row_level_source_is_rejected_before_rendering(self):
        source = ROOT / 'results/multi_objective_v2/reports/explainability/current'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'current'
            directory.mkdir()
            for name in ['manifest.json', 'feature_importance.csv', 'shap_values.csv']:
                (directory / name).write_bytes((source / name).read_bytes())
            (directory / 'shap_values.csv').write_text('corrupted source')
            with self.assertRaisesRegex(ValueError, 'row-level SHAP hash mismatch'):
                load_saved_reports(root, include_shap_values=True)


if __name__ == '__main__':
    unittest.main()

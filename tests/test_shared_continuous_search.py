"""Scientific parity and real optimizer coverage for the approved GP path."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/08_multi_objective'))
from helper.batch_gp import BatchGP, torch_bundle
from helper.config import load_optimization_config
from helper.continuous_search import (MaskAcquisition, ContinuousSearchError,
                                     mixture_constraints, optimize_shared_candidates,
                                     select_refinement_masks)
from helper.feasibility import feasibility_report, ROUND5_POLICY_VERSION
from helper.gp_strategy import (shared_acquisition, shared_scores, validate_decision,
                                validate_search_config, StrategyDecisionRequired)
from helper.intact_policy import resolve_intact_combination_policy, annotate_intact_combination_evidence
from helper.penalties import constraint_report
from helper.registry import load_registry
from helper.selection_scoring import _mechanics_phase_scores, _continuous_mechanics_candidates, _candidate_masks
from helper.candidate_workflow import CandidateSelectionOptions, run_candidate_selection


class SharedContinuousTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.registry = load_registry()
        cls.config = load_optimization_config()
        cls.policy = resolve_intact_combination_policy(cls.config, 13)
        cls.mask = tuple(cls.registry.feature_names.index(f) for f in ('ectoin_M', 'ethylene_glycol_M'))
        x = np.zeros((12, len(cls.registry.feature_names)))
        x[:, cls.mask[0]] = np.tile([.08, .18, .28, .42], 3)
        x[:, cls.mask[1]] = np.tile([.2, .6, 1., 1.5], 3)
        b = np.repeat(['ROUND_001', 'ROUND_002', 'ROUND_003'], 4)
        y = 65 - 60 * (x[:, cls.mask[0]] - .2) ** 2 + np.repeat([-6., 0., 6.], 4)
        controls = np.zeros((3, x.shape[1]))
        v = BatchGP.fit(np.vstack([x, controls]), np.r_[y, [44., 50., 56.]],
                        np.r_[b, ['ROUND_001', 'ROUND_002', 'ROUND_003']],
                        np.r_[np.zeros(12, bool), np.ones(3, bool)], strategy='control')
        mechanical = .04 + .06 * x[:, cls.mask[1]]
        m = BatchGP.fit(x, mechanical, b, strategy='current', noise=np.full(12, .001))
        frame = pd.DataFrame(x, columns=cls.registry.feature_names)
        frame['viability_percent'] = y
        frame['critical_axial_load_N_per_needle'] = mechanical
        cls.models = SimpleNamespace(acquisition_bundle=torch_bundle([v, m]), training_frame=frame,
                                     gp_strategy_decision={'viability_strategy': 'control'})
        cls.train_x = x
        cls.evidence = pd.DataFrame([
            dict(zip(cls.registry.feature_names, row), active_ingredient_set=('ectoin_M', 'ethylene_glycol_M'),
                 intact_patch_formation_pass=passed)
            for row, passed in zip(x[:3], [1., 0., 1.])
        ])
        cls.lo = np.array([.001, .001])
        cls.hi = np.array([.5, 2.5])

    def wrapper(self):
        return MaskAcquisition(shared_acquisition(self.models, self.train_x, (0., 0.)),
                               self.registry, self.config, self.policy, self.evidence,
                               self.mask, self.lo, self.hi)

    def test_same_posterior_score_and_intact_chemistry_adjustment(self):
        wrapper = self.wrapper()
        X = torch.tensor([[[.32, .36]], [[.71, .62]]], dtype=torch.double, requires_grad=True)
        physical = wrapper.physical(X).detach().numpy()[:, 0, :]
        frame = annotate_intact_combination_evidence(pd.DataFrame(physical, columns=self.registry.feature_names),
                                                     self.evidence, self.registry, self.policy)
        frame['screening_acquisition_penalty'] = [constraint_report(row, self.registry, self.config)['screening_acquisition_penalty'] for _, row in frame.iterrows()]
        expected, _ = _mechanics_phase_scores(frame, self.models, self.registry, self.config, self.policy)
        np.testing.assert_allclose(wrapper(X).detach().numpy(), expected, atol=1e-9, rtol=1e-9)
        direct, _ = shared_scores(self.models, self.train_x, physical, (0., 0.))
        np.testing.assert_allclose(wrapper.acquisition(wrapper.physical(X)).detach().numpy(), direct, atol=1e-9)
        gradient = torch.autograd.grad(wrapper(X).sum(), X)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        for i in range(2):
            plus, minus = X.detach().clone(), X.detach().clone()
            plus[0, 0, i] += 1e-5
            minus[0, 0, i] -= 1e-5
            finite_difference = ((wrapper(plus)[0] - wrapper(minus)[0]) / 2e-5).item()
            self.assertAlmostEqual(gradient[0, 0, i].item(), finite_difference, delta=2e-3)

    def test_finite_gradients_at_evidence_and_bounds(self):
        wrapper = self.wrapper()
        points = np.vstack([self.train_x[0, list(self.mask)], self.lo, self.hi])
        x = torch.tensor((points - self.lo) / (self.hi - self.lo), dtype=torch.double).unsqueeze(1).requires_grad_()
        value = wrapper(x)
        value.sum().backward()
        self.assertTrue(torch.isfinite(value).all())
        self.assertTrue(torch.isfinite(x.grad).all())

    def run_optimizer(self, feasible=None, pool=None, target=2, config=None):
        if pool is None:
            pool = pd.DataFrame(self.train_x[:4], columns=self.registry.feature_names)
        config = self.config if config is None else config
        lower = np.zeros(len(self.registry.feature_names))
        upper = np.array([self.registry.get_by_feature(f).upper_bound for f in self.registry.feature_names])
        def validate(vector):
            return feasibility_report(dict(zip(self.registry.feature_names, vector)), self.registry,
                                      self.config, True, ROUND5_POLICY_VERSION)['feasibility_pass']
        return optimize_shared_candidates(models=self.models, train_x=self.train_x, candidate_pool=pool,
            registry=self.registry, config=config, policy=self.policy, evidence=self.evidence,
            masks=_candidate_masks(pool, self.registry, config, limit=False),
            lower_bounds=lower, upper_bounds=upper, reference=(0., 0.),
            target=target, feasible=feasible or validate, policy_version=ROUND5_POLICY_VERSION)

    def test_single_ingredient_surplus_cannot_hide_promising_mixtures(self):
        scores = {(i,): -float(i) for i in range(21)}
        mixtures = {(0, 1): 10., (0, 1, 2): 12., (0, 1, 2, 3): 11., (1, 2): 9.}
        scores.update(mixtures)
        pool = np.zeros((len(scores), len(self.registry.feature_names)))
        for row, mask in zip(pool, scores):
            row[list(mask)] = .2
        frame = pd.DataFrame(pool, columns=self.registry.feature_names)
        eligible = _candidate_masks(frame, self.registry, self.config, limit=False)
        self.assertEqual(set(eligible), set(scores))
        selected = select_refinement_masks({mask: scores[mask] for mask in eligible}, 16)
        self.assertEqual(len(selected), 16)
        self.assertTrue(set(mixtures).issubset(selected))
        self.assertEqual(selected[0], (0, 1, 2))
        self.assertEqual({len(mask) for mask in selected}, {1, 2, 3, 4})
        self.assertEqual(selected, select_refinement_masks(dict(reversed(list(scores.items()))), 16))

    def test_ingredient_count_coverage_and_small_budgets_are_explicit(self):
        scores = {(i,): 100. - i for i in range(21)}
        scores.update({(0, 1): 4., (0, 1, 2): 3., (0, 1, 2, 3): 2.})
        selected = select_refinement_masks(scores, 16)
        self.assertEqual({len(mask) for mask in selected}, {1, 2, 3, 4})
        self.assertEqual(select_refinement_masks(scores, 2), [(0,), (0, 1)])
        with self.assertRaisesRegex(ValueError, 'finite'):
            select_refinement_masks({(0,): float('nan')}, 16)

    def test_real_optimizer_attempts_single_pair_triple_and_quadruple_sets(self):
        recipes = [dict(ectoin_M=.12), dict(ectoin_M=.12, ethylene_glycol_M=.6),
                   dict(ectoin_M=.12, ethylene_glycol_M=.6, glycerol_M=.15),
                   dict(ectoin_M=.12, ethylene_glycol_M=.6, glycerol_M=.15, dmso_M=.05)]
        pool = pd.DataFrame(recipes).reindex(columns=self.registry.feature_names).fillna(0.)
        for _, row in pool.iterrows():
            self.assertTrue(feasibility_report(row, self.registry, self.config, True, ROUND5_POLICY_VERSION)['feasibility_pass'])
        with patch('helper.batch_gp.BatchGP.fit', side_effect=AssertionError('must not refit')):
            _, metadata = self.run_optimizer(pool=pool, target=4)
        self.assertEqual(metadata['optimized_mask_size_counts'], {1: 1, 2: 1, 3: 1, 4: 1})
        self.assertEqual(metadata['optimized_mask_count'], metadata['masks_requested'])
        self.assertTrue(all(record['attempts'] > 0 for record in metadata['mask_records']))
        annotated = annotate_intact_combination_evidence(pool, self.evidence, self.registry, self.policy)
        annotated['screening_acquisition_penalty'] = [constraint_report(row, self.registry, self.config)['screening_acquisition_penalty'] for _, row in pool.iterrows()]
        expected, _ = _mechanics_phase_scores(annotated, self.models, self.registry, self.config, self.policy)
        for record in metadata['mask_records']:
            self.assertAlmostEqual(record['best_seed_score'], expected[len(record['features']) - 1], places=9)

    def test_real_optimizer_uses_approved_model_and_is_reproducible(self):
        with patch('helper.batch_gp.BatchGP.fit', side_effect=AssertionError('must not refit')):
            first, metadata = self.run_optimizer()
            second, _ = self.run_optimizer()
        np.testing.assert_allclose(first, second, atol=1e-10, rtol=0.)
        self.assertGreater(len(first), 0)
        self.assertGreater(metadata['optimizer_attempt_count'], 0)
        self.assertFalse(metadata['continuous_optimizer_fallback'])
        inactive = [i for i in range(first.shape[1]) if i not in self.mask]
        np.testing.assert_array_equal(first[:, inactive], 0.)
        for row in first:
            self.assertTrue(feasibility_report(dict(zip(self.registry.feature_names, row)), self.registry, self.config, True, ROUND5_POLICY_VERSION)['feasibility_pass'])

    def test_polymer_sugar_constraint_matches_existing_feasibility(self):
        mask = tuple(sorted(self.registry.feature_names.index(f) for f in ('trehalose_M', 'pvp_pct')))
        lo = np.array([.001, .1]); hi = np.array([.5, 10.])
        constraints = mixture_constraints(self.registry, self.config, mask, lo, hi, ROUND5_POLICY_VERSION)
        for sugar, expected in ((.29, True), (.31, False)):
            physical = np.array([sugar, 1.])
            normalized = torch.tensor((physical - lo) / (hi - lo), dtype=torch.double)
            passes = all(float((normalized[indices] * weights).sum()) >= bound - 1e-12 for indices, weights, bound in constraints)
            self.assertEqual(passes, expected)

    def test_errors_and_zero_survivors_are_not_fallbacks(self):
        with patch('helper.continuous_search.optimize_acqf', side_effect=RuntimeError('optimizer broke')):
            with self.assertRaisesRegex(ContinuousSearchError, 'optimizer broke'):
                self.run_optimizer()
        with self.assertRaisesRegex(ContinuousSearchError, 'no feasible ingredient sets'):
            self.run_optimizer(feasible=lambda row: False)

    def test_versioned_decisions_and_contradictory_settings(self):
        legacy = json.loads((ROOT / 'results/multi_objective_v2/gp_strategy_decision.json').read_text())
        new = dict(legacy, version=2, acquisition='shared_qlognehvi', search_mode='continuous_required', decision_id='test-continuous')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'decision.json'
            for decision in (legacy, new):
                path.write_text(json.dumps(decision))
                self.assertEqual(validate_decision(path, 13)['version'], decision['version'])
            path.write_text(json.dumps(dict(legacy, search_mode='continuous_required')))
            with self.assertRaises(StrategyDecisionRequired):
                validate_decision(path, 13)
        config = dict(self.config, gp_strategy_decision=new, continuous_qlognehvi={'enabled': False})
        with self.assertRaisesRegex(StrategyDecisionRequired, 'contradicts'):
            validate_search_config(config)
        with patch('helper.acquisition.botorch_available', return_value=False):
            with self.assertRaisesRegex(StrategyDecisionRequired, 'needs torch'):
                validate_search_config(dict(self.config, gp_strategy_decision=new))

    def test_legacy_shared_decision_remains_deliberate_finite_only(self):
        config = dict(self.config, gp_strategy_decision={'version': 1, 'viability_strategy': 'control'})
        pool = pd.DataFrame(self.train_x, columns=self.registry.feature_names)
        generated, metadata = _continuous_mechanics_candidates(pool, pd.DataFrame(), pd.DataFrame(),
                                                               self.models, self.registry, config, True)
        self.assertTrue(generated.empty)
        self.assertEqual(metadata['search_mode_actual'], 'finite_pool')
        self.assertFalse(metadata['continuous_optimizer_fallback'])

    def test_workflow_failure_saves_diagnostics_without_a_proposal(self):
        legacy = json.loads((ROOT / 'results/multi_objective_v2/gp_strategy_decision.json').read_text())
        decision = dict(legacy, version=2, acquisition='shared_qlognehvi', search_mode='continuous_required', decision_id='test-failure')
        row = dict.fromkeys(self.registry.feature_names, 0.)
        row.update(ectoin_M=.123456, betaine_M=.321654, candidate_id='failure_test',
                   formulation_id='failure_test', candidate_origin='sparse_exploration')
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ('formulations.csv', 'observations.csv'):
                (folder / name).write_bytes((ROOT / 'data/processed_v2' / name).read_bytes())
            path = folder / 'decision.json'
            path.write_text(json.dumps(decision))
            pool = folder / 'pool.csv'
            pd.DataFrame([row]).to_csv(pool, index=False)
            options = CandidateSelectionOptions(formulations_path=folder / 'formulations.csv',
                observations_path=folder / 'observations.csv', candidate_pool_path=pool,
                output_dir=folder / 'next_round', total_candidate_pool_path=folder / 'total.csv',
                gp_strategy_decision_path=path, batch_id='ROUND_013')
            error = ContinuousSearchError('simulated optimizer failure', {'optimizer_attempt_count': 1})
            with patch('helper.candidate_workflow.select_next_round', side_effect=error) as selection:
                with self.assertRaisesRegex(ContinuousSearchError, 'simulated optimizer failure'):
                    run_candidate_selection(options)
                selection.assert_called_once()
            self.assertFalse((folder / 'next_round/next_round_candidates.csv').exists())
            self.assertFalse((folder / 'rounds/ROUND_013/proposal/proposal.csv').exists())
            self.assertFalse((folder / 'total.csv').exists())
            failure = json.loads((folder / 'continuous_search_failure.json').read_text())
            self.assertEqual(failure['optimizer_attempt_count'], 1)

    def test_comparison_plot_records_actual_gp_phase_search_and_origin(self):
        from helper.plot_reporting import write_decision
        frame = pd.DataFrame([dict(candidate_id='finite', formulation_id='finite', selection_rank=1,
                                   candidate_origin='finite_pool', generation_origin='local_perturbation')])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            metadata = dict(active_phase='mechanics_enabled', optimizer_mode='finite_pool_plus_continuous_qlognehvi',
                            gp_strategy_decision=dict(viability_strategy='control', decision_id='comparison-only'))
            (folder / 'next_round_metadata.json').write_text(json.dumps(metadata))
            write_decision(frame, folder / 'plots')
            source = pd.read_csv(folder / 'plots/candidate_decisions_source.csv')
            self.assertEqual(source.production_gp.iloc[0], 'Control-informed batch-effect viability GP')
            self.assertEqual(source.search_method.iloc[0], metadata['optimizer_mode'])
            self.assertEqual(source.evidence_phase.iloc[0], 'mechanics_enabled')
            self.assertEqual(source.candidate_origin.iloc[0], 'local_perturbation')


if __name__ == '__main__':
    unittest.main()

"""Behavioral checks for cumulative retirement, bounded confirmations and rescues."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/08_multi_objective'))
from helper.automatic_repeats import (annotate_automatic_repeats, filter_retired_recipes,
                                      load_completed_retest_history)
from helper.registry import load_registry
from helper.retest import build_retest_candidates
from helper.candidates import generate_rescue_candidate_pool
from helper.group10_selection import _allocate_full_phase_slate
from helper.artifacts import supersede_unstarted_proposal, ArtifactConflictError


class AutomaticRepeatTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry()
        self.forms = pd.DataFrame([dict(formulation_id=f'f{i}', **{
            f: (.2 + i * .1 if f == 'glycerol_M' else 0.) for f in self.registry.feature_names})
            for i in range(4)])
        self.config = {'automatic_repeat_policy': {'enabled': True, 'max_cumulative_intact_failures': 2},
                       'retest': {'max_completed_automatic_retests_per_formulation': 1},
                       '_completed_automatic_retest_history': {}}
        prediction = lambda x: SimpleNamespace(mean=np.full(len(x), 50.), std=np.full(len(x), 5.))
        self.models = SimpleNamespace(viability=SimpleNamespace(predict=prediction),
            critical_load=SimpleNamespace(predict=prediction),
            intact=SimpleNamespace(predict_proba=lambda x: np.full(len(x), .8)))

    def observation(self, batch, endpoint, value, fid='f0'):
        return dict(formulation_id=fid, batch_id=f'ROUND_{batch:03d}', endpoint=endpoint,
                    value=value, source_type='wetlab_feedback')

    def test_cumulative_failures_survive_intervening_pass_and_duplicate_readouts(self):
        rows = [self.observation(1, 'intact_patch_formation_pass', 0),
                self.observation(1, 'intact_patch_formation_pass', 0),
                self.observation(2, 'intact_patch_formation_pass', 1),
                self.observation(3, 'intact_patch_formation_pass', 0),
                self.observation(4, 'intact_patch_formation_pass', np.nan),
                self.observation(5, 'intact_patch_formation_pass', ' ')]
        frame = annotate_automatic_repeats(self.forms, self.forms, pd.DataFrame(rows), self.registry, self.config)
        self.assertEqual(frame.iloc[0].cumulative_intact_failure_count, 2)
        self.assertTrue(frame.iloc[0].automatic_repeat_retired)
        for origin in ('retest', 'finite_pool', 'screened_hit_mechanics', 'rescue_dilution'):
            candidate = self.forms.head(1).assign(candidate_origin=origin)
            self.assertTrue(filter_retired_recipes(candidate, self.forms, pd.DataFrame(rows), self.registry, self.config).empty)

    def test_exact_recipe_aliases_share_counts_and_future_batches_are_excluded(self):
        alias = self.forms.head(1).assign(formulation_id='alias')
        forms = pd.concat([self.forms, alias], ignore_index=True)
        rows = pd.DataFrame([self.observation(1, 'intact_patch_formation_pass', 0),
                             self.observation(1, 'intact_patch_formation_pass', 0, 'alias'),
                             self.observation(5, 'intact_patch_formation_pass', 0, 'alias')])
        config = {**self.config, '_gp_target_round': 5}
        checked = annotate_automatic_repeats(forms, forms, rows, self.registry, config)
        self.assertEqual(checked.iloc[0].cumulative_intact_failure_count, 1)
        self.assertEqual(checked.iloc[-1].cumulative_intact_failure_count, 1)

    def test_retest_history_requires_frozen_role_and_finite_viability(self):
        rows = pd.DataFrame([self.observation(1, 'viability_percent', 50),
                             self.observation(1, 'intact_patch_formation_pass', 0),
                             self.observation(2, 'viability_percent', np.nan),
                             self.observation(3, 'viability_percent', 60),
                             self.observation(4, 'viability_percent', float('inf'))])
        with tempfile.TemporaryDirectory() as tmp:
            for batch in range(1, 6):
                path = Path(tmp) / f'ROUND_{batch:03d}/proposal/proposal.csv'
                path.parent.mkdir(parents=True)
                with path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=['formulation_id', 'experimental_role'])
                    writer.writeheader()
                    writer.writerow(dict(formulation_id='f0', experimental_role='ordinary' if batch == 3 else 'retest_priority'))
            history, hashes = load_completed_retest_history(rows, tmp, target_round=5)
        self.assertEqual(history, {'f0': ['ROUND_001']})
        self.assertEqual(len(hashes), 2)

    def test_confirmation_cap_blocks_persistent_historical_range(self):
        rows = pd.DataFrame([self.observation(b, 'viability_percent', v)
                             for b, v in [(1, 20), (2, 60), (3, 55)]])
        initial = build_retest_candidates(self.forms, rows, self.models, self.registry, self.config)
        self.assertEqual(initial.formulation_id.tolist(), ['f0'])
        self.assertIn('completed_automatic_retests=0/1', initial.iloc[0].selection_explanation)
        self.assertNotIn('viability_std_tiebreak', initial.iloc[0].selection_explanation)
        confirmed = {**self.config, '_completed_automatic_retest_history': {'f0': ['ROUND_003']}}
        self.assertTrue(build_retest_candidates(self.forms, rows, self.models, self.registry, confirmed).empty)

    def test_existing_numeric_trigger_boundaries_and_retirement(self):
        sd_rows = pd.DataFrame([self.observation(1, 'viability_percent', v) for v in (42, 50, 58)])
        result = build_retest_candidates(self.forms, sd_rows, self.models, self.registry, self.config)
        self.assertEqual(result.formulation_id.tolist(), ['f0'])
        self.assertEqual(result.iloc[0].retest_priority_score, 1.)
        cross = pd.DataFrame([self.observation(1, 'viability_percent', 50), self.observation(2, 'viability_percent', 65)])
        self.assertEqual(build_retest_candidates(self.forms, cross, self.models, self.registry, self.config).formulation_id.tolist(), ['f0'])
        failed = pd.concat([cross, pd.DataFrame([self.observation(b, 'intact_patch_formation_pass', 0) for b in (1, 2)])])
        self.assertTrue(build_retest_candidates(self.forms, failed, self.models, self.registry, self.config).empty)

    def test_supplied_retest_rows_cannot_bypass_completed_confirmation_cap(self):
        obs = pd.DataFrame([self.observation(1, 'viability_percent', 50)])
        config = {**self.config, '_completed_automatic_retest_history': {'f0': ['ROUND_001']}}
        for tag in ({'recommendation_type': 'retest_priority'}, {'experimental_role': 'retest_priority'}, {'candidate_origin': 'retest'}):
            supplied = self.forms.head(1).assign(**tag)
            self.assertTrue(filter_retired_recipes(supplied, self.forms, obs, self.registry, config).empty)
        self.assertEqual(len(filter_retired_recipes(self.forms.head(1).assign(candidate_origin='screened_hit_mechanics'), self.forms, obs, self.registry, config)), 1)

    def test_neighbor_trigger_uses_three_other_tested_recipes(self):
        rows = pd.DataFrame([self.observation(1, 'viability_percent', 50 if i == 0 else 30, f'f{i}') for i in range(4)])
        result = build_retest_candidates(self.forms, rows, self.models, self.registry, self.config)
        self.assertEqual(result.formulation_id.tolist(), ['f0'])
        self.assertAlmostEqual(result.iloc[0].local_neighbor_residual, 20.)

    def test_retired_parent_can_generate_new_rescues_but_tested_variants_are_excluded(self):
        forms = self.forms.head(1).copy()
        forms.loc[0, 'glycerol_M'] = .4
        half = forms.assign(formulation_id='half', glycerol_M=.2)
        forms = pd.concat([forms, half], ignore_index=True)
        obs = pd.DataFrame([self.observation(1, 'viability_percent', 70),
                            self.observation(1, 'intact_patch_formation_pass', 0),
                            self.observation(2, 'viability_percent', 70),
                            self.observation(2, 'intact_patch_formation_pass', 0),
                            self.observation(1, 'viability_percent', 60, 'half')])
        def feasible(frame, *args, **kwargs):
            return frame.assign(feasibility_pass=True)
        with patch('helper.candidates.annotate_feasibility', side_effect=feasible), patch('helper.candidates.annotate_support', side_effect=lambda frame,*a:frame):
            rescue = generate_rescue_candidate_pool(self.registry, forms, obs, self.config, SimpleNamespace())
        self.assertEqual(sorted(rescue.glycerol_M.round(8).tolist()), [.1, .3])
        self.assertTrue(rescue.cumulative_intact_failure_count.eq(0).all())

    def test_optional_slots_backfill_and_control_remains_required(self):
        extra = pd.DataFrame([dict(candidate_id='control', formulation_id='control', experimental_role='campaign_control')])
        pool = pd.DataFrame([dict(candidate_id=f'n{i}', formulation_id=f'n{i}', experimental_role='ordinary',
                                 candidate_origin='finite_pool', mechanics_phase_score=100-i) for i in range(15)])
        chosen=[]
        def add(row): chosen.append(row.to_dict()); return True
        meta = _allocate_full_phase_slate(extra, pool, add, chosen, optional_special_slots=True)
        self.assertEqual(len(chosen), 12)
        self.assertEqual(meta['ordinary_slots'], 11)
        self.assertEqual(meta['selected_retest_or_rescue_kind'], 'none')
        with self.assertRaisesRegex(ValueError, 'control'):
            _allocate_full_phase_slate(extra.iloc[:0], pool, add, [], optional_special_slots=True)

    def test_shared_slot_prefers_retest_then_rescue_then_ordinary(self):
        extra = pd.DataFrame([dict(candidate_id='control', formulation_id='control', experimental_role='campaign_control')])
        pool = pd.DataFrame([dict(candidate_id=f'n{i}', formulation_id=f'n{i}', experimental_role='ordinary',
                                 candidate_origin='finite_pool', mechanics_phase_score=100-i, retest_priority_score=0.) for i in range(15)] +
                            [dict(candidate_id='confirm', formulation_id='confirm', experimental_role='retest_priority', candidate_origin='retest', mechanics_phase_score=-100., retest_priority_score=2.),
                             dict(candidate_id='rescue', formulation_id='rescue', experimental_role='ordinary', candidate_origin='rescue_dilution', mechanics_phase_score=1000., retest_priority_score=0.)])
        for reject, expected in [(set(), 'retest'), ({'confirm'}, 'rescue'), ({'confirm','rescue'}, 'none')]:
            chosen=[]
            def add(row):
                if row.candidate_id in reject: return False
                chosen.append(row.to_dict()); return True
            meta = _allocate_full_phase_slate(extra, pool, add, chosen, optional_special_slots=True)
            self.assertEqual(meta['selected_retest_or_rescue_kind'], expected)
            self.assertEqual(len(chosen), 12)

    def test_superseding_unstarted_proposal_preserves_linked_model_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'rounds/ROUND_014'
            proposal = folder / 'proposal'
            comparison = folder / 'gp_comparison'
            proposal.mkdir(parents=True)
            comparison.mkdir()
            (proposal / 'proposal.csv').write_text('candidate_id\nold\n')
            data = b'old frozen model evidence\n'
            (comparison / 'current.json').write_bytes(data)
            (comparison / 'manifest.json').write_text(json.dumps({'hashes': {
                'current.json': hashlib.sha256(data).hexdigest()}}))
            archive = supersede_unstarted_proposal('ROUND_014', observations=pd.DataFrame(),
                active_worksheet=None, policy_versions=[], reason='Selection policy refresh', results_root=root)
            self.assertEqual((archive / 'gp_comparison/current.json').read_bytes(), data)
            self.assertTrue((archive / 'proposal/proposal.csv').exists())
            self.assertFalse(comparison.exists())

    def test_rescue_acquisition_ties_use_formulation_identity(self):
        extra = pd.DataFrame([dict(candidate_id='control', formulation_id='control', experimental_role='campaign_control')])
        pool = pd.DataFrame([dict(candidate_id=f'n{i}', formulation_id=f'n{i}', experimental_role='ordinary',
                                 candidate_origin='finite_pool', mechanics_phase_score=10-i) for i in range(12)] +
                            [dict(candidate_id='a', formulation_id='z_recipe', experimental_role='ordinary', candidate_origin='rescue_dilution', mechanics_phase_score=100.),
                             dict(candidate_id='z', formulation_id='a_recipe', experimental_role='ordinary', candidate_origin='rescue_dilution', mechanics_phase_score=100.)])
        chosen=[]
        def add(row): chosen.append(row.to_dict()); return True
        meta = _allocate_full_phase_slate(extra, pool, add, chosen, optional_special_slots=True)
        self.assertEqual(meta['selected_retest_or_rescue_id'], 'a_recipe')

    def test_prefilled_endpoint_id_is_not_a_result_but_edits_and_measurements_block_superseding(self):
        for entry, allowed in [({}, True), ({'viability_percent': 60.}, False),
                               ({'mechanical_definition_id': 'different'}, False)]:
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                proposal = root / 'rounds/ROUND_014/proposal'
                proposal.mkdir(parents=True)
                frame = pd.DataFrame([dict(candidate_id='candidate', batch_id='ROUND_014',
                    mechanical_definition_id='endpoint', viability_percent=np.nan)])
                frame.to_csv(proposal / 'proposal.csv', index=False)
                for key, value in entry.items(): frame.loc[0,key] = value
                worksheet = root / 'worksheet.csv'
                frame.to_csv(worksheet,index=False)
                def supersede():
                    return supersede_unstarted_proposal('ROUND_014', observations=pd.DataFrame(),
                        active_worksheet=worksheet, policy_versions=[], reason='Selection refresh', results_root=root)
                if allowed:
                    self.assertIsNotNone(supersede())
                else:
                    with self.assertRaises(ArtifactConflictError): supersede()
                    self.assertTrue((proposal / 'proposal.csv').exists())


if __name__ == '__main__':
    unittest.main()

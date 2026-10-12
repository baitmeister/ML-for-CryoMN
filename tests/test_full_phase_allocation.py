"""Full-phase role budgets and origin-neutral final selection."""
from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/08_multi_objective'))
from helper.group10_selection import _allocate_full_phase_slate


class FullPhaseAllocationTests(unittest.TestCase):
    def setUp(self):
        def row(name, role='ordinary', origin='finite_pool', score=0., priority=0.):
            return dict(candidate_id=name, formulation_id=name, experimental_role=role,
                        candidate_origin=origin, mechanics_phase_score=score,
                        retest_priority_score=priority)
        self.extra = pd.DataFrame([row('control', 'campaign_control'),
                                   row('followup', 'screened_hit_mechanics')])
        self.pool = pd.DataFrame([
            row('retest_low', 'retest_priority', 'retest', 500., 1.),
            row('retest_high', 'retest_priority', 'retest', -500., 10.),
            row('rescue', origin='rescue_dilution', score=1000.),
            *[row(f'continuous_{i}', origin='continuous_qlognehvi', score=100. - i) for i in range(12)],
            *[row(f'finite_{i}', score=10. - i) for i in range(12)],
        ])

    def allocate(self, pool=None, rejected=(), extra=None):
        chosen = []
        def add(row):
            if row.candidate_id in rejected:
                return False
            chosen.append(row.to_dict())
            return True
        metadata = _allocate_full_phase_slate(self.extra if extra is None else extra,
                                             self.pool if pool is None else pool, add, chosen)
        return pd.DataFrame(chosen), metadata

    def test_one_shared_special_slot_and_nine_origins_compete_on_score(self):
        slate, metadata = self.allocate()
        self.assertEqual(slate.iloc[:3].candidate_id.tolist(), ['control', 'followup', 'retest_high'])
        self.assertEqual(slate.iloc[3:].candidate_id.tolist(), [f'continuous_{i}' for i in range(9)])
        self.assertEqual(metadata['ordinary_slots'], 9)
        self.assertEqual(len(slate), 12)
        self.assertFalse(slate.candidate_origin.eq('rescue_dilution').any())
        # Changing source labels alone cannot change selected recipe identities.
        swapped = self.pool.replace({'candidate_origin': {'continuous_qlognehvi': 'finite_pool', 'finite_pool': 'continuous_qlognehvi'}})
        pd.testing.assert_series_equal(slate.candidate_id, self.allocate(pool=swapped)[0].candidate_id)

    def test_constraint_rejections_apply_to_special_and_ordinary_rows(self):
        slate, metadata = self.allocate(rejected=['retest_high', 'continuous_0', 'continuous_1', 'continuous_2', 'continuous_3'])
        self.assertEqual(metadata['selected_retest_or_rescue_id'], 'retest_low')
        self.assertEqual(slate.iloc[3].candidate_id, 'continuous_4')
        self.assertEqual(slate.iloc[-1].candidate_id, 'finite_0')

    def test_rescue_fills_shared_slot_only_when_retests_cannot(self):
        slate, metadata = self.allocate(rejected=['retest_high', 'retest_low'])
        self.assertEqual(metadata['selected_retest_or_rescue_kind'], 'rescue')
        self.assertEqual(slate.candidate_origin.eq('rescue_dilution').sum(), 1)
        self.assertEqual(slate.experimental_role.eq('retest_priority').sum(), 0)

    def test_missing_required_reservations_stop_instead_of_changing_budget(self):
        for role in ('control', 'followup'):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, 'cannot fill required'):
                self.allocate(rejected=[role])
        with self.assertRaisesRegex(ValueError, 'retest/rescue'):
            self.allocate(rejected=['retest_high', 'retest_low', 'rescue'])
        with self.assertRaisesRegex(ValueError, 'control'):
            self.allocate(extra=pd.DataFrame())

    def test_nonfinite_scores_and_insufficient_ordinary_rows_stop(self):
        invalid = self.pool.copy()
        invalid.loc[invalid.candidate_id.eq('continuous_0'), 'mechanics_phase_score'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite mechanics'):
            self.allocate(pool=invalid)
        short = self.pool.loc[~self.pool.experimental_role.eq('ordinary') | self.pool.candidate_origin.eq('rescue_dilution')]
        with self.assertRaisesRegex(ValueError, '3/12'):
            self.allocate(pool=short)


if __name__ == '__main__':
    unittest.main()

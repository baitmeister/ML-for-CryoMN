"""Scientific display invariants for the selected production suite."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/08_multi_objective'))
from helper import plot_data as original
from helper import campaign_plots as review
from helper.paths import OBSERVATIONS_PATH, RESULTS_V2_DIR
from helper.evaluation_metrics import compute_fixed_reference_hypervolume


class ProductionEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prospective=pd.read_csv(RESULTS_V2_DIR/'reports/prospective/tables/prospective_evaluation_table.csv')
        cls.metrics=pd.read_csv(RESULTS_V2_DIR/'reports/prospective/tables/prospective_metrics.csv')
        cls.candidates=pd.read_csv(RESULTS_V2_DIR/'rounds/ROUND_009/proposal/proposal.csv')
        cls.observations=pd.read_csv(OBSERVATIONS_PATH)

    def tearDown(self):
        plt.close('all')

    def test_zero_area_is_evidence_not_missing(self):
        for values,reference in [((40,0),(0,0)),((10,1),(20,2)),((20,2),(20,2))]:
            frame=pd.DataFrame({'viability_percent':[values[0]],'critical_axial_load_N_per_needle':[values[1]]})
            metric=compute_fixed_reference_hypervolume(frame,dict(zip(frame.columns,reference)))
            self.assertEqual(metric['status'],'estimated')
            self.assertEqual(metric['hypervolume'],0)
        frame=frame.astype(float)
        frame.iloc[0]=[np.inf,np.nan]
        metric=compute_fixed_reference_hypervolume(frame,{'viability_percent':0,'critical_axial_load_N_per_needle':0})
        self.assertEqual(metric['status'],'not_estimable')

    def test_decision_assignments_alignment_and_unknowns(self):
        table=original.decision_table(self.candidates)
        self.assertEqual(table.mechanical_assignment.tolist(),[
            'Screen only','Screen only','Backup 6','Backup 10','Primary 1','Backup 9',
            'Backup 8','Primary 2','Primary 3','Backup 7','Backup 5','Primary 4'])
        unknown=table.viability_prediction_status.str.startswith('unknown_')
        self.assertTrue(table.loc[unknown,'public_viability_mean'].isna().all())
        self.assertTrue(table.loc[unknown,'public_viability_std'].isna().all())
        self.assertEqual(table.prior_only.sum(),6)
        self.assertTrue(set(self.candidates.columns).issubset(table.columns))
        fig=review.decision_figure(table);fig.canvas.draw()
        axes=[ax for ax in fig.axes if ax.axison]
        self.assertEqual(len(axes),5)
        for row in (0,11):
            ys=[ax.transData.transform((0,row))[1] for ax in axes]
            np.testing.assert_allclose(ys,[ys[0]]*5)
        # Only known rows get errorbar Line2D data; unknown labels use axis-relative x.
        plotted_rows=[]
        for line in axes[1].lines:
            plotted_rows.extend(np.asarray(line.get_ydata(),dtype=float))
        self.assertTrue(set(plotted_rows).issubset(set(table.index[~unknown])))

    def test_timeline_preserves_gaps_and_data(self):
        table=original._timeline_table(self.observations,self.prospective,self.metrics,self.candidates)
        before=table.copy(deep=True)
        for _ in range(1):
            fig=review.timeline_figure(table);fig.canvas.draw()
            pd.testing.assert_frame_equal(table,before)
        gap=table.loc[~table.round_number.eq(5)]
        series=review.round_series(gap,'mae',range(1,10))
        self.assertTrue(pd.isna(series.loc[5,'value']))

    def test_variants_retain_metrics_and_cohort(self):
        table=original.trust_table(self.prospective)
        old=original._trust_table(self.prospective)
        pd.testing.assert_frame_equal(table[old.columns],old)
        v=table.loc[table.panel.eq('viability_prediction')]
        self.assertEqual(len(v),72)
        self.assertAlmostEqual(v.absolute_error.mean(),27.45603818528404)
        self.assertAlmostEqual(v.signed_error.mean(),25.22353261643731)
        before=table.copy(deep=True)
        for _ in range(1):
            fig=review.trust_figure(table,self.metrics);fig.canvas.draw()
            pd.testing.assert_frame_equal(table,before)

    def test_empty_missing_uncertainty_and_constant_cases(self):
        for case in ['empty','missing_std','constant']:
            data=self.prospective.copy()
            if case=='empty': data=data.iloc[:0]
            elif case=='missing_std': data['prediction_std']=np.nan
            else:
                data['prediction_mean']=50.;data['observed_mean']=30.
                data['prediction_std']=0.;data['absolute_error']=20.
            table=original.trust_table(data)
            for _ in range(1):
                fig=review.trust_figure(table,self.metrics.iloc[:0])
                fig.canvas.draw();plt.close(fig)
        timeline=original._timeline_table(self.observations,self.prospective,self.metrics,self.candidates)
        for _ in range(1):
            review.timeline_figure(timeline.iloc[:0]).canvas.draw()

    def test_a_and_e_use_only_real_evidence(self):
        review.pareto_figure(pd.DataFrame(),pd.DataFrame(),pd.DataFrame()).canvas.draw()
        timeline=original.timeline_table(self.observations,self.prospective,self.metrics,self.candidates)
        trust=original.trust_table(self.prospective)
        fig=review.publication_figure(timeline,trust,self.metrics,pd.DataFrame())
        fig.canvas.draw()
        pending=next(ax for ax in fig.axes if any('Mechanical evidence pending' in t.get_text() for t in ax.texts))
        self.assertEqual(len(pending.collections),0)
        renderer=fig.canvas.get_renderer()
        bounds=[t.get_window_extent(renderer) for t in pending.texts]
        for i,first in enumerate(bounds):
            for second in bounds[i+1:]:self.assertFalse(first.overlaps(second))

if __name__=='__main__':unittest.main()

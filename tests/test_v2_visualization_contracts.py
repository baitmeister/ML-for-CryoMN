"""Production integration, transparent export, and safe historical backfill."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/08_multi_objective'))
from helper.visualization import generate_proposal_artifacts, generate_completed_round_artifacts
from helper.prospective_evaluation import generate_round_prospective_artifacts, generate_campaign_prospective_artifacts
from helper import plot_reporting as reports, campaign_plots as plots
from helper.plot_backfill import restyle_existing, archived_observations, _promote
from helper.paths import RESULTS_V2_DIR
from helper.evaluation_metrics import _round_metrics
from helper.forecast_diagnostics import write_full_mechanics_diagnostics

class V2VisualizationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates=pd.read_csv(RESULTS_V2_DIR/'next_round/next_round_candidates.csv')
        cls.table=pd.read_csv(RESULTS_V2_DIR/'reports/prospective/tables/prospective_evaluation_table.csv')
        cls.metrics=pd.read_csv(RESULTS_V2_DIR/'reports/prospective/tables/prospective_metrics.csv')
        cls.observations=archived_observations(cls.table)

    def tearDown(self):plt.close('all')

    def assert_png(self,path):
        with Image.open(path) as im:
            self.assertEqual(im.mode,'RGBA')
            self.assertEqual(im.getchannel('A').getextrema()[0],0)
            self.assertAlmostEqual(im.info['dpi'][0],300,delta=1)
            self.assertEqual(im.getpixel((0,0))[3],0)

    def test_proposal_png_and_complete_source(self):
        with tempfile.TemporaryDirectory() as name:
            generated=generate_proposal_artifacts(self.candidates,name)
            self.assertEqual([p.name for p in generated],['candidate_decisions.png','candidate_decisions_source.csv','candidate_decisions_metadata.json'])
            self.assert_png(generated[0])
            self.assertTrue(set(self.candidates).issubset(pd.read_csv(generated[1]).columns))
            self.assertFalse(list(Path(name).rglob('*.pdf')))

    def test_optional_e_and_no_pdf_exports(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name)
            reports.write_prospective(self.observations,self.table,self.metrics,self.candidates,output,'Test')
            self.assertFalse((output/'publication_summary.png').exists())
            reports.write_prospective(self.observations,self.table,self.metrics,self.candidates,output,'Test',True)
            self.assertEqual(len(list(output.glob('*.png'))),3)
            for p in output.glob('*.png'):self.assert_png(p)
            self.assertFalse(list(output.glob('*.pdf')))

    def test_single_round_and_campaign_entrypoints(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            with patch('helper.prospective_evaluation.build_round_prospective_table',return_value=self.table[self.table.round_id.eq('ROUND_008')]),patch('helper.prospective_evaluation._completed_round_ids',return_value=['ROUND_008']):
                generated=generate_round_prospective_artifacts('ROUND_008',self.observations,root)
                self.assertIn('surrogate_trust.png',[p.name for p in generated])
                source=pd.read_csv(root/'rounds/ROUND_008/reports/plots/campaign_timeline_source.csv')
                self.assertEqual(set(source.round_id),{'ROUND_008'})
                generated=generate_campaign_prospective_artifacts(self.observations,root,include_publication_summary=True)
                self.assertIn('publication_summary.png',[p.name for p in generated])

            rounds = ['ROUND_008', 'ROUND_018']
            with patch('helper.prospective_evaluation.build_round_prospective_table',
                       side_effect=lambda batch_id, *_args, **_kwargs: self.table.loc[self.table.round_id.eq(batch_id)]), \
                 patch('helper.prospective_evaluation._completed_round_ids', return_value=rounds):
                generated = generate_campaign_prospective_artifacts(self.observations, root)
            publication_source = next(path for path in generated if path.name == 'publication_summary_source.csv')
            source = pd.read_csv(publication_source)
            parity = source.loc[source.panel.eq('viability_prediction')]
            expected = self.table.loc[
                self.table.round_id.eq('ROUND_018') & self.table.endpoint.eq('viability_percent')
                & self.table.formal_metric_eligible
            ]
            self.assertEqual(set(parity.round_id), {'ROUND_018'})
            self.assertEqual(set(parity.parity_cohort), {'full_mechanics'})
            np.testing.assert_allclose(parity.prediction, expected.prediction_mean)
            np.testing.assert_allclose(parity.observed, expected.observed_mean)

    def test_full_mechanics_diagnostics_score_frozen_matches_and_cutoff(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            frozen = root / 'rounds/ROUND_013/gp_comparison'
            frozen.mkdir(parents=True)
            data = pd.DataFrame([
                ('ROUND_012', 'f1', 'bootstrap', 'True', 25, 50, 42, 58),
                ('ROUND_013', 'f1', 'mechanics_enabled', 'True', 38, 48, 40, 56),
                ('ROUND_013', 'f2', 'mechanics_enabled', 'True', 40, 50, 42, 58),
                ('ROUND_014', 'f3', 'mechanics_enabled', 'True', 65, 60, 52, 68),
                ('ROUND_014', 'f4', 'mechanics_enabled', 'False', 0, 100, 90, 110),
            ], columns=['round_id', 'formulation_id', 'active_phase', 'formal_metric_eligible',
                        'observed_mean', 'prediction_mean', 'interval_95_lower', 'interval_95_upper'])
            data['endpoint'] = 'viability_percent'
            evidence = 'pre_outcome_challenger_after_slate_selection'
            pd.DataFrame([
                ('f1', 'control', 'before_batch', 48, 35, 61),
                ('f1', 'control', 'reference_conditioned', 48, 0, 90),
                ('f1', 'batch', 'before_batch', 48, 0, 90),
                ('f2', 'control', 'before_batch', 49, 37, 61),
            ], columns=['formulation_id', 'strategy', 'stage', 'predicted', 'future_lower', 'future_upper']).assign(
                batch_id='ROUND_013', evidence=evidence, is_control=False,
            ).to_csv(frozen / 'predictions.csv', index=False)
            write_full_mechanics_diagnostics(data, root, root, {'ROUND_013': {'viability_strategy': 'control'}}, .95)

            scores = pd.read_csv(root / 'interval_scores.csv').set_index(['round_id', 'interval_type'])
            pooled = scores.loc[('FULL_MECHANICS', 'proposal')]
            self.assertEqual((pooled.n_eligible, pooled.n_scored, pooled.n_covered), (3, 3, 1))
            self.assertAlmostEqual(pooled.mean_interval_score, (96 + 96 + 16) / 3)
            future = scores.loc[('FULL_MECHANICS', 'future_batch')]
            self.assertEqual((future.n_eligible, future.n_scored), (3, 1))
            self.assertEqual(future.mean_interval_score, 26)
            self.assertTrue(np.isnan(scores.loc[('ROUND_014', 'future_batch'), 'mean_interval_score']))

            segments = pd.read_csv(root / 'viability_segments.csv').set_index('segment')
            below = segments.loc['observed_below_cutoff']
            above = segments.loc['observed_at_or_above_cutoff']
            self.assertEqual((below.n, below.n_hits, above.n, above.n_hits), (1, 1, 2, 2))
            self.assertEqual(above.observed_min_percent, 40)
            self.assertTrue(np.isnan(below.r2))
            self.assertAlmostEqual(above.r2, 0.6)

    def test_completed_entrypoint_uses_prepared_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            inputs=({}, {}, pd.DataFrame(), pd.DataFrame())
            with patch('helper.visualization._build_model_evaluation_frames',return_value={}),patch('helper.visualization.prepare_diagnostics',return_value=inputs),patch('helper.visualization._write_best_performers_summary',return_value=None),patch('helper.visualization._write_visualization_summary',return_value=Path(name)/'report_summary.txt'):
                generated=generate_completed_round_artifacts(pd.DataFrame(),self.observations,self.candidates,name,'ROUND_008')
            self.assertEqual({p.stem for p in generated if p.suffix=='.png'},{'observed_tradeoff','diagnostics_1','diagnostics_2'})

    def test_backfill_does_not_fit_and_preserves_unsupported(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);round_dir=root/'rounds/ROUND_008'
            (round_dir/'reports/tables').mkdir(parents=True)
            table=self.table[self.table.round_id.eq('ROUND_008')]
            src=round_dir/'reports/tables/prospective_evaluation_table.csv';table.to_csv(src,index=False)
            self.metrics[self.metrics.scope.eq('round') & self.metrics.round_id.eq('ROUND_008')].to_csv(src.with_name('prospective_metrics.csv'),index=False)
            (round_dir/'proposal/plots').mkdir(parents=True)
            pd.DataFrame({'candidate_id':['old']}).to_csv(round_dir/'proposal/proposal.csv',index=False)
            old=round_dir/'proposal/plots/next_round_candidate_screen.png';old.write_bytes(b'preserve')
            before=src.read_bytes()
            with patch('helper.plot_reporting._cross_validated_predictions',side_effect=AssertionError('Historical fit forbidden')):
                manifest=restyle_existing(root)
            self.assertEqual(src.read_bytes(),before);self.assertEqual(old.read_bytes(),b'preserve')
            self.assertTrue(any(r['status']=='skipped' for r in json.loads(manifest.read_text())['records']))

    def test_invalid_export_does_not_replace_old_plot(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);stage=root/'stage';dest=root/'dest';stage.mkdir();dest.mkdir()
            old=dest/'next_round_candidate_screen.png';old.write_bytes(b'old')
            candidate=stage/'candidate_decisions.png';Image.new('RGB',(10,10),'white').save(candidate)
            with self.assertRaises(ValueError):_promote(stage,dest,[candidate])
            self.assertEqual(old.read_bytes(),b'old')

    def test_diagnostics_preserve_negative_r2_and_metric_sources(self):
        cv=pd.DataFrame({'actual':[0,1,2],'predicted':[2,1,0]})
        frames={'viability_percent':cv}
        paired=pd.DataFrame({'batch_id':['ROUND_001','ROUND_003'],'viability_percent':[30,50],'critical_axial_load_N_per_needle':[1,2]})
        metrics=_round_metrics(paired);before=metrics.copy(deep=True)
        r2=pd.DataFrame({'batch_id':['ROUND_001','ROUND_003'],'viability_r2':[-3,np.nan],'load_r2':[np.nan,.5]})
        figs=plots.diagnostic_figures(frames,frames,metrics,r2)
        for fig in figs:fig.canvas.draw()
        pd.testing.assert_frame_equal(metrics,before)
        trend=[ax for ax in figs[0].axes if ax.get_ylabel().startswith('R²')][0]
        self.assertEqual(trend.get_ylabel(), 'R² from grouped CV')
        self.assertLess(trend.get_ylim()[0],-3)
        self.assertTrue(np.isnan(trend.lines[0].get_ydata()[1]))
        source=reports.diagnostic_source(frames,frames,metrics,r2)
        restored=reports.diagnostic_inputs(source)
        np.testing.assert_allclose(restored[2].normalized_hypervolume,metrics.normalized_hypervolume)

    def test_a_hides_untrained_mechanics_fallback(self):
        fig=plots.pareto_figure(pd.DataFrame(),pd.DataFrame(),self.candidates)
        self.assertFalse(any(ax.collections for ax in fig.axes))
        self.assertTrue(any('Mechanical evidence pending' in t.get_text() for ax in fig.axes for t in ax.texts))

    def test_backfill_rejects_conflicting_archived_aggregates(self):
        table=self.table.iloc[[0,0]].copy();table.iloc[1,table.columns.get_loc('observed_mean')]=1234
        with self.assertRaises(ValueError):archived_observations(table)

if __name__=='__main__':unittest.main()

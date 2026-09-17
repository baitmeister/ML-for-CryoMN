from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/08_multi_objective'))
from helper.audit_sensitivity import settings,training_rows,fit_sensitivity,Setting
from helper.registry import load_registry


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        self.registry=load_registry()
        self.frame=pd.DataFrame([{**{c:0. for c in self.registry.feature_names},'ectoin_M':x,'batch_id':batch,
            'viability_percent':y,'viability_percent__noise':1.,'critical_axial_load_N_per_needle':m}
            for x,y,m,batch in [(0.,20,.1,'ROUND_001'),(.1,40,np.nan,'ROUND_002'),(.2,55,.2,'ROUND_003'),(.3,65,.3,'ROUND_004')]])
        self.x=self.frame[self.registry.feature_names].to_numpy(float)

    def test_major_axes_and_endpoint_specific_noise(self):
        v=settings('viability_percent');m=settings('critical_axial_load_N_per_needle')
        self.assertTrue({'source','scaling','smoothness','length','length_fitting','length_parameterization','amplitude','mean','recency','training_coverage','noise','target_transform'}.issubset({s.factor for s in v.values()}))
        self.assertIn('sens_noise_5pp',v);self.assertNotIn('sens_noise_5pp',m)
        self.assertEqual(len(training_rows(self.frame,'viability_percent',v['sens_paired_only'])),3)
        recent=training_rows(self.frame,'viability_percent',v['sens_recent3'])
        self.assertEqual(set(recent.batch_id),{'ROUND_002','ROUND_003','ROUND_004'})

    def test_variance_is_squared_and_future_interval_adds_noise(self):
        _,_,detail=fit_sensitivity(self.frame,self.x,self.registry,'viability_percent',Setting(noise='fixed',noise_sd=5.))
        self.assertAlmostEqual(detail['noise_variance_model_space'],25.)
        latent=detail['latent_upper']-detail['latent_lower'];future=detail['future_upper']-detail['future_lower']
        self.assertTrue(np.all(future>latent))

    def test_transformed_predictions_are_in_physical_units(self):
        mean,sd,detail=fit_sensitivity(self.frame,self.x,self.registry,'viability_percent',Setting(transform='logit'))
        self.assertTrue(np.all((mean>0)&(mean<100)))
        self.assertTrue(np.isfinite(sd).all())
        self.assertTrue(np.all((detail['future_lower']>=0)&(detail['future_upper']<=100)))

    def test_constant_response_and_force_scale_do_not_use_percent_floor(self):
        frame=self.frame.copy();frame['critical_axial_load_N_per_needle']=.1
        mean,sd,detail=fit_sensitivity(frame,self.x,self.registry,'critical_axial_load_N_per_needle',Setting(noise='spread',noise_sd=.1))
        self.assertTrue(np.isfinite(mean).all());self.assertTrue(np.isfinite(sd).all())
        self.assertLess(detail['noise_variance_model_space'],1e-10)

    def test_selection_stops_without_botorch(self):
        from helper.selection_scoring import _mechanics_phase_scores
        frame=self.frame.dropna(subset=['critical_axial_load_N_per_needle'])
        candidates=frame.copy()
        candidates['viability_ucb']=[25.,50.,70.]
        candidates['critical_axial_load_ucb']=[.1,.3,.2]
        candidates['acquisition_penalty']=0.
        with patch('helper.acquisition.botorch_available',return_value=False),patch('helper.selection_scoring.botorch_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'Selection stopped'):
                _mechanics_phase_scores(candidates,SimpleNamespace(training_frame=frame),self.registry,{})

    def test_selection_stops_when_botorch_scoring_fails(self):
        from helper.selection_scoring import _mechanics_phase_scores
        frame=self.frame.dropna(subset=['critical_axial_load_N_per_needle'])
        with patch('helper.selection_scoring.try_botorch_qlognehvi_scores', return_value=(None, {'botorch_error':'fit failed'})):
            with self.assertRaisesRegex(RuntimeError, 'fit failed'):
                _mechanics_phase_scores(frame,SimpleNamespace(training_frame=frame),self.registry,{})



if __name__=='__main__':unittest.main()

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/08_multi_objective'))
from helper.gp_comparison import freeze,evaluate,load_inputs,reference_subset
from helper.gp_strategy import StrategyDecisionRequired, shared_scores
from helper.candidate_workflow import CandidateSelectionOptions,run_candidate_selection
from helper.models import train_endpoint_models

class ComparisonTests(unittest.TestCase):
    def test_freeze_conditioning_no_leakage_and_immutable(self):
        forms,obs,registry=load_inputs(ROOT)
        obs=obs.loc[~obs.batch_id.eq('ROUND_010')]
        with tempfile.TemporaryDirectory() as d,patch('helper.gp_comparison_plots.render'):
            frozen=Path(d)/'frozen';freeze(ROOT,'ROUND_010',frozen,forms=forms,obs=obs,registry=registry)
            with self.assertRaises(FileExistsError):freeze(ROOT,'ROUND_010',frozen,forms=forms,obs=obs,registry=registry)
            p=pd.read_csv(frozen/'predictions.csv');slate=pd.read_csv(frozen/'slate.csv')
            self.assertEqual(set(p.formulation_id),set(slate.formulation_id))
            fake=pd.DataFrame(dict(formulation_id=slate.formulation_id,batch_id='ROUND_010',endpoint='viability_percent',value=np.arange(len(slate))+40.))
            result=Path(d)/'evaluated';evaluate(frozen,fake,result)
            e=pd.read_csv(result/'predictions.csv')
            for row in e.loc[e.stage.eq('reference_conditioned')].itertuples():self.assertNotIn(row.formulation_id,json.loads(row.conditioning_ids))
            fake.value+=30
            other=Path(d)/'other';evaluate(frozen,fake,other)
            second=pd.read_csv(other/'predictions.csv')
            np.testing.assert_allclose(e.loc[e.stage.eq('before_batch'),'predicted'],second.loc[second.stage.eq('before_batch'),'predicted'])
            (frozen/'batch.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):evaluate(frozen,fake,Path(d)/'tampered')
    def test_gate_before_any_writes(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(StrategyDecisionRequired):run_candidate_selection(CandidateSelectionOptions(batch_id='ROUND_011',output_dir=Path(d)/'next_round'))
            self.assertEqual(list(Path(d).iterdir()),[])
    def test_production_shared_prediction_parity(self):
        import torch
        forms,obs,registry=load_inputs(ROOT)
        from helper.group10_config import production_observations
        config={'gp_strategy_decision':{'viability_strategy':'batch'},'_gp_raw_observations':obs}
        m=train_endpoint_models(forms,production_observations(obs),registry,config)
        x=forms[registry.feature_names].fillna(0).to_numpy(float)[:3]
        prediction=m.viability.predict(x);p=m.acquisition_bundle.posterior(torch.tensor(x,dtype=torch.double))
        np.testing.assert_allclose(prediction.mean,p.mean[:,0].detach().numpy(),atol=1e-7)
        np.testing.assert_allclose(m.critical_load.predict(x).mean,p.mean[:,1].detach().numpy(),atol=1e-7)
        scores,_=shared_scores(m,x,x,(0.,0.));self.assertTrue(np.isfinite(scores).all())
if __name__=='__main__':unittest.main()

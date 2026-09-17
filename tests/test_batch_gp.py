import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/08_multi_objective'))
from helper.batch_gp import BatchGP,torch_bundle
from helper.gp_strategy import validate_decision,StrategyDecisionRequired

class BatchTests(unittest.TestCase):
    def dataset(self,shift=12):
        x=np.tile(np.linspace(0,1,6),3)[:,None]
        b=np.repeat(['ROUND_001','ROUND_002','ROUND_003'],6)
        y=50+10*x[:,0]+np.repeat([-shift,0,shift],6)+np.tile([.2,-.1,.1,-.2,.1,-.1],3)
        return x,y,b
    def test_recovers_shared_shifts(self):
        x,y,b=self.dataset();m=BatchGP.fit(x,y,b)
        np.testing.assert_allclose([r['offset'] for r in m.offsets()],[-12,0,12],atol=1)
        mean,_=m.posterior(np.array([[.5]]));self.assertAlmostEqual(mean[0],55,delta=1)
    def test_no_batch_shift(self):
        x,y,b=self.dataset(0);m=BatchGP.fit(x,y,b)
        self.assertLess(m.batch_sd,1)
    def test_controls_excluded_or_separate(self):
        x,y,b=self.dataset();controls=np.r_[np.zeros(len(y),bool),True]
        xx=np.vstack([x,[[100]]]);yy=np.r_[y,0];bb=np.concatenate([b,['ROUND_003']])
        a=BatchGP.fit(x,y,b);c=BatchGP.fit(xx,yy,bb,controls)
        np.testing.assert_allclose(a.predict(x),c.predict(x))
        d=BatchGP.fit(xx,yy,bb,controls,strategy='control')
        self.assertIn('insufficient',d.calibration_status)
        self.assertEqual(d.controls.sum(),1)
    def test_condition_snapshot_and_future(self):
        x,y,b=self.dataset();m=BatchGP.fit(x,y,b)
        p=np.array([[.2],[.8]])
        before,bc=m.posterior(p,batch='ROUND_004',future=True)
        c=m.condition(p[:1],[42.],'ROUND_004')
        after,ac=c.posterior(p,batch='ROUND_004',future=True)
        self.assertLess(after[1],before[1]);self.assertLess(ac[1,1],bc[1,1]);self.assertEqual(c.batch_sd,m.batch_sd)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'m.json';m.save(path);loaded=BatchGP.load(path)
            np.testing.assert_allclose(m.predict(p),loaded.predict(p))
    def test_torch_joint_covariance_and_qlognehvi(self):
        import torch
        from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
        x,y,b=self.dataset();m=BatchGP.fit(x,y,b);bundle=torch_bundle([m,m])
        q=torch.tensor([[.2],[.8]],dtype=torch.double,requires_grad=True)
        mean,cov=m.posterior(q.detach().numpy());post=bundle.posterior(q)
        np.testing.assert_allclose(post.mean[:,0].detach().numpy(),mean,atol=1e-7)
        full_cov=post.distribution.covariance_matrix.detach().numpy()
        selected=full_cov[::2,::2] if post.distribution._interleaved else full_cov[:2,:2]
        np.testing.assert_allclose(selected,cov,atol=1e-6)
        samples=post.rsample(torch.Size([8]));self.assertEqual(samples.shape,(8,2,2))
        acq=qLogNoisyExpectedHypervolumeImprovement(bundle,[0.,0.],torch.tensor([[0.],[1.]],dtype=torch.double),cache_root=False,prune_baseline=False)
        val=acq(q);val.sum().backward();self.assertTrue(torch.isfinite(val).all());self.assertTrue(torch.isfinite(q.grad).all())
    def test_gate(self):
        with self.assertRaises(StrategyDecisionRequired):validate_decision(None,11)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'decision.json';p.write_text(json.dumps(dict(version=1,model_version='batch_gp_v1',decision_id='chosen-1',chosen_by='user',decided_at='2026-09-16',effective_round=11,viability_strategy='batch',mechanical_strategy='current_comparable_endpoint',acquisition='shared_qlognehvi_finite_pool',objective='expected_formulation_response',evidence=['review'],acceptance_thresholds=None)))
            self.assertEqual(validate_decision(p,11)['viability_strategy'],'batch')

if __name__=='__main__':unittest.main()

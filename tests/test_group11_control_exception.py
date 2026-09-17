import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/08_multi_objective'))
from helper.batch_gp import BatchGP,torch_bundle
from helper.gp_strategy import check_control_history,StrategyDecisionRequired

class SparseControlTests(unittest.TestCase):
    def model(self,controls=True):
        return BatchGP.fit(np.array([[0.],[1.],[0.],[1.],[.5]]),np.array([40.,60.,30.,50.,35.]),np.array(['A','A','B','B','B']),np.array([False,False,False,False,controls]),strategy='control')
    def test_exception_scoped_to_one_batch_and_round11(self):
        d={'sparse_control_exception':{'authorized':True,'round':11,'minimum_control_batches':1}}
        check_control_history(self.model(),d,11)
        for model,decision,round_ in [(self.model(),{},11),(self.model(),d,12),(self.model(False),d,11)]:
            with self.assertRaises(StrategyDecisionRequired):check_control_history(model,decision,round_)
    def test_prior_applied_before_fit_and_latent_posterior(self):
        import torch
        m=self.model();self.assertEqual(m.reference_sd,50)
        q=np.array([[.2],[.8]]);p=torch_bundle([m]).posterior(torch.tensor(q,dtype=torch.double))
        mean,cov=m.posterior(q)
        np.testing.assert_allclose(p.mean.detach().numpy().ravel(),mean,atol=1e-8)
        np.testing.assert_allclose(p.distribution.covariance_matrix.detach().numpy(),cov,atol=1e-6)
        with self.assertRaises(ValueError):torch_bundle([m]).posterior(torch.tensor(q,dtype=torch.double),observation_noise=True)

if __name__=='__main__':unittest.main()

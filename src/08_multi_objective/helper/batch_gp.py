"""Exact additive batch GP. Absolute labels; controls have a separate intercept.

All fitting is in original endpoint units. Batch/noise hyperparameters are bounded
maximum-likelihood estimates (empirical Bayes), not fully Bayesian samples.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

VERSION = 'batch_gp_v1'
NAMES = {
 'current': 'Current GP',
 'campaign': 'Proposed GP — campaign-only',
 'campaign_noise': 'Proposed GP — campaign-only + fitted residual noise (matched comparison)',
 'batch': 'Proposed GP — campaign-only + batch effect',
 'control': 'Proposed GP — campaign-only + batch effect + control calibration',
}

def matern(a,b):
    d=cdist(a,b);r=np.sqrt(5.)*d
    return (1+r+r*r/3)*np.exp(-r)

@dataclass
class BatchGP:
    x: np.ndarray
    y: np.ndarray
    batches: np.ndarray
    controls: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    mean: float
    amplitude: float
    batch_sd: float
    noise_sd: float
    known_variance: np.ndarray
    reference_sd: float = 50.
    strategy: str = 'batch'
    fit_status: str = 'fixed'
    future_noise_variance: float | None = None

    @classmethod
    def fit(cls,x,y,batches,controls=None,strategy='batch',noise=None):
        x=np.asarray(x,float);y=np.asarray(y,float);batches=np.asarray(batches,str)
        c=np.zeros(len(y),bool) if controls is None else np.asarray(controls,bool)
        if strategy!='control':
            keep=~c;x=x[keep];y=y[keep];batches=batches[keep]
            if noise is not None:noise=np.asarray(noise)[keep]
            c=c[keep]
        ordinary=~c
        if ordinary.sum()<2 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('At least two finite ordinary labels required')
        center=x[ordinary].mean(0);scale=x[ordinary].std(0);scale[scale<1e-12]=1.
        amp=max(float(y[ordinary].std()),1.)
        sd=np.ones(len(y)) if noise is None else np.asarray(noise,float)
        sd=np.where(np.isfinite(sd)&(sd>0),sd,amp)
        model=cls(x,y,batches,c,center,scale,float(y[ordinary].mean()),amp,0.,0.,sd**2,strategy=strategy)
        if strategy in ('batch','control','campaign_noise'):
            # Fit one total residual SD; do not double count recorded per-row noise.
            model.known_variance=np.zeros(len(y))
            with_batch=strategy in ('batch','control')
            def objective(logs):
                model.noise_sd=float(np.exp(logs[-1]));model.batch_sd=float(np.exp(logs[0])) if with_batch else 0.
                try:
                    cf=cho_factor(model.training_cov(),lower=True)
                    r=y-model.mean
                    return .5*r@cho_solve(cf,r)+np.log(np.diag(cf[0])).sum()+len(y)*.5*np.log(2*np.pi)
                except np.linalg.LinAlgError:return 1e30
            initial=np.log([5.,5.] if with_batch else [5.])
            bounds=[(np.log(.01),np.log(50.)),(np.log(.5),np.log(30.))] if with_batch else [(np.log(.5),np.log(30.))]
            fit=minimize(objective,initial,bounds=bounds,method='L-BFGS-B')
            if not fit.success or not np.isfinite(fit.fun):raise RuntimeError('Batch GP fit failed: '+str(fit.message))
            objective(fit.x);model.fit_status='bounded maximum likelihood; '+str(fit.message)
        model.refresh();return model

    def training_cov(self):
        k=self.amplitude**2*matern((self.x-self.center)/self.scale,(self.x-self.center)/self.scale)
        k*=np.outer(~self.controls,~self.controls)
        k+=self.reference_sd**2*np.outer(self.controls,self.controls)
        k+=self.batch_sd**2*(self.batches[:,None]==self.batches[None,:])
        return k+np.diag(self.known_variance+self.noise_sd**2+1e-7)

    def refresh(self):
        self.factor=cho_factor(self.training_cov(),lower=True)
        self.alpha=cho_solve(self.factor,self.y-self.mean)

    @property
    def calibration_status(self):
        n=len(np.unique(self.batches[self.controls]))
        return 'not used' if self.strategy!='control' else ('supported by repeated reference batches' if n>=2 else 'insufficient reference history; prior-sensitive')

    def posterior(self,x,batch=None,controls=None,future=False):
        x=np.asarray(x,float);c=np.zeros(len(x),bool) if controls is None else np.asarray(controls,bool)
        cross=self.amplitude**2*matern((x-self.center)/self.scale,(self.x-self.center)/self.scale)*np.outer(~c,~self.controls)
        prior=self.amplitude**2*matern((x-self.center)/self.scale,(x-self.center)/self.scale)*np.outer(~c,~c)
        cross+=self.reference_sd**2*np.outer(c,self.controls);prior+=self.reference_sd**2*np.outer(c,c)
        if batch is not None:
            b=np.broadcast_to(np.asarray(batch,str),(len(x),))
            cross+=self.batch_sd**2*(b[:,None]==self.batches[None,:])
            prior+=self.batch_sd**2*(b[:,None]==b[None,:])
        if future:
            residual=self.noise_sd**2 if self.noise_sd else float(np.median(self.known_variance))
            prior+=np.eye(len(x))*residual
        mean=self.mean+cross@self.alpha
        cov=prior-cross@cho_solve(self.factor,cross.T)
        cov=(cov+cov.T)/2
        return mean,cov

    def condition(self,x,y,batch,controls=None):
        """Condition on designated measurements only; never retune hyperparameters."""
        import copy
        m=copy.deepcopy(self);x=np.asarray(x,float);y=np.asarray(y,float)
        c=np.zeros(len(y),bool) if controls is None else np.asarray(controls,bool)
        if c.any() and self.strategy!='control':raise ValueError('This model excludes controls')
        m.x=np.vstack([m.x,x]);m.y=np.r_[m.y,y];m.controls=np.r_[m.controls,c]
        m.batches=np.r_[m.batches,np.broadcast_to(np.asarray(batch,str),(len(y),))]
        m.known_variance=np.r_[m.known_variance,np.full(len(y),np.median(self.known_variance))]
        m.refresh();return m

    def offsets(self):
        rows=[]
        for b in np.unique(self.batches):
            k=self.batch_sd**2*(self.batches==b).astype(float)
            rows.append(dict(batch_id=b,offset=float(k@self.alpha),sd=float(np.sqrt(max(0,self.batch_sd**2-k@cho_solve(self.factor,k)))),
                controls=int(((self.batches==b)&self.controls).sum()),ordinary=int(((self.batches==b)&~self.controls).sum())))
        return rows

    def save(self,path):
        path=Path(path)
        d={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in self.__dict__.items() if k not in ('factor','alpha')}
        path.write_text(json.dumps({'version':VERSION,**d},indent=2)+'\n')

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text())
        if d.pop('version')!=VERSION:raise ValueError('Unsupported model snapshot')
        for k in ('x','y','batches','controls','center','scale','known_variance'):d[k]=np.asarray(d[k])
        m=cls(**d);m.refresh();return m

    def predict(self,x,return_std=False):
        mean,cov=self.posterior(x)
        return (mean,np.sqrt(np.maximum(np.diag(cov),0))) if return_std else mean


def torch_bundle(models):
    """Differentiable exact formulation posterior; excludes fresh batch/assay noise.

    Independent outputs, full covariance across baseline/pending/candidates.
    cache_root=False is used by the caller for this custom BoTorch Model.
    """
    import torch
    from botorch.models.model import Model
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from gpytorch.distributions import MultivariateNormal,MultitaskMultivariateNormal
    class Bundle(Model):
        def __init__(self):
            super().__init__()
            for i,m in enumerate(models):
                for key,val in dict(x=(m.x-m.center)/m.scale,center=m.center,scale=m.scale,
                    ordinary=(~m.controls).astype(float),alpha=m.alpha,chol=np.linalg.cholesky(m.training_cov())).items():
                    self.register_buffer(f'{key}{i}',torch.as_tensor(val,dtype=torch.double))
        @property
        def num_outputs(self):return len(models)
        @property
        def batch_shape(self):return torch.Size()
        def posterior(self,X,output_indices=None,observation_noise=False,posterior_transform=None,**kwargs):
            if observation_noise is not False:raise ValueError('Acquisition bundle exposes formulation response only')
            outputs=[]
            for i in (range(len(models)) if output_indices is None else output_indices):
                m=models[i];z=(X-getattr(self,f'center{i}'))/getattr(self,f'scale{i}')
                def kernel(a,b):
                    r=5**.5*torch.linalg.vector_norm(a.unsqueeze(-2)-b.unsqueeze(-3),dim=-1)
                    return m.amplitude**2*(1+r+r*r/3)*torch.exp(-r)
                cross=kernel(z,getattr(self,f'x{i}'))*getattr(self,f'ordinary{i}')
                mean=m.mean+cross@getattr(self,f'alpha{i}')
                chol=getattr(self,f'chol{i}')
                solved=torch.cholesky_solve(cross.transpose(-1,-2),chol)
                cov=kernel(z,z)-cross@solved
                cov=(cov+cov.transpose(-1,-2))/2+torch.eye(X.shape[-2],device=X.device,dtype=X.dtype)*max(m.amplitude**2*1e-10,1e-14)
                outputs.append(MultivariateNormal(mean,cov))
            distribution=outputs[0] if len(outputs)==1 else MultitaskMultivariateNormal.from_independent_mvns(outputs)
            p=GPyTorchPosterior(distribution)
            return posterior_transform(p) if posterior_transform is not None else p
    return Bundle()

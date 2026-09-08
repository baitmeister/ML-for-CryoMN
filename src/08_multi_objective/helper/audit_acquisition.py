"""Offline-only multiple-GP acquisition; production imports are prohibited."""
from dataclasses import dataclass
import numpy as np

@dataclass
class AuditBundle:
    model: object
    baseline: object
    lower: np.ndarray
    ranges: np.ndarray
    reference: tuple

    def posterior(self,x):
        import torch
        return self.model.posterior(torch.tensor((np.asarray(x)-self.lower)/self.ranges,dtype=torch.double))

    def scores(self,x,pending=None):
        import torch
        from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
        p=None if pending is None else torch.tensor((np.asarray(pending)-self.lower)/self.ranges,dtype=torch.double)
        acq=qLogNoisyExpectedHypervolumeImprovement(self.model,list(self.reference),self.baseline,X_pending=p)
        tx=torch.tensor((np.asarray(x)-self.lower)/self.ranges,dtype=torch.double)
        with torch.no_grad(): return acq(tx.unsqueeze(-2)).cpu().numpy()

def build_bundle(vx,vy,mx,my,paired_baseline,lower,upper,vvar=None,mvar=None,reference=(0.,0.)):
    import torch
    from botorch.models import SingleTaskGP,ModelListGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import SumMarginalLogLikelihood
    torch.manual_seed(42)
    lower=np.asarray(lower,float);ranges=np.maximum(np.asarray(upper,float)-lower,1e-12)
    models=[]
    for x,y,var in [(vx,vy,vvar),(mx,my,mvar)]:
        if len(x)<2: raise ValueError('Each endpoint requires at least two measured rows')
        tx=torch.tensor((np.asarray(x)-lower)/ranges,dtype=torch.double)
        ty=torch.tensor(np.asarray(y).reshape(-1,1),dtype=torch.double)
        tv=None if var is None else torch.tensor(np.asarray(var).reshape(-1,1),dtype=torch.double)
        models.append(SingleTaskGP(tx,ty,train_Yvar=tv,outcome_transform=Standardize(m=1)))
    model=ModelListGP(*models);fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood,model))
    baseline=torch.tensor((np.asarray(paired_baseline)-lower)/ranges,dtype=torch.double)
    if len(baseline)==0:raise ValueError('Observed paired baseline required')
    return AuditBundle(model,baseline,lower,ranges,reference)

def select_from_pool(bundle,pool,n=4):
    """Sequential finite-pool qLogNEHVI, with selected points pending."""
    pool=np.asarray(pool);chosen=[];remaining=list(range(len(pool)))
    for _ in range(min(n,len(pool))):
        scores=bundle.scores(pool[remaining],pool[chosen] if chosen else None)
        i=remaining.pop(int(np.argmax(scores)));chosen.append(i)
    return chosen

def propose_with_fallback(bundle, pool, n=4, feasibility=None):
    """Offline continuous qLogNEHVI; finite-pool retry uses the identical model.

    The final heuristic is explicitly exploration by distance from observed paired
    locations, and is never described as hypervolume improvement.
    """
    pool=np.asarray(pool,float)
    if feasibility is not None: pool=np.asarray([x for x in pool if feasibility(x)])
    if len(pool)==0: raise ValueError('No feasible candidates')
    errors=[]
    try:
        import torch
        from botorch.optim import optimize_acqf
        from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
        selected=[]
        for _ in range(n):
            pending=None if not selected else torch.tensor((np.asarray(selected)-bundle.lower)/bundle.ranges,dtype=torch.double)
            acq=qLogNoisyExpectedHypervolumeImprovement(bundle.model,list(bundle.reference),bundle.baseline,X_pending=pending)
            bounds=torch.stack([torch.zeros(len(bundle.lower),dtype=torch.double),torch.ones(len(bundle.lower),dtype=torch.double)])
            x,_=optimize_acqf(acq,bounds,q=1,num_restarts=3,raw_samples=32,options={'maxiter':100})
            value=bundle.lower+x.detach().numpy().reshape(-1)*bundle.ranges
            if feasibility is not None and not feasibility(value): raise ValueError('Continuous candidate rejected by feasibility')
            selected.append(value)
        return np.asarray(selected),{'mode':'qlognehvi_continuous','errors':errors}
    except Exception as exc: errors.append(str(exc))
    try:
        return pool[select_from_pool(bundle,pool,n)],{'mode':'qlognehvi_finite_pool','errors':errors}
    except Exception as exc: errors.append(str(exc))
    # Conservative deterministic coverage fallback, not an alternative GP fit.
    baseline=bundle.baseline.detach().cpu().numpy() if bundle is not None else np.empty((0,pool.shape[1]))
    scaled=(pool-bundle.lower)/bundle.ranges if bundle is not None else pool
    remaining=list(range(len(pool)));selected=[]
    for _ in range(min(n,len(pool))):
        scores=[float(np.min(np.linalg.norm(baseline-scaled[i],axis=1))) if len(baseline) else 0. for i in remaining]
        i=remaining.pop(int(np.argmax(scores)));selected.append(i);baseline=np.vstack([baseline,scaled[i]])
    return pool[selected],{'mode':'heuristic_coverage_not_qlognehvi','errors':errors}

"""Offline comparison models only. Production code must not import this module."""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .models import _fit_regression

VARIANTS=('current_mixed','campaign_mean','campaign_median','campaign_existing',
          'campaign_bounds_learned','legacy_correction')

def fit_predict(variant, frame, x, registry, use_adaptive=False):
    f=frame.dropna(subset=['viability_percent']).copy()
    campaign=f.batch_id.astype(str).str.startswith('ROUND_')
    if variant!='current_mixed' and variant!='legacy_correction': f=f.loc[campaign]
    cols=registry.feature_names
    xx=f[cols].to_numpy(float); y=f.viability_percent
    noise=f.viability_percent__noise.copy()
    if use_adaptive and 'adaptive_sd' in f: noise=f.adaptive_sd.fillna(noise)
    if variant in ('campaign_mean','campaign_median'):
        value=y.mean() if variant=='campaign_mean' else y.median()
        return np.full(len(x),value),np.full(len(x),max(float(y.std()),1.))
    if variant in ('current_mixed','campaign_existing'):
        p=_fit_regression(xx,y,y_noise=noise).predict(x);return p.mean,p.std
    bounds=np.array([registry.get_by_feature(c).upper_bound-registry.get_by_feature(c).lower_bound for c in cols])
    bounds=np.maximum(bounds,1e-12)
    def learned(a,b,sd,test):
        scale=max(float(np.std(b)),1.)
        # One learned shared length scale; bounded search avoids fitting 24 free sensitivities.
        gp=GaussianProcessRegressor(kernel=ConstantKernel(1.,(.1,10))*Matern(1.,(.1,10),nu=2.5),
            alpha=(np.asarray(sd)/scale)**2,normalize_y=True,n_restarts_optimizer=0,random_state=42)
        gp.fit(a/bounds,b)
        return gp.predict(test/bounds,return_std=True)
    if variant=='campaign_bounds_learned': return learned(xx,y,noise,x)
    lit=f.loc[~f.batch_id.astype(str).str.startswith('ROUND_')]
    cam=f.loc[f.batch_id.astype(str).str.startswith('ROUND_')]
    base=_fit_regression(lit[cols].to_numpy(float),lit.viability_percent,y_noise=lit.viability_percent__noise)
    cp=base.predict(cam[cols].to_numpy(float)); bp=base.predict(x)
    residual=cam.viability_percent-cp.mean
    correction=_fit_regression(cam[cols].to_numpy(float),residual,y_noise=cam.viability_percent__noise)
    p=correction.predict(x)
    return bp.mean+p.mean,np.sqrt(bp.std**2+p.std**2)

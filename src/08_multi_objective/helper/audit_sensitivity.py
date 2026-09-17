"""Offline one-factor GP sensitivity study; never used by production selection."""
from dataclasses import asdict, dataclass, replace
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Setting:
    factor: str = 'matched_baseline'
    source: str = 'campaign'
    scaling: str = 'standard'
    smoothness: float = 2.5
    length: float = 1.0
    fit_length: bool = False
    ard: bool = False
    length_bounds: tuple = (.1, 10.)
    fit_amplitude: bool = False
    amplitude: float = 1.0
    noise: str = 'recorded'
    noise_sd: float = 1.0
    mean: str = 'mean'
    transform: str = 'identity'
    recent_batches: int = 0
    paired_only: bool = False
    restarts: int = 0


def describe(setting):
    result=asdict(setting)
    if np.isinf(result['smoothness']):result['smoothness']='RBF'
    return result


def settings(endpoint):
    base=Setting()
    cases={'sens_baseline':base}
    def add(name,factor,**kw):cases['sens_'+name]=replace(base,factor=factor,**kw)
    add('mixed_sources','source',source='mixed')
    add('bounds','scaling',scaling='bounds')
    add('matern15','smoothness',smoothness=1.5)
    add('rbf','smoothness',smoothness=float('inf'))
    add('length_short','length',length=.3)
    add('length_long','length',length=3.)
    add('length_learned','length_fitting',fit_length=True)
    add('ard_learned','length_parameterization',fit_length=True,ard=True)
    add('amplitude_learned','amplitude',fit_amplitude=True)
    add('mean_median','mean',mean='median')
    add('recent3','recency',recent_batches=3)
    add('paired_only','training_coverage',paired_only=True)
    add('noise_learned','noise',noise='learned')
    add('bounds_learned','combined_proposal',scaling='bounds',fit_length=True,fit_amplitude=True)
    # A second matched family isolates fitting/regularization choices around the
    # proposed learned/bounds model rather than claiming a full factorial search.
    proposed=cases['sens_bounds_learned']
    cases['sens_bounds_learned_wide']=replace(proposed,factor='bounds_vs_bounds_learned_regularization',length_bounds=(.03,30.))
    cases['sens_bounds_learned_restarts']=replace(proposed,factor='bounds_vs_bounds_learned_fit_stability',restarts=2)
    if endpoint=='viability_percent':
        for sd in (1.,3.,5.,10.):add('noise_'+str(int(sd))+'pp','noise',noise='fixed',noise_sd=sd)
        add('logit','target_transform',transform='logit')
    else:
        # Force noise sensitivities are fractions of prior target spread,
        # never viability percentage-point floors masquerading as Newtons.
        for fraction in (.1,.25,.5):
            add('noise_spread_'+str(fraction),'noise',noise='spread',noise_sd=fraction)
        add('log','target_transform',transform='log')
    return cases


def training_rows(frame, endpoint, setting):
    f=frame.loc[pd.to_numeric(frame[endpoint],errors='coerce').notna()].copy()
    if setting.source=='campaign':f=f.loc[f.batch_id.astype(str).str.startswith('ROUND_')]
    if setting.paired_only:
        other='critical_axial_load_N_per_needle' if endpoint=='viability_percent' else 'viability_percent'
        values=f.get(other,pd.Series(np.nan,index=f.index))
        if other=='viability_percent' and 'mechanical_pair_viability_percent' in f:
            values=values.combine_first(f.mechanical_pair_viability_percent)
        f=f.loc[values.notna()]
    if setting.recent_batches:
        rounds=sorted(f.batch_id.astype(str).unique(),key=lambda s:int(s.split('_')[-1]))
        f=f.loc[f.batch_id.isin(rounds[-setting.recent_batches:])]
    return f


def _optimizer(objective, initial, bounds):
    result=minimize(objective,initial,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':150})
    return result.x,result.fun


def fit_sensitivity(frame, x, registry, endpoint, setting):
    f=training_rows(frame,endpoint,setting)
    if len(f)<2:raise ValueError('Fewer than two eligible preceding training rows')
    features=registry.feature_names
    xx=f[features].to_numpy(float); x=np.asarray(x,float)
    y=f[endpoint].to_numpy(float)
    raw_scale=max(float(np.std(y)),1e-8)
    recorded=pd.to_numeric(f.get(endpoint+'__noise',pd.Series(np.nan,index=f.index)),errors='coerce').to_numpy(float)
    raw_sd=np.where(np.isfinite(recorded)&(recorded>0),recorded,raw_scale)
    if setting.noise=='fixed':raw_sd=np.full(len(y),setting.noise_sd)
    if setting.noise=='spread':raw_sd=np.full(len(y),setting.noise_sd*raw_scale)
    if setting.scaling=='standard':
        scaler=StandardScaler().fit(xx);xx=scaler.transform(xx);x=scaler.transform(x)
    else:
        lower=np.array([registry.get_by_feature(c).lower_bound for c in features])
        widths=np.maximum(np.array([registry.get_by_feature(c).upper_bound for c in features])-lower,1e-12)
        xx=(xx-lower)/widths;x=(x-lower)/widths
    if setting.transform=='logit':
        bounded=np.clip(y,.5,99.5)
        target=np.log(bounded/(100-bounded))
        transformed_sd=raw_sd*100/(bounded*(100-bounded))
        inverse=lambda a:100*expit(a)
    elif setting.transform=='log':
        if np.any(y<0):raise ValueError('Log transform requires nonnegative mechanics')
        positive=np.maximum(y,1e-6)
        target=np.log(positive);transformed_sd=raw_sd/positive
        inverse=np.exp
    else:
        target=y;transformed_sd=raw_sd;inverse=lambda a:a
    center=float(np.median(target) if setting.mean=='median' else np.mean(target))
    scale=max(float(np.std(target)),1e-8)
    length=np.full(len(features),setting.length) if setting.ard else setting.length
    bounds=setting.length_bounds if setting.fit_length else 'fixed'
    shape=RBF(length,bounds) if np.isinf(setting.smoothness) else Matern(length,bounds,nu=setting.smoothness)
    signal=ConstantKernel(setting.amplitude,(.1,10.) if setting.fit_amplitude else 'fixed')*shape
    kernel=signal
    alpha=np.square(transformed_sd/scale)
    if setting.noise=='learned':
        # One fitted noise variance in standardized target space. This is an
        # explicit bounded-ML sensitivity, not a validated Bayesian noise prior.
        kernel=signal+WhiteKernel(.25,(1e-4,4.))
        alpha=1e-8
    fitted=setting.fit_length or setting.fit_amplitude or setting.noise=='learned'
    gp=GaussianProcessRegressor(kernel=kernel,alpha=alpha,normalize_y=False,
        optimizer=_optimizer if fitted else None,n_restarts_optimizer=setting.restarts,random_state=42)
    gp.fit(xx,(target-center)/scale)
    mean,returned_sd=gp.predict(x,return_std=True);mean=center+mean*scale
    returned_var=np.square(returned_sd*scale)
    campaign_mask=f.batch_id.astype(str).str.startswith('ROUND_').to_numpy()
    future_sd=transformed_sd[campaign_mask] if campaign_mask.any() else transformed_sd
    noise_var=float(gp.kernel_.k2.noise_level)*scale**2 if setting.noise=='learned' else float(np.median(future_sd**2))
    latent_var=np.maximum(returned_var-noise_var,0) if setting.noise=='learned' else returned_var
    future_var=latent_var+noise_var
    # Monotone transformed Gaussian intervals; physical-space moments by fixed
    # Gauss-Hermite quadrature avoid reporting inverse(mean) as a predicted mean.
    nodes,weights=np.polynomial.hermite.hermgauss(24)
    samples=inverse(mean[:,None]+np.sqrt(2*latent_var[:,None])*nodes)
    physical_mean=samples@weights/np.sqrt(np.pi)
    physical_var=np.maximum(samples**2@weights/np.sqrt(np.pi)-physical_mean**2,0)
    details=dict(training_rows=len(f),fitted_kernel=str(gp.kernel_),settings=describe(setting),
        normalized_log_marginal_likelihood=float(gp.log_marginal_likelihood_value_),
        noise_variance_model_space=noise_var,noise_space=setting.transform,
        future_noise_basis='fitted shared noise' if setting.noise=='learned' else 'median preceding campaign noise variance in model space',
        missing_recorded_noise_count=int((~np.isfinite(recorded)|(recorded<=0)).sum()),
        latent_lower=inverse(mean-1.96*np.sqrt(latent_var)),latent_upper=inverse(mean+1.96*np.sqrt(latent_var)),
        future_lower=inverse(mean-1.96*np.sqrt(future_var)),future_upper=inverse(mean+1.96*np.sqrt(future_var)))
    return physical_mean,np.sqrt(physical_var),details

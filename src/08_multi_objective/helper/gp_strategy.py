"""Explicit production strategy decision and shared endpoint model integration."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .batch_gp import BatchGP, NAMES, VERSION, torch_bundle

class StrategyDecisionRequired(RuntimeError):pass


def validate_decision(path,round_number):
    if path is None or not Path(path).is_file():
        raise StrategyDecisionRequired('GP strategy decision required before Group 11/hybrid generation. Ingestion/reporting may proceed; record your choice using --gp-strategy-decision PATH.')
    d=json.loads(Path(path).read_text())
    required={'version','model_version','decision_id','chosen_by','decided_at','effective_round','viability_strategy','mechanical_strategy','acquisition','objective','evidence','acceptance_thresholds'}
    if not required.issubset(d):raise StrategyDecisionRequired('Incomplete GP decision: '+str(sorted(required-set(d))))
    if d['model_version']!=VERSION:raise StrategyDecisionRequired('Decision model version does not match implemented configuration')
    if d['version']!=1 or d['effective_round']>round_number or d['effective_round']<11:raise StrategyDecisionRequired('Invalid strategy version/effective round')
    if any(not d[k] for k in ('decision_id','chosen_by','decided_at','evidence')):raise StrategyDecisionRequired('Decision provenance must be nonempty')
    if d['viability_strategy'] not in NAMES:raise StrategyDecisionRequired('Unsupported viability strategy')
    if d['objective']!='expected_formulation_response':raise StrategyDecisionRequired('Unsupported optimization objective')
    if d['mechanical_strategy']!='current_comparable_endpoint':raise StrategyDecisionRequired('Unsupported mechanical strategy')
    expected='current_paired_qlognehvi' if d['viability_strategy']=='current' else 'shared_qlognehvi_finite_pool'
    if d['acquisition']!=expected:raise StrategyDecisionRequired('GP and acquisition configuration mismatch')
    if d['acceptance_thresholds'] not in ({},None):raise StrategyDecisionRequired('New selection thresholds need explicit policy implementation; use null for no new threshold')
    d['sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return d


def control_mask(frame):
    role=frame.get('experimental_role',pd.Series('',index=frame.index)).fillna('')
    return role.eq('campaign_control') | frame.formulation_id.astype(str).str.startswith('reference_')


def viability_rows(forms,obs,features,campaign=True):
    o=obs.loc[obs.endpoint.isin(['viability_percent','mechanical_pair_viability_percent'])].copy()
    if campaign:o=o.loc[o.batch_id.astype(str).str.match(r'^ROUND_\d+$')]
    o['is_control']=control_mask(o)
    o['observation_noise']=pd.to_numeric(o.get('observation_noise',pd.Series(np.nan,index=o.index)),errors='coerce')
    o['value']=pd.to_numeric(o.value,errors='coerce');o=o.loc[np.isfinite(o.value)]
    rows=o.groupby(['formulation_id','batch_id','is_control'],as_index=False).agg(value=('value','mean'),noise=('observation_noise','mean'),n=('value','size'))
    return rows.merge(forms[['formulation_id']+list(features)].drop_duplicates('formulation_id'),on='formulation_id',validate='many_to_one')


def fit_viability(forms,obs,features,strategy):
    rows=viability_rows(forms,obs,features,campaign=strategy!='current')
    model=BatchGP.fit(rows[features].to_numpy(float),rows.value.to_numpy(),rows.batch_id.to_numpy(),
        rows.is_control.to_numpy(),strategy=strategy,noise=rows.noise.to_numpy())
    return model


def install_strategy(endpoint_models,forms,obs,registry,config):
    decision=(config or {}).get('gp_strategy_decision')
    if not decision or decision['viability_strategy']=='current':return endpoint_models
    from .models import RegressionSurrogate
    strategy=decision['viability_strategy']
    raw=(config or {}).get('_gp_raw_observations',obs)
    v=fit_viability(forms,raw,registry.feature_names,strategy)
    if strategy=='control' and v.calibration_status!='supported by repeated reference batches':
        raise StrategyDecisionRequired('Control calibration has fewer than two historical control batches; select another supported strategy or collect ordinary scheduled evidence.')
    # Reconstruct the Current GP mechanical posterior exactly (no force noise floor added).
    frame=endpoint_models.training_frame
    target='critical_axial_load_N_per_needle'
    f=frame.loc[frame[target].notna()]
    if len(f)<2:raise StrategyDecisionRequired('Two comparable mechanical observations required')
    m=snapshot_current(endpoint_models.critical_load,f[registry.feature_names].to_numpy(float),f[target].to_numpy(float),f.batch_id.to_numpy())
    endpoint_models.viability=RegressionSurrogate(v,residual_std=v.noise_sd or 1.,fitted=True,fallback_mean=v.mean)
    endpoint_models.critical_load=RegressionSurrogate(m,residual_std=m.noise_sd or 1.,fitted=True,fallback_mean=m.mean)
    endpoint_models.acquisition_bundle=torch_bundle([v,m])
    endpoint_models.gp_strategy_decision=decision
    return endpoint_models


def shared_scores(models,train_x,candidate_x,reference):
    import torch
    from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
    if len(train_x)<2:raise RuntimeError('Comparable observed paired acquisition baseline required')
    from botorch.sampling.normal import SobolQMCNormalSampler
    acq=qLogNoisyExpectedHypervolumeImprovement(models.acquisition_bundle,list(reference),
        torch.tensor(np.array(train_x,copy=True),dtype=torch.double),cache_root=False,prune_baseline=False,sampler=SobolQMCNormalSampler(torch.Size([128]),seed=42))
    with torch.no_grad():
        chunks=[acq(torch.tensor(np.array(candidate_x[i:i+64],copy=True),dtype=torch.double).unsqueeze(-2)).cpu().numpy() for i in range(0,len(candidate_x),64)]
        scores=np.concatenate(chunks) if chunks else np.empty(0)
    if not np.isfinite(scores).all():raise RuntimeError('Nonfinite shared qLogNEHVI scores')
    return scores,{'backend':'shared_formulation_posterior','strategy':models.gp_strategy_decision['viability_strategy']}


def snapshot_current(surrogate,x,y,batches):
    """Lossless exact-GP representation of an already fitted sklearn Current GP."""
    pipeline=surrogate.model
    if not hasattr(pipeline,'named_steps'):raise ValueError('Current GP is not fitted as a GP')
    scaler=pipeline.named_steps['standardscaler'];gp=pipeline.named_steps['gaussianprocessregressor']
    n=len(y)
    m=BatchGP(np.asarray(x,float),np.asarray(y,float),np.asarray(batches,str),np.zeros(n,bool),
        scaler.mean_.copy(),scaler.scale_.copy(),float(gp._y_train_mean),float(gp._y_train_std),
        0.,0.,np.broadcast_to(np.asarray(gp.alpha,float),(n,)).copy()*float(gp._y_train_std)**2,strategy='current')
    m.refresh();return m

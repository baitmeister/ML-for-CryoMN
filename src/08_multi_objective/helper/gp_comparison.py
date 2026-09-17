"""Immutable challenger snapshots, forward evaluation and report-only rendering."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import os
import numpy as np
import pandas as pd
from .batch_gp import BatchGP, NAMES, VERSION
from .gp_strategy import viability_rows, fit_viability, control_mask, snapshot_current
from .models import train_endpoint_models
from .registry import load_registry
from .group10_config import production_observations


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def round_numbers(obs):return pd.to_numeric(obs.batch_id.astype(str).str.extract(r'^ROUND_(\d+)$')[0],errors='coerce')

def attach_roles(obs,root):
    obs=obs.copy()
    if 'experimental_role' not in obs:obs['experimental_role']=''
    for path in sorted((Path(root)/'results/multi_objective_v2/rounds').glob('*/proposal/proposal.csv')):
        p=pd.read_csv(path)
        if 'experimental_role' in p:
            roles=p.drop_duplicates('formulation_id').set_index('formulation_id').experimental_role
            mask=obs.batch_id.eq(path.parents[1].name)&obs.experimental_role.fillna('').eq('')
            obs.loc[mask,'experimental_role']=obs.loc[mask,'formulation_id'].map(roles).fillna('')
    return obs


def load_inputs(root):
    root=Path(root);forms=pd.read_csv(root/'data/processed_v2/formulations.csv')
    obs=attach_roles(pd.read_csv(root/'data/processed_v2/observations.csv'),root)
    registry=load_registry(root/'config_v2/ingredients.yaml')
    return forms,obs,registry


def fit_models(forms,prior,registry):
    result={};errors=[]
    for name in NAMES:
        try:
            if name=='current':
                clean=production_observations(prior)
                models=train_endpoint_models(forms,clean,registry)
                f=models.training_frame.dropna(subset=['viability_percent'])
                m=snapshot_current(models.viability,f[registry.feature_names].to_numpy(float),f.viability_percent.to_numpy(float),f.batch_id.to_numpy())
                # Future campaign measurement noise must not inherit literature noise.
                camp=f.batch_id.str.startswith('ROUND_').to_numpy()
                m.future_noise_variance=float(np.median(m.known_variance[camp])) if camp.any() else float(np.median(m.known_variance))
            else:m=fit_viability(forms,prior,registry.feature_names,name)
            result[name]=m
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            errors.append({'strategy':name,'reason':str(exc)})
    return result,errors


def forecast(m,slate,features,batch,strategy,stage,provenance,refs=()):
    x=slate[features].to_numpy(float);ctrl=control_mask(slate).to_numpy()
    # Control predictions require its separately calibrated reference model.
    keep=~ctrl if strategy!='control' else np.ones(len(slate),bool)
    s=slate.loc[keep];x=x[keep];ctrl=ctrl[keep]
    latent,lc=m.posterior(x,batch=batch,controls=ctrl)
    mean,fc=m.posterior(x,batch=batch,controls=ctrl,future=True)
    if m.future_noise_variance is not None:
        fc=lc+np.eye(len(x))*m.future_noise_variance
    ls=np.sqrt(np.maximum(0,np.diag(lc)));fs=np.sqrt(np.maximum(0,np.diag(fc)))
    rows=[]
    for i,(_,r) in enumerate(s.iterrows()):
        rows.append(dict(strategy=strategy,model_name=NAMES[strategy],batch_id=batch,formulation_id=r.formulation_id,
            stage=stage,evidence=provenance,predicted=float(mean[i]),latent_lower=float(latent[i]-1.96*ls[i]),latent_upper=float(latent[i]+1.96*ls[i]),
            future_lower=float(mean[i]-1.96*fs[i]),future_upper=float(mean[i]+1.96*fs[i]),is_control=bool(ctrl[i]),
            conditioning_ids=json.dumps(list(refs)),calibration_status=m.calibration_status))
    return rows


def evaluate_models(models,slate,features,batch,actual,reference_ids,provenance):
    records=[];offsets=[]
    actual=actual.set_index('formulation_id')
    for name,m in models.items():
        before=forecast(m,slate,features,batch,name,'before_batch',provenance)
        for row in before:
            row['actual']=float(actual.loc[row['formulation_id'],'value']) if row['formulation_id'] in actual.index else np.nan
            row['evaluation_subset']='full_slate'
            records.append(row)
        chosen=slate.loc[slate.formulation_id.isin(reference_ids)&slate.formulation_id.isin(actual.index)].copy()
        if name!='control':chosen=chosen.loc[~control_mask(chosen)]
        # Only batch-effect strategies have the reference-conditioned use case.
        if name not in ('batch','control') or chosen.empty:continue
        refs=chosen.formulation_id.tolist()
        cond=m.condition(chosen[features].to_numpy(float),actual.loc[refs,'value'].to_numpy(float),batch,control_mask(chosen).to_numpy())
        target=slate.loc[~slate.formulation_id.isin(refs)&~control_mask(slate)]
        after=forecast(cond,target,features,batch,name,'reference_conditioned',provenance,refs)
        for row in after:
            row['actual']=float(actual.loc[row['formulation_id'],'value']) if row['formulation_id'] in actual.index else np.nan
            row['evaluation_subset']='matched_nonreference';records.append(row)
        for row in before:
            if row['formulation_id'] in set(target.formulation_id):
                records.append({**row,'evaluation_subset':'matched_nonreference','conditioning_ids':json.dumps(refs)})
        offsets.extend([{**o,'strategy':name,'stage':'reference_conditioned','target_batch':batch} for o in cond.offsets() if o['batch_id']==batch])
    return records,offsets


def reference_subset(slate,prior):
    # Pre-outcome rule: exact ID previously measured in a preceding campaign batch,
    # plus the declared control. New near-neighbours are not silently called retests.
    measured=prior.loc[prior.endpoint.eq('viability_percent')&prior.batch_id.str.startswith('ROUND_')]
    return slate.loc[control_mask(slate)|slate.formulation_id.isin(measured.formulation_id),'formulation_id'].tolist()


def freeze(root,batch,output,slate_path=None,forms=None,obs=None,registry=None):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Frozen GP comparison already exists; never overwrite: '+str(output))
    if forms is None:forms,obs,registry=load_inputs(root)
    n=int(batch.split('_')[-1]);numbers=round_numbers(obs)
    if numbers.eq(n).any():raise ValueError('Target outcomes already present: cannot freeze pre-outcome predictions')
    prior=obs.loc[numbers.lt(n)|numbers.isna()]
    slate_path=Path(slate_path) if slate_path else root/f'results/multi_objective_v2/rounds/{batch}/proposal/proposal.csv'
    slate=pd.read_csv(slate_path).drop_duplicates('formulation_id')
    models,errors=fit_models(forms,prior,registry)
    if not {'current','campaign','campaign_noise','batch'}.issubset(models):raise RuntimeError('Required challenger fit failed: '+str(errors))
    refs=reference_subset(slate,prior);output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.gp_freeze_',dir=output.parent) as tmp:
        temp=Path(tmp);records=[];offsets=[]
        for name,m in models.items():
            m.save(temp/(name+'.json'))
            records.extend(forecast(m,slate,registry.feature_names,batch,name,'before_batch','pre_outcome_challenger_after_slate_selection'))
            offsets.extend([{**o,'strategy':name,'stage':'training','target_batch':batch} for o in m.offsets()])
        created_at=datetime.now(timezone.utc).isoformat()
        training_hash=hashlib.sha256(prior.to_csv(index=False).encode()).hexdigest()
        for record in records:
            record.update(model_version=VERSION,training_cutoff=n-1,training_sha256=training_hash,created_at=created_at)
        pd.DataFrame(records).to_csv(temp/'predictions.csv',index=False)
        pd.DataFrame(offsets).to_csv(temp/'offsets.csv',index=False)
        slate.to_csv(temp/'slate.csv',index=False)
        prior.to_csv(temp/'training_observations.csv',index=False)
        manifest=dict(version=VERSION,batch_id=batch,cutoff=n-1,created_at=datetime.now(timezone.utc).isoformat(),
            features=registry.feature_names,reference_ids=refs,models=list(models),fit_errors=errors,
            proposal_sha256=digest(slate_path),training_sha256=digest(temp/'training_observations.csv'),
            code_sha256={p.name:digest(p) for p in [Path(__file__),Path(__file__).with_name('batch_gp.py'),Path(__file__).with_name('gp_strategy.py')]},
            data_source_sha256=digest(root/'data/processed_v2/observations.csv'),
            assumptions={'one_biological_batch_per_group':True,'no_target_outcomes_viewed':True,
                'residual_sd_bounds_pp':[.5,30.],'batch_sd_bounds_pp':[.01,50.],'reference_prior_sd_pp':50.,
                'hyperparameters':'bounded maximum likelihood; parameter uncertainty not integrated',
                'replicates':'formulation-batch means; no unverified independence-based variance division'},
            hashes={p.name:digest(p) for p in temp.iterdir()})
        (temp/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        os.rename(temp,output)
    from .gp_comparison_plots import render
    render(output)
    return output


def evaluate(frozen,obs,output):
    frozen=Path(frozen);output=Path(output)
    manifest=json.loads((frozen/'manifest.json').read_text())
    for name,h in manifest['hashes'].items():
        if digest(frozen/name)!=h:raise ValueError('Frozen artifact hash mismatch: '+name)
    if output.exists():raise FileExistsError('Evaluation exists; use render to redraw saved evidence')
    features=manifest['features'];slate=pd.read_csv(frozen/'slate.csv')
    models={name:BatchGP.load(frozen/(name+'.json')) for name in manifest['models']}
    target=obs.loc[obs.batch_id.eq(manifest['batch_id'])]
    actual=target.loc[target.endpoint.eq('viability_percent')].groupby('formulation_id',as_index=False).value.mean()
    if actual.empty:raise ValueError('No target viability outcomes available')
    records,offsets=evaluate_models(models,slate,features,manifest['batch_id'],actual,manifest['reference_ids'],'pre_outcome_challenger_after_slate_selection')
    # Unconditioned means must equal saved forecasts, not merely a later compatible refit.
    frozen_predictions=pd.read_csv(frozen/'predictions.csv')
    observation_ids=target.loc[target.endpoint.eq('viability_percent')].groupby('formulation_id').apply(
        lambda g:g.get('observation_id',pd.Series([],dtype=str)).dropna().astype(str).tolist(),include_groups=False).to_dict()
    for record in records:
        refs=json.loads(record['conditioning_ids'])
        record['conditioning_observation_ids']=json.dumps([oid for fid in refs for oid in observation_ids.get(fid,[])])
        record.update(model_version=manifest['version'],training_sha256=manifest['training_sha256'],training_cutoff=manifest['cutoff'],created_at=manifest['created_at'])
    comparison=pd.DataFrame(records)
    check=comparison.loc[(comparison.stage=='before_batch')&(comparison.evaluation_subset=='full_slate')].merge(frozen_predictions,on=['strategy','formulation_id'],suffixes=('','_frozen'))
    if len(check)!=len(frozen_predictions) or not np.allclose(check.predicted,check.predicted_frozen,rtol=0,atol=1e-8):raise ValueError('Frozen forecast reproduction mismatch')
    output.mkdir(parents=True)
    comparison.to_csv(output/'predictions.csv',index=False);pd.DataFrame(offsets).to_csv(output/'offsets.csv',index=False)
    (output/'manifest.json').write_text(json.dumps({'frozen_manifest_sha256':digest(frozen/'manifest.json'),'stage':'evaluation','evaluated_at':datetime.now(timezone.utc).isoformat()},indent=2))
    from .gp_comparison_plots import render
    render(output);return output


def audit(root,output,start=3,end=9):
    root=Path(root);output=Path(output)
    if output.exists():raise FileExistsError('Audit output already exists')
    forms,obs,registry=load_inputs(root);numbers=round_numbers(obs)
    records=[];offsets=[];errors=[]
    for n in range(start,end+1):
        batch=f'ROUND_{n:03d}';path=root/f'results/multi_objective_v2/rounds/{batch}/proposal/proposal.csv'
        if not path.exists() or not numbers.eq(n).any():continue
        slate=pd.read_csv(path).drop_duplicates('formulation_id')
        prior=obs.loc[numbers.lt(n)|numbers.isna()]
        models,failures=fit_models(forms,prior,registry);errors.extend([{'batch_id':batch,**e} for e in failures])
        actual=obs.loc[numbers.eq(n)&obs.endpoint.eq('viability_percent')].groupby('formulation_id',as_index=False).value.mean()
        rows,off=evaluate_models(models,slate,registry.feature_names,batch,actual,reference_subset(slate,prior),'historical_forward_refit')
        records.extend(rows);offsets.extend(off)
    output.mkdir(parents=True);pd.DataFrame(records).to_csv(output/'predictions.csv',index=False)
    pd.DataFrame(offsets).to_csv(output/'offsets.csv',index=False)
    (output/'manifest.json').write_text(json.dumps(dict(version=VERSION,stage='historical_forward_refit',start=start,end=end,fit_errors=errors,
        observations_sha256=digest(root/'data/processed_v2/observations.csv'),assumptions='Empirical Bayes; fixed formulation kernel; parameter uncertainty not integrated'),indent=2))
    from .gp_comparison_plots import render
    render(output);return output

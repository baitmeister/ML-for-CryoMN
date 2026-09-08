"""Offline forward audit. Required output directory; never promotes a model."""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from helper.registry import load_registry
from helper.models import build_training_frame
from helper.audit_models import VARIANTS,fit_predict
from helper.observation_noise import estimate_noise
from helper.group10_config import production_observations

def metrics(g):
    y=g.actual.to_numpy(float);p=g.predicted.to_numpy(float);s=g['std'].to_numpy(float);e=p-y
    spread=np.sum((y-y.mean())**2)
    return dict(n=len(g),mae=float(abs(e).mean()),bias=float(e.mean()),rmse=float(np.sqrt(np.mean(e**2))),
        r2=float(1-np.sum(e**2)/spread) if len(g)>1 and spread>0 else None,
        interval_95_coverage=float(np.mean(abs(e)<=1.96*s)),interval_95_width=float(np.mean(3.92*s)),
        spearman=float(pd.Series(y).corr(pd.Series(p),method='spearman')) if np.std(p)>0 and np.std(y)>0 else None)

def run(root,output):
    root=Path(root).resolve(); output=Path(output).resolve()
    if output==root or root/'data'==output or (root/'data') in output.parents or (root/'results/multi_objective_v2') in output.parents:
        raise ValueError('Audit output must not be a live data/results directory')
    output.mkdir(parents=True,exist_ok=True)
    registry=load_registry(root/'config_v2/ingredients.yaml')
    forms=pd.read_csv(root/'data/processed_v2/formulations.csv'); obs=production_observations(pd.read_csv(root/'data/processed_v2/observations.csv'))
    records=[]; failures=[]; noise_tables=[]
    for number in range(3,9):
        batch=f'ROUND_{number:03}'
        proposal=pd.read_csv(root/f'results/multi_objective_v2/rounds/{batch}/proposal/proposal.csv')
        prior=obs.loc[~obs.batch_id.str.startswith('ROUND_')|obs.batch_id.lt(batch)]
        frame=build_training_frame(forms,prior,registry)
        noise=estimate_noise(prior.loc[prior.endpoint.eq('viability_percent')]);noise['cutoff']=batch;noise_tables.append(noise)
        frame=frame.merge(noise[['formulation_id','batch_id','mean_sd']].rename(columns={'mean_sd':'adaptive_sd'}),on=['formulation_id','batch_id'],how='left')
        actual=obs.loc[obs.batch_id.eq(batch)&obs.endpoint.eq('viability_percent')].groupby('formulation_id').value.mean()
        proposal=proposal.loc[proposal.formulation_id.isin(actual.index)]
        x=proposal[registry.feature_names].to_numpy(float)
        for variant,adaptive in [(v,False) for v in VARIANTS]+[('campaign_existing',True)]:
            start=time.monotonic(); name=variant+('_adaptive' if adaptive else '')
            try:
                with warnings.catch_warnings(record=True) as caught:
                    mean,sd=fit_predict(variant,frame,x,registry,adaptive)
                for i,(_,r) in enumerate(proposal.iterrows()):
                    records.append(dict(round_id=batch,variant=name,formulation_id=r.formulation_id,actual=float(actual[r.formulation_id]),
                        predicted=float(mean[i]),std=float(sd[i]),support=r.get('viability_prediction_status','unclassified_historical'),
                        role=r.get('recommendation_type','unknown'),seconds=time.monotonic()-start,cutoff=batch,
                        warnings='; '.join(sorted(set(str(w.message) for w in caught)))))
            except Exception as exc: failures.append({'round_id':batch,'variant':name,'error':str(exc)})
    table=pd.DataFrame(records);table.to_csv(output/'predictions.csv',index=False)
    rows=[]
    for keys,g in table.groupby(['variant','round_id']): rows.append(dict(variant=keys[0],cohort=keys[1],**metrics(g)))
    for name,g in table.groupby('variant'):
        rows.append(dict(variant=name,cohort='pooled',**metrics(g)))
        for role,h in g.groupby('role'): rows.append(dict(variant=name,cohort='role:'+role,**metrics(h)))
        for support,h in g.groupby('support'): rows.append(dict(variant=name,cohort='support:'+support,**metrics(h)))
    summary=pd.DataFrame(rows);summary.to_csv(output/'metrics.csv',index=False)
    pd.concat(noise_tables).to_csv(output/'derived_noise.csv',index=False)
    (output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    pooled=summary.loc[summary.cohort.eq('pooled')].sort_values('mae')
    report=['# Offline model audit','', 'Development evidence: frozen candidates from Groups 3–8. No policy or model is promoted.',
        'Group 9 outcomes are not used. This is not a replay of alternative experimental selections.', '',
        '| Variant | MAE (points) | Bias | R² | 95% coverage | Mean width |', '|---|---:|---:|---:|---:|---:|']
    for _,r in pooled.iterrows():report.append(f'| {r.variant} | {r.mae:.3f} | {r.bias:.3f} | {r.r2:.3f} | {r.interval_95_coverage:.3f} | {r.interval_95_width:.3f} |')
    report+=['','Review the simplest campaign-only alternatives first; the observed prior/source mismatch is material. Learned-kernel and correction variants require fresh prospective validation.',
        'Adaptive noise assumes independent preparations only when explicitly identified. Historical unknown independence uses a conservative fallback.',
        'The correction uncertainty uses an independent-component approximation; it is diagnostic, not validated calibration.',
        'No repeated configured controls or supplementary mechanical endpoints exist in this snapshot: batch adjustment and mechanical-target comparison are not estimable.',
        'The separate ModelListGP implementation is tested on synthetic unequal endpoint datasets; it has not replaced production acquisition.',
        f'Failed fits: {len(failures)}. Details and convergence warnings are retained in CSV/JSON.',
        'User decision before the first full-mechanics proposal is required before any production switch.']
    (output/'audit_report.md').write_text('\n'.join(report)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args();run(a.root,a.output_dir)

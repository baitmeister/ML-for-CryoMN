"""Read-only model sensitivity followed by evidence export; never writes proposals."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .models import train_endpoint_models, paired_objective_frame
from .gp_strategy import fit_viability, shared_scores, control_mask, StrategyDecisionRequired
from .batch_gp import torch_bundle


def check_sensitivity(forms,obs,pool,registry,config,output):
    models=train_endpoint_models(forms,obs,registry,config)
    raw=config['_gp_raw_observations']
    candidates=pool.loc[~control_mask(pool)].drop_duplicates('formulation_id').copy()
    if len(candidates)<20:raise StrategyDecisionRequired('Sensitivity requires at least 20 feasible ordinary candidates')
    x=candidates[registry.feature_names].to_numpy(float)
    paired=paired_objective_frame(models.training_frame)
    paired=paired.dropna(subset=['viability_percent','critical_axial_load_N_per_needle'])
    train_x=paired[registry.feature_names].to_numpy(float)
    ref=config.get('selection',{}).get('reference_point',{})
    reference=(float(ref.get('viability_percent',0)),float(ref.get('critical_axial_load_N_per_needle',0)))
    mechanical=models.critical_load.model
    records=[];offsets=[];predictions={};top={}
    for sd in (25.,50.,100.):
        viability=fit_viability(forms,raw,registry.feature_names,'control',reference_sd=sd)
        models.acquisition_bundle=torch_bundle([viability,mechanical])
        mean=viability.predict(x)
        scores,_=shared_scores(models,train_x,x,reference)
        if not np.isfinite(mean).all():raise RuntimeError('Nonfinite sensitivity predictions')
        table=pd.DataFrame({'formulation_id':candidates.formulation_id.to_numpy(),'reference_sd_pp':sd,'latent_viability':mean,'qlognehvi':scores})
        table=table.sort_values(['qlognehvi','formulation_id'],ascending=[False,True],kind='stable')
        table['acquisition_rank']=np.arange(1,len(table)+1)
        records.append(table);predictions[sd]=mean;top[sd]=set(table.head(20).formulation_id)
        offsets.extend([{**r,'reference_sd_pp':sd} for r in viability.offsets() if r['batch_id']=='ROUND_010'])
    comparisons=[]
    for sd in (25.,100.):
        delta=float(np.max(np.abs(predictions[sd]-predictions[50.])))
        overlap=len(top[sd]&top[50.])
        comparisons.append(dict(reference_sd_pp=sd,max_absolute_latent_change_pp=delta,top20_overlap=overlap,flagged=delta>5 or overlap<15))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    candidates.to_csv(output/'candidate_pool.csv',index=False)
    pd.concat(records).to_csv(output/'scores.csv',index=False)
    pd.DataFrame(offsets).to_csv(output/'group10_offsets.csv',index=False)
    flagged=any(r['flagged'] for r in comparisons)
    summary=dict(flagged=flagged,comparisons=comparisons,candidates=len(x),production_reference_sd_pp=50,
                 acquisition_seed=42,mc_samples=128,reference_point=reference,
                 decision=config['gp_strategy_decision'],
                 observations_sha256=hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest(),
                 candidate_pool_sha256=hashlib.sha256((output/'candidate_pool.csv').read_bytes()).hexdigest())
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Group 11 control-prior sensitivity','',f'Feasible ordinary candidate count: {len(x)}. Production prior SD: 50 viability percentage points.',
           'Identical candidates, mechanical posterior, acquisition reference point and Monte Carlo seed for all three fits. Only control-baseline prior SD changes; batch and residual variance are refitted.',
           '', '| Prior SD | Largest latent mean change (pp) | Top-20 overlap | Flagged |','|---|---:|---:|---|']
    lines += [f"| {r['reference_sd_pp']:.0f} | {r['max_absolute_latent_change_pp']:.6f} | {r['top20_overlap']}/20 | {r['flagged']} |" for r in comparisons]
    lines += ['', 'Flags are operational review checks, not experimental acceptance thresholds.',
              'STOP: review sensitivity before generation.' if flagged else 'PASS: generation may proceed with the authorized 50-point prior.',
              '', 'One control batch remains weak evidence. This check does not establish calibration accuracy.',
              '[All scores](scores.csv) · [Group 10 batch shifts](group10_offsets.csv) · [Candidate pool](candidate_pool.csv)']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    if flagged:raise StrategyDecisionRequired('Control-prior sensitivity flagged; no official proposal generated. See '+str(output/'README.md'))
    return summary

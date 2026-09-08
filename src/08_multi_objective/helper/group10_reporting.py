"""Additional reporting only; historical primary metrics are not replaced."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .acceptance import assess_acceptance
from .batch_effects import control_diagnostics
from .group10_config import load_group10_config
from .endpoints import aggregate_intact_patch_replicates

def additional_reports(observations, prospective_table, results_root, output_dir):
    if 'experimental_role' not in observations: return []
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    records=[];settings=load_group10_config()
    for (fid,batch),g in observations.groupby(['formulation_id','batch_id']):
        if not g.experimental_role.fillna('').ne('').any(): continue
        metadata=Path(results_root)/'rounds'/str(batch)/'proposal/selection_metadata.json'
        cfg=settings if not metadata.exists() else json.loads(metadata.read_text()).get('group10',{}).get('effective_config',settings)
        req=cfg['application_requirements']
        vals=g.groupby('endpoint').value.mean().to_dict()
        intact=g.loc[g.endpoint.eq('intact_patch_formation_pass'),'value']
        definition=req.get('fracture_force_definition')
        endpoint='supported_axial_load_1mm_N_per_needle' if definition=='supported_load_1mm_v1' else 'critical_axial_load_N_per_needle'
        row={'formulation_id':fid,'batch_id':batch,**vals,'intact_patch_formation_pass':aggregate_intact_patch_replicates(intact),
             'mechanical_value':vals.get(endpoint),'mechanical_definition_id':'supported_load_1mm_v1' if endpoint.startswith('supported') else 'legacy_curve_maximum_v1'}
        records.append(assess_acceptance(row,req))
    pd.DataFrame(records).to_csv(out/'application_acceptance.csv',index=False)
    control_diagnostics(observations).to_csv(out/'reference_monitoring.csv',index=False)
    # Count ingested CSV replicates separately for each endpoint; do not invent
    # the number underlying an already-averaged measurement.
    counts=observations.loc[observations.experimental_role.fillna('').ne('')].copy()
    counts=counts.loc[pd.to_numeric(counts.value,errors='coerce').notna()]
    counts=counts.drop_duplicates(['formulation_id','batch_id','endpoint','replicate_id'])
    counts=counts.groupby(['formulation_id','batch_id','experimental_role','endpoint']).size().reset_index(name='recorded_replicate_count')
    counts['count_basis']='ingested_completed_csv_replicates; not independent preparations'
    counts.to_csv(out/'recorded_replicate_counts.csv',index=False)
    cohorts=[]
    if not prospective_table.empty:
        t=prospective_table.copy();t['diagnostic_cohort']='unclassified_historical'
        for batch,g in t.groupby('round_id'):
            p=Path(results_root)/'rounds'/str(batch)/'proposal/proposal.csv'
            if not p.exists():continue
            proposal=pd.read_csv(p).set_index('candidate_id')
            for i,row in g.iterrows():
                if row.candidate_id not in proposal.index:continue
                r=proposal.loc[row.candidate_id];role=r.get('experimental_role',r.get('recommendation_type',''))
                status=r.get('viability_prediction_status','unclassified_historical')
                t.loc[i,'diagnostic_cohort']=role if role in ('campaign_control','screened_hit_mechanics','retest_priority') else str(status)
        for keys,g in t.groupby(['endpoint','diagnostic_cohort']):
            valid=g.loc[g.evaluation_eligible.eq(True)].dropna(subset=['prediction_mean','observed_mean'])
            if valid.empty:continue
            e=valid.prediction_mean-valid.observed_mean;spread=((valid.observed_mean-valid.observed_mean.mean())**2).sum()
            cohorts.append({'endpoint':keys[0],'cohort':keys[1],'n':len(valid),'mae':e.abs().mean(),'bias':e.mean(),
                'rmse':np.sqrt((e**2).mean()),'r2':1-(e**2).sum()/spread if spread>0 and len(valid)>1 else None,
                'interval_95_coverage':pd.to_numeric(valid.interval_95_covered,errors='coerce').mean(),
                'interval_95_width':(valid.interval_95_upper-valid.interval_95_lower).mean()})
    pd.DataFrame(cohorts).to_csv(out/'diagnostic_cohorts.csv',index=False)
    return [out/n for n in ['application_acceptance.csv','reference_monitoring.csv','diagnostic_cohorts.csv','recorded_replicate_counts.csv']]

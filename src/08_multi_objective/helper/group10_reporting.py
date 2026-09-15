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
        mechanics=g.loc[g.endpoint.eq(endpoint)]
        actual=mechanics.get('mechanical_definition_id',pd.Series('',index=mechanics.index)).fillna('').replace('','legacy_curve_maximum_v1')
        identities=actual.unique()
        actual_definition=str(identities[0]) if len(identities)==1 else 'mixed_or_missing_definition'
        mechanical_value=vals.get(endpoint) if len(identities)==1 else None
        row={'formulation_id':fid,'batch_id':batch,**vals,'intact_patch_formation_pass':aggregate_intact_patch_replicates(intact),
             'mechanical_value':mechanical_value,'mechanical_definition_id':actual_definition}
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
    terminal=observations.loc[observations.endpoint.eq('terminal_force_08mm_status')].copy()
    if not terminal.empty:
        details=[]
        for row in terminal.itertuples():
            analysis=json.loads(row.analysis_provenance)
            details.append({'formulation_id':row.formulation_id,'batch_id':row.batch_id,'replicate_id':row.replicate_id,
                'experimental_role':row.experimental_role, 'mechanical_definition_id':analysis['definition_id'],
                **{k:analysis.get(k) for k in ('status','reason','endpoint_N_total','endpoint_N_per_needle',
                    'apparent_secant_stiffness_N_per_mm_total',
                    'apparent_secant_stiffness_N_per_mm_per_needle',
                    'stiffness_definition_id','stiffness_formula','stiffness_qc','loaded_needle_count',
                    'trigger_time_s','loading_duration_s','source_file','source_file_hash')}})
        pd.DataFrame(details).to_csv(out/'terminal_force_measurements.csv',index=False)
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
                mechanical = row.endpoint in ('critical_axial_load_N_per_needle','initial_stiffness_N_per_mm_per_needle')
                if mechanical and role == 'campaign_control':
                    t.loc[i,'diagnostic_cohort']='ordinary_mechanics'
                else:
                    t.loc[i,'diagnostic_cohort']=role if role in ('campaign_control','screened_hit_mechanics','retest_priority') else str(status)
        if 'mechanical_definition_id' not in t:
            t['mechanical_definition_id']=''
        t['mechanical_definition_id']=t.mechanical_definition_id.fillna('')
        for keys,g in t.groupby(['endpoint','diagnostic_cohort','mechanical_definition_id']):
            valid=g.loc[g.evaluation_eligible.eq(True)].dropna(subset=['prediction_mean','observed_mean'])
            if valid.empty:continue
            e=valid.prediction_mean-valid.observed_mean;spread=((valid.observed_mean-valid.observed_mean.mean())**2).sum()
            cohorts.append({'endpoint':keys[0],'cohort':keys[1],'mechanical_definition_id':keys[2],'n':len(valid),'mae':e.abs().mean(),'bias':e.mean(),
                'rmse':np.sqrt((e**2).mean()),'r2':1-(e**2).sum()/spread if spread>0 and len(valid)>1 else None,
                'interval_95_coverage':pd.to_numeric(valid.interval_95_covered,errors='coerce').mean(),
                'interval_95_width':(valid.interval_95_upper-valid.interval_95_lower).mean()})
    pd.DataFrame(cohorts).to_csv(out/'diagnostic_cohorts.csv',index=False)
    return [out/n for n in ['application_acceptance.csv','reference_monitoring.csv','diagnostic_cohorts.csv','recorded_replicate_counts.csv']]

"""Run synthetic outcomes for the frozen Group 10 plan and generate Group 11 in isolation.

Not discovered by unittest. Usage: python3 tests/verify_group10_endpoint_workflow.py
--output-dir /private/tmp/new-empty-validation-directory
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/08_multi_objective'))
from helper.group10_config import production_observations
from helper.terminal_force import DEFINITION, MODEL_FIELD


def main(output):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
               for name in ('data','results') for p in (ROOT/name).rglob('*') if p.is_file()}
    data=output/'data';data.mkdir()
    for name in ('formulations.csv','observations.csv'):
        shutil.copy2(ROOT/'data/processed_v2'/name,data/name)
    results=output/'results';next_dir=results/'next_round'
    # Retain only the real preceding evidence snapshot, then invent Group 10
    # outcomes inside the isolated copy. Never re-ingest the live Group 9 rows.
    obs=pd.read_csv(data/'observations.csv')
    numbers=pd.to_numeric(obs.batch_id.astype(str).str.extract(r'^ROUND_(\d+)$')[0],errors='coerce')
    obs.loc[numbers.isna() | numbers.le(9)].to_csv(data/'observations.csv',index=False)
    source=ROOT/'results/multi_objective_v2/rounds/ROUND_010/proposal'
    shutil.copytree(source,results/'rounds/ROUND_010/proposal')
    next_dir.mkdir(parents=True)
    shutil.copy2(source/'proposal.csv',next_dir/'next_round_candidates.csv')
    shared=['--formulations',str(data/'formulations.csv'),'--observations',str(data/'observations.csv'),
            '--output-dir',str(next_dir),'--total-candidate-pool',str(results/'pool.csv')]
    def run(label,script,args):
        completed=subprocess.run([sys.executable,str(ROOT/script),*args],cwd=ROOT,capture_output=True,text=True)
        (output/(label+'.log')).write_text(completed.stdout+completed.stderr)
        if completed.returncode:
            raise RuntimeError(f'{label} failed: '+completed.stderr[-5000:])
        print(label+' passed',flush=True)
    ten=pd.read_csv(next_dir/'next_round_candidates.csv')
    assert ten.batch_id.unique().tolist()==['ROUND_010']
    assert ten.experimental_role.eq('campaign_control').sum()==1
    assert ten.experimental_role.eq('screened_hit_mechanics').sum()==1
    assert ten.mechanical_prediction_definition_id.eq(DEFINITION).all()
    mechanical=ten.sort_values('mechanical_selection_rank').head(4)
    assert mechanical.experimental_role.value_counts().to_dict()=={'ordinary':2,'campaign_control':1,'screened_hit_mechanics':1}
    originals=ten.copy()
    ten['viability_percent']=65+np.arange(len(ten))*.3
    ten['intact_patch_formation_pass']=1
    ten['replicate_id']='synthetic_rep1'
    raw_dir=output/'synthetic_curves';raw_dir.mkdir()
    extra=[]
    for j,i in enumerate(mechanical.index):
        for rep in (1,2):
            row=ten.loc[i].copy()
            row['replicate_id']=f'synthetic_rep{rep}'
            row['mechanical_test_attempted']=True
            row['mechanical_test_id']=f'synthetic_{j}_{rep}'
            row['needles_compressed']=100
            row['instron_file']=''
            if not (j==3 and rep==2):  # Lost specimen represented only as an attempt.
                n=150 if j==2 and rep==2 else 205
                t=np.arange(n)*.025;d=np.arange(n)*.005+2
                f=np.maximum(0.,1+(d-d[4])*(10+j+rep*.1));f[:4]=[.1,.4,.6,.9]
                raw=raw_dir/f'curve_{j}_{rep}.csv'
                pd.DataFrame({'Time':t,'Displacement':d,'Force':f}).to_csv(raw,index=False)
                row['instron_file']=str(raw)
            if rep==1:
                for col,val in row.items():
                    if col in ('replicate_id','mechanical_test_attempted','mechanical_test_id','needles_compressed','instron_file'):
                        if col not in ten or ten[col].dtype != object:ten[col]=ten[col].astype(object)
                        ten.at[i,col]=val
            else:
                extra.append(row)
    ten=pd.concat([ten,pd.DataFrame(extra)],ignore_index=True)
    ten.to_csv(output/'synthetic_group10.csv',index=False)
    run('group10_update_and_group11','src/08_multi_objective/03_run_round/run_round.py',
        [str(output/'synthetic_group10.csv'),*shared,'--seed','42'])
    obs=pd.read_csv(data/'observations.csv')
    new=obs.loc[obs.batch_id.eq('ROUND_010')]
    active=production_observations(obs)
    assert active.loc[active.endpoint.eq(MODEL_FIELD),'mechanical_definition_id'].eq(DEFINITION).all()
    control_evidence=active.loc[active.experimental_role.fillna('').eq('campaign_control')]
    assert not control_evidence.endpoint.eq('viability_percent').any()
    assert control_evidence.endpoint.eq(MODEL_FIELD).any()
    eleven=pd.read_csv(next_dir/'next_round_candidates.csv')
    assert eleven.batch_id.unique().tolist()==['ROUND_011']
    control=eleven.loc[eleven.experimental_role.eq('campaign_control')]
    assert len(control)==1 and control.mechanical_selection_rank.isna().all()
    assert control.mechanical_test_recommended.eq(False).all()
    primaries=eleven.loc[eleven.mechanical_test_recommended.eq(True)]
    assert len(primaries)==4 and not primaries.experimental_role.eq('campaign_control').any()
    eleven_metadata=json.loads((results/'rounds/ROUND_011/proposal/selection_metadata.json').read_text())
    from helper.models import build_training_frame, paired_objective_frame
    from helper.registry import load_registry
    from helper.phase import resolve_phase_mode
    from helper.config import load_optimization_config
    forms=pd.read_csv(data/'formulations.csv')
    phase=resolve_phase_mode(forms,obs,load_registry(),load_optimization_config(),target_round_number=11)
    assert phase.paired_observation_count==8
    assert phase.distinct_formulation_count==8
    assert phase.batch_count==2
    assert phase.active_phase=='mechanics_hybrid'
    assert eleven_metadata['active_phase']=='mechanics_hybrid'
    manifest=json.loads((results/'rounds/ROUND_010/completed/mechanical_execution_manifest.json').read_text())
    assert manifest['workload']['specimen_runs_attempted']==8
    assert manifest['workload']['tests_with_interpretable_terminal_force']==6
    assert manifest['workload']['usable_formulation_batch_observations']==4
    assert (results/'rounds/ROUND_010/reports/tables/terminal_force_measurements.csv').exists()
    assert (results/'rounds/ROUND_010/reports/tables/prospective_evaluation_table.csv').exists()
    changed=[p for p,h in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    summary=dict(synthetic_only=True,group10_original_frozen_contract_ingestible=True,
        group10_frozen_definition=DEFINITION,group10_roles=originals.experimental_role.value_counts().to_dict(),
        mechanical_primary_roles=mechanical.experimental_role.value_counts().to_dict(),
        production_gp_methodology='unchanged',legacy_mechanics_excluded_from_new_target=True,
        control_viability_excluded_from_training=True,control_mechanics_retained_as_endpoint_evidence=True,
        group11_control_unranked=True,group11_mechanical_primaries=len(primaries),
        group11_phase=phase.active_phase,paired_observations=phase.paired_observation_count,next_proposal='ROUND_011',workload=manifest['workload'],
        protected_source_files=len(protected),protected_hash_mismatches=changed,
        evidence_limitation='Synthetic outcomes on the actual frozen Group 10 slate; Group 11 is an isolated test proposal')
    (output/'verification_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    main(parser.parse_args().output_dir)

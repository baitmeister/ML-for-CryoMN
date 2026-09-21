from pathlib import Path
import sys, json, tempfile, shutil, subprocess, threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src/08_multi_objective'))
from helper import artifacts
from helper.registry import load_registry
from helper.models import build_training_frame, paired_objective_frame, _fit_regression
from helper.terminal_force import analyze_file
from helper.phase import _completed_screening_round_count

evidence = {}
forms = pd.read_csv(ROOT/'data/processed_v2/formulations.csv')
obs = pd.read_csv(ROOT/'data/processed_v2/observations.csv')
registry = load_registry()
p = pd.read_csv(ROOT/'results/multi_objective_v2/reports/prospective/tables/prospective_evaluation_table.csv')
v = p.loc[p.endpoint.eq('viability_percent') & p.formal_metric_eligible]
evidence['prospective_recomputed'] = dict(n=len(v), mae=float((v.prediction_mean-v.observed_mean).abs().mean()),
    coverage=float(((v.observed_mean>=v.interval_95_lower)&(v.observed_mean<=v.interval_95_upper)).mean()),
    width=float((v.interval_95_upper-v.interval_95_lower).mean()))

raw_checks = []
for _, row in obs.loc[obs.endpoint.eq('terminal_force_08mm_status')].iterrows():
    stored = json.loads(row.analysis_provenance)
    path = Path(stored['source_file'])
    result = {'observation_id': row.observation_id, 'source_exists': path.exists()}
    if path.exists():
        actual = analyze_file(path, stored['settings'], stored['loaded_needle_count'])
        result.update(status=actual['status'], hash_matches=actual['source_file_hash']==stored['source_file_hash'],
            status_matches=actual['status']==stored['status'], force_matches=actual['endpoint_N_total']==stored['endpoint_N_total'])
    raw_checks.append(result)
evidence['raw_terminal_reproduction'] = raw_checks

frame = paired_objective_frame(build_training_frame(forms, obs, registry)).dropna(subset=['viability_percent','critical_axial_load_N_per_needle'])
feature_x=frame[registry.feature_names].to_numpy(float)
y=frame.critical_axial_load_N_per_needle.reset_index(drop=True)
preds=[];means=[]
for i in range(len(y)):
    keep=np.arange(len(y))!=i
    model=_fit_regression(feature_x[keep],y[keep].reset_index(drop=True),y_noise=pd.Series(np.nan,index=range(sum(keep))))
    preds.append(model.predict(feature_x[i:i+1]).mean[0]);means.append(y[keep].mean())
evidence['mechanics_exploratory_loo']={'n':len(y),'gp_mae':float(np.mean(np.abs(y-np.array(preds)))),
    'training_mean_mae':float(np.mean(np.abs(y-np.array(means)))),'gp_predictions':preds,'actual':y.tolist(),
    'limitation':'Post-hoc eight-row diagnostic, not prospective validation or a reliable model-selection comparison.'}

with tempfile.TemporaryDirectory(prefix='cryomn_review_') as directory:
    tmp=Path(directory);out=tmp/'results';proposal=out/'rounds/ROUND_011/proposal';proposal.mkdir(parents=True)
    source=ROOT/'results/multi_objective_v2/rounds/ROUND_011/proposal'
    for name in ('proposal.csv','selection_metadata.json'):
        shutil.copy2(source/name,proposal/name)
    sheet=pd.read_csv(proposal/'proposal.csv')
    sheet['viability_percent']=sheet.viability_percent.astype(object)
    sheet.loc[5,'viability_percent']='fifty'
    sheet.loc[5,'intact_patch_formation_pass']=1
    sheet_path=tmp/'partial.csv';sheet.to_csv(sheet_path,index=False)
    forms_path=tmp/'formulations.csv';obs_path=tmp/'observations.csv'
    forms.to_csv(forms_path,index=False);obs.to_csv(obs_path,index=False)
    command=[sys.executable,str(ROOT/'src/08_multi_objective/03_run_round/run_round.py'),str(sheet_path),
        '--formulations',str(forms_path),'--observations',str(obs_path),'--output-dir',str(out/'next_round'),
        '--total-candidate-pool',str(out/'pool.csv'),'--skip-review','--skip-generate']
    first=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    after=pd.read_csv(obs_path);new=after.loc[after.batch_id.eq('ROUND_011')]
    evidence['partial_and_invalid_number']={'exit_code':first.returncode,'stdout':first.stdout,'stderr':first.stderr,
        'new_observations':new[['formulation_id','endpoint','value']].to_dict('records'),
        'completed_archive_exists':(proposal.parent/'completed/completed.csv').exists(),
        'status':json.loads((out/'current_round_status.json').read_text()) if (out/'current_round_status.json').exists() else None}
    sheet.loc[5,'viability_percent']=55;sheet.to_csv(sheet_path,index=False)
    second=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    evidence['partial_completion_retry']={'exit_code':second.returncode,'stdout':second.stdout,'stderr':second.stderr}

    # Force both writers past the existence check before either creates the destination.
    a=tmp/'a';b=tmp/'b';dest=tmp/'frozen';a.write_text('A');b.write_text('B')
    barrier=threading.Barrier(2);original=artifacts._atomic_copy
    def synchronized_copy(source,destination):
        barrier.wait(timeout=5)
        return original(source,destination)
    with patch.object(artifacts,'_atomic_copy',synchronized_copy):
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(artifacts.copy_frozen,path,dest) for path in (a,b)]
            outcomes=[str(job.result()) for job in jobs]
    evidence['frozen_copy_race']={'successful_writers':len(outcomes),'final_bytes':dest.read_text(),
        'method':'Controlled concurrent interleaving in a temporary directory; no production artifacts touched.'}

tiny=pd.DataFrame([{'batch_id':'ROUND_001','formulation_id':'a','endpoint':'viability_percent','value':60},
                   {'batch_id':'ROUND_001','formulation_id':'b','endpoint':'intact_patch_formation_pass','value':1}])
evidence['screening_completion_gate_probe']={'rows':2,'distinct_formulations':2,
    'complete_viability_intact_pairs':0,'counted_completed_rounds':_completed_screening_round_count(tiny)}
with tempfile.TemporaryDirectory(prefix='cryomn_rebuild_probe_') as directory:
    tmp=Path(directory)
    for name in ('formulations.csv','observations.csv'):
        shutil.copy2(ROOT/'data/processed_v2'/name,tmp/name)
    before=len(pd.read_csv(tmp/'observations.csv'))
    command=[sys.executable,str(ROOT/'src/08_multi_objective/01_build_database/build_database.py'),'--output-dir',directory]
    run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    after=pd.read_csv(tmp/'observations.csv')
    evidence['initializer_overwrite_probe']={'exit_code':run.returncode,'before_observations':before,
        'after_observations':len(after),'remaining_campaign_observations':int(after.batch_id.str.match(r'^ROUND_\d+$').sum()),'stdout':run.stdout}
Path('/private/tmp/cryomn_review_evidence.json').write_text(json.dumps(evidence,indent=2,default=str)+'\n')
print(json.dumps(evidence,indent=2,default=str))

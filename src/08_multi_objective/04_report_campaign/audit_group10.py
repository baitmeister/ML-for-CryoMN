"""Dynamic offline forward audit. Never promotes a model or changes a proposal.

Available observed rounds with frozen proposals are evaluated at their preceding
cutoffs. This is a retrospective refit, not frozen challenger prediction evidence.
"""
import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from helper.registry import load_registry
from helper.models import build_training_frame
from helper.audit_models import VARIANTS, fit_predict
from helper.audit_sensitivity import settings, describe, training_rows, fit_sensitivity
from helper.observation_noise import estimate_noise
from helper.group10_config import production_observations
from helper.terminal_force import MODEL_FIELD, DEFINITION, LEGACY_DEFINITION

ENDPOINTS = ('viability_percent', MODEL_FIELD)


def metrics(g):
    y=g.actual.to_numpy(float);p=g.predicted.to_numpy(float);s=g['std'].to_numpy(float);e=p-y
    spread=np.sum((y-y.mean())**2)
    lower=g.get('latent_lower',pd.Series(np.nan,index=g.index)).fillna(pd.Series(p-1.96*s,index=g.index)).to_numpy()
    upper=g.get('latent_upper',pd.Series(np.nan,index=g.index)).fillna(pd.Series(p+1.96*s,index=g.index)).to_numpy()
    return dict(n=len(g),mae=float(abs(e).mean()),bias=float(e.mean()),rmse=float(np.sqrt(np.mean(e**2))),
        r2=float(1-np.sum(e**2)/spread) if len(g)>1 and spread>0 else None,
        interval_95_coverage=float(np.mean((y>=lower)&(y<=upper))),interval_95_width=float(np.mean(upper-lower)),
        spearman=float(pd.Series(y).corr(pd.Series(p),method='spearman')) if len(g)>1 and np.ptp(p)>1e-10 and np.ptp(y)>1e-10 else None,
        future_interval_95_coverage=float(((g.actual>=g.future_lower)&(g.actual<=g.future_upper)).loc[g.future_lower.notna()].mean()) if 'future_lower' in g else None,
        future_interval_95_width=float((g.future_upper-g.future_lower).mean()) if 'future_lower' in g else None)


def observed_rounds(obs, start_round=3, end_round=None):
    rounds = pd.to_numeric(obs.batch_id.astype(str).str.extract(r'^ROUND_(\d+)$')[0], errors='coerce')
    valid = pd.to_numeric(obs.value, errors='coerce').notna() & obs.endpoint.isin(ENDPOINTS)
    return sorted(int(n) for n in rounds.loc[valid].dropna().unique()
                  if n >= start_round and (end_round is None or n <= end_round))


def run(root, output, start_round=3, end_round=None, endpoints=ENDPOINTS, suite='expanded'):
    root=Path(root).resolve(); output=Path(output).resolve()
    if suite not in ('historical','expanded'):raise ValueError('Unknown audit suite')
    if start_round < 1 or (end_round is not None and end_round < start_round):
        raise ValueError('Invalid audit round range')
    if not endpoints or not set(endpoints).issubset(ENDPOINTS):
        raise ValueError('Unsupported audit endpoints')
    if output==root or root/'data'==output or (root/'data') in output.parents or output==root/'results/multi_objective_v2' or (root/'results/multi_objective_v2') in output.parents:
        raise ValueError('Audit output must not be a live data/results directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty audit output directory to preserve earlier decision evidence')
    output.mkdir(parents=True,exist_ok=True)
    registry=load_registry(root/'config_v2/ingredients.yaml')
    forms=pd.read_csv(root/'data/processed_v2/formulations.csv')
    raw_obs=pd.read_csv(root/'data/processed_v2/observations.csv')
    numbers=pd.to_numeric(raw_obs.batch_id.astype(str).str.extract(r'^ROUND_(\d+)$')[0],errors='coerce')
    rounds=observed_rounds(raw_obs,start_round,end_round)
    records=[]; failures=[]; skipped=[]; noise_tables=[]; proposal_hashes={}
    for number in rounds:
        batch=f'ROUND_{number:03d}'
        proposal_path=root/f'results/multi_objective_v2/rounds/{batch}/proposal/proposal.csv'
        if not proposal_path.exists():
            skipped.append(dict(round_id=batch,reason='No frozen proposal available'));continue
        proposal_hashes[batch]=hashlib.sha256(proposal_path.read_bytes()).hexdigest()
        proposal=pd.read_csv(proposal_path)
        # Determine a mechanical cohort from this round, then restrict all prior
        # mechanical evidence to that definition. Future outcomes never enter fit.
        current=raw_obs.loc[numbers.eq(number)]
        prior_raw=raw_obs.loc[numbers.lt(number) | ~raw_obs.batch_id.astype(str).str.startswith('ROUND_')]
        for endpoint in endpoints:
            endpoint_rows=current.loc[current.endpoint.eq(endpoint)]
            if endpoint_rows.empty:
                skipped.append(dict(round_id=batch,endpoint=endpoint,reason='No measured outcomes'));continue
            definitions=[''] if endpoint=='viability_percent' else sorted(
                endpoint_rows.get('mechanical_definition_id',pd.Series('',index=endpoint_rows.index)).fillna('').replace('',LEGACY_DEFINITION).unique())
            for definition in definitions:
                if definition and definition not in (DEFINITION, LEGACY_DEFINITION):
                    skipped.append(dict(round_id=batch,endpoint=endpoint,mechanical_definition_id=definition,reason='Unsupported mechanical definition'));continue
                prior=production_observations(prior_raw,mechanical_definition=definition or None)
                actual_rows=production_observations(current,mechanical_definition=definition or None)
                frame=build_training_frame(forms,prior,registry)
                if endpoint not in frame:
                    skipped.append(dict(round_id=batch,endpoint=endpoint,mechanical_definition_id=definition,reason='No preceding comparable target measurements'));continue
                measured=prior.loc[prior.endpoint.eq(endpoint)]
                # The offline adaptive estimator is specified in viability
                # percentage points; do not apply its floors to force in N.
                noise=estimate_noise(measured) if endpoint=='viability_percent' else pd.DataFrame()
                if not noise.empty:
                    noise['cutoff']=batch;noise['audit_endpoint']=endpoint;noise['mechanical_definition_id']=definition
                    noise_tables.append(noise)
                    frame=frame.merge(noise[['formulation_id','batch_id','mean_sd']].rename(columns={'mean_sd':'adaptive_sd'}),on=['formulation_id','batch_id'],how='left')
                actual=actual_rows.loc[actual_rows.endpoint.eq(endpoint)].groupby('formulation_id').value.mean()
                slate=proposal.loc[proposal.formulation_id.isin(actual.index)]
                if slate.empty:
                    skipped.append(dict(round_id=batch,endpoint=endpoint,mechanical_definition_id=definition,reason='No eligible measured proposal rows'));continue
                x=slate[registry.feature_names].to_numpy(float)
                variants=VARIANTS if endpoint=='viability_percent' else ('campaign_mean','campaign_median','campaign_existing','campaign_bounds_learned')
                comparisons=[(v,False) for v in variants]
                if endpoint=='viability_percent': comparisons.append(('campaign_existing',True))
                sensitivity=settings(endpoint) if suite=='expanded' else {}
                comparisons.extend((name,False) for name in sensitivity)
                for variant,adaptive in comparisons:
                    start=time.monotonic();name=variant+('_adaptive' if adaptive else '')
                    training=frame.loc[frame[endpoint].notna()]
                    if variant not in ('current_mixed','legacy_correction'):
                        training=training.loc[training.batch_id.astype(str).str.startswith('ROUND_')]
                    if variant in sensitivity:training=training_rows(frame,endpoint,sensitivity[variant])
                    identity=dict(round_id=batch,endpoint=endpoint,mechanical_definition_id=definition,variant=name)
                    if len(training)<2:
                        skipped.append(dict(**identity,reason='Fewer than two preceding comparable training rows',training_rows=len(training)));continue
                    try:
                        detail={}
                        with warnings.catch_warnings(record=True) as caught:
                            if variant in sensitivity:
                                mean,sd,detail=fit_sensitivity(frame,x,registry,endpoint,sensitivity[variant])
                            else:
                                mean,sd=fit_predict(variant,frame,x,registry,adaptive,endpoint=endpoint)
                        if not np.isfinite(mean).all() or not np.isfinite(sd).all():
                            raise ValueError('Nonfinite predictions or uncertainty')
                        for i,(_,r) in enumerate(slate.iterrows()):
                            records.append(dict(**identity,formulation_id=r.formulation_id,actual=float(actual[r.formulation_id]),
                                predicted=float(mean[i]),std=float(sd[i]),training_rows=len(training),
                                factor=sensitivity[variant].factor if variant in sensitivity else 'historical_comparison',
                                fit_details=json.dumps({k:v for k,v in detail.items() if k not in ('latent_lower','latent_upper','future_lower','future_upper')},sort_keys=True),
                                latent_lower=float(detail['latent_lower'][i]) if detail else np.nan,
                                latent_upper=float(detail['latent_upper'][i]) if detail else np.nan,
                                future_lower=float(detail['future_lower'][i]) if detail else np.nan,
                                future_upper=float(detail['future_upper'][i]) if detail else np.nan,
                                uncertainty_kind='latent_sd_plus_explicit_future_interval' if detail else 'historical_spread' if variant in ('campaign_mean','campaign_median') else 'returned_model_sd_not_future_observation_interval',
                                support=r.get('viability_prediction_status' if endpoint=='viability_percent' else 'mechanical_prediction_status','unclassified_historical'),
                                role=('ordinary_mechanics' if endpoint==MODEL_FIELD and r.get('recommendation_type')=='campaign_control' else r.get('recommendation_type','unknown')),seconds=time.monotonic()-start,cutoff=batch,
                                warnings='; '.join(sorted(set(str(w.message) for w in caught)))))
                    except Exception as exc: failures.append(dict(**identity,error=str(exc)))
    columns=['round_id','endpoint','mechanical_definition_id','variant','formulation_id','actual','predicted','std','training_rows','uncertainty_kind','support','role','seconds','cutoff','warnings','factor','fit_details','latent_lower','latent_upper','future_lower','future_upper']
    table=pd.DataFrame(records,columns=columns);table.to_csv(output/'predictions.csv',index=False)
    rows=[]
    for keys,g in table.groupby(['endpoint','mechanical_definition_id','variant'],dropna=False):
        identity=dict(endpoint=keys[0],mechanical_definition_id=keys[1],variant=keys[2])
        rows.append(dict(**identity,cohort='pooled',**metrics(g)))
        for batch,h in g.groupby('round_id'): rows.append(dict(**identity,cohort=batch,**metrics(h)))
        for role,h in g.groupby('role'): rows.append(dict(**identity,cohort='role:'+str(role),**metrics(h)))
        for support,h in g.groupby('support'): rows.append(dict(**identity,cohort='support:'+str(support),**metrics(h)))
    summary=pd.DataFrame(rows,columns=['endpoint','mechanical_definition_id','variant','cohort','n','mae','bias','rmse','r2','interval_95_coverage','interval_95_width','spearman','future_interval_95_coverage','future_interval_95_width'])
    summary.to_csv(output/'metrics.csv',index=False)
    noise_output=pd.concat(noise_tables,ignore_index=True) if noise_tables else pd.DataFrame(columns=['cutoff','audit_endpoint','mechanical_definition_id'])
    noise_output.to_csv(output/'derived_noise.csv',index=False)
    (output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    (output/'skipped.json').write_text(json.dumps(skipped,indent=2)+'\n')
    catalogue={endpoint:{name:describe(setting) for name,setting in settings(endpoint).items()} for endpoint in endpoints} if suite=='expanded' else {}
    (output/'sensitivity_settings.json').write_text(json.dumps(catalogue,indent=2)+'\n')
    comparisons_out=[]
    for (endpoint,definition,batch),group in table.groupby(['endpoint','mechanical_definition_id','round_id'],dropna=False):
        for variant in group.variant.unique():
            if not variant.startswith('sens_') or variant=='sens_baseline':continue
            baseline='sens_bounds_learned' if variant in ('sens_bounds_learned_wide','sens_bounds_learned_restarts') else 'sens_baseline'
            merged=group.loc[group.variant.eq(variant)].merge(group.loc[group.variant.eq(baseline)],on='formulation_id',suffixes=('_test','_base'))
            if merged.empty:continue
            comparisons_out.append(dict(endpoint=endpoint,mechanical_definition_id=definition,round_id=batch,variant=variant,baseline=baseline,n=len(merged),
                mae_change=float((merged.predicted_test-merged.actual_test).abs().mean()-(merged.predicted_base-merged.actual_base).abs().mean()),
                mean_absolute_prediction_change=float((merged.predicted_test-merged.predicted_base).abs().mean())))
    pd.DataFrame(comparisons_out,columns=['endpoint','mechanical_definition_id','round_id','variant','baseline','n','mae_change','mean_absolute_prediction_change']).to_csv(output/'sensitivity_comparisons.csv',index=False)
    ranking=[]
    for keys,group in table.loc[table.endpoint.eq('viability_percent')].groupby(['round_id','variant']):
        selected_base=None
        for kappa in (.35,0.,1.,2.):
            scored=group.assign(score=group.predicted+kappa*group['std']).sort_values(['score','formulation_id'],ascending=[False,True])
            selected=scored.head(4)
            identities=set(selected.formulation_id)
            if selected_base is None:selected_base=identities
            ranking.append(dict(round_id=keys[0],variant=keys[1],kappa=kappa,selected_count=len(selected),
                top4_mean_actual=float(selected.actual.mean()),overlap_with_kappa035=len(identities & selected_base),
                selected_formulations=';'.join(selected.formulation_id.astype(str)),
                scope='rerank_measured_frozen_slate_without_production_constraints_or_new_candidates'))
    pd.DataFrame(ranking,columns=['round_id','variant','kappa','selected_count','top4_mean_actual','overlap_with_kappa035','selected_formulations','scope']).to_csv(output/'ucb_sensitivity.csv',index=False)
    scope=dict(suite=suite,audit_kind='retrospective_forward_refit',start_round=start_round,end_round=end_round,
        discovered_observed_rounds=rounds,evaluated_rounds=sorted(table.round_id.unique().tolist()),
        endpoints=list(endpoints),proposal_sha256=proposal_hashes,
        observations_sha256=hashlib.sha256((root/'data/processed_v2/observations.csv').read_bytes()).hexdigest(),
        model_promotion=False,production_artifacts_modified=False,
        untested_dimensions=['cross-output correlation / multi-task transfer','batch-effect normalization','qLogNEHVI versus qLogNParEGO on a common candidate pool','candidate-pool generation / feasibility / diversity policy changes','fully Bayesian hyperparameter priors'],
        limitation='Outcome-known strategy comparison; not frozen challenger evidence or a counterfactual acquisition trial')
    (output/'audit_scope.json').write_text(json.dumps(scope,indent=2)+'\n')
    report=['# Dynamic offline GP audit','',
        'Reading guide: docs/group10/audit_reading_guide.md explains all columns and model names. MAE = mean absolute error; Bias = prediction minus observation; R² = explained squared variation; n = evaluated observations. Interval coverage below uses original/latent intervals; future-observation coverage is a separate metrics.csv column.','',
        'Observed rounds discovered: '+(', '.join(str(n) for n in rounds) or 'none')+'.',
        'Each model is refitted using preceding outcomes only. Current-round outcomes are used only for evaluation.',
        'No strategy is promoted. Original production predictions remain in the staged prospective reports.',
        'Mechanical definitions are evaluated separately. Unavailable training is recorded in skipped.json, not replaced by fabricated forecasts.',
        '', '| Endpoint / definition | Model | n | MAE | Bias | R² | Interval coverage |',
        '|---|---|---:|---:|---:|---:|---:|']
    for _,r in summary.loc[summary.cohort.eq('pooled')].iterrows():
        report.append(f'| {r.endpoint} / {r.mechanical_definition_id} | {r.variant} | {r.n} | {r.mae:.4g} | {r.bias:.4g} | {r.r2:.3f} | {r.interval_95_coverage:.3f} |')
    report += ['', 'Viability errors are percentage points; mechanical errors are N per loaded needle.',
        'Historical intervals use mean ± 1.96 × returned SD. Expanded cases also store explicit latent and future intervals, including inverse-transformed intervals. Neither is automatically calibrated; future coverage and width have separate metric columns.',
        'Select using bias, within-round ranking, uncertainty and simple baselines together. Pooled R² alone does not establish useful selection.',
        'The acquisition ModelListGP remains offline-only. This audit compares prediction strategies, not the experimental benefit of alternative candidate slates.',
        'Expanded sensitivity settings and paired one-factor comparisons are in sensitivity_settings.json and sensitivity_comparisons.csv. Negative MAE change favors the challenger on matched outcomes.',
        'Expanded cases distinguish latent uncertainty from future intervals under their stated noise assumptions. These intervals require empirical validation; unknown-noise fallback and transformed-noise approximations are recorded.',
        'This is a one-factor study plus selected combined configurations, not an exhaustive factorial or a nested model-selection validation. Source selection also changes empirical centering/scaling when those are data-dependent.',
        'ucb_sensitivity.csv varies the uncertainty bonus on the measured historical slate only; it is not a replay of the full constrained selector or unknown alternative formulations.',
        'Acquisition, correlated-output transfer, batch effects and future-outcome benefit are not inferred from this prediction audit; see the decision guide for the required complementary checks.',
        'Freeze challenger predictions before future outcomes for a prospective strategy comparison.',
        f'Failed fits: {len(failures)}; skipped cohorts/comparisons: {len(skipped)}.']
    (output/'audit_report.md').write_text('\n'.join(report)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--start-round',type=int,default=3)
    p.add_argument('--end-round',type=int,default=None,help='Default: latest available observed round')
    p.add_argument('--endpoints',nargs='+',choices=ENDPOINTS,default=list(ENDPOINTS))
    p.add_argument('--suite',choices=['historical','expanded'],default='expanded')
    a=p.parse_args();run(a.root,a.output_dir,a.start_round,a.end_round,a.endpoints,a.suite)

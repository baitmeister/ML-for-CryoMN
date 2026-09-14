"""Group 10 experimental allocation. Production scoring remains unchanged."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from itertools import combinations
import hashlib
import numpy as np
import pandas as pd
from .group10_config import active, reference_readiness, production_observations, ROLES
from .models import train_endpoint_models, build_training_frame
from .registry import IngredientRegistry, presence_threshold
from .feasibility import annotate_feasibility
from .selection_scoring import annotate_candidates
from .similarity import resolve_similarity_policy, build_history_similarity_index, validate_selected_similarity
from .selection_policy import select_mechanical_tests
from .phase import resolve_phase_mode
from .intact_policy import resolve_intact_combination_policy, build_intact_evidence, annotate_intact_combination_evidence


def _combo(row,registry):
    return tuple(f for f in registry.feature_names if float(row.get(f,0) or 0)>=presence_threshold(f))

def reference_candidate(config, registry):
    state=reference_readiness(config)
    if not state['ready']: return pd.DataFrame(),state
    r=config['reference']; row={f:0. for f in registry.feature_names}
    row.update(dmso_M=r['dmso_volume_percent']*10*r['density_g_mL']*r['purity_fraction']/r['molecular_weight_g_mol'],sucrose_M=.1)
    key=','.join(f'{f}:{row[f]:.12g}' for f in registry.feature_names)
    row.update(formulation_id='reference_'+hashlib.sha256(key.encode()).hexdigest()[:12],candidate_id='reference_control',
               recommendation_type='campaign_control',experimental_role='campaign_control',candidate_origin='campaign_control',
               preparation_basis='2.5% v/v DMSO + 100 mM sucrose; final volume; common campaign base medium')
    return pd.DataFrame([row]),state

def reference_feasibility(frame, registry, optimization, config):
    expected, state = reference_candidate(config, registry)
    if not state["ready"] or len(frame)!=1 or not np.allclose(frame[registry.feature_names].to_numpy(float), expected[registry.feature_names].to_numpy(float),rtol=0,atol=1e-12):
        raise ValueError("Reference exception applies only to the exact configured recipe")
    # Only the exact declared reference gets a DMSO ceiling exception.
    modified=IngredientRegistry([replace(i,upper_bound=max(i.upper_bound,float(frame.iloc[0].dmso_M)),
                          practical_upper_bound=max(i.practical_upper_bound or 0,float(frame.iloc[0].dmso_M)))
                          if i.feature_name=='dmso_M' else i for i in registry.ingredients])
    return annotate_feasibility(frame,modified,optimization,policy_active=True)

def apply_group10(result, formulations, observations, registry, optimization, config, round_number, unavailable=()):
    if not active(config,round_number): return result
    observations=production_observations(observations,target_round_number=round_number)
    models=train_endpoint_models(formulations,observations,registry,optimization)
    phase=resolve_phase_mode(formulations,observations,registry,optimization,target_round_number=round_number)
    reference,state=reference_candidate(config,registry)
    if not reference.empty:
        reference=reference_feasibility(reference,registry,optimization,config)
        if not reference.feasibility_pass.all(): raise ValueError('Reference fails unrelated preparation constraints')
        if any(reference.iloc[0].get(f,0)>0 for f in unavailable): raise ValueError('Reference contains unavailable ingredient')
    frame=build_training_frame(formulations,observations,registry)
    frame=frame.loc[frame.batch_id.astype(str).str.startswith('ROUND_')].copy()
    frame=frame.dropna(subset=['viability_percent','intact_patch_formation_pass'])
    load='critical_axial_load_N_per_needle'
    prior=set(frame.loc[frame[load].notna(),'formulation_id']) if load in frame else set()
    failed=set(frame.loc[frame.preparation_feasibility_pass.eq(0),'formulation_id']) if 'preparation_feasibility_pass' in frame else set()
    hits=[]
    for fid,g in frame.groupby('formulation_id',sort=True):
        if fid in failed or not g.intact_patch_formation_pass.ge(.5).any(): continue
        if fid in prior: continue
        row=g.iloc[0].to_dict()
        if any(row.get(f,0)>0 for f in unavailable): continue
        # Rank means across batches; retain disagreements in provenance.
        row.update(candidate_id='followup_'+str(fid),recommendation_type='screened_hit_mechanics',
                   experimental_role='screened_hit_mechanics',candidate_origin='screened_hit_mechanics',
                   observed_hit_mean=float(g.viability_percent.mean()),observed_hit_sd=float(g.viability_percent.std()) if len(g)>1 else 0.,
                   selection_explanation=f'First mechanics follow-up; prior batches={len(g)}, intact passing={int(g.intact_patch_formation_pass.ge(.5).sum())}; mean viability={g.viability_percent.mean():.3f}')
        # No old endpoint values may survive into a fresh worksheet.
        for k in ['viability_percent','intact_patch_formation_pass',load,'initial_stiffness_N_per_mm_per_needle']:
            row.pop(k,None)
        hits.append(row)
    hits=pd.DataFrame(hits)
    if not hits.empty:
        hits=annotate_feasibility(hits,registry,optimization,policy_active=True)
        hits=hits.loc[hits.feasibility_pass].sort_values(['observed_hit_mean','observed_hit_sd','formulation_id'],ascending=[False,True,True])
    policy=resolve_intact_combination_policy(optimization,round_number)
    legacy_anchor=result.viability_screen.loc[result.viability_screen.recommendation_type.eq('mechanics_anchor')].copy() if reference.empty else pd.DataFrame()
    if not legacy_anchor.empty: legacy_anchor['experimental_role']='mechanics_anchor'
    extra=pd.concat([reference,legacy_anchor,hits],ignore_index=True,sort=False)
    if not extra.empty:
        evidence=build_intact_evidence(formulations,observations,registry,policy,round_number)
        extra=annotate_intact_combination_evidence(extra,evidence,registry,policy)
        extra=annotate_candidates(extra,models,registry,optimization,policy_active=True)
        history_counts=frame.loc[frame[load].notna()].groupby('formulation_id').size() if load in frame else pd.Series(dtype=int)
        extra['prior_mechanical_observation_count']=extra.formulation_id.map(history_counts).fillna(0).astype(int)
        extra['mechanical_repeat_allowed']=extra.experimental_role.isin(['campaign_control','mechanics_anchor'])
        extra['mechanical_repeat_status']='intentional_followup'
        extra['viability_prediction_status']='observed_followup'
        extra.loc[extra.experimental_role.eq('campaign_control'),'viability_prediction_status']='reference_not_modeled'
        extra['raw_surrogate_viability_mean']=extra.predicted_viability_percent
        extra['raw_surrogate_viability_std']=extra.viability_std
        extra.loc[extra.experimental_role.eq('campaign_control'),['raw_surrogate_viability_mean','raw_surrogate_viability_std','predicted_viability_percent']]=np.nan
    ordinary=result.candidate_pool.copy()
    ordinary=ordinary.loc[~ordinary.recommendation_type.fillna('').isin(ROLES|{'mechanics_anchor'})].copy()
    ordinary['experimental_role']=ordinary.recommendation_type.where(ordinary.recommendation_type.eq('retest_priority'),'ordinary')
    selected=[]; used=set(); combos=Counter(); ingredients=Counter(); pairs=Counter(); cold_counts=Counter()
    measured=frame.loc[frame.viability_percent.notna()]
    cold={f for f in registry.feature_names if measured.loc[measured[f]>=presence_threshold(f),'formulation_id'].nunique()<3}
    similarity=build_history_similarity_index(formulations,observations,registry,resolve_similarity_policy(optimization,round_number))
    def add(row):
        fid=str(row.formulation_id)
        if fid in used: return False
        combo=_combo(row,registry); role=row.get('experimental_role','ordinary')
        if any(ingredients[f]>=optimization['selection']['max_candidates_per_ingredient'] for f in combo): return False
        if role=='ordinary' and row.get('candidate_origin')!='rescue_dilution' and any(cold_counts[f]>=2 for f in combo if f in cold): return False
        if row.get('support_status')=='boundary' and sum(x.get('support_status')=='boundary' for x in selected)>=1: return False
        if role!='retest_priority':
            cap=optimization['selection']['max_candidates_per_larger_ingredient_combination'] if len(combo)>2 else optimization['selection']['max_candidates_per_ingredient_combination']
            if len(combo)>1 and combos[combo]>=cap: return False
            if any(pairs[p]>=optimization['selection']['max_candidates_per_shared_ingredient_pair'] for p in combinations(combo,2)): return False
        if role not in ROLES and role!='retest_priority':
            trial=pd.DataFrame(selected+[row.to_dict()])
            try: validate_selected_similarity(trial,formulations,observations,registry,resolve_similarity_policy(optimization,round_number))
            except ValueError: return False
        selected.append(row.to_dict());used.add(fid);ingredients.update(combo)
        if role=='ordinary' and row.get('candidate_origin')!='rescue_dilution': cold_counts.update(f for f in combo if f in cold)
        if role!='retest_priority': combos.update([combo]); pairs.update(combinations(combo,2))
        return True
    if not extra.empty:
        for _,r in extra.loc[extra.experimental_role.eq('campaign_control')].iterrows():
            if not add(r): raise ValueError('Reference conflicts with slate constraints')
        for _,r in extra.loc[extra.experimental_role.isin(['mechanics_anchor'])].iterrows(): add(r)
        hit_target=max(0,(1 if len(reference) else 2)-len(legacy_anchor))
        for _,r in extra.loc[extra.experimental_role.eq('screened_hit_mechanics')].iterrows():
            if sum(x['experimental_role']=='screened_hit_mechanics' for x in selected)>=hit_target: break
            add(r)
    # Reserve up to two evidence-driven viability retests. Their identities stay separate.
    for _,r in ordinary.loc[ordinary.experimental_role.eq('retest_priority')].sort_values('retest_priority_score',ascending=False).head(2).iterrows(): add(r)
    rescue_filled=0
    for _,r in ordinary.loc[ordinary.candidate_origin.eq('rescue_dilution')].iterrows():
        if rescue_filled>=2: break
        if add(r): rescue_filled+=1
    reserved=len(selected)
    score='mechanics_phase_score' if phase.active_phase=='mechanics_enabled' else 'hybrid_phase_score' if phase.active_phase=='mechanics_hybrid' else 'screening_phase_score'
    ranked=ordinary.sort_values([score,'candidate_id'],ascending=[False,True],na_position='last')
    # Residual origin quotas preserve the frozen ordinary proposal's origin proportions.
    origins=result.viability_screen.loc[~result.viability_screen.recommendation_type.isin(ROLES|{'retest_priority','mechanics_anchor'}) & ~result.viability_screen.candidate_origin.eq('rescue_dilution'),'candidate_origin'].value_counts()
    total=12-reserved
    weights=origins/origins.sum(); quotas={k:int(np.floor(v*total)) for k,v in weights.items()}
    for k in sorted(quotas,key=lambda k:(-(weights[k]*total-quotas[k]),k))[:total-sum(quotas.values())]: quotas[k]+=1
    for origin,q in quotas.items():
        accepted=0
        for _,r in ranked.loc[ranked.candidate_origin.eq(origin)&ranked.experimental_role.eq('ordinary')].iterrows():
            if accepted>=q: break
            if add(r): accepted+=1
        if accepted<q: raise ValueError(f'Group10 residual origin/chemistry quota unsatisfied: {origin} {accepted}/{q}; no proposal written')
    if len(selected)!=12: raise ValueError(f'Group10 slate has {len(selected)}/12 rows')
    slate=pd.DataFrame(selected).reset_index(drop=True);slate['selection_rank']=range(1,13)
    mechanical,meta=select_mechanical_tests(slate,models,registry,optimization,phase,4,policy)
    if not mechanical.empty:
        protected=mechanical.loc[mechanical.experimental_role.isin(ROLES|{'mechanics_anchor'})]
        fresh=mechanical.loc[mechanical.experimental_role.eq('ordinary')]
        # Preserve existing mechanically ranked fresh coverage ordering.
        first=pd.concat([protected,fresh.head(max(0,4-len(protected)))])
        rest=mechanical.loc[~mechanical.candidate_id.isin(first.candidate_id)]
        mechanical=pd.concat([first,rest]).reset_index(drop=True)
        mechanical['mechanical_selection_rank']=range(1,len(mechanical)+1)
        mechanical['mechanical_primary_recommended']=mechanical.mechanical_selection_rank<=4
        mechanical['mechanical_backup_status']=np.where(mechanical.mechanical_primary_recommended,'primary','ordered_backup')
        mechanical['mechanical_transition_role']=mechanical.experimental_role.where(mechanical.experimental_role.isin(ROLES|{'mechanics_anchor'}),mechanical.mechanical_transition_role)
    metadata=deepcopy(result.metadata)
    metadata['group10']={'policy_version':config['policy_version'],'effective_config':deepcopy(config),
        'reference_readiness':state,'reserved_screen_rows':reserved,'ordinary_origin_quotas':quotas,
        'model_revision':'unchanged',
        'mechanical_formulation_capacity':4,'replicate_count_source':'completed_round_csv',
        'training_cutoff':f'completed observations before ROUND_{round_number:03d}',
        'reference_exception':'exact recipe DMSO ceiling only',
        'mechanical_definition_id':config['mechanical_endpoint']['definition_id'],
        'endpoint_revision':config['production_endpoint_revision']}
    metadata['group10']['effective_optimization_config']=deepcopy(optimization)
    metadata['group10']['feature_bounds']={f:[registry.get_by_feature(f).lower_bound,registry.get_by_feature(f).upper_bound] for f in registry.feature_names}
    metadata['group10']['training_observations_sha256']=hashlib.sha256(observations.to_csv(index=False).encode()).hexdigest()
    metadata['proposal_schema_version']=config['proposal_schema_version']
    slate['mechanical_definition_id']=config['mechanical_endpoint']['definition_id']
    slate['mechanical_prediction_definition_id']=config['mechanical_endpoint']['definition_id']
    slate['mechanical_prediction_status']='fitted' if models.critical_load.fitted else 'untrained_placeholder'
    mechanical['mechanical_definition_id']=config['mechanical_endpoint']['definition_id']
    metadata['selected_candidate_ids']=slate.candidate_id.tolist()
    metadata['mechanical_test_count']=int(mechanical.mechanical_primary_recommended.sum()) if not mechanical.empty else 0
    metadata['group10']['realized_roles']=slate.experimental_role.value_counts().to_dict()
    metadata['group10']['rescue_rows']=rescue_filled
    metadata['group10']['ingredient_counts']=dict(ingredients)
    metadata['group10']['pair_counts']={' + '.join(k):v for k,v in pairs.items()}
    metadata['viability_prediction_labeling']['selected_status_counts']=slate.viability_prediction_status.value_counts().to_dict()
    metadata['shared_ingredient_pair_diversity']['selected_pair_counts']={' + '.join(k):v for k,v in pairs.items()}
    metadata['shared_ingredient_pair_diversity']['maximum_pair_multiplicity']=max(pairs.values(),default=0)
    metadata['ingredient_frequency_diversity']['counts_after']=dict(ingredients)
    metadata['ingredient_frequency_diversity']['maximum_ingredient_frequency']=max(ingredients.values(),default=0)
    metadata['boundary_slate_cap']['selected_support_boundary_rows']=int(slate.get('support_status',pd.Series('',index=slate.index)).eq('boundary').sum())
    metadata['boundary_slate_cap']['selected_boundary_probe_origin_rows']=int(slate.candidate_origin.eq('boundary_probe').sum())
    metadata['formulation_similarity']['final_validation']=validate_selected_similarity(slate,formulations,observations,registry,resolve_similarity_policy(optimization,round_number))
    metadata['mechanical_policy']=meta
    if not reference.empty:
        metadata['mechanics_transition']['anchor_selection']={'enabled':False,'selected':False,'reason':'Recurring reference supersedes one-time anchor'}
    return replace(result,viability_screen=slate,mechanical_tests=mechanical,
                   candidate_pool=pd.concat([ordinary,extra],ignore_index=True),metadata=metadata)

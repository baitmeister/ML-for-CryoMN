"""Forward control policy and outcome-cutoff regression checks."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/08_multi_objective'))
from helper.group10_config import load_group10_config,load_group10_config_for_round,assert_frozen_group10_protocol,production_observations,assert_group10_can_proceed
from helper.config import load_optimization_config
from helper.selection_policy import _mechanical_eligibility_mask
from helper.batch_effects import control_diagnostics
from helper.terminal_force import MODEL_FIELD,DEFINITION,LEGACY_DEFINITION

spec=importlib.util.spec_from_file_location('dynamic_gp_audit',ROOT/'src/08_multi_objective/04_report_campaign/audit_group10.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class ControlPolicyTests(unittest.TestCase):
    def test_existing_group10_contract_is_exactly_preserved(self):
        frozen=json.loads((ROOT/'results/multi_objective_v2/rounds/ROUND_010/proposal/selection_metadata.json').read_text())
        self.assertEqual(load_group10_config_for_round(10),frozen['group10']['effective_config'])
        assert_frozen_group10_protocol(load_group10_config_for_round(10),10,frozen)
        future=load_group10_config_for_round(11)
        self.assertFalse(future['reference']['mechanical'])
        self.assertEqual(future['policy_version'],'group10_workflow_v5')
        with self.assertRaisesRegex(RuntimeError,'requires policy'):
            assert_group10_can_proceed(load_group10_config_for_round(10),11)

    def test_reference_excluded_even_with_no_previous_mechanics(self):
        frame=pd.DataFrame({'recommendation_type':['campaign_control','ordinary','screened_hit_mechanics','retest_priority'],
                            'prior_mechanical_observation_count':[0,0,0,0],
                            'mechanical_repeat_allowed':[True,False,False,False]})
        mask,meta=_mechanical_eligibility_mask(frame,load_optimization_config())
        self.assertEqual(mask.tolist(),[False,True,True,False])
        self.assertEqual(meta['viability_control_excluded_count'],1)
        # Only the explicitly preserved historical selection path can opt in.
        frame['mechanical_reference_eligible']=[True,False,False,False]
        self.assertTrue(_mechanical_eligibility_mask(frame,load_optimization_config())[0].iloc[0])

    def test_control_monitoring_is_viability_only_and_mechanics_are_retained(self):
        rows=[dict(formulation_id='reference',batch_id='ROUND_010',experimental_role='campaign_control',
                   endpoint=e,value=v,mechanical_definition_id=DEFINITION) for e,v in [('viability_percent',70),(MODEL_FIELD,.2)]]
        obs=pd.DataFrame(rows)
        self.assertEqual(set(production_observations(obs).endpoint),{MODEL_FIELD,'mechanical_pair_viability_percent'})
        self.assertEqual(control_diagnostics(obs).endpoint.tolist(),['viability_percent'])

    def test_group10_four_pairs_reach_hybrid_without_training_control_viability(self):
        from helper.models import build_training_frame, paired_objective_frame, train_endpoint_models
        from helper.phase import resolve_phase_mode
        from helper.registry import load_registry
        from helper.evaluation_metrics import build_feasible_paired_objectives
        from helper.selection_scoring import _mechanics_phase_scores
        registry=load_registry();config=load_optimization_config()
        forms=pd.read_csv(ROOT/'data/processed_v2/formulations.csv')
        raw=pd.read_csv(ROOT/'data/processed_v2/observations.csv')
        numbers=pd.to_numeric(raw.batch_id.astype(str).str.extract(r'^ROUND_(\d+)$')[0],errors='coerce')
        raw=raw.loc[numbers.isna() | numbers.le(9)].copy()
        slate=pd.read_csv(ROOT/'results/multi_objective_v2/rounds/ROUND_010/proposal/proposal.csv')
        forms=pd.concat([forms,slate[['formulation_id',*registry.feature_names]]],ignore_index=True).drop_duplicates('formulation_id')
        rows=[]
        primary=slate.loc[slate.mechanical_test_recommended.eq(True)]
        reference=primary.loc[primary.experimental_role.eq('campaign_control'),'formulation_id'].iloc[0]
        for j,r in enumerate(primary.itertuples()):
            for endpoint,value in [('viability_percent',60+j),('intact_patch_formation_pass',1),(MODEL_FIELD,.12+j*.01)]:
                rows.append(dict(formulation_id=r.formulation_id,batch_id='ROUND_010',endpoint=endpoint,value=value,
                    replicate_id='r1',source_type='wetlab_feedback',observation_noise=.01,
                    experimental_role=r.experimental_role,mechanical_definition_id=DEFINITION))
        # A repeated specimen is still one formulation-batch sample.
        rows.append({**rows[0],'replicate_id':'r2'})
        obs=pd.concat([raw,pd.DataFrame(rows)],ignore_index=True)
        filtered=production_observations(obs,target_round_number=11)
        pd.testing.assert_frame_equal(filtered,production_observations(filtered))
        frame=build_training_frame(forms,filtered,registry)
        ref=frame.loc[frame.formulation_id.eq(reference) & frame.batch_id.eq('ROUND_010')]
        self.assertTrue(ref.viability_percent.isna().all())
        self.assertEqual(ref.mechanical_pair_viability_percent.tolist(),[60])
        pairs=paired_objective_frame(frame).dropna(subset=['viability_percent',MODEL_FIELD])
        self.assertEqual(len(pairs),8)
        phase=resolve_phase_mode(forms,filtered,registry,config,target_round_number=11)
        self.assertEqual((phase.paired_observation_count,phase.distinct_formulation_count,phase.batch_count),(8,8,2))
        self.assertEqual(phase.active_phase,'mechanics_hybrid')
        self.assertFalse(phase.full_gate_met)
        feasible,_=build_feasible_paired_objectives(forms,obs)
        self.assertEqual(len(feasible),8)
        self.assertIn(reference,feasible.formulation_id.tolist())
        models=train_endpoint_models(forms,filtered,registry,config)
        with patch('helper.selection_scoring.try_botorch_qlognehvi_scores',return_value=(np.ones(len(primary)),{})) as acquisition:
            _mechanics_phase_scores(primary,models,registry,config)
        train_y=acquisition.call_args.kwargs['train_y']
        self.assertEqual(len(train_y),8)
        self.assertTrue(np.any(np.isclose(train_y[:,0],60) & np.isclose(train_y[:,1],.12)))
        # Companions from another batch cannot manufacture a mechanical pair.
        bad=obs.copy()
        bad.loc[bad.formulation_id.eq(reference) & bad.endpoint.eq('viability_percent'),'batch_id']='ROUND_011'
        self.assertEqual(resolve_phase_mode(forms,bad,registry,config).paired_observation_count,7)


    def test_acquisition_preserves_measured_reference_outside_proposal_bounds(self):
        if importlib.util.find_spec('botorch') is None:
            self.skipTest('Optional BoTorch not installed')
        from helper.acquisition import try_botorch_optimize_qlognehvi
        # The measured 0.352 M recipe must not be fitted as if it contained
        # the ordinary candidate ceiling of 0.1 M DMSO.
        with patch('botorch.models.SingleTaskGP',side_effect=RuntimeError('captured training inputs')) as constructor:
            _,metadata=try_botorch_optimize_qlognehvi(
                np.array([[.352],[.05]]),np.array([[60,.2],[50,.1]]),
                np.array([0.]),np.array([.1]),[(0,)],(0.,0.),1,lambda x: True)
        self.assertIn('captured training inputs',metadata['botorch_error'])
        self.assertAlmostEqual(float(constructor.call_args.args[0][0,0]),3.52)


    def test_group11_rejects_mechanical_entry_on_viability_reference(self):
        from helper.feedback import ingest_feedback
        from helper.group10_selection import reference_candidate
        from helper.registry import load_registry
        registry=load_registry();config=load_group10_config_for_round(11)
        candidate,_=reference_candidate(config,registry)
        forms=pd.read_csv(ROOT/'data/processed_v2/formulations.csv')
        obs=pd.read_csv(ROOT/'data/processed_v2/observations.csv')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);candidate.to_csv(root/'proposal.csv',index=False)
            pd.DataFrame([dict(formulation_id=candidate.iloc[0].formulation_id,
                viability_percent=60,mechanical_test_attempted=True)]).to_csv(root/'sheet.csv',index=False)
            with self.assertRaisesRegex(ValueError,'Viability-only reference'):
                ingest_feedback(root/'sheet.csv',[root/'proposal.csv'],forms,obs,registry,
                    'ROUND_011',proposal_metadata={'group10':{'effective_config':config}})


class DynamicAuditTests(unittest.TestCase):
    def test_discovers_later_rounds_and_excludes_unmeasured_rows(self):
        obs=pd.DataFrame({'batch_id':['ROUND_003','ROUND_009','ROUND_010','ROUND_011','legacy_wetlab'],
                          'endpoint':['viability_percent']*5,'value':[1,2,3,np.nan,4]})
        self.assertEqual(audit.observed_rounds(obs),[3,9,10])
        self.assertEqual(audit.observed_rounds(obs,9,9),[9])

    def test_dynamic_endpoint_audit_uses_prior_only_and_separates_definitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shutil.copytree(ROOT/'config_v2',root/'config_v2')
            data=root/'data/processed_v2';data.mkdir(parents=True)
            forms=pd.read_csv(ROOT/'data/processed_v2/formulations.csv').iloc[:2].copy()
            forms.to_csv(data/'formulations.csv',index=False)
            rows=[]
            for n in (8,9,10):
                for j,fid in enumerate(forms.formulation_id):
                    for endpoint,value in [('viability_percent',n+j),(MODEL_FIELD,n*.01+j*.02)]:
                        rows.append(dict(formulation_id=fid,batch_id=f'ROUND_{n:03}',endpoint=endpoint,value=value,
                            replicate_id='r1',observation_noise=.01,source_type='wetlab_feedback',experimental_role='ordinary',
                            mechanical_definition_id=LEGACY_DEFINITION if n==8 else DEFINITION))
                proposal=root/f'results/multi_objective_v2/rounds/ROUND_{n:03}/proposal'
                proposal.mkdir(parents=True)
                forms.assign(candidate_id=['a','b'],recommendation_type='ordinary').to_csv(proposal/'proposal.csv',index=False)
            obs=pd.DataFrame(rows);obs.to_csv(data/'observations.csv',index=False)
            calls=[]
            def fake_fit(variant,frame,x,registry,adaptive,endpoint='viability_percent'):
                calls.append((endpoint,set(frame.loc[frame[endpoint].notna(),'batch_id'])))
                return np.ones(len(x)),np.ones(len(x))
            with patch.object(audit,'fit_predict',side_effect=fake_fit):
                audit.run(root,root/'audit',start_round=10)
            self.assertTrue(calls)
            for endpoint,batches in calls:
                self.assertNotIn('ROUND_010',batches)
                if endpoint==MODEL_FIELD:self.assertEqual(batches,{'ROUND_009'})
            table=pd.read_csv(root/'audit/predictions.csv')
            self.assertEqual(set(table.endpoint),{'viability_percent',MODEL_FIELD})
            self.assertTrue(table.loc[table.endpoint.eq(MODEL_FIELD),'mechanical_definition_id'].eq(DEFINITION).all())
            with self.assertRaisesRegex(ValueError,'empty audit'):
                audit.run(root,root/'audit',start_round=10)
            audit.run(root,root/'empty_audit',start_round=11)
            self.assertTrue(pd.read_csv(root/'empty_audit/metrics.csv').empty)


if __name__=='__main__':unittest.main()

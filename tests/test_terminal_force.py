"""Endpoint contract and historical separation; no live artifact writes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/08_multi_objective'))
from helper.group10_config import load_group10_config, production_observations, validate_group10_config
from helper.terminal_force import (
    analyze_terminal, analyze_frame, DEFINITION, MODEL_FIELD,
    STIFFNESS_DEFINITION, STIFFNESS_MODEL_FIELD,
)
from helper.feedback import ingest_feedback
from helper.registry import load_registry
from helper.evaluation_metrics import build_feasible_paired_objectives, summarize_prospective_metrics
from helper.mechanics_execution import update_endpoint_workload


class TerminalForceTests(unittest.TestCase):
    def setUp(self):
        self.c = load_group10_config()
        self.e = self.c['mechanical_endpoint']
        self.t = np.arange(205)*.025
        self.d = np.arange(205)*.005 + 2
        self.f = np.maximum(0., 1 + (self.d-self.d[4])*10)
        self.f[:4] = [.1, .4, .6, .9]

    def result(self, f=None, d=None, t=None, count=100):
        return analyze_terminal(self.f if f is None else f, self.d if d is None else d,
                                self.t if t is None else t, self.e, count)

    def test_endpoint_is_terminal_not_maximum_and_no_reset_after_drop(self):
        f = self.f.copy(); f[40:50] = 100; f[70:80] = .2; f[180:] = 200
        r = self.result(f=f)
        self.assertEqual(r['status'], 'complete')
        self.assertEqual(r['trigger_index'], 4)
        self.assertAlmostEqual(r['endpoint_N_total'], 9.)
        self.assertAlmostEqual(r['endpoint_N_per_needle'], .09)
        self.assertAlmostEqual(r['apparent_secant_stiffness_N_per_mm_total'], 10.)
        self.assertAlmostEqual(r['apparent_secant_stiffness_N_per_mm_per_needle'], .1)
        self.assertEqual(r['stiffness_definition_id'], STIFFNESS_DEFINITION)
        self.assertEqual(r['full_window_maximum_N'], 100.)

    def test_force_drop_preserves_terminal_force_and_signed_secant(self):
        for terminal_force in (0., .5, 1.):
            for count in (100, None):
                with self.subTest(terminal_force=terminal_force, count=count):
                    force = self.f.copy()
                    force[80:] = terminal_force
                    result = self.result(f=force, count=count)
                    self.assertEqual(result['status'], 'complete')
                    self.assertAlmostEqual(result['endpoint_N_total'], terminal_force)
                    self.assertAlmostEqual(
                        result['apparent_secant_stiffness_N_per_mm_total'],
                        (terminal_force - 1.) / .8,
                    )
                    self.assertEqual(result['stiffness_qc'], 'negative_secant' if terminal_force < 1 else 'ok')
                    if count is None:
                        self.assertIsNone(result['endpoint_N_per_needle'])
                        self.assertIsNone(result['apparent_secant_stiffness_N_per_mm_per_needle'])
                    else:
                        self.assertAlmostEqual(result['endpoint_N_per_needle'], terminal_force / count)
                        self.assertAlmostEqual(
                            result['apparent_secant_stiffness_N_per_mm_per_needle'],
                            (terminal_force - 1.) / .8 / count,
                        )
        force = self.f.copy()
        force[80:] = -.5
        self.assertEqual(self.result(f=force)['status'], 'ambiguous_curve')
        self.assertIsNone(self.result(f=force)['endpoint_N_total'])

    def test_trigger_ignores_brief_spike_and_pretrigger_motion(self):
        f = self.f.copy(); f[:4] = [.1, 2, .3, .9]
        d = self.d.copy(); d[:4] = [3, 2.5, 2, 2.01]
        self.assertEqual(self.result(f=f, d=d)['trigger_index'], 4)

    def test_trigger_needs_duration_and_samples(self):
        f = self.f.copy(); f[:] = .9; f[5:9] = 1
        self.assertEqual(self.result(f=f)['status'], 'incomplete_test')
        f[20:25] = 1
        self.assertEqual(self.result(f=f)['trigger_index'], 20)

    def test_incomplete_and_missing_trigger_never_produce_force(self):
        r = self.result(f=self.f[:100], d=self.d[:100], t=self.t[:100])
        self.assertEqual(r['status'], 'incomplete_test')
        self.assertIsNone(r['endpoint_N_total'])
        self.assertIsNone(self.result(f=np.full_like(self.f,.9))['endpoint_N_total'])

    def test_interpolation_not_extrapolation_or_post_terminal_processing(self):
        d = self.d.copy(); d[164:] += .002
        r = self.result(d=d)
        expected = self.f[163] + .005/.007*(self.f[164]-self.f[163])
        self.assertAlmostEqual(r['endpoint_N_total'], expected)
        f = self.f.copy(); f[180] = np.nan
        self.assertEqual(self.result(f=f)['status'], 'complete')

    def test_invalid_gaps_reversals_and_units(self):
        f = self.f.copy(); f[30] = np.nan
        self.assertEqual(self.result(f=f)['status'], 'invalid_curve')
        t = self.t.copy(); t[30:] += .1
        self.assertEqual(self.result(t=t)['status'], 'invalid_curve')
        d = self.d.copy(); d[30] = d[29]-.01
        self.assertEqual(self.result(d=d)['status'], 'ambiguous_curve')
        t = self.t.copy(); t[30] = t[29]
        self.assertEqual(self.result(t=t)['status'], 'invalid_curve')

    def test_bluehill_units_row_and_ambiguous_strain_column(self):
        frame = pd.DataFrame({'Time':['(s)', *self.t], 'Force':['(N)', *self.f],
                              'Displacement':['(mm)', *self.d], 'Compressive strain (Displacement)':['(%)', *self.d]})
        r = analyze_frame(frame,self.e)
        self.assertTrue(r['units_row_removed'])
        self.assertEqual(r['status'],'complete')
        self.assertAlmostEqual(r['endpoint_N_total'],9)
        self.assertIsNone(r['endpoint_N_per_needle'])
        frame.loc[0,'Force']='(kN)'
        self.assertEqual(analyze_frame(frame,self.e)['status'],'invalid_curve')
        with self.assertRaises(ValueError):
            analyze_frame(frame.drop(columns='Time'),self.e)

    def test_loaded_count_and_frozen_settings(self):
        self.assertIsNone(self.result(count=None)['endpoint_N_per_needle'])
        for count in [0,-1,1.5,True]:
            self.assertEqual(self.result(count=count)['status'],'invalid_curve')
        for key,value in [('displacement_limit_mm',1),('trigger_force_N',2),('trigger_duration_s',.2)]:
            c=deepcopy(self.c);c['mechanical_endpoint'][key]=value
            with self.assertRaises(ValueError):validate_group10_config(c)

    def test_readiness_distinguishes_configured_code_from_frozen_live_policy(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('readiness',ROOT/'src/08_multi_objective/04_report_campaign/check_group10_readiness.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        state=module.inspect_readiness(ROOT)
        self.assertTrue(state['endpoint_code_configured'])
        self.assertTrue(state['reference']['ready'])
        self.assertFalse(state['ready_for_group10_wetlab'])
        self.assertEqual(state['active_worksheet_groups'],['ROUND_009'])
        self.assertFalse(state['group10_frozen_with_current_endpoint'])

    def test_training_and_pareto_never_mix_endpoint_definitions(self):
        rows=[]
        for batch,definition,value in [('ROUND_009','',100),('ROUND_010',DEFINITION,2)]:
            for endpoint,y in [('viability_percent',60),(MODEL_FIELD,value),
                               (STIFFNESS_MODEL_FIELD,value*10),('intact_patch_formation_pass',1)]:
                rows.append(dict(formulation_id='f',batch_id=batch,endpoint=endpoint,value=y,mechanical_definition_id=definition))
        obs=pd.DataFrame(rows)
        g9=production_observations(obs,target_round_number=9)
        g10=production_observations(obs,target_round_number=10)
        self.assertEqual(g9.loc[g9.endpoint.eq(MODEL_FIELD),'value'].tolist(),[100])
        self.assertEqual(g10.loc[g10.endpoint.eq(MODEL_FIELD),'value'].tolist(),[2])
        self.assertEqual(g9.loc[g9.endpoint.eq(STIFFNESS_MODEL_FIELD),'value'].tolist(),[1000])
        self.assertEqual(g10.loc[g10.endpoint.eq(STIFFNESS_MODEL_FIELD),'value'].tolist(),[20])
        self.assertEqual(len(g10.loc[g10.endpoint.eq('viability_percent')]),2)
        feasible,_=build_feasible_paired_objectives(pd.DataFrame(),obs)
        self.assertEqual(feasible[MODEL_FIELD].tolist(),[2])
        self.assertEqual(feasible.mechanical_definition_id.tolist(),[DEFINITION])

    def test_prospective_pools_keep_definitions_separate(self):
        rows=[]
        for batch,definition,error in [('ROUND_009','legacy_curve_maximum_v1',1),('ROUND_010',DEFINITION,10)]:
            rows.append(dict(round_id=batch,endpoint=MODEL_FIELD,mechanical_definition_id=definition,
                endpoint_role='objective',metric_type='continuous',provenance_class='formal_frozen',
                formal_cohort=True,evaluation_eligible=True,formal_metric_eligible=True,
                observed_mean=2.,prediction_mean=2.+error,interval_95_covered=False,
                interval_95_lower=0.,interval_95_upper=1.))
        metrics=summarize_prospective_metrics(pd.DataFrame(rows))
        pooled=metrics.loc[metrics.scope.eq('pooled_formal')].set_index('mechanical_definition_id')
        self.assertEqual(len(pooled),2)
        self.assertEqual(pooled.loc[DEFINITION,'mae'],10)
        self.assertEqual(pooled.loc['legacy_curve_maximum_v1','mae'],1)

    def test_ingestion_dispatch_count_missing_and_legacy_preservation(self):
        forms=pd.read_csv(ROOT/'data/processed_v2/formulations.csv')
        old=pd.read_csv(ROOT/'data/processed_v2/observations.csv')
        candidate=forms.iloc[[0]].copy();candidate['experimental_role']='ordinary'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            raw=pd.DataFrame({'Time':self.t,'Force':self.f,'Displacement':self.d})
            raw.to_csv(p/'raw.csv',index=False);candidate.to_csv(p/'proposal.csv',index=False)
            row=dict(formulation_id=candidate.iloc[0].formulation_id,viability_percent=60,
                     intact_patch_formation_pass=1,instron_file=str(p/'raw.csv'),needles_compressed=100)
            pd.DataFrame([row]).to_csv(p/'sheet.csv',index=False)
            args=(p/'sheet.csv',[p/'proposal.csv'],forms,old,load_registry())
            meta={'group10':{'effective_config':self.c}}
            _,updated=ingest_feedback(*args,'ROUND_010',proposal_metadata=meta)
            new=updated.loc[updated.batch_id.eq('ROUND_010')]
            self.assertAlmostEqual(new.loc[new.endpoint.eq(MODEL_FIELD),'value'].iloc[0],.09)
            self.assertAlmostEqual(new.loc[new.endpoint.eq(STIFFNESS_MODEL_FIELD),'value'].iloc[0],.1)
            self.assertEqual(new.loc[new.endpoint.eq(MODEL_FIELD),'mechanical_definition_id'].iloc[0],DEFINITION)
            self.assertEqual(new.loc[new.endpoint.eq(STIFFNESS_MODEL_FIELD),'metric_definition_id'].iloc[0],STIFFNESS_DEFINITION)
            self.assertTrue(new.loc[new.endpoint.eq(MODEL_FIELD),'raw_file_hash'].str.len().eq(64).all())
            workload=update_endpoint_workload({},updated,'ROUND_010')['workload']
            self.assertEqual(workload['tests_with_interpretable_terminal_force'],1)
            _,legacy=ingest_feedback(*args,'ROUND_009')
            historical=legacy.loc[legacy.batch_id.eq('ROUND_009') & legacy.endpoint.eq(MODEL_FIELD)]
            self.assertAlmostEqual(historical.value.iloc[0],(self.f.max()-self.f[0])/100)
            with self.assertRaisesRegex(ValueError,'pre-Group-10'):
                ingest_feedback(*args,'ROUND_009',proposal_metadata=meta)
            pd.DataFrame([{**row,'critical_axial_load_N_per_needle':99}]).to_csv(p/'sheet.csv',index=False)
            with self.assertRaisesRegex(ValueError,'conflicts'):
                ingest_feedback(*args,'ROUND_010',proposal_metadata=meta)
            pd.DataFrame([{**row,'needles_compressed':None}]).to_csv(p/'sheet.csv',index=False)
            _,unknown=ingest_feedback(*args,'ROUND_010',proposal_metadata=meta)
            unknown=unknown.loc[unknown.batch_id.eq('ROUND_010')]
            self.assertFalse(unknown.endpoint.eq(MODEL_FIELD).any())
            self.assertTrue(unknown.endpoint.eq('terminal_force_08mm_N_total').any())
            pd.DataFrame([row,row]).to_csv(p/'sheet.csv',index=False)
            with self.assertRaisesRegex(ValueError,'multiple specimens'):
                ingest_feedback(*args,'ROUND_010',proposal_metadata=meta)


if __name__ == '__main__':
    unittest.main()

from copy import deepcopy
from pathlib import Path
import sys,tempfile,unittest,json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/08_multi_objective'))
from helper.group10_config import load_group10_config,validate_group10_config,reference_readiness,production_observations
from helper.acceptance import assess_acceptance
from helper.mechanical_events import analyze_curve
from helper.observation_noise import estimate_noise
from helper.group10_selection import reference_candidate,reference_feasibility
from helper.config import load_optimization_config
from helper.registry import load_registry
from helper.models import build_training_frame

class Group10Tests(unittest.TestCase):
    def setUp(self):
        self.c=load_group10_config()
        self.e={**self.c['mechanical_endpoint'],'protocol_id':'test','test_mode':'single_needle',
            'contact_force_N':.1,'baseline_points':1,'absolute_drop_N':.01,'drop_window_mm':.25,
            'persistence_points':2,'force_sign':1}
    def analyze(self,f,d=None,**kw):
        if d is None:d=[0,.1,.3,.5,.6,.7,.9,1.1]
        return analyze_curve(f,d,{**self.e,**kw},1)
    def test_config_gates(self):
        self.assertFalse(reference_readiness(self.c)['ready'])
        for k,v in [('activation_round',9),('production_model_revision',True),('production_noise_revision',True),('production_endpoint_revision',True)]:
            c=deepcopy(self.c);c[k]=v
            with self.assertRaises(ValueError):validate_group10_config(c)
    def test_requirements(self):
        req=self.c['application_requirements'];r={'viability_percent':60,'mechanical_value':.1,'intact_patch_formation_pass':1,'mechanical_definition_id':'supported_load_1mm_v1'}
        self.assertEqual(assess_acceptance(r,req)['application_status'],'requirements_pending')
        req={**req,'minimum_viability_percent':60,'minimum_fracture_force_N_per_needle':.1,'fracture_force_definition':'supported_load_1mm_v1'}
        self.assertEqual(assess_acceptance(r,req)['application_status'],'meets_measured_requirements')
        self.assertEqual(assess_acceptance({**r,'mechanical_definition_id':'legacy'},req)['application_status'],'insufficient_evidence')
        self.assertEqual(assess_acceptance({**r,'intact_patch_formation_pass':0},req)['application_status'],'below_requirements')
    def test_first_event_ignores_later_peak(self):
        r=self.analyze([0,.1,.5,1,.8,.8,3,4]);self.assertEqual(r['status'],'event_detected');self.assertEqual(r['endpoint_N_total'],1)
    def test_exact_ten_percent(self):
        r=self.analyze([0,.1,.5,1,.9,.9,1.2,1.3]);self.assertEqual(r['status'],'event_detected')
    def test_spike_and_subthreshold(self):
        for f in ([0,.1,.5,1,.89,1,1.2,1.3],[0,.1,.5,1,.91,.91,1.2,1.3]):
            self.assertEqual(self.analyze(f)['status'],'no_event_detected')
    def test_incomplete_and_contact_reference(self):
        self.assertEqual(self.analyze([0,.1,.2,.3,.4,.5,.6,.7],d=[0,.2,.3,.4,.5,.6,.7,1.1])['status'],'incomplete_test')
        self.assertEqual(self.analyze([0,.1,.2,.3,.4,.5,.6,.7])['status'],'no_event_detected')
    def test_gaps_unloading_and_missing_protocol(self):
        self.assertEqual(self.analyze([0,.1,.2,np.nan,.4,.5,.6,.7])['status'],'invalid_curve')
        self.assertEqual(self.analyze([0,.1,.2,.3,.4,.5,.6,.7],d=[0,.1,.3,.2,.6,.7,.9,1.1])['status'],'ambiguous_curve')
        self.assertEqual(self.analyze([0,.1,.2,.3,.4,.5,.6,.7],protocol_id=None)['status'],'protocol_incomplete')
    def test_array_and_interpolated_end(self):
        e={**self.e,'test_mode':'array'}
        r=analyze_curve([0,.1,1,100],[0,.1,1.,1.2],e,100)
        self.assertEqual(r['status'],'no_event_detected');self.assertLess(r['endpoint_N_total'],100)
        self.assertEqual(r['endpoint_N_per_needle'],r['endpoint_N_total']/100)
    def test_noise_independence(self):
        o=pd.DataFrame([dict(formulation_id='a',batch_id='ROUND_001',endpoint='viability_percent',value=x,preparation_id=p,specimen_id='s') for p,x in [('p1',10),('p1',10),('p2',20),('p2',20)]])
        r=estimate_noise(o).iloc[0];self.assertEqual(r.n_independent,2);self.assertGreater(r.mean_variance,0)
        r=estimate_noise(o.drop(columns='preparation_id')).iloc[0];self.assertEqual(r.n_independent,1);self.assertEqual(r.noise_source,'pooled_conservative_fallback')
    def test_reference_only_exception_and_training_exclusion(self):
        c=deepcopy(self.c);c['reference'].update(density_g_mL=1.1,purity_fraction=1.,base_medium='test medium',independent_preparations=2,specimens_per_preparation=2)
        reg=load_registry();f,state=reference_candidate(c,reg)
        self.assertTrue(state['ready']);self.assertGreater(f.iloc[0].dmso_M,.1)
        self.assertTrue(reference_feasibility(f,reg,load_optimization_config(),c).iloc[0].feasibility_pass)
        obs=pd.DataFrame([dict(formulation_id=f.iloc[0].formulation_id,batch_id='ROUND_010',endpoint='viability_percent',value=90,experimental_role='campaign_control')])
        self.assertTrue(production_observations(obs).empty)
        self.assertEqual(reg.get_by_feature('dmso_M').upper_bound,.1)

    def test_supplementary_ingestion_reproduces_and_preserves_legacy(self):
        from helper.feedback import ingest_feedback
        from helper.mechanical_events import export_analysis
        root=Path(__file__).resolve().parents[1]
        formulations=pd.read_csv(root/'data/processed_v2/formulations.csv')
        observations=pd.read_csv(root/'data/processed_v2/observations.csv')
        candidate=formulations.iloc[[0]].copy()
        candidate['experimental_role']='campaign_control'
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            raw=pd.DataFrame({'Force (N)':[0,.1,.5,1,.8,.8,3,4],
                              'Displacement (mm)':[0,.1,.3,.5,.6,.7,.9,1.1]})
            source=td/'raw.csv';raw.to_csv(source,index=False)
            export_analysis(raw,'Force (N)','Displacement (mm)',self.e,1,source,td/'analysis')
            candidate.to_csv(td/'candidate.csv',index=False)
            sheet=pd.DataFrame([{'formulation_id':candidate.iloc[0].formulation_id,
                'viability_percent':60,'intact_patch_formation_pass':1,
                'needles_compressed':1,'instron_file':str(source),
                'supplementary_analysis_file':str(td/'analysis/mechanical_analysis.json')}])
            sheet.to_csv(td/'sheet.csv',index=False)
            args=(td/'sheet.csv',[td/'candidate.csv'],formulations,observations,load_registry(),'ROUND_010')
            _,updated=ingest_feedback(*args,proposal_metadata={'group10':{'effective_config':{'mechanical_endpoint':self.e}}})
            new=updated[updated.batch_id=='ROUND_010']
            self.assertEqual(new.loc[new.endpoint=='supported_axial_load_1mm_N_per_needle','value'].iloc[0],1)
            self.assertEqual(new.loc[new.endpoint=='critical_axial_load_N_per_needle','value'].iloc[0],4)
            self.assertTrue(production_observations(new).empty)
            analysis_path=td/'analysis/mechanical_analysis.json'
            analysis=json.loads(analysis_path.read_text());analysis['endpoint_N_total']=9
            analysis_path.write_text(json.dumps(analysis))
            with self.assertRaisesRegex(ValueError,'does not reproduce'):
                ingest_feedback(*args,proposal_metadata={'group10':{'effective_config':{'mechanical_endpoint':self.e}}})

class AuditBundleTests(unittest.TestCase):
    def test_unavailable_bundle_has_explicit_heuristic(self):
        from helper.audit_acquisition import propose_with_fallback
        points, metadata = propose_with_fallback(None, np.array([[0.], [.5], [1.]]), n=2)
        self.assertEqual(metadata['mode'], 'heuristic_coverage_not_qlognehvi')
        self.assertEqual(len(points), 2)
        self.assertEqual(len(metadata['errors']), 2)

    def test_unequal_training_sets_and_shared_posterior(self):
        import importlib.util
        if importlib.util.find_spec('botorch') is None: self.skipTest('Optional BoTorch not installed')
        from helper.audit_acquisition import build_bundle, select_from_pool
        b=build_bundle(np.array([[0.],[.2],[.5],[.8],[1.]]),[10,20,40,65,70],np.array([[0.],[.5],[1.]]),[1,.7,.1],np.array([[0.],[.5],[1.]]),[0.],[1.])
        self.assertEqual(len(b.model.models[0].train_inputs[0]),5)
        self.assertEqual(len(b.model.models[1].train_inputs[0]),3)
        pool=np.array([[.25],[.6],[.9]])
        self.assertEqual(tuple(b.posterior(pool).mean.shape),(3,2))
        self.assertTrue(np.isfinite(b.scores(pool)).all())
        chosen=select_from_pool(b,pool,2)
        self.assertEqual(len(set(chosen)),2)

if __name__=='__main__':unittest.main()

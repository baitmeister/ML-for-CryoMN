"""Forward-only workflow settings. No production model switch is implemented."""
from copy import deepcopy
from pathlib import Path
import math
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[3] / 'config_v2' / 'group10.yaml'
ROLES = {'campaign_control', 'screened_hit_mechanics'}
METADATA_FIELDS = ['preparation_id', 'specimen_id', 'readout_id', 'cell_batch_id',
                   'protocol_id', 'mechanical_test_id', 'mechanical_test_attempted',
                   'mechanical_definition_id', 'raw_file_hash']

def load_group10_config(path=CONFIG_PATH):
    with Path(path).open() as f:
        config = yaml.safe_load(f)
    with (Path(path).parent / "endpoints.yaml").open() as f:
        config["application_requirements"] = yaml.safe_load(f).get("application_requirements", {})
    validate_group10_config(config)
    return config

def validate_group10_config(c):
    activation = c.get('activation_round')
    if activation is not None and (type(activation) is not int or activation < 10):
        raise ValueError('Group 10 activation must be null or an integer >= 10')
    for k in ['production_model_revision','production_noise_revision','production_endpoint_revision']:
        if c.get(k) is not False:
            raise ValueError(k + ' must remain false pending the user decision')
    r=c['reference']
    if r['dmso_volume_percent'] != 2.5 or r['sucrose_M'] != .1:
        raise ValueError('Reference exception is restricted to 2.5% v/v DMSO + 100 mM sucrose')
    for k in ['density_g_mL','purity_fraction','molecular_weight_g_mol']:
        v=r.get(k)
        if v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0):
            raise ValueError('Invalid reference '+k)
    if r.get('purity_fraction') is not None and r['purity_fraction']>1:
        raise ValueError('purity_fraction must not exceed 1')
    for k in ['replicate_count']:
        if r.get(k) is not None and (type(r[k]) is not int or r[k]<1):
            raise ValueError(k+' must be a positive integer')
    req=c['application_requirements']
    for k, upper in [('minimum_viability_percent',100),('minimum_fracture_force_N_per_needle',float('inf'))]:
        v=req.get(k)
        if v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=upper):
            raise ValueError('Invalid '+k)
    definition=req.get('fracture_force_definition')
    if definition is not None and (not isinstance(definition,str) or not definition.strip()):
        raise ValueError('fracture_force_definition must be null or nonempty')
    e=c['mechanical_endpoint']
    if e['displacement_limit_mm']!=1 or e['relative_drop']!=.1 or e['supplementary_only'] is not True:
        raise ValueError('Endpoint must remain supplementary, contact-relative 1 mm, 10% drop')
    replicas=c.get('mechanical_replicates_per_formulation')
    if replicas is not None and (type(replicas) is not int or replicas<1):
        raise ValueError('mechanical_replicates_per_formulation must be a positive integer')
    if c.get('production_decision_boundary') != 'beginning_of_full_mechanics':
        raise ValueError('Production decision belongs before the first full-mechanics proposal')

def active(c, round_number):
    return c.get('activation_round') is not None and round_number is not None and round_number >= c['activation_round']

def reference_readiness(c):
    r=c['reference']
    missing=[k for k in ['density_g_mL','purity_fraction','replicate_count'] if r.get(k) in (None,'')]
    return {'ready':not missing,'missing_settings':missing}

def production_observations(obs):
    result = obs
    if 'experimental_role' in result:
        result = result.loc[~result.experimental_role.fillna('').eq('campaign_control')]
    if 'mechanical_definition_id' in result:
        definition = result.mechanical_definition_id.fillna('')
        incompatible = result.endpoint.eq('critical_axial_load_N_per_needle') & ~definition.isin(['', 'legacy_curve_maximum_v1'])
        result = result.loc[~incompatible]
    return result.copy() if result is not obs else obs

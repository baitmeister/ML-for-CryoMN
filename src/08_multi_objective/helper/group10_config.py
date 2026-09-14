"""Forward-only workflow settings. No production model switch is implemented."""
from copy import deepcopy
from pathlib import Path
import math
import yaml
import pandas as pd

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
    for k in ['production_model_revision','production_noise_revision']:
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
    req=c['application_requirements']
    for k, upper in [('minimum_viability_percent',100),('minimum_fracture_force_N_per_needle',float('inf'))]:
        v=req.get(k)
        if v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=upper):
            raise ValueError('Invalid '+k)
    definition=req.get('fracture_force_definition')
    if definition is not None and (not isinstance(definition,str) or not definition.strip()):
        raise ValueError('fracture_force_definition must be null or nonempty')
    e=c['mechanical_endpoint']
    from .terminal_force import DEFINITION, validate_settings
    if e.get('definition_id') == DEFINITION:
        validate_settings(e)
        if c.get('production_endpoint_revision') is not True:
            raise ValueError('The authorized terminal endpoint must be active for Group 10')
    elif e.get('definition_id') == 'supported_load_1mm_v1':
        if e.get('displacement_limit_mm') != 1 or e.get('relative_drop') != .1 or e.get('supplementary_only') is not True or c.get('production_endpoint_revision') is not False:
            raise ValueError('Historical supplementary definition must remain unchanged')
    else:
        raise ValueError('Unknown mechanical endpoint definition')
    if c.get('production_decision_boundary') != 'beginning_of_full_mechanics':
        raise ValueError('Production decision belongs before the first full-mechanics proposal')

def active(c, round_number):
    return c.get('activation_round') is not None and round_number is not None and round_number >= c['activation_round']

def reference_readiness(c):
    r=c['reference']
    missing=[k for k in ['density_g_mL','purity_fraction','molecular_weight_g_mol'] if r.get(k) in (None,'')]
    return {'ready':not missing,'missing_settings':missing}

def production_observations(obs, target_round_number=None, mechanical_definition=None):
    """Select a comparable endpoint cohort without changing stored measurements.

    Explicit proposal round wins. Otherwise retain an upstream selected cohort or
    use the new definition once it is present; historical-only data stay historical.
    """
    from .terminal_force import DEFINITION, LEGACY_DEFINITION, MODEL_FIELD
    result = obs
    if 'experimental_role' in result:
        result = result.loc[~result.experimental_role.fillna('').eq('campaign_control')]
    definition = result.get('mechanical_definition_id', pd.Series('', index=result.index)).fillna('')
    selected = mechanical_definition or obs.attrs.get('mechanical_definition_id')
    if target_round_number is not None:
        config = load_group10_config()
        selected = DEFINITION if active(config, target_round_number) and config['production_endpoint_revision'] else LEGACY_DEFINITION
    if selected is None:
        selected = DEFINITION if definition.eq(DEFINITION).any() else LEGACY_DEFINITION
    if 'endpoint' in result:
        compatible = definition.eq(DEFINITION) if selected == DEFINITION else definition.isin(['', LEGACY_DEFINITION])
        result = result.loc[~result.endpoint.eq(MODEL_FIELD) | compatible]
    result = result.copy()
    result.attrs['mechanical_definition_id'] = selected
    return result

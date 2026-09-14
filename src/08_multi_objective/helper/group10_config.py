"""Forward-only workflow settings. No production model switch is implemented."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
from collections.abc import Mapping
import yaml
import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parents[3] / 'config_v2' / 'group10.yaml'
ROLES = {'campaign_control', 'screened_hit_mechanics'}
METADATA_FIELDS = ['preparation_id', 'specimen_id', 'readout_id', 'cell_batch_id',
                   'protocol_id', 'mechanical_test_id', 'mechanical_test_attempted',
                   'mechanical_definition_id', 'raw_file_hash']
ROUND_ENDPOINT_ADDENDUM = 'mechanical_endpoint_addendum.json'


class Group10HardStop(RuntimeError):
    """Raised before artifacts or observations can advance under an unsafe contract."""


def _missing_settings(config, required, prefix=''):
    if not isinstance(config, Mapping):
        return [prefix.rstrip('.') or 'configuration']
    return [
        prefix + key
        for key in required
        if key not in config or config.get(key) is None
        or (isinstance(config.get(key), str) and not config.get(key).strip())
    ]

def load_group10_config(path=CONFIG_PATH):
    with Path(path).open() as f:
        config = yaml.safe_load(f)
    with (Path(path).parent / "endpoints.yaml").open() as f:
        config["application_requirements"] = yaml.safe_load(f).get("application_requirements", {})
    validate_group10_config(config)
    return config


def load_group10_config_for_round(round_number, path=CONFIG_PATH):
    """Load the contract with a Group 10-specific fail-closed error."""
    try:
        return load_group10_config(path)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        if round_number is not None and round_number >= 10:
            raise Group10HardStop(
                'GROUP 10 HARD STOP: settings/protocol could not be loaded and validated: '
                + str(exc)
            ) from exc
        raise


def load_round_mechanical_endpoint(proposal_dir, batch_id):
    """Load a hash-bound pre-Group-10 endpoint addendum, if one exists."""
    proposal_dir = Path(proposal_dir)
    path = proposal_dir / ROUND_ENDPOINT_ADDENDUM
    if not path.exists():
        return None
    try:
        addendum = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise Group10HardStop(f'ROUND ENDPOINT HARD STOP: invalid {path}: {exc}') from exc
    blockers = []
    if addendum.get('schema_version') != 1:
        blockers.append('schema_version must be 1')
    if addendum.get('batch_id') != batch_id:
        blockers.append(f'batch_id must be {batch_id}')
    if addendum.get('effective_at') != 'validation_and_ingestion':
        blockers.append('effective_at must be validation_and_ingestion')
    bound = addendum.get('bound_files')
    if not isinstance(bound, Mapping) or not bound:
        blockers.append('bound_files must contain proposal artifact hashes')
    else:
        for name, expected in bound.items():
            source = proposal_dir / str(name)
            if Path(str(name)).name != str(name) or not source.is_file():
                blockers.append(f'bound proposal file is missing or unsafe: {name}')
                continue
            actual = hashlib.sha256(source.read_bytes()).hexdigest()
            if actual != expected:
                blockers.append(f'bound proposal file hash mismatch: {name}')
    endpoint = addendum.get('mechanical_endpoint')
    try:
        current = load_group10_config()['mechanical_endpoint']
        from .terminal_force import validate_settings
        validate_settings(endpoint)
        if endpoint != current:
            blockers.append('mechanical_endpoint does not match the reviewed Group 10 endpoint')
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append('mechanical_endpoint is invalid: ' + str(exc))
    if blockers:
        raise Group10HardStop('ROUND ENDPOINT HARD STOP: ' + '; '.join(blockers))
    return deepcopy(endpoint)

def validate_group10_config(c):
    if not isinstance(c, Mapping):
        raise ValueError('Group 10 configuration must be a mapping')
    missing = _missing_settings(
        c,
        [
            'policy_version', 'activation_round', 'proposal_schema_version',
            'production_model_revision', 'production_noise_revision',
            'production_endpoint_revision', 'reference', 'replicate_count_source',
            'production_decision_boundary', 'mechanical_endpoint',
            'application_requirements',
        ],
    )
    missing += _missing_settings(
        c.get('reference'),
        [
            'role', 'dmso_volume_percent', 'sucrose_M', 'density_g_mL',
            'purity_fraction', 'molecular_weight_g_mol', 'mechanical',
        ],
        'reference.',
    )
    missing += _missing_settings(
        c.get('mechanical_endpoint'),
        [
            'definition_id', 'detector_version', 'protocol_id',
            'supplementary_only', 'displacement_limit_mm', 'trigger_force_N',
            'trigger_minimum_samples', 'trigger_duration_s', 'force_unit',
            'displacement_unit', 'time_unit', 'force_sign', 'baseline',
            'smoothing', 'test_mode', 'max_sample_gap_s',
            'displacement_reversal_tolerance_mm', 'nominal_needle_height_mm',
        ],
        'mechanical_endpoint.',
    )
    if missing:
        raise ValueError(
            'Group 10 settings/protocol are incomplete; missing: '
            + ', '.join(sorted(set(missing)))
        )
    activation = c.get('activation_round')
    if activation is not None and (type(activation) is not int or activation < 10):
        raise ValueError('Group 10 activation must be null or an integer >= 10')
    if c.get('policy_version') != 'group10_workflow_v4':
        raise ValueError('Group 10 requires policy_version=group10_workflow_v4')
    if c.get('proposal_schema_version') != 4:
        raise ValueError('Group 10 requires proposal_schema_version=4')
    for k in ['production_model_revision','production_noise_revision']:
        if c.get(k) is not False:
            raise ValueError(k + ' must remain false pending the user decision')
    r=c['reference']
    if r.get('role') != 'campaign_control' or r.get('mechanical') is not True:
        raise ValueError('Group 10 reference must be an active mechanical campaign_control')
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
    height = e.get('nominal_needle_height_mm')
    if isinstance(height, bool) or not isinstance(height, (int, float)) or not math.isfinite(height) or height <= 0:
        raise ValueError('mechanical_endpoint.nominal_needle_height_mm must be a positive number')
    if c.get('replicate_count_source') != 'completed_round_csv':
        raise ValueError('Group 10 replicate counts must come from completed_round_csv')
    if c.get('production_decision_boundary') != 'beginning_of_full_mechanics':
        raise ValueError('Production decision belongs before the first full-mechanics proposal')

def active(c, round_number):
    return c.get('activation_round') is not None and round_number is not None and round_number >= c['activation_round']

def reference_readiness(c):
    r=c['reference']
    missing=[k for k in ['role','dmso_volume_percent','sucrose_M','density_g_mL','purity_fraction','molecular_weight_g_mol','mechanical'] if r.get(k) in (None,'')]
    return {'ready':not missing,'missing_settings':missing}


def assert_group10_can_proceed(c, round_number, observations=None):
    """Fail closed before selecting or running Group 10+.

    The round CSV ingestion path is the validation boundary for prior wet-lab
    evidence. Requiring the immediately preceding observed round prevents an
    explicit batch override from bypassing the Group 9 -> Group 10 transition.
    """
    if round_number is None or round_number < 10:
        return
    blockers = []
    try:
        validate_group10_config(c)
    except (KeyError, TypeError, ValueError) as exc:
        blockers.append(str(exc))
    if c.get('activation_round') != 10:
        blockers.append('activation_round must be 10 so the reviewed rules are live for Group 10')
    try:
        state = reference_readiness(c)
    except (KeyError, TypeError):
        state = {'ready': False, 'missing_settings': ['reference']}
    if not state['ready']:
        blockers.append('reference settings are incomplete: ' + ', '.join(state['missing_settings']))
    if observations is not None:
        observed_rounds = set()
        if not observations.empty and 'batch_id' in observations:
            parsed = pd.to_numeric(
                observations.batch_id.astype(str).str.extract(r'^ROUND_(\d+)$')[0],
                errors='coerce',
            ).dropna()
            observed_rounds = set(parsed.astype(int).tolist())
        required_round = round_number - 1
        if required_round not in observed_rounds:
            blockers.append(
                f'ROUND_{required_round:03d} validated results are not present in observations.csv'
            )
    if blockers:
        raise Group10HardStop('GROUP 10 HARD STOP: ' + '; '.join(dict.fromkeys(blockers)))


def assert_frozen_group10_protocol(c, round_number, proposal_metadata):
    """Require a complete, current frozen contract before Group 10+ ingestion."""
    if round_number is None or round_number < 10:
        return
    assert_group10_can_proceed(c, round_number)
    frozen = (proposal_metadata or {}).get('group10', {}).get('effective_config')
    blockers = []
    if not isinstance(frozen, Mapping):
        blockers.append('frozen proposal metadata has no complete Group 10 effective_config')
    else:
        try:
            validate_group10_config(frozen)
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append('frozen proposal settings/protocol are invalid: ' + str(exc))
        if frozen != c:
            blockers.append('frozen proposal settings/protocol do not match the live reviewed configuration')
    if blockers:
        raise Group10HardStop('GROUP 10 HARD STOP: ' + '; '.join(dict.fromkeys(blockers)))

def production_observations(obs, target_round_number=None, mechanical_definition=None):
    """Select a comparable endpoint cohort without changing stored measurements.

    Explicit proposal round wins. Otherwise retain an upstream selected cohort or
    use the new definition once it is present; historical-only data stay historical.
    """
    from .terminal_force import (
        DEFINITION, LEGACY_DEFINITION, MODEL_FIELD, STIFFNESS_MODEL_FIELD,
    )
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
        mechanical = result.endpoint.isin([MODEL_FIELD, STIFFNESS_MODEL_FIELD])
        result = result.loc[~mechanical | compatible]
    result = result.copy()
    result.attrs['mechanical_definition_id'] = selected
    return result

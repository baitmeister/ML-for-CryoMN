"""Measured application requirements, independent of Pareto rank and acquisition."""
import math

def _number(v):
    try:
        n=float(v)
        return n if math.isfinite(n) else None
    except (ValueError,TypeError):
        return None

def assess_acceptance(row, requirements):
    v=_number(row.get('viability_percent'))
    f=_number(row.get('mechanical_value'))
    vt=requirements.get('minimum_viability_percent')
    ft=requirements.get('minimum_fracture_force_N_per_needle')
    definition=requirements.get('fracture_force_definition')
    compatible=bool(definition and definition==row.get('mechanical_definition_id'))
    intact=_number(row.get('intact_patch_formation_pass'))
    result={'requirements_version':requirements.get('policy_version','application_requirements_v1'),
            'formulation_id':row.get('formulation_id'),'batch_id':row.get('batch_id'),
            'requirements_status':'unspecified' if vt is None and ft is None else 'partial' if vt is None or ft is None else 'complete',
            'viability_margin':None if v is None or vt is None else v-vt,
            'mechanical_margin':None if f is None or ft is None or not compatible else f-ft,
            'mechanical_compatibility_status':'compatible' if compatible else 'unknown_or_incompatible',
            'application_status':'requirements_pending'}
    if result['requirements_status']!='complete': return result
    if v is None or f is None or intact is None or not compatible:
        result['application_status']='insufficient_evidence'
    elif intact<.5 or v<vt or f<ft:
        result['application_status']='below_requirements'
    else:
        result['application_status']='meets_measured_requirements'
    return result

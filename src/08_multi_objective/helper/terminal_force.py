"""Versioned whole-patch terminal force, preserving the acquired trajectory."""
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd

DEFINITION = 'terminal_force_08mm_after_1N_v1'
LEGACY_DEFINITION = 'legacy_curve_maximum_v1'
MODEL_FIELD = 'critical_axial_load_N_per_needle'  # Compatibility key, not a fracture claim.
STIFFNESS_MODEL_FIELD = 'initial_stiffness_N_per_mm_per_needle'  # Compatibility key.
STIFFNESS_DEFINITION = 'apparent_secant_stiffness_08mm_after_1N_v1'
STIFFNESS_FORMULA = '(terminal_force_N-trigger_force_N)/0.8_mm'


def validate_settings(c):
    locked = dict(definition_id=DEFINITION, detector_version='terminal_force_v1',
                  protocol_id='whole_patch_1N_08mm_v1', supplementary_only=False,
                  displacement_limit_mm=0.8, trigger_force_N=1.0,
                  trigger_minimum_samples=5, trigger_duration_s=0.1,
                  force_sign=1, force_unit='N', displacement_unit='mm', time_unit='s',
                  baseline='instrument_export_no_additional_subtraction', smoothing='none',
                  test_mode='array', max_sample_gap_s=0.0375,
                  displacement_reversal_tolerance_mm=0.0001)
    for key, value in locked.items():
        if c.get(key) != value or (isinstance(c.get(key), bool) and not isinstance(value, bool)):
            raise ValueError(f'{DEFINITION} requires {key}={value!r}; changes require a new definition version')


def analyze_terminal(force, displacement, time, config, needles_compressed=None):
    """Return total force even when the worksheet's loaded-needle count is absent."""
    result = dict(definition_id=DEFINITION, detector_version='terminal_force_v1',
                  status='protocol_incomplete', endpoint_N_total=None, endpoint_N_per_needle=None,
                  apparent_secant_stiffness_N_per_mm_total=None,
                  apparent_secant_stiffness_N_per_mm_per_needle=None,
                  stiffness_definition_id=STIFFNESS_DEFINITION,
                  stiffness_formula=STIFFNESS_FORMULA,
                  stiffness_qc='not_calculated',
                  event_displacement_mm=None, settings=dict(config), loaded_needle_count=needles_compressed,
                  interpretation='Whole-patch terminal compression resistance; nominal N per loaded needle when count is supplied')
    try:
        validate_settings(config)
    except ValueError as exc:
        return {**result, 'reason': str(exc)}
    if needles_compressed is not None and (type(needles_compressed) is not int or needles_compressed <= 0):
        return {**result, 'status': 'invalid_curve', 'reason': 'Loaded needle count must be a positive integer'}
    if time is None:
        return {**result, 'reason': 'Time in seconds is required to verify the sustained trigger'}
    try:
        f, d, t = (np.asarray(v, dtype=float) for v in (force, displacement, time))
    except (ValueError, TypeError):
        return {**result, 'status': 'invalid_curve', 'reason': 'Nonnumeric acquisition data'}
    if f.ndim != 1 or d.shape != f.shape or t.shape != f.shape or len(f) < 5:
        return {**result, 'status': 'invalid_curve', 'reason': 'Insufficient or mismatched acquisition arrays'}
    start = None
    trigger = None
    end = None
    tolerance = 1e-9
    # Stop validation at the terminal bracket: later unloading is not part of this endpoint.
    for i in range(len(f)):
        if not np.isfinite([f[i], d[i], t[i]]).all():
            return {**result, 'status': 'invalid_curve', 'reason': 'Nonfinite sample; gaps are not stitched'}
        if i and (t[i] <= t[i-1] or t[i]-t[i-1] > config['max_sample_gap_s'] + tolerance):
            return {**result, 'status': 'invalid_curve', 'reason': 'Nonmonotone time or acquisition gap'}
        if trigger is None:
            if f[i] >= config['trigger_force_N']:
                if start is None:
                    start = i
                if i-start+1 >= config['trigger_minimum_samples'] and t[i]-t[start] >= config['trigger_duration_s']-tolerance:
                    trigger = start
                    result.update(trigger_index=int(trigger), trigger_confirmed_index=i,
                                  trigger_time_s=float(t[trigger]), contact_displacement_mm=float(d[trigger]),
                                  trigger_force_N=float(f[trigger]))
            else:
                start = None
        if trigger is not None:
            window = d[trigger:i+1]
            if np.any(np.diff(window) < -config['displacement_reversal_tolerance_mm']-tolerance):
                return {**result, 'status': 'ambiguous_curve', 'reason': 'Displacement reversal after trigger'}
            if d[i]-d[trigger] >= config['displacement_limit_mm']-tolerance:
                end = i
                break
    if trigger is None:
        return {**result, 'status': 'incomplete_test', 'reason': 'No sustained 1 N trigger'}
    if end is None:
        return {**result, 'status': 'incomplete_test', 'reason': 'Trace does not reach +0.8 mm after trigger',
                'analyzed_displacement_mm': float(d[-1]-d[trigger])}
    x = d[trigger] + config['displacement_limit_mm']
    if end <= trigger or d[end] <= d[end-1]:
        return {**result, 'status': 'ambiguous_curve', 'reason': 'No increasing terminal interpolation bracket'}
    weight = float(np.clip((x-d[end-1])/(d[end]-d[end-1]), 0., 1.))
    value = float(f[end-1] + weight*(f[end]-f[end-1]))
    terminal_time = float(t[end-1] + weight*(t[end]-t[end-1]))
    if value < 0:
        return {**result, 'status': 'ambiguous_curve', 'reason': 'Negative terminal compressive force'}
    stiffness = (value-float(f[trigger]))/config['displacement_limit_mm']
    # A force drop can produce a negative secant without invalidating the
    # measured terminal force. Preserve the signed result and flag it for QC.
    return {**result, 'status': 'complete', 'reason': 'Interpolated force at +0.8 mm after sustained 1 N',
            'endpoint_N_total': value,
            'endpoint_N_per_needle': value/needles_compressed if needles_compressed else None,
            'apparent_secant_stiffness_N_per_mm_total': stiffness,
            'apparent_secant_stiffness_N_per_mm_per_needle': stiffness/needles_compressed if needles_compressed else None,
            'stiffness_qc': 'negative_secant' if stiffness < 0 else 'ok',
            'terminal_time_s': terminal_time, 'loading_duration_s': terminal_time-t[trigger],
            'terminal_displacement_mm': float(x), 'terminal_bracket_indices': [end-1, end],
            'analyzed_displacement_mm': 0.8,
            'full_window_maximum_N': float(max(np.max(f[trigger:end]), value))}


def curve_columns(frame, force_column=None, displacement_column=None, time_column=None):
    """Avoid confusing Bluehill's strain column with measured displacement."""
    def choose(explicit, words):
        if explicit is not None:
            if explicit not in frame:
                raise ValueError(f'Column not found: {explicit}')
            return explicit
        matches = [str(c) for c in frame if any(w in str(c).lower() for w in words)
                   and 'strain' not in str(c).lower()]
        if len(matches) != 1:
            raise ValueError(f'Expected one {words[0]} column, found {matches}; supply the exact column')
        return matches[0]
    return (choose(force_column, ['force', 'load']),
            choose(displacement_column, ['displacement', 'extension', 'position']),
            choose(time_column, ['time']))


def numeric_curve(frame, columns, config):
    """Remove only a recognized units row. Never drop malformed measurement rows."""
    selected = frame.loc[:, list(columns)].copy()
    expected = [config['force_unit'], config['displacement_unit'], config.get('time_unit', 's')][:len(columns)]
    first = selected.iloc[0] if len(selected) else []
    def unit(value):
        return str(value).strip().strip('()[]').strip()
    units_row = len(first) and all(unit(v) == u for v, u in zip(first, expected))
    if units_row:
        selected = selected.iloc[1:]
    for column, target in zip(columns, expected):
        tokens = re.findall(r'[\[(]([^\])]+)[\])]', str(column))
        if tokens and tokens[-1].strip() != target:
            raise ValueError(f'{column}: expected unit {target}')
    return [pd.to_numeric(selected[c], errors='coerce').to_numpy(float) for c in columns], bool(units_row)


def analyze_frame(frame, config, needles_compressed=None, force_column=None, displacement_column=None, time_column=None):
    columns = curve_columns(frame, force_column, displacement_column, time_column)
    arrays, units_row = numeric_curve(frame, columns, config)
    result = analyze_terminal(*arrays, config, needles_compressed)
    result.update(force_column=columns[0], displacement_column=columns[1], time_column=columns[2],
                  units_row_removed=units_row, index_basis='zero-based numeric acquisition rows after units row')
    return result


def analyze_file(source, config, needles_compressed=None):
    from .instron import _read_bluehill_csv
    source = Path(source)
    result = analyze_frame(_read_bluehill_csv(source), config, needles_compressed)
    result.update(source_file=str(source.resolve()), source_file_hash=hashlib.sha256(source.read_bytes()).hexdigest())
    return result

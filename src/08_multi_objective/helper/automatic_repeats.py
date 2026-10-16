"""Cumulative intact-failure retirement and completed automatic-retest evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

import pandas as pd

from .config import nested_get
from .endpoints import aggregate_intact_patch_replicates
from .paths import RESULTS_V2_DIR


POLICY_VERSION = 'cumulative_failure_bounded_retest_v1'


def repeat_policy_enabled(config: Mapping) -> bool:
    return bool(nested_get(config, 'automatic_repeat_policy.enabled', False))


def _campaign_observations(observations, target_round=None):
    if observations.empty or not {'formulation_id', 'batch_id', 'endpoint', 'value'}.issubset(observations):
        return pd.DataFrame(columns=['formulation_id', 'batch_id', 'endpoint', 'value'])
    rows = observations.copy()
    rows['value'] = rows.value.map(lambda value: pd.NA if isinstance(value, str) and not value.strip() else value)
    rounds = pd.to_numeric(rows.batch_id.astype(str).str.extract(r'^ROUND_(\d+)$')[0], errors='coerce')
    keep = rounds.notna()
    if target_round is not None:
        keep &= rounds.lt(int(target_round))
    if 'source_type' in rows:
        keep &= rows.source_type.eq('wetlab_feedback')
    return rows.loc[keep].copy()


def load_completed_retest_history(observations, rounds_dir, target_round=None):
    """Read frozen roles; only batches with finite measured viability consume a retest."""
    rows = _campaign_observations(observations, target_round)
    viability = rows.loc[rows.endpoint.eq('viability_percent')].copy()
    values = pd.to_numeric(viability.value, errors='coerce')
    viability = viability.loc[values.notna() & values.abs().lt(float('inf'))]
    measured = set(zip(viability.formulation_id.astype(str), viability.batch_id.astype(str)))
    batches = {batch for _, batch in measured}
    completed = {}
    inputs = {}
    for batch in sorted(batches):
        path = Path(rounds_dir) / batch / 'proposal/proposal.csv'
        if not path.exists():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = path.with_name('selection_commit.json')
        if receipt.exists():
            expected = json.loads(receipt.read_text()).get('hashes', {}).get('proposal/proposal.csv')
            if expected and expected != digest:
                raise ValueError(f'Frozen retest history changed: {path}')
        inputs[str(path.resolve())] = digest
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                role = row.get('experimental_role') or row.get('recommendation_type', '')
                fid = str(row.get('formulation_id', ''))
                if role == 'retest_priority' and (fid, batch) in measured:
                    completed.setdefault(fid, set()).add(batch)
    return {fid: sorted(batches) for fid, batches in completed.items()}, inputs


def repeat_evidence(formulations, observations, registry, config):
    # Local import avoids a cycle with rescue generation.
    from .candidates import formulation_fingerprint

    rows = _campaign_observations(observations, config.get('_gp_target_round'))
    recipe_by_id = {str(row.formulation_id): formulation_fingerprint(row, registry)
                    for _, row in formulations.iterrows()}
    intact = rows.loc[rows.endpoint.eq('intact_patch_formation_pass')].copy()
    intact['recipe_key'] = intact.formulation_id.astype(str).map(recipe_by_id)
    intact = intact.dropna(subset=['recipe_key'])
    failures = {}
    for (recipe, batch), group in intact.groupby(['recipe_key', 'batch_id']):
        if aggregate_intact_patch_replicates(group.value) == 0.0:
            failures.setdefault(recipe, set()).add(str(batch))
    completed = config.get('_completed_automatic_retest_history')
    if completed is None:
        completed, _ = load_completed_retest_history(rows, RESULTS_V2_DIR / 'rounds', config.get('_gp_target_round'))
    measured = rows.loc[rows.endpoint.eq('viability_percent')].copy()
    values = pd.to_numeric(measured.value, errors='coerce')
    measured = measured.loc[values.notna() & values.abs().lt(float('inf'))]
    measured_keys = set(zip(measured.formulation_id.astype(str), measured.batch_id.astype(str)))
    confirmations = {}
    for fid, batches in completed.items():
        recipe = recipe_by_id.get(str(fid))
        if recipe is not None:
            confirmations.setdefault(recipe, set()).update(
                str(batch) for batch in batches if (str(fid), str(batch)) in measured_keys)
    return recipe_by_id, failures, confirmations


def annotate_automatic_repeats(candidates, formulations, observations, registry, config):
    if not repeat_policy_enabled(config) or candidates.empty:
        return candidates.copy()
    from .candidates import formulation_fingerprint

    _, failures, confirmations = repeat_evidence(formulations, observations, registry, config)
    result = candidates.copy()
    keys = result.apply(lambda row: formulation_fingerprint(row, registry), axis=1)
    result['cumulative_intact_failure_count'] = keys.map(lambda key: len(failures.get(key, ())))
    result['completed_automatic_retest_count'] = keys.map(lambda key: len(confirmations.get(key, ())))
    maximum = int(nested_get(config, 'automatic_repeat_policy.max_cumulative_intact_failures', 2))
    result['automatic_repeat_retired'] = result.cumulative_intact_failure_count.ge(maximum)
    return result


def filter_retired_recipes(candidates, formulations, observations, registry, config):
    """Filter repeat eligibility before supplied or generated rows can be reserved."""
    annotated = annotate_automatic_repeats(candidates, formulations, observations, registry, config)
    if not repeat_policy_enabled(config) or annotated.empty:
        return annotated
    is_retest = pd.Series(False, index=annotated.index)
    for column, label in (('recommendation_type', 'retest_priority'), ('experimental_role', 'retest_priority'), ('candidate_origin', 'retest')):
        if column in annotated:
            is_retest |= annotated[column].eq(label)
    cap = int(nested_get(config, 'retest.max_completed_automatic_retests_per_formulation', 1))
    eligible = ~annotated.automatic_repeat_retired & (~is_retest | annotated.completed_automatic_retest_count.lt(cap))
    return annotated.loc[eligible].reset_index(drop=True)


def automatic_repeat_metadata(formulations, observations, registry, config):
    if not repeat_policy_enabled(config):
        return {'enabled': False}
    recipe_by_id, failures, confirmations = repeat_evidence(formulations, observations, registry, config)
    maximum = int(nested_get(config, 'automatic_repeat_policy.max_cumulative_intact_failures', 2))
    cap = int(nested_get(config, 'retest.max_completed_automatic_retests_per_formulation', 1))
    records = []
    for fid, recipe in sorted(recipe_by_id.items()):
        failed = sorted(failures.get(recipe, ()))
        retested = sorted(confirmations.get(recipe, ()))
        if failed or retested:
            records.append(dict(formulation_id=fid, failed_batches=failed,
                cumulative_intact_failure_count=len(failed), completed_retest_batches=retested,
                completed_automatic_retest_count=len(retested), retired=len(failed) >= maximum,
                automatic_retest_exhausted=len(retested) >= cap))
    return dict(enabled=True, policy_version=nested_get(config, 'automatic_repeat_policy.policy_version', POLICY_VERSION),
        failure_count_unit='distinct exact-recipe batch; conservative all-pass aggregation',
        max_cumulative_intact_failures=maximum, passes_reset_failure_count=False,
        max_completed_automatic_retests_per_formulation=cap,
        retest_completion_rule='frozen retest_priority role and finite measured viability in that batch',
        shared_retest_rescue_capacity=1, empty_special_slots='backfill by constrained acquisition ranking',
        history=records)

#!/usr/bin/env python3
"""Isolated, matched ROUND_013 finite versus continuous-refinement comparison."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import ModuleType

import numpy as np
import pandas as pd
import torch
import botorch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/08_multi_objective'))
from helper.candidates import (generate_rescue_candidate_pool, generate_support_aware_candidate_pool,
                               unavailable_features_from_config)
from helper.config import load_optimization_config, load_availability_config
from helper.feasibility import annotate_feasibility, annotate_support, build_support_context, policy_activation, feasibility_report
from helper.gp_strategy import validate_decision
from helper.group10_config import production_observations, load_group10_config_for_round
from helper.group10_selection import apply_group10
from helper.models import train_endpoint_models
from helper.registry import load_registry, presence_threshold
from helper.selection import select_next_round, write_selection_result
from helper.selection_scoring import _mechanics_phase_scores, annotate_candidates
from helper.intact_policy import build_intact_evidence, resolve_intact_combination_policy, annotate_intact_combination_evidence
from helper.similarity import SimilarityAudit, build_history_similarity_index, resolve_similarity_policy
from helper.visualization import generate_proposal_artifacts

BASE = ROOT / 'results/multi_objective_v2/comparisons/pre_full_phase_check'
PREVIOUS_ALLOCATION_COMMIT = '6b759c7d5ff7e266e4743aed008f01f3acf19059'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_campaign_manifest():
    """Protect current inputs and official outputs without requiring old artifacts."""
    comparisons = ROOT / 'results/multi_objective_v2/comparisons'
    roots = ('config_v2', 'data/processed_v2', 'data/raw/instron/ROUND_012',
             'data/raw/viability/ROUND_012', 'results/multi_objective_v2')
    files = {}
    for name in roots:
        for path in sorted((ROOT / name).rglob('*')):
            if path.is_file() and not path.is_relative_to(comparisons):
                files[str(path.relative_to(ROOT))] = digest(path)
    return dict(created_at=datetime.now(timezone.utc).isoformat(),
                scope='Current campaign inputs and official outputs at comparison start',
                protected_campaign_files=files)


def verify_campaign(manifest):
    changed = [name for name, expected in manifest['protected_campaign_files'].items()
               if not (ROOT / name).is_file() or digest(ROOT / name) != expected]
    if changed:
        raise RuntimeError('Protected campaign files changed: ' + ', '.join(changed))
    return len(manifest['protected_campaign_files'])


def starting_pool(forms, obs, registry, config, unavailable):
    active, version, _ = policy_activation(config, 13)
    support = build_support_context(forms, registry, config, obs)
    similarity = resolve_similarity_policy(config, 13)
    index = build_history_similarity_index(forms, obs, registry, similarity)
    audit = SimilarityAudit(similarity, history_reference_count=len(index))
    kwargs = dict(formulations=forms, observations=obs, optimization_config=config, support=support,
                  unavailable_feature_names=unavailable, similarity_index=index, similarity_audit=audit,
                  policy_version=version)
    rescue = generate_rescue_candidate_pool(registry, **kwargs)
    kwargs.pop('observations')
    pool = generate_support_aware_candidate_pool(registry, n_candidates=config['selection']['generated_candidate_pool_size'],
                                                random_seed=config['random_seed'], **kwargs)
    pool = pd.concat([rescue, pool], ignore_index=True).drop_duplicates('formulation_id', keep='first')
    pool = annotate_support(annotate_feasibility(pool, registry, config, active, version), registry, support)
    return pool.loc[pool.feasibility_pass].reset_index(drop=True), audit, version


def verify_refinement_coverage(metadata):
    """A successful solve alone is insufficient: inspect what was searched."""
    records = [row for row in metadata['mask_ranking_records'] if 'best_seed_score' in row]
    searched = [row for row in metadata['mask_records'] if row['attempts'] > 0]
    expected = set(map(int, metadata['feasible_mask_size_counts']))
    actual = {len(row['features']) for row in searched}
    if metadata['masks_requested'] >= len(expected) and actual != expected:
        raise RuntimeError(f'Ingredient-count coverage failed: eligible {expected}, searched {actual}')
    if len(searched) != metadata['masks_requested']:
        raise RuntimeError('A selected ingredient set never reached the optimizer')
    if not min(records, key=lambda row: row['acquisition_rank'])['selected_for_refinement']:
        raise RuntimeError('The highest-scoring ingredient set was not refined')
    if metadata['masks_requested'] >= len(expected):
        if not all(row['selected_for_refinement'] for row in records if row['best_for_ingredient_count']):
            raise RuntimeError('A best-scoring ingredient-count representative was not refined')
    return True


def previous_allocator(output):
    """Replay the immutable prior allocator without a legacy production flag."""
    source = subprocess.check_output(['git', 'show', PREVIOUS_ALLOCATION_COMMIT + ':src/08_multi_objective/helper/group10_selection.py'], cwd=ROOT)
    path = output / 'previous_group10_selection.py'
    path.write_bytes(source)
    module = ModuleType('helper.comparison_previous_group10')
    module.__package__ = 'helper'
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module.apply_group10


def verify_full_allocation(result, registry, config, version, unavailable):
    slate = result.viability_screen
    roles = slate.experimental_role
    special = roles.eq('retest_priority') | slate.candidate_origin.eq('rescue_dilution')
    assert len(slate) == 12 and roles.eq('campaign_control').sum() == 1
    assert roles.eq('screened_hit_mechanics').sum() == 1 and special.sum() == 1
    ordinary = slate.loc[roles.eq('ordinary') & ~slate.candidate_origin.eq('rescue_dilution')]
    assert len(ordinary) == 9
    assert np.all(np.diff(ordinary.mechanics_phase_score.to_numpy(float)) <= 1e-9)
    assert result.metadata['group10']['ordinary_origin_quotas'] == {}
    assert result.metadata['group10']['reserved_screen_rows'] == 3
    assert {row['formulation_id'] for row in result.metadata['retest_policy']['selected_candidates']} == set(slate.loc[roles.eq('retest_priority'), 'formulation_id'])
    ingredients, sets, pairs = Counter(), Counter(), Counter()
    for _, row in slate.iterrows():
        features = tuple(f for f in registry.feature_names if abs(row[f]) >= presence_threshold(f))
        assert all(abs(row[f]) <= 1e-12 for f in unavailable)
        if row.experimental_role != 'campaign_control':
            assert feasibility_report(row, registry, config, True, version)['feasibility_pass']
        ingredients.update(features)
        if row.experimental_role != 'retest_priority':
            sets.update([features])
            pairs.update(combinations(features, 2))
    limits = config['selection']
    assert max(ingredients.values(), default=0) <= limits['max_candidates_per_ingredient']
    assert max(pairs.values(), default=0) <= limits['max_candidates_per_shared_ingredient_pair']
    for features, count in sets.items():
        if len(features) > 1:
            cap = limits['max_candidates_per_larger_ingredient_combination'] if len(features) > 2 else limits['max_candidates_per_ingredient_combination']
            assert count <= cap
    assert slate.support_status.eq('boundary').sum() <= config['support_policy']['max_boundary_candidates_per_slate']


def verify_earlier_allocations(initial, forms, obs, registry, config, group_config, models, unavailable, old_allocator):
    """Compare phase routing in the changed final allocator on a shared pool."""
    verified = []
    for phase in ('screening_only', 'mechanics_bootstrap', 'mechanics_hybrid'):
        forced = deepcopy(config)
        forced['phase_mode'] = phase
        before = old_allocator(deepcopy(initial), forms, obs, registry, forced, group_config, 13, unavailable, endpoint_models=models)
        after = apply_group10(deepcopy(initial), forms, obs, registry, forced, group_config, 13, unavailable, endpoint_models=models)
        for field in ('viability_screen', 'mechanical_tests', 'candidate_pool'):
            pd.testing.assert_frame_equal(getattr(before, field), getattr(after, field), check_exact=False, atol=1e-9, rtol=1e-9)
        assert before.metadata['group10']['ordinary_origin_quotas'] == after.metadata['group10']['ordinary_origin_quotas']
        assert after.metadata['group10']['full_phase_allocation'] is None
        verified.append(phase)
    return verified


def main(output):
    output = output.resolve()
    if not output.is_relative_to(BASE.resolve()) or output in (BASE.resolve(), (BASE / 'baseline').resolve()):
        raise ValueError('Output must be a new subdirectory inside the comparison workspace')
    manifest = capture_campaign_manifest()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'campaign_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verified = verify_campaign(manifest)
    torch.set_num_threads(1)
    forms = pd.read_csv(ROOT / 'data/processed_v2/formulations.csv')
    raw = pd.read_csv(ROOT / 'data/processed_v2/observations.csv')
    obs = production_observations(raw, target_round_number=13)
    registry = load_registry()
    config = load_optimization_config()
    unavailable = unavailable_features_from_config(load_availability_config(), registry)
    legacy_path = ROOT / 'results/multi_objective_v2/gp_strategy_decision.json'
    config.update(gp_strategy_decision=validate_decision(legacy_path, 13), _gp_raw_observations=raw, _gp_target_round=13)
    models = train_endpoint_models(forms, obs, registry, config)
    config.pop('_gp_raw_observations')
    models.viability.model.save(output / 'shared_viability_model.json')
    models.critical_load.model.save(output / 'shared_mechanical_model.json')
    print('Approved models fitted once and saved.', flush=True)
    start = time.perf_counter()
    pool, audit, version = starting_pool(forms, obs, registry, config, unavailable)
    generation_seconds = time.perf_counter() - start
    pool.to_csv(output / 'starting_pool.csv', index=False)
    print(f'Shared starting pool: {len(pool)} recipes ({generation_seconds:.1f}s).', flush=True)
    frozen = pd.read_csv(ROOT / 'results/multi_objective_v2/next_round/next_round_candidates.csv')
    group_config = load_group10_config_for_round(13)
    old_allocator = previous_allocator(output)
    results, receipts = {}, {}
    upstream = {}
    cases = [('old_finite', 'finite_pool', old_allocator), ('old_refined', 'continuous_required', old_allocator),
             ('finite', 'finite_pool', apply_group10), ('refined', 'continuous_required', apply_group10),
             ('refined_repeat', 'continuous_required', apply_group10)]
    for label, mode, allocator in cases:
        folder = output / label
        folder.mkdir()
        decision = json.loads(legacy_path.read_text())
        decision.update(version=2, decision_id=f'round13-{mode}-comparison-v2', effective_round=13,
                        acquisition='shared_qlognehvi', search_mode=mode,
                        chosen_by='user: full phase 1 control, 1 mechanics follow-up, 1 retest/rescue; comparison only',
                        evidence=['src/08_multi_objective/02_select_candidates/README.md',
                                  'current campaign inputs recorded in campaign_manifest.json'])
        decision_path = folder / 'gp_strategy_decision.json'
        decision_path.write_text(json.dumps(decision, indent=2) + '\n')
        run_config = deepcopy(config)
        run_config['gp_strategy_decision'] = validate_decision(decision_path, 13)
        models.gp_strategy_decision = run_config['gp_strategy_decision']
        started = time.perf_counter()
        print(f'Running {label}...', flush=True)
        try:
            # Old/new allocation consumes an identical already-scored pool.
            # The repeat independently regenerates refinements for reproducibility.
            reuse = mode in upstream and label != 'refined_repeat'
            if not reuse:
                initial = select_next_round(forms, obs, pool.copy(), registry, run_config, target_round_number=13,
                    policy_active=True, policy_version=version, similarity_audit=deepcopy(audit),
                    unavailable_feature_names=unavailable, endpoint_models=models)
                if label == 'refined_repeat':
                    pd.testing.assert_frame_equal(initial.candidate_pool, upstream[mode].candidate_pool,
                                                  check_exact=False, atol=1e-9, rtol=1e-9)
                else:
                    upstream[mode] = initial
            else:
                initial = upstream[mode]
            result = allocator(deepcopy(initial), forms, obs, registry, run_config, group_config, 13,
                               unavailable, endpoint_models=models)
            if allocator is apply_group10:
                verify_full_allocation(result, registry, config, version, unavailable)
            if mode == 'continuous_required':
                verify_refinement_coverage(result.metadata['continuous_qlognehvi'])
        except Exception as exc:
            (folder / 'failure.json').write_text(json.dumps(dict(error=str(exc), metadata=getattr(exc, 'metadata', {})), indent=2))
            verify_campaign(manifest)
            raise
        elapsed = time.perf_counter() - started
        result.metadata.update(formulation_feasibility_policy_active=True,
                               formulation_feasibility_policy_version=version,
                               gp_strategy_decision=run_config['gp_strategy_decision'], comparison_only=True)
        write_selection_result(result, folder, batch_id='ROUND_013', registry=registry,
                               total_candidate_pool_path=folder / 'scored_pool.csv')
        if mode == 'continuous_required':
            pd.DataFrame(result.metadata['continuous_qlognehvi']['mask_ranking_records']).to_csv(
                folder / 'ingredient_set_ranking.csv', index=False)
        sheet = pd.read_csv(folder / 'next_round_candidates.csv')
        generate_proposal_artifacts(sheet, folder)
        if label == 'old_finite':
            for column in ('formulation_id', 'mechanical_test_recommended', 'mechanical_selection_rank'):
                pd.testing.assert_series_equal(sheet[column], frozen[column], check_names=False)
        results[label] = result
        receipts[label] = dict(seconds=elapsed, selected=sheet.formulation_id.tolist(),
                               reused_upstream_candidates=reuse,
                               allocation=result.metadata['group10'],
                               mechanical_primary_positions=sheet.loc[sheet.mechanical_test_recommended, 'selection_rank'].astype(int).tolist(),
                               mechanical_primaries=sheet.loc[sheet.mechanical_test_recommended, 'formulation_id'].tolist(),
                               continuous=result.metadata['continuous_qlognehvi'])
        print(f'{label} completed in {elapsed:.1f}s; {receipts[label]["continuous"].get("selected_candidate_count", 0)} continuous recipes selected.', flush=True)
        verify_campaign(manifest)
    for name in ('viability_screen', 'mechanical_tests', 'candidate_pool'):
        pd.testing.assert_frame_equal(getattr(results['refined'], name), getattr(results['refined_repeat'], name),
                                      check_exact=False, atol=1e-9, rtol=1e-9)
    earlier_phases = verify_earlier_allocations(upstream['finite_pool'], forms, obs, registry, config,
                                               group_config, models, unavailable, old_allocator)

    # Common final scoring: no pending points; same model, baseline, sampler and penalties.
    union = pd.concat([results[key].viability_screen.assign(comparison_run=key) for key in ('old_finite', 'old_refined', 'finite', 'refined')], ignore_index=True)
    ordinary = union.loc[~union.experimental_role.eq('campaign_control')].copy()
    policy = resolve_intact_combination_policy(config, 13)
    evidence = build_intact_evidence(forms, obs, registry, policy, 13)
    scored = annotate_candidates(annotate_intact_combination_evidence(ordinary, evidence, registry, policy), models, registry, config, True)
    common_score, _ = _mechanics_phase_scores(scored, models, registry, config, policy)
    ordinary['common_individual_log_acquisition_score'] = common_score
    fields = ['comparison_run', 'selection_rank', 'formulation_id', 'candidate_id', 'experimental_role',
              'candidate_origin', 'generation_origin', 'common_individual_log_acquisition_score',
              'raw_surrogate_viability_mean', 'predicted_critical_axial_load_N_per_needle',
              'feasibility_pass', 'support_status', *registry.feature_names]
    ordinary.reindex(columns=fields).to_csv(output / 'candidate_comparison.csv', index=False)
    overlap = len(set(receipts['finite']['selected']) & set(receipts['refined']['selected']))
    removed = results['old_refined'].viability_screen.loc[~results['old_refined'].viability_screen.formulation_id.isin(receipts['refined']['selected'])]
    added = results['refined'].viability_screen.loc[~results['refined'].viability_screen.formulation_id.isin(receipts['old_refined']['selected'])]
    def recipe(row):
        return ' + '.join(f'{registry.get_by_feature(f).display_name} {row[f]:.6g} {registry.get_by_feature(f).unit}'
                          for f in registry.feature_names if abs(row[f]) >= presence_threshold(f))
    change_lines = [f"| Removed | {int(row.selection_rank)} | {recipe(row)} | {row.generation_origin} |" for _, row in removed.iterrows()]
    change_lines += [f"| Added | {int(row.selection_rank)} | {recipe(row)} | {row.generation_origin} |" for _, row in added.iterrows()]
    summary = dict(status='passed', created_at=datetime.now(timezone.utc).isoformat(),
                   branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                   code_base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   versions=dict(python=platform.python_version(), torch=torch.__version__, botorch=botorch.__version__),
                   shared_pool_sha256=digest(output / 'starting_pool.csv'), pool_generation_seconds=generation_seconds,
                   old_finite_replays_frozen_slate=True, old_refined_reconstructed_from_commit=True,
                   previous_allocation_commit=PREVIOUS_ALLOCATION_COMMIT,
                   previous_allocation_source_sha256=digest(output / 'previous_group10_selection.py'),
                   repeat_reproduces_refined_run=True, full_phase_allocation_verified=True,
                   earlier_phase_allocation_parity=earlier_phases,
                   ingredient_set_coverage_verified=True,
                   selected_formulation_overlap=overlap, protected_campaign_files_verified=verified,
                   overlap_with_previous_refined=len(set(receipts['old_refined']['selected']) & set(receipts['refined']['selected'])),
                   overlap_with_official=len(set(receipts['old_finite']['selected']) & set(receipts['refined']['selected'])),
                   mechanical_primaries_unchanged=receipts['finite']['mechanical_primaries'] == receipts['refined']['mechanical_primaries'],
                   source_sha256={str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT / 'src/08_multi_objective').rglob('*.py'))},
                   input_sha256={name: digest(ROOT / name) for name in ('data/processed_v2/formulations.csv', 'data/processed_v2/observations.csv')},
                   runs=receipts)
    (output / 'comparison_receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
    info = receipts['refined']['continuous']
    mask_sizes = dict(sorted((str(k), v) for k, v in info.get('mask_size_counts', {}).items()))
    refined_pool = results['refined'].candidate_pool
    best_continuous = float(refined_pool.loc[refined_pool.candidate_origin.eq('continuous_qlognehvi'), 'mechanics_phase_score'].max())
    allocation_lines = []
    for key in ('old_finite', 'old_refined', 'finite', 'refined'):
        slate = results[key].viability_screen
        roles = slate.experimental_role
        origin_counts = results[key].metadata['group10'].get('ordinary_origin_counts')
        if origin_counts is None:
            origin_counts = slate.loc[roles.eq('ordinary') & ~slate.candidate_origin.eq('rescue_dilution'), 'candidate_origin'].value_counts().to_dict()
        allocation_lines.append(f"| {key} | {int(roles.eq('campaign_control').sum())} | {int(roles.eq('screened_hit_mechanics').sum())} | {int(roles.eq('retest_priority').sum())} | {int(slate.candidate_origin.eq('rescue_dilution').sum())} | {origin_counts} | {receipts[key]['mechanical_primary_positions']} |")
    report = f'''# ROUND_013 full-phase allocation comparison

Full phase now reserves **one control, one mechanics follow-up and one shared retest/rescue slot**. An eligible evidence-ranked retest takes priority; otherwise an acquisition-ranked rescue fills that slot. The remaining **nine candidates compete across origins on the shared mechanics acquisition score**, with the existing feasibility, availability, similarity, support, cold-ingredient, ingredient-frequency, exact-combination and shared-pair constraints. There is no finite-versus-continuous quota. If a required role or the 12-row slate cannot be filled under these constraints, selection stops with a diagnostic.

The old allocator was loaded from immutable commit `{PREVIOUS_ALLOCATION_COMMIT[:7]}` and reproduced the official finite slate. Its refined slate was reconstructed in this run; no archived comparison output is required or verified. All runs share the fitted control-informed viability GP, comparable-endpoint mechanical GP and starting pool. Old/new allocation consumes the same scored candidate pool for each search method; the repeated refinement regenerates its pool independently.

| Run | Control | Follow-up | Retest | Rescue | Ordinary sources (realized counts, not quotas) | Mechanical primary positions |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(allocation_lines)}

`old_finite` is the official baseline; `old_refined` reconstructs the previous allocation with origin quotas; `finite` and `refined` both use the new full-phase allocation.

- Recipes retained from the **reconstructed previous refined allocation**: **{summary['overlap_with_previous_refined']}/12**.
- Recipes retained from the **official finite proposal**: **{summary['overlap_with_official']}/12**.
- Recipes shared by **new finite and new refined allocations**: **{overlap}/12**.
- Continuous candidates accepted before combined-pool filtering: **{info.get('accepted_candidate_count', 0)}**; surviving all candidate filters: **{info.get('accepted_after_all_candidate_filters', 0)}**.
- Continuous candidates selected: **{info.get('selected_candidate_count', 0)}**; mechanical primaries: **{info.get('selected_mechanical_primary_count', 0)}**.
- Mechanical primary identities identical between new finite and new refined allocation: **{summary['mechanical_primaries_unchanged']}**.
- Ingredient sets searched, grouped by number of ingredients: **{mask_sizes}**. Sets are ranked by the best feasible seed's shared acquisition score, with the best set of each available ingredient count reserved first. Every chosen set reached the optimizer; coverage and top-score inclusion were checked.
- Best surviving continuous individual log-acquisition score: **{best_continuous:.6f}**. Compare it with the finite recipes in `candidate_comparison.csv`; larger is better under this shared score.
- New final allocation runtime from the shared scored pools: finite **{receipts['finite']['seconds']:.1f} s**, refined **{receipts['refined']['seconds']:.1f} s**. Candidate generation, scoring and plotting are excluded from these two timings.
- A second refinement run reproduced candidate values, selection and mechanical assignments within 1e-9 numerical tolerance.
- Screening, bootstrap and hybrid final-allocation checks on the shared scored pool matched the previous allocator, including recipe identities, mechanical assignments and source quotas.
- All **{verified}** protected campaign files remained unchanged.

Changes below are relative to the reconstructed previous refined allocation. Positions refer to each run separately.

| Change | Position in that run | Recipe | Source |
| --- | --- | --- | --- |
{chr(10).join(change_lines)}

The new allocation records `ordinary_origin_quotas={{}}`; origin counts are outcomes of scoring and constraints. Other phases retain their previous reservations and origin allocation. The best continuous score describes a model-based search utility, not a measured improvement.

Inspect `candidate_comparison.csv` for recipes and common individual log-acquisition scores, each run's `next_round_candidates.csv` for the complete slate and mechanical assignments, and `comparison_receipt.json` for execution details. Candidate-decision plots are inside each run's `plots` directory.

Scope: this is finite-pool exploration plus continuous concentration refinement within the existing ingredient-set budget. All eligible sets with feasible seeds are scored before selecting up to 16 sets (at most four ingredients each); the run does not search every ingredient combination. `ingredient_set_ranking.csv` records the scores, choices and actual optimizer attempts. Existing similarity conflict priority gives rescue candidates precedence, then continuous candidates, then ordinary finite candidates. All survivors are subsequently ranked with a common score and existing slate policies. The scores are individual model-based utilities, not joint batch hypervolume or measured experimental improvement. This comparison does not replace the official ROUND_013 proposal.
'''
    (output / 'README.md').write_text(report)
    verify_campaign(manifest)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    main(parser.parse_args().output_dir)

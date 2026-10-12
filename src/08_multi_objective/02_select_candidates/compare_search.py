#!/usr/bin/env python3
"""Isolated, matched ROUND_013 finite versus continuous-refinement comparison."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import botorch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/08_multi_objective'))
from helper.candidates import (generate_rescue_candidate_pool, generate_support_aware_candidate_pool,
                               unavailable_features_from_config)
from helper.config import load_optimization_config, load_availability_config
from helper.feasibility import annotate_feasibility, annotate_support, build_support_context, policy_activation
from helper.gp_strategy import validate_decision
from helper.group10_config import production_observations, load_group10_config_for_round
from helper.group10_selection import apply_group10
from helper.models import train_endpoint_models
from helper.registry import load_registry
from helper.selection import select_next_round, write_selection_result
from helper.selection_scoring import _mechanics_phase_scores, annotate_candidates
from helper.intact_policy import build_intact_evidence, resolve_intact_combination_policy, annotate_intact_combination_evidence
from helper.similarity import SimilarityAudit, build_history_similarity_index, resolve_similarity_policy
from helper.visualization import generate_proposal_artifacts

BASE = ROOT / 'results/multi_objective_v2/comparisons/pre_full_phase_check'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def main(output):
    output = output.resolve()
    if not output.is_relative_to(BASE.resolve()) or output in (BASE.resolve(), (BASE / 'baseline').resolve()):
        raise ValueError('Output must be a new subdirectory inside the comparison workspace')
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((BASE / 'baseline/manifest.json').read_text())
    assert digest(BASE / 'baseline' / manifest['archive']) == manifest['archive_sha256']
    verified = verify_campaign(manifest)
    # Configuration is part of the matched experiment and must match the baseline.
    for path, expected in manifest['files'].items():
        if path.startswith('config_v2/'):
            assert digest(ROOT / path) == expected, path
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
    results, receipts = {}, {}
    for label, mode in [('finite', 'finite_pool'), ('refined', 'continuous_required'), ('refined_repeat', 'continuous_required')]:
        folder = output / label
        folder.mkdir()
        decision = json.loads(legacy_path.read_text())
        decision.update(version=2, decision_id=f'round13-{mode}-comparison-v2', effective_round=13,
                        acquisition='shared_qlognehvi', search_mode=mode,
                        chosen_by='user: implement the latest simplified plan; comparison only',
                        evidence=['docs/plans/pre_full_phase_check_repair_plan.md', 'preserved post-Group-12 baseline'])
        decision_path = folder / 'gp_strategy_decision.json'
        decision_path.write_text(json.dumps(decision, indent=2) + '\n')
        run_config = deepcopy(config)
        run_config['gp_strategy_decision'] = validate_decision(decision_path, 13)
        models.gp_strategy_decision = run_config['gp_strategy_decision']
        started = time.perf_counter()
        print(f'Running {label}...', flush=True)
        try:
            result = select_next_round(forms, obs, pool.copy(), registry, run_config, target_round_number=13,
                policy_active=True, policy_version=version, similarity_audit=deepcopy(audit),
                unavailable_feature_names=unavailable, endpoint_models=models)
            result = apply_group10(result, forms, obs, registry, run_config, group_config, 13,
                                   unavailable, endpoint_models=models)
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
        sheet = pd.read_csv(folder / 'next_round_candidates.csv')
        generate_proposal_artifacts(sheet, folder)
        if label == 'finite':
            for column in ('formulation_id', 'mechanical_test_recommended', 'mechanical_selection_rank'):
                pd.testing.assert_series_equal(sheet[column], frozen[column], check_names=False)
        results[label] = result
        receipts[label] = dict(seconds=elapsed, selected=sheet.formulation_id.tolist(),
                               mechanical_primaries=sheet.loc[sheet.mechanical_test_recommended, 'formulation_id'].tolist(),
                               continuous=result.metadata['continuous_qlognehvi'])
        print(f'{label} completed in {elapsed:.1f}s; {receipts[label]["continuous"].get("selected_candidate_count", 0)} continuous recipes selected.', flush=True)
        verify_campaign(manifest)
    for name in ('viability_screen', 'mechanical_tests', 'candidate_pool'):
        pd.testing.assert_frame_equal(getattr(results['refined'], name), getattr(results['refined_repeat'], name),
                                      check_exact=False, atol=1e-9, rtol=1e-9)

    # Common final scoring: no pending points; same model, baseline, sampler and penalties.
    union = pd.concat([results[key].viability_screen.assign(comparison_run=key) for key in ('finite', 'refined')], ignore_index=True)
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
    summary = dict(status='passed', created_at=datetime.now(timezone.utc).isoformat(),
                   branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                   code_base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   versions=dict(python=platform.python_version(), torch=torch.__version__, botorch=botorch.__version__),
                   shared_pool_sha256=digest(output / 'starting_pool.csv'), pool_generation_seconds=generation_seconds,
                   finite_replays_frozen_slate=True, repeat_reproduces_refined_run=True,
                   selected_formulation_overlap=overlap, protected_campaign_files_verified=verified,
                   source_sha256={str(path.relative_to(ROOT)): digest(path) for path in sorted((ROOT / 'src/08_multi_objective').rglob('*.py'))},
                   input_sha256={name: digest(ROOT / name) for name in ('data/processed_v2/formulations.csv', 'data/processed_v2/observations.csv')},
                   runs=receipts)
    (output / 'comparison_receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
    info = receipts['refined']['continuous']
    mask_sizes = dict(sorted((str(k), v) for k, v in info.get('mask_size_counts', {}).items()))
    refined_pool = results['refined'].candidate_pool
    best_continuous = float(refined_pool.loc[refined_pool.candidate_origin.eq('continuous_qlognehvi'), 'mechanics_phase_score'].max())
    report = f'''# ROUND_013 search comparison

The finite replay reproduced the frozen slate and mechanical assignments. The continuous run used the same fitted control-informed viability GP, comparable-endpoint mechanical GP, starting pool and qLogNEHVI settings.

- Shared formulations: **{overlap}/12**.
- Continuous candidates accepted before combined-pool filtering: **{info.get('accepted_candidate_count', 0)}**; surviving all candidate filters: **{info.get('accepted_after_all_candidate_filters', 0)}**.
- Continuous candidates selected: **{info.get('selected_candidate_count', 0)}**; mechanical primaries: **{info.get('selected_mechanical_primary_count', 0)}**.
- Ingredient sets searched, grouped by number of ingredients: **{mask_sizes}**. The retained ordering chooses the smallest sets first; if all 16 are single-ingredient sets, this comparison does not assess multi-ingredient continuous refinement.
- Best surviving continuous individual log-acquisition score: **{best_continuous:.6f}**. Compare it with the finite recipes in `candidate_comparison.csv`; larger is better under this shared score.
- Finite selection runtime: **{receipts['finite']['seconds']:.1f} s**; selection with refinement: **{receipts['refined']['seconds']:.1f} s**. Shared pool generation and plotting are excluded.
- A second refinement run reproduced candidate values, selection and mechanical assignments within 1e-9 numerical tolerance.
- All **{verified}** protected campaign files remained unchanged.

Inspect `candidate_comparison.csv` for recipes and common individual log-acquisition scores, each run's `next_round_candidates.csv` for the complete slate and mechanical assignments, and `comparison_receipt.json` for execution details. Candidate-decision plots are inside each run's `plots` directory.

Scope: this is finite-pool exploration plus continuous concentration refinement within the existing ingredient-set budget. The original mask ordering is retained; the run does not search every ingredient combination. Existing similarity conflict priority gives rescue candidates precedence, then continuous candidates, then ordinary finite candidates. All survivors are subsequently ranked with a common score and existing slate policies. The scores are individual model-based utilities, not joint batch hypervolume or measured experimental improvement. This comparison does not replace the official ROUND_013 proposal.
'''
    (output / 'README.md').write_text(report)
    verify_campaign(manifest)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    main(parser.parse_args().output_dir)

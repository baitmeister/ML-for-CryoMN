"""Lifecycle regression tests use isolated databases, never live campaign writes."""
import csv
import importlib.util
import io
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'src/08_multi_objective'
sys.path.insert(0, str(V2))
from helper import persistence as storage
from helper.artifacts import copy_frozen, ArtifactConflictError
from helper.feedback import validate_numeric_feedback, NUMERIC_FEEDBACK_COLUMNS
from helper.finalization import finalization_manifest
from helper.transfer import write_v2_tables, OBSERVATION_COLUMNS
from helper.registry import load_registry


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('value', ['fifty', 'NaN', 'NA', 'inf', '-Infinity', '1e999'])
@pytest.mark.parametrize('column', NUMERIC_FEEDBACK_COLUMNS)
def test_nonblank_numbers_never_become_missing(tmp_path, value, column):
    path = tmp_path / 'sheet.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['candidate_id', column])
        writer.writeheader()
        writer.writerow({'candidate_id': 'test_candidate', column: value})
    with pytest.raises(ValueError, match=f'Row 2, candidate test_candidate, column {column}'):
        validate_numeric_feedback(path)


def test_blank_and_valid_numeric_representations(tmp_path):
    path = tmp_path / 'sheet.csv'
    path.write_text('candidate_id,viability_percent\na,\nb,  \nc,5.5e1\nd,+50.0\n')
    validate_numeric_feedback(path)


def test_frozen_publication_race_has_exactly_one_winner(tmp_path):
    barrier = threading.Barrier(2)
    destination = tmp_path / 'frozen.csv'
    def write(data):
        barrier.wait()
        try:
            storage.publish_bytes(destination, data, frozen=True)
            return 'won'
        except storage.PersistenceConflict:
            return 'conflict'
    with ThreadPoolExecutor(2) as workers:
        results = list(workers.map(write, [b'first\n', b'second\n']))
    assert sorted(results) == ['conflict', 'won']
    source = tmp_path / 'source.csv'
    source.write_bytes(destination.read_bytes())
    copy_frozen(source, destination)
    source.write_bytes(b'different\n')
    with pytest.raises(ArtifactConflictError):
        copy_frozen(source, destination)


def test_process_lock_prevents_lost_read_modify_write(tmp_path):
    counter = tmp_path / 'counter'
    counter.write_text('0')
    script = '''import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from helper.persistence import writer_lock, publish_bytes
p=Path(sys.argv[2])
for _ in range(8):
    with writer_lock():
        value=int(p.read_text())
        time.sleep(.005)
        publish_bytes(p,str(value+1).encode())
'''
    processes = [subprocess.Popen([sys.executable, '-c', script, str(V2), str(counter)]) for _ in range(2)]
    assert [p.wait(timeout=20) for p in processes] == [0, 0]
    assert counter.read_text() == '16'


@pytest.mark.parametrize('boundary', range(1, 9))
@pytest.mark.parametrize('after', [False, True])
def test_interruption_at_every_publication_boundary_recovers(tmp_path, boundary, after):
    # Three payload writes, prepared manifest, three promotions, committed manifest.
    root = tmp_path / 'journal'
    form, obs, archive = [tmp_path / name for name in ('forms.csv', 'obs.csv', 'completed.csv')]
    form.write_bytes(b'old forms')
    obs.write_bytes(b'old observations')
    outputs = [(form, b'new forms', False), (obs, b'new observations', False), (archive, b'completed', True)]
    publish = storage.publish_bytes
    count = 0
    def interrupt(*args, **kwargs):
        nonlocal count
        count += 1
        if count == boundary and not after:
            raise OSError('simulated interruption')
        result = publish(*args, **kwargs)
        if count == boundary and after:
            raise OSError('simulated interruption')
        return result
    with patch.object(storage, 'publish_bytes', side_effect=interrupt):
        with pytest.raises(OSError, match='simulated'):
            storage.commit_files(root, outputs)
    prepared = bool(list(root.glob('*/manifest.json')))
    storage.recover_transactions(root)
    if prepared:
        assert [p.read_bytes() for p, _, _ in outputs] == [data for _, data, _ in outputs]
    else:
        assert form.read_bytes() == b'old forms'
        assert obs.read_bytes() == b'old observations'
        assert not archive.exists()
    storage.recover_transactions(root)  # Recovery is itself idempotent.


def test_recovery_refuses_unrelated_edits_and_read_only_never_recovers(tmp_path):
    root, target = tmp_path / 'journal', tmp_path / 'table.csv'
    target.write_bytes(b'old')
    publish = storage.publish_bytes
    def interrupt(path, *args, **kwargs):
        if Path(path) == target:
            raise OSError('interrupted')
        return publish(path, *args, **kwargs)
    with patch.object(storage, 'publish_bytes', side_effect=interrupt), pytest.raises(OSError):
        storage.commit_files(root, [(target, b'new', False)])
    before = snapshot(tmp_path)
    with pytest.raises(storage.PersistenceConflict, match='requires recovery'):
        storage.recover_transactions(root, read_only=True)
    assert snapshot(tmp_path) == before
    target.write_bytes(b'operator edit')
    with pytest.raises(storage.PersistenceConflict, match='external edit'):
        storage.recover_transactions(root)
    assert target.read_bytes() == b'operator edit'


def test_initializer_protects_either_nonempty_table(tmp_path):
    forms, obs = pd.DataFrame({'formulation_id': ['a']}), pd.DataFrame({'value': [55]})
    write_v2_tables(forms, obs, tmp_path)
    before = snapshot(tmp_path)
    with pytest.raises(FileExistsError, match='separate empty directory'):
        write_v2_tables(forms, obs, tmp_path)
    assert snapshot(tmp_path) == before
    (tmp_path / 'formulations.csv').unlink()
    before = snapshot(tmp_path)
    with pytest.raises(FileExistsError):
        write_v2_tables(forms, obs, tmp_path)
    assert snapshot(tmp_path) == before


def test_finalization_does_not_require_force_from_every_screen_or_lost_attempt():
    sheet = pd.DataFrame([
        dict(candidate_id='a', viability_percent=60, intact_patch_formation_pass='pass'),
        dict(candidate_id='a', viability_percent=61, intact_patch_formation_pass=None),
        dict(candidate_id='b', viability_percent=55, intact_patch_formation_pass='fail'),
    ])
    audit = dict(expected_actual_intact_test_ids=['a'], recorded_mechanical_test_ids=['a'])
    completion = finalization_manifest(sheet, audit, finalize=True)
    assert completion['unmeasured_outcomes'] == []
    audit['recorded_mechanical_test_ids'] = []
    with pytest.raises(ValueError, match='mechanical_test_attempt'):
        finalization_manifest(sheet, audit, finalize=True)
    completion = finalization_manifest(sheet, audit, finalize=True, unmeasured_reason='instrument unavailable')
    assert completion['unmeasured_outcomes'][0]['endpoints'] == ['mechanical_test_attempt']


@pytest.fixture
def round_case(tmp_path):
    spec = importlib.util.spec_from_file_location('round_lifecycle', V2 / '03_run_round/run_round.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    features = load_registry().feature_names
    candidates = pd.DataFrame([
        dict(candidate_id=f'c{i}', formulation_id=f'f{i}', batch_id='ROUND_001',
             viability_percent=None, intact_patch_formation_pass=None, replicate_id=None,
             **{f: float(f == 'dmso_M') for f in features}) for i in (1, 2)
    ])
    proposal = tmp_path / 'results/rounds/ROUND_001/proposal'
    proposal.mkdir(parents=True)
    candidates.to_csv(proposal / 'proposal.csv', index=False)
    (proposal / 'selection_metadata.json').write_text('{}')
    working = tmp_path / 'results/next_round/next_round_candidates.csv'
    working.parent.mkdir(parents=True)
    candidates.to_csv(working, index=False)
    forms = tmp_path / 'formulations.csv'
    pd.DataFrame(columns=['formulation_id', 'source', *features]).to_csv(forms, index=False)
    obs = tmp_path / 'observations.csv'
    pd.DataFrame(columns=OBSERVATION_COLUMNS).to_csv(obs, index=False)
    argv = ['run_round.py', str(working), '--formulations', str(forms), '--observations', str(obs),
            '--output-dir', str(working.parent), '--skip-generate']
    def run(*extra):
        with patch.object(sys, 'argv', argv + list(extra)), redirect_stdout(io.StringIO()):
            module.main()
    return module, candidates, working, obs, run


def test_partial_results_fail_without_persistent_writes_and_can_be_completed(tmp_path, round_case):
    module, sheet, working, obs, run = round_case
    sheet.loc[0, 'viability_percent'] = 55
    sheet.to_csv(working, index=False)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match='not finalized'):
        run('--skip-review')
    assert snapshot(tmp_path) == before
    with pytest.raises(ValueError, match='Unresolved outcomes'):
        run('--skip-review', '--finalize')
    assert snapshot(tmp_path) == before
    sheet['viability_percent'] = [55, 65]
    sheet['intact_patch_formation_pass'] = [0, 0]
    sheet.to_csv(working, index=False)
    run('--finalize', '--skip-review')
    assert len(pd.read_csv(obs)) == 4
    assert (tmp_path / 'results/rounds/ROUND_001/completed/ingestion_commit.json').exists()


def test_documented_missing_outcomes_can_finalize(tmp_path, round_case):
    _, sheet, working, obs, run = round_case
    sheet.loc[0, 'viability_percent'] = 55
    sheet.to_csv(working, index=False)
    run('--finalize', '--unmeasured-reason', 'remaining samples lost in transfer', '--skip-review')
    manifest = json.loads((tmp_path / 'results/rounds/ROUND_001/completed/finalization.json').read_text())
    assert len(manifest['unmeasured_outcomes']) == 2
    assert len(pd.read_csv(obs)) == 1


def test_invalid_number_stops_before_any_archival_or_table_write(tmp_path, round_case):
    _, sheet, working, _, run = round_case
    sheet['viability_percent'] = ['fifty', '']
    sheet.to_csv(working, index=False)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match='Row 2, candidate c1, column viability_percent'):
        run('--finalize', '--skip-review')
    assert snapshot(tmp_path) == before


def test_retry_after_report_failure_skips_ingest_without_duplicate_rows(tmp_path, round_case):
    module, sheet, working, obs, run = round_case
    sheet['viability_percent'], sheet['intact_patch_formation_pass'] = [55, 65], [0, 0]
    sheet.to_csv(working, index=False)
    with patch.object(module, 'generate_completed_round_artifacts', side_effect=OSError('report interrupted')):
        with pytest.raises(OSError, match='report interrupted'):
            run('--finalize')
    committed = obs.read_bytes()
    with patch.object(module, 'ingest_feedback', side_effect=AssertionError('must not reingest')), \
         patch.object(module, 'generate_completed_round_artifacts', return_value=[]) as report, \
         patch.object(module, 'generate_round_prospective_artifacts', return_value=[]), \
         patch.object(module, 'generate_campaign_prospective_artifacts', return_value=[]):
        run()
        report.assert_called_once()
    assert obs.read_bytes() == committed
    with pytest.raises(storage.PersistenceConflict, match='inputs differ'):
        run('--skip-review', '--viability-noise', '2')
    assert obs.read_bytes() == committed
    changed = pd.read_csv(obs)
    changed.loc[0, 'value'] = 7
    changed.to_csv(obs, index=False)
    with pytest.raises(storage.PersistenceConflict, match='observations changed'):
        run('--skip-review')


@pytest.fixture
def selection_case(tmp_path):
    from helper.candidate_workflow import CandidateSelectionOptions
    import helper.candidate_workflow as workflow
    from helper.selection import SelectionResult
    import shutil
    source = ROOT / 'results/multi_objective_v2/rounds/ROUND_011'
    for name in ('formulations.csv', 'observations.csv'):
        shutil.copyfile(ROOT / 'data/processed_v2' / name, tmp_path / name)
    decision = tmp_path / 'decision.json'
    # The archived decision has a validated sha256 field as well as the user's settings.
    shutil.copyfile(source / 'proposal/gp_strategy_decision.json', decision)
    slate = pd.read_csv(source / 'proposal/proposal.csv')
    metadata = json.loads((source / 'proposal/selection_metadata.json').read_text())
    options = CandidateSelectionOptions(
        formulations_path=tmp_path / 'formulations.csv', observations_path=tmp_path / 'observations.csv',
        output_dir=tmp_path / 'results/next_round', total_candidate_pool_path=tmp_path / 'pool.csv',
        pool_size=20, seed=42, batch_id='ROUND_011', gp_strategy_decision_path=decision)
    def selected(**kwargs):
        return SelectionResult(slate.copy(), slate.loc[slate.mechanical_test_recommended].copy(),
                               slate.copy(), json.loads(json.dumps(metadata)))
    def write(result, output, *, batch_id, total_candidate_pool_path, registry):
        output.mkdir(parents=True)
        result.viability_screen.to_csv(output / 'next_round_candidates.csv', index=False)
        (output / 'next_round_summary.txt').write_text('Synthetic selection for lifecycle test\n')
        (output / 'next_round_metadata.json').write_text(json.dumps(result.metadata))
        result.candidate_pool.to_csv(total_candidate_pool_path, index=False)
    def sensitivity(*args):
        output = args[-1]
        output.mkdir(parents=True, exist_ok=True)
        (output / 'summary.json').write_text('{"flagged":false}')
    def freeze(root, batch, output, **kwargs):
        output.mkdir(parents=True)
        pd.DataFrame({'predicted': [60.], 'created_at': [str(len(list(tmp_path.rglob('manifest.json'))))]}).to_csv(output / 'predictions.csv', index=False)
        (output / 'model.json').write_text('{"fixed":true}')
        manifest = {k: 'fixed' for k in ('cutoff', 'features', 'reference_ids', 'models', 'training_sha256', 'assumptions')}
        manifest.update(batch_id=batch, proposal_sha256=storage.file_hash(kwargs['slate_path']),
                        hashes={p.name: storage.file_hash(p) for p in output.iterdir()})
        (output / 'manifest.json').write_text(json.dumps(manifest))
    with patch.object(workflow, 'select_next_round', side_effect=selected) as select, \
         patch('helper.group10_selection.apply_group10', side_effect=lambda result, *args: result), \
         patch('helper.control_sensitivity.check_sensitivity', side_effect=sensitivity), \
         patch.object(workflow, 'write_selection_result', side_effect=write), \
         patch('helper.gp_comparison.freeze', side_effect=freeze) as challenger, \
         patch.object(workflow, 'generate_proposal_artifacts', return_value=[]) as report:
        yield workflow, options, select, challenger, report


def test_round11_selection_retry_reuses_frozen_forecasts_and_preserves_lab_edits(tmp_path, selection_case):
    workflow, options, select, challenger, _ = selection_case
    first = workflow.run_candidate_selection(options)
    frozen_root = tmp_path / 'results/rounds/ROUND_011'
    frozen = snapshot(frozen_root)
    working = Path(options.output_dir) / 'next_round_candidates.csv'
    sheet = pd.read_csv(working)
    sheet.loc[0, 'viability_percent'] = 52
    sheet.to_csv(working, index=False)
    entered = working.read_bytes()
    second = workflow.run_candidate_selection(options)
    assert second.round_id == first.round_id == 'ROUND_011'
    assert select.call_count == challenger.call_count == 1
    assert working.read_bytes() == entered
    assert snapshot(frozen_root) == frozen
    with pytest.raises(storage.PersistenceConflict, match='inputs/decision differ'):
        workflow.run_candidate_selection(replace(options, seed=43))
    assert working.read_bytes() == entered
    assert snapshot(frozen_root) == frozen


def test_selection_report_failure_can_resume_without_refit(tmp_path, selection_case):
    workflow, options, select, challenger, report = selection_case
    report.side_effect = OSError('plot interrupted')
    with pytest.raises(OSError, match='plot interrupted'):
        workflow.run_candidate_selection(options)
    report.side_effect = None
    workflow.run_candidate_selection(options)
    assert select.call_count == challenger.call_count == 1


def test_pre_journal_comparison_reuse_ignores_only_new_timestamps(tmp_path, selection_case):
    workflow, options, _, challenger, _ = selection_case
    workflow.run_candidate_selection(options)
    proposal = tmp_path / 'results/rounds/ROUND_011/proposal'
    (proposal / 'selection_commit.json').unlink()
    frozen = snapshot(proposal.parent / 'gp_comparison')
    workflow.run_candidate_selection(options)
    assert challenger.call_count == 2  # Old evidence is reproduced once before adoption.
    assert snapshot(proposal.parent / 'gp_comparison') == frozen


def test_selector_refuses_changed_frozen_artifact(tmp_path, selection_case):
    workflow, options, _, _, _ = selection_case
    workflow.run_candidate_selection(options)
    frozen = tmp_path / 'results/rounds/ROUND_011/gp_comparison/predictions.csv'
    frozen.write_text('tampered')
    with pytest.raises(storage.PersistenceConflict, match='artifact changed'):
        workflow.run_candidate_selection(options)


def test_committed_retry_refuses_changed_recipe(tmp_path, round_case):
    _, sheet, working, _, run = round_case
    sheet['viability_percent'], sheet['intact_patch_formation_pass'] = [55, 65], [0, 0]
    sheet.to_csv(working, index=False)
    run('--finalize', '--skip-review')
    forms_path = tmp_path / 'formulations.csv'
    forms = pd.read_csv(forms_path)
    forms.loc[0, 'dmso_M'] = 2.0
    forms.to_csv(forms_path, index=False)
    before = snapshot(tmp_path)
    with pytest.raises(storage.PersistenceConflict, match='concentrations changed'):
        run('--skip-review')
    assert snapshot(tmp_path) == before


def test_concurrent_round_ingestions_preserve_both_batches(tmp_path, round_case):
    _, sheet, working, obs, _ = round_case
    sheet['viability_percent'], sheet['intact_patch_formation_pass'] = [55, 65], [0, 0]
    sheet.to_csv(working, index=False)
    second_proposal = tmp_path / 'results/rounds/ROUND_002/proposal'
    second_proposal.mkdir(parents=True)
    second = sheet.copy()
    second['batch_id'] = 'ROUND_002'
    second[['viability_percent', 'intact_patch_formation_pass']] = None
    second.to_csv(second_proposal / 'proposal.csv', index=False)
    (second_proposal / 'selection_metadata.json').write_text('{}')
    second['viability_percent'], second['intact_patch_formation_pass'] = [57, 67], [0, 0]
    worksheet2 = tmp_path / 'round2.csv'
    second.to_csv(worksheet2, index=False)
    common = ['--formulations', str(tmp_path / 'formulations.csv'), '--observations', str(obs),
              '--output-dir', str(working.parent), '--finalize', '--skip-review', '--skip-generate']
    processes = [subprocess.Popen([sys.executable, str(V2 / '03_run_round/run_round.py'), str(path), *common],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for path in (working, worksheet2)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
    observations = pd.read_csv(obs)
    assert observations.batch_id.value_counts().to_dict() == {'ROUND_001': 4, 'ROUND_002': 4}
    assert observations.observation_id.is_unique


def test_process_death_releases_lock_and_recovers_prepared_commit(tmp_path):
    target = tmp_path / 'table.csv'
    target.write_bytes(b'old')
    script = '''import os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from helper import persistence as p
root=Path(sys.argv[2]);target=root/'table.csv'
original=p.publish_bytes
def die(path,*args,**kwargs):
    result=original(path,*args,**kwargs)
    if Path(path)==target:os._exit(23)
    return result
p.publish_bytes=die
p.commit_files(root/'journal',[(target,b'new',False),(root/'archive.csv',b'evidence',True)])
'''
    result = subprocess.run([sys.executable, '-c', script, str(V2), str(tmp_path)], timeout=20)
    assert result.returncode == 23
    assert target.read_bytes() == b'new'
    assert not (tmp_path / 'archive.csv').exists()
    storage.recover_transactions(tmp_path / 'journal')
    assert (tmp_path / 'archive.csv').read_bytes() == b'evidence'

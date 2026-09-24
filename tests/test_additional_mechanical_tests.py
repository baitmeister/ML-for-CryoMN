"""Extra acquired backups require explicit, scoped authorization."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/08_multi_objective'))
from helper.artifacts import ProposalValidationError
from helper.mechanics_execution import validate_mechanics_execution


@pytest.fixture
def case():
    proposal = pd.DataFrame({
        'candidate_id': [f'c{i}' for i in range(1, 8)],
        'mechanical_selection_rank': [1, 2, 3, 4, 5, 6, None],
        'mechanical_test_recommended': [True] * 4 + [False] * 3,
    })
    completed = proposal[['candidate_id']].copy()
    completed['intact_patch_formation_pass'] = 'pass'
    completed['mechanical_test_attempted'] = [True] * 5 + [False] * 2
    return completed, proposal


def test_explicit_extra_preserves_frozen_budget_and_original_deviation(case):
    completed, proposal = case
    with pytest.raises(ProposalValidationError, match='exceeds capacity 4'):
        validate_mechanics_execution(completed, proposal)
    audit = validate_mechanics_execution(completed, proposal,
        additional_test_ids=['c5'], additional_test_reason='Operator measured backup 1 as an extra')
    assert audit['configured_mechanical_capacity'] == 4
    assert audit['program_requested_test_count'] == 4
    assert audit['expected_actual_intact_test_ids'] == ['c1', 'c2', 'c3', 'c4']
    assert audit['recorded_mechanical_test_ids'] == ['c1', 'c2', 'c3', 'c4', 'c5']
    assert audit['promoted_backup_ids'] == []
    assert len(audit['additional_test_authorization']['original_policy_violations']) == 2
    assert audit['violations'] == []
    # The authorization is not sticky on subsequent calls or proposals.
    with pytest.raises(ProposalValidationError):
        validate_mechanics_execution(completed, proposal)


@pytest.mark.parametrize('ids,reason', [([], 'reason'), (['c5'], ''),
    (['c5', 'c5'], 'reason'), (['c6'], 'reason'), (['missing'], 'reason')])
def test_invalid_or_nonmatching_authorization_fails(case, ids, reason):
    with pytest.raises(ProposalValidationError):
        validate_mechanics_execution(*case, additional_test_ids=ids, additional_test_reason=reason)


@pytest.mark.parametrize('problem', ['failed', 'unconfirmed', 'unranked', 'skipped_backup', 'missing_primary'])
def test_extra_authorization_never_waives_eligibility_or_priority(case, problem):
    completed, proposal = case
    extra = 'c5'
    if problem in ('failed', 'unconfirmed'):
        completed.loc[4, 'intact_patch_formation_pass'] = 'fail' if problem == 'failed' else None
    elif problem in ('unranked', 'skipped_backup'):
        completed.loc[4, 'mechanical_test_attempted'] = False
        extra = 'c7' if problem == 'unranked' else 'c6'
        completed.loc[completed.candidate_id.eq(extra), 'mechanical_test_attempted'] = True
    else:
        completed.loc[0, 'mechanical_test_attempted'] = False
    with pytest.raises(ProposalValidationError):
        validate_mechanics_execution(completed, proposal,
            additional_test_ids=[extra], additional_test_reason='Operator requested extra')

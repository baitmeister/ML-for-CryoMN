"""Explicit completion attestation, without inventing replicate counts."""
import pandas as pd

from .mechanics_execution import _row_intact_measurement


def finalization_manifest(completed, mechanics_audit, *, finalize=False,
                          unmeasured_reason='', validate_only=False):
    unresolved = []
    expected = set(mechanics_audit['expected_actual_intact_test_ids'])
    attempted = set(mechanics_audit['recorded_mechanical_test_ids'])
    for candidate_id, rows in completed.groupby('candidate_id', sort=False):
        missing = []
        viability = pd.to_numeric(rows.get('viability_percent', pd.Series(dtype=float)), errors='raise')
        if not viability.notna().any():
            missing.append('viability_percent')
        if not any(_row_intact_measurement(row) is not None for _, row in rows.iterrows()):
            missing.append('intact_patch_formation_pass')
        if str(candidate_id) in expected - attempted:
            missing.append('mechanical_test_attempt')
        if missing:
            unresolved.append(dict(candidate_id=str(candidate_id), endpoints=missing,
                                   csv_rows=[int(i) + 2 for i in rows.index]))
    reason = unmeasured_reason.strip()
    details = '; '.join(f'{r["candidate_id"]} (rows {r["csv_rows"]}): {", ".join(r["endpoints"])}'
                        for r in unresolved)
    if not validate_only:
        if not finalize:
            raise ValueError('Round is not finalized. Run --validate-only while editing; use --finalize '
                             'only when the planned measurements are finished. '
                             + ('Unresolved outcomes: ' + details if unresolved else ''))
        if unresolved and not reason:
            raise ValueError('Unresolved outcomes: ' + details + '. Finish the measurements or supply '
                             '--unmeasured-reason documenting why these outcomes will remain unmeasured.')
    return dict(version=1, completion_attested=bool(finalize),
                unmeasured_outcomes=unresolved, unmeasured_reason=reason,
                replicate_count_policy='operator attests planned measurements finished; no inferred quota')

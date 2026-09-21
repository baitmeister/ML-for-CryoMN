**Five workflow fixes implemented — 20 September 2026**

1. Nonblank malformed or nonfinite numeric entries are rejected with CSV row, candidate and column diagnostics. Literal `NaN` and `NA` cannot silently turn into missing values. Genuine blanks and finite numbers remain supported.
2. Stage 03 requires `--finalize` before first ingestion. Missing viability/intact outcomes or expected mechanical attempts are listed; intentionally unmeasured outcomes require `--unmeasured-reason`, archived with that list. This is an operator attestation, not an inferred replicate quota. Lost mechanical attempts and endpoint-specific replicate rows remain supported. An entirely unmeasured round still needs a campaign decision before advancement.
3. Stage 01 refuses either nonempty database table. Intentional rebuilds use a separate empty `--output-dir`; there is no destructive force option. CLI verification confirmed a fresh build works and a repeated initialization leaves all existing outputs byte-identical.
4. Workflow writers share a process lock. Table/archive/proposal commits stage durable payloads and record input and before/after output hashes. Prepared commits recover before normal selection/ingestion; read-only validation and audits refuse pending recovery. Frozen publication uses atomic no-clobber creation. Conflicting external edits stop recovery. Readers that bypass these entry points and manual editors do not participate in the lock.
5. Committed ingestion retries validate source evidence, batch observations and formulation concentrations before resuming reports/generation. Selection retries reuse saved state and immutable forecasts, preserve entered worksheet values, and reject changed inputs. Old proposals are adopted only when reproduction agrees with frozen selection/model evidence, ignoring new creation timestamps. Unknown or conflicting partial historical outputs remain conflicts.

The models, two objectives, phase gates and live campaign data were not changed. No uncertain compatibility code was removed. The finalization flag intentionally changes the round command; the operator guides and synthetic workflow harness now include it.

**Verified**

- Full suite: **207 passed, 7 failed, 12 subtests passed** with third-party pytest plugin autoload disabled. The seven failures match the pre-implementation baseline in `test_summary.txt`; they concern stale campaign shape/schema/phase/metric fixtures and the legacy supplementary-mechanics fixture.
- 75 new lifecycle cases cover numeric errors, finalization, initialization, frozen publication, recovery, retries and concurrent writes. Fault injection checks before/after all eight publication boundaries of a three-file commit. A separate process exiting after its first promotion releases the lock and its prepared commit recovers. Two concurrent round ingestions retain both batches.
- Actual frozen Round 11 replayed in an isolated copy, then reran identically. No existing copied Round 11 artifact changed; only the new receipt and saved selection state were added. This check used the real model/selection path.
- Synthetic outcomes on the actual Group 10 slate passed ingestion, reporting and Group 11 generation. One lost attempt and one incomplete trace remained represented, with four usable formulation/batch mechanical observations. All **766** protected source files retained their hashes. The synthetic strategy decision was confined to the test copy.
- After adding the remaining read-only audit guards, 14 focused audit/control tests also passed. Python compilation and `git diff --check` passed.

**Limits**

Physical power loss and failing storage hardware were not tested. Completion of intended biological/technical replicates still depends on the operator's explicit attestation. The seven existing suite failures remain unresolved; these fixes do not validate experimental efficacy or GP predictive accuracy.

[Operator instructions](../../multi_objective_operator_guide.md) · [Recorded checks](implementation_checks.json) · [Integration output](implementation_integration_summary.json) · [Suite summary](implementation_test_summary.txt)

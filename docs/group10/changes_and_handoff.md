# Implementation handoff and policy changes

## Completion ledger

| Package | Implementation | Activation |
|---|---|---|
| Cleanup and branch | Preliminary edits removed from original checkout; rebuilt on requested branch | Original Group 9 checkout unchanged |
| Configuration and frozen metadata | Group10 workflow/schema v3, full effective settings, evidence hash | Earliest Group 10 |
| Requirements and acceptance | Null-safe paired evaluator and separate tables | Reporting only |
| Reference control | Exact recipe exception, readiness, exclusion from production models | Pending ordinary reference replicate count |
| Screened-hit follow-up | Existing screened chemistry can enter mechanical slots, with ordinary constraints and backups | Group 10 |
| Confirmation | Removed at user request | No scheduling or batch-count requirement |
| Replicate/run metadata | Compatible worksheet fields and workload manifest | Group 10 |
| Operational 1 mm endpoint | Versioned pure detector, plot/JSON CLI, reproducible supplementary ingestion | Pending detector settings; supplementary only |
| Model/noise/control audit | Seven forward variants including adaptive-noise sensitivity, diagnostic control shifts | Offline only |
| Shared multiple GPs | Unequal training sets, common posterior, sequential acquisition and explicit fallback | Offline only |
| Reporting | Acceptance/control/support-role diagnostics, historic metric preserved | Group 10 evidence |

## Important implementation decisions

Production GP fitting remains unchanged apart from excluding explicitly reference-role evidence and incompatible-definition mechanical labels. Historical rows lacking role/definition metadata retain their old interpretation. Supplementary force uses its own endpoint name and cannot enter the existing critical-load GP by accident.

The new workflow assembles the final slate after existing candidate generation/scoring. It reserves typed roles, retests and available rescues, then fills residual origin quotas with original scores while enforcing novelty and chemistry/diversity caps. Existing ordinary prediction/acquisition methods are not replaced. Final metadata updates role, ingredient, pair, support and prediction-label counts. Frozen Group 9 proposals bypass this wrapper.

When no recurring reference is ready, an already-scheduled legacy bootstrap anchor remains eligible and replaces one hit slot, preserving the two fresh positions. Once reference activation is ready, it supersedes that one-time anchor. This avoids silently removing a historical bootstrap-control mechanism merely because the new reference settings are pending.

Mechanical array results are operational nominal loads, not inferred individual fracture strengths. No event within a completed window is usable for the new operational endpoint, while incomplete/ambiguous traces are not exact labels. Legacy maximum-force results remain separate.

Existing mechanical R² reporting contained a missing `r2_score` import which appeared only after nonconstant mechanical data were present. The isolated end-to-end mechanical test exposed it; the import is fixed without changing the metric calculation.

The historical endpoint configuration hash test now checks the byte-identical original endpoint section while separately allowing the new reporting-only requirements prefix. Original data/proposal/metric hashes are not updated.

## Later decision report

Read `audit/audit_report.md` and its supporting CSVs before choosing any future production model. The audit reproduced mixed-source MAE 27.456 points. The bounded learned campaign GP yielded 14.989 in the historical replay, versus 15.854 for the existing-kernel campaign GP. All still have negative pooled R²; these are development comparisons, not proof of improved experimental discovery.

No shared-model, adaptive-noise, batch-adjustment, or new mechanical-target activation is implemented in the production selector. A later explicit user decision before the first full-mechanics proposal is required. No four-unknown exploration cap was introduced.

## Operational transition

1. Finish Group 9 under its frozen contract.
2. Ingest normally, adding `--skip-generate` if Group 10 settings are not ready.
3. Bring actual Group 9 evidence into the reviewed branch before finalizing a Group 10 proposal.
4. Review the settings register. Reference and detector capability may remain pending independently.
5. Generate a proposal using an explicit isolated output root first; all output paths, including total pool, must point there.
6. Activate the tested workflow for the next unstarted group. Never edit a frozen/started proposal in place.
7. Use raw reference monitoring, not normalization. Review the model audit separately before the first full-mechanics proposal.

`audit_group10.py --output-dir <explicit-audit-directory>` runs the offline historical comparison. `analyze_mechanical.py --help` describes explicit-source/column/output arguments. Stage 3's existing `--formulations`, `--observations`, `--output-dir`, `--total-candidate-pool`, and `--skip-generate` options support isolated verification and safe group boundaries.

## User simplification, workflow v2

Base medium is identical throughout the campaign, so it is not a reference setup field. One preparation supplies ordinary testing replicates. Reference `replicate_count` and `mechanical_replicates_per_formulation` replace the preparation/specimen planning hierarchy. Optional existing provenance columns and raw records remain readable. The confirmation feature, novelty/mechanical exceptions, and acceptance-status placeholder were removed. Test replicates are not treated as independent preparations by the offline uncertainty audit. The model decision boundary is now the first full-mechanics proposal, not a later group within that phase. Existing frozen manifests retain their recorded historical settings.

## Replicates from completed CSV (workflow v3)

No advance replicate counts are required. The completed round CSV supplies actual replicates; reports count ingested replicate IDs separately by formulation, batch and endpoint. An aggregate mean counts as one recorded measurement; underlying unrecorded replicates are not guessed. These are test replicates, not independent preparations. The reference is ready for new proposals from Group 10 on this development branch, with no replicate-count gate. No live Group 10 proposal has been generated; the live checkout still has Group 9 frozen. The reference receives one screen slot and a mechanical slot conditional on actual intact formation.

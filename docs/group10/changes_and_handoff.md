# Implementation handoff — Group 10 v4 / Group 11+ v5

The operator contract is [README.md](README.md), with the extraction rule in
[mechanical_sop.md](mechanical_sop.md). This ledger records endpoint and role
activation. The [round progression register](../multi_objective_round_progression.md)
also records GP choices, candidate search and full-phase allocation.

## Implementation and activation

| Component | Final implementation | Activation |
|---|---|---|
| Policy and proposal schema | Frozen Group 10 `group10_workflow_v4` / 4 retained; `group10_workflow_v5` / 5 makes the reference viability-only | v5 from Group 11 |
| Mechanical definition | `terminal_force_08mm_after_1N_v1`; force at +0.8 mm after sustained 1 N | Group 10 |
| Extraction and protocol | `terminal_force_v1` / `whole_patch_1N_08mm_v1`; raw CSV parsing, interpolation, quality status and provenance | Frozen Group 10 contract |
| Reference | Exact 2.5% v/v DMSO + 100 mM sucrose exception; exclude control viability from ordinary formulation-response labels and metrics; retain comparable mechanics as ordinary evidence | Group 10; Group 11 `control` uses viability references through a separate batch/reference component |
| Screened-hit mechanics | Measured viable/intact chemistry receiving characterization under the selected endpoint; stable formulation IDs | Group 10 |
| Experimental allocation | Group 10 unchanged; Group 11+ has four experimental mechanical positions, no control rank or repeat exemption | v5 from Group 11 |
| Replicates and workload | Counts from completed CSV rows; separate formulation allocation, attempts, interpretable outcomes and usable endpoint observations | Group 10 |
| Requirements | Null-safe thresholds and definition-aware, same-batch acceptance reporting | Reporting only |
| Confirmation scheduling | Removed; no cadence or independent-batch setup | Inactive |
| Production GP methodology | Campaign-only, control-informed batch-effect Matérn-5/2 viability GP; comparable-endpoint mechanical GP | Decision `group11-control-informed-user-v1`, effective Group 11; Group 10 retains its frozen pooled GP |
| Model/noise/control audit | Dynamic forward refits over observed rounds, definition-separated endpoint comparisons and viability-only control monitoring | Offline/diagnostic only |
| Shared-posterior acquisition | Approved independent endpoint posteriors supply qLogNEHVI | Finite-pool search from Group 11 under the version-1 decision |
| Continuous concentration refinement | Same GP and acquisition; ranked ingredient sets with mixture coverage; explicit failure handling | Version-2 `continuous_required`, hybrid/full phase, implemented in `98a3405` and `6b759c7` |
| Full-phase role allocation | One control, one mechanics follow-up, one retest/rescue, nine ordinary candidates; no ordinary origin quotas | `mechanics_enabled` under repair commit `95ecf9c` |
| Reporting | Definition-separated training/Pareto/prospective cohorts, control exclusion and placeholder-prediction exclusion | Group 10 evidence |
| Readiness | Read-only status plus fail-closed checks for complete settings, prior-round evidence and frozen protocol | Enforced from Group 10 |

## Compatibility and behavior

Frozen Group 9 proposal CSV/metadata keep their original bytes. A separate hash-bound
addendum applies the terminal endpoint during Group 9 validation/ingestion.
Historical rows without definition metadata retain the legacy maximum-force meaning.

The existing `critical_axial_load_N_per_needle` field is a compatibility key. Under the Group 9 terminal addendum and Group 10 definition it holds nominal terminal force per loaded needle, with total patch force stored separately. The existing stiffness key holds the terminal-window apparent secant and carries its formula in provenance. The mechanical training view selects one comparable definition for both metrics; old maxima/slopes remain outside the new target's cohort. Eligible viability/intact information remains usable. Phase thresholds are unchanged, but fewer comparable mechanical observations can postpone phase progression.

Slate assembly uses the scored candidate pool. Protected roles, retests and rescues
count toward the applicable slate/chemistry limits. Bootstrap/hybrid retain
residual source quotas; the repaired full phase uses 1/1/1/9 role allocation and
ranks ordinary origins together. Similarity/diversity constraints remain in force,
and allocation shortfalls stop proposal generation. Group 10 retains its frozen
reference treatment. Group 11+ reserves neither a mechanical control nor an anchor.

Round ingestion recomputes the endpoint from the raw file and frozen settings. It rejects mismatched manual values and reused raw files, preserves incomplete/invalid statuses, and never replaces missing mechanics with zero. A missing loaded count permits total-force reporting but supplies no nominal per-needle training label.

The proposal writer obtains specimen counts from result rows. The mechanical R²
report imports `r2_score` explicitly. Isolated workflow verification covers both behaviors.

## Evidence and recorded decisions

The [explained model audit](audit/explained_comparison.md) compares Groups 3–8 at
forward training cutoffs. It supplies decision evidence without promoting a model.
The Group 11 GP decision selects `control`; the endpoint contract and mechanical
phase gates do not replace that decision. Search mode is separately explicit in version 2.

Application viability and force thresholds remain unset. The force requirement must name its endpoint. No advance replicate count, separate base-medium setting, independent-preparation requirement or confirmation cadence is requested. Actual specimen count and raw paths come from the completed CSV; temperature/handling remain experimental protocol metadata.

See [verification.md](verification.md) for 105 passing tests, actual trace replay, data-hash checks and synthetic Group 9→10→11 verification. The [settings register](settings_register.md) lists resolved and remaining inputs. Follow [transition_group9_to_10.md](transition_group9_to_10.md) to incorporate actual Group 9 evidence and freeze the real Group 10 proposal.

## 2026-09-15 clarification

The user retains Group 10 as planned. No frozen or working campaign artifact was changed.
The reference is viability-only from Group 11; no recurring mechanical control or anchor
uses its four-test quota. The dynamic audit remains offline. See
[readiness and GP decisions](readiness_and_gp_decisions.md) for the architecture, staged-report
distinction, decision prerequisites and verification.


### Group 10 mechanical evidence clarification

The DMSO/sucrose row is a viability control only. Its valid Group 10 mechanical
result and measured same-batch viability/intact companions count as an ordinary
mechanical pair for phase progression, paired acquisition and observed trade-offs.
Control viability remains outside ordinary formulation-response labels and viability
metrics; the Group 11 `control` model uses it as reference/batch evidence.
Four valid Group 10 pairs plus four Group 9 pairs yield **8 pairs / 8
formulations / 2 batches**, meeting the unchanged hybrid gate for Group 11.
Group 10's planned tests stay unchanged; Group 11+ allocates no mechanical rank
to the recurring viability reference. Technical replicates still aggregate within
formulation/batch and do not inflate gate counts.

# Implementation handoff — Group 10 v4 / Group 11+ v5

The current operator contract is [README.md](README.md), with the exact extraction rule in [mechanical_sop.md](mechanical_sop.md). This ledger describes the final implementation. Superseded planning notes and preview slates are retained in Git history rather than duplicated in current guidance.

## Implementation and activation

| Component | Final implementation | Activation |
|---|---|---|
| Policy and proposal schema | Frozen Group 10 `group10_workflow_v4` / 4 retained; `group10_workflow_v5` / 5 makes the reference viability-only | v5 from Group 11 |
| Mechanical definition | `terminal_force_08mm_after_1N_v1`; force at +0.8 mm after sustained 1 N | Group 10 |
| Extraction and protocol | `terminal_force_v1` / `whole_patch_1N_08mm_v1`; raw CSV parsing, interpolation, quality status and provenance | Frozen Group 10 contract |
| Reference | Exact 2.5% v/v DMSO + 100 mM sucrose exception; monitor viability separately and exclude control viability from its GP; retain already collected mechanics as ordinary mechanical evidence | Configured for Group 10 |
| Screened-hit mechanics | Measured viable/intact chemistry receiving characterization under the selected endpoint; stable formulation IDs | Group 10 |
| Experimental allocation | Group 10 unchanged; Group 11+ has four experimental mechanical positions, no control rank or repeat exemption | v5 from Group 11 |
| Replicates and workload | Counts from completed CSV rows; separate formulation allocation, attempts, interpretable outcomes and usable endpoint observations | Group 10 |
| Requirements | Null-safe thresholds and definition-aware, same-batch acceptance reporting | Reporting only |
| Confirmation scheduling | Removed; no cadence or independent-batch setup | Inactive |
| Production GP methodology | Existing architecture, scaling, source treatment, fixed noise and acquisition retained | Any change requires a decision before full mechanics |
| Model/noise/control audit | Dynamic forward refits over observed rounds, definition-separated endpoint comparisons and viability-only control monitoring | Offline/diagnostic only |
| Multiple-GP acquisition | Shared endpoint-specific models, qLogNEHVI and explicit fallbacks | Offline only |
| Reporting | Definition-separated training/Pareto/prospective cohorts, control exclusion and placeholder-prediction exclusion | Group 10 evidence |
| Readiness | Read-only status plus fail-closed selection/ingestion checks for complete current settings, prior-round evidence and frozen protocol | Enforced from Group 10 |

## Compatibility and behavior

Frozen Group 9 proposal CSV/metadata keep their original bytes. A separate hash-bound addendum now applies the terminal endpoint when the newly staged Group 9 worksheet is validated/ingested. Historical rows without definition metadata retain the legacy maximum-force meaning.

The existing `critical_axial_load_N_per_needle` field is a compatibility key. Under the Group 9 terminal addendum and Group 10 definition it holds nominal terminal force per loaded needle, with total patch force stored separately. The existing stiffness key holds the terminal-window apparent secant and carries its formula in provenance. The mechanical training view selects one comparable definition for both metrics; old maxima/slopes remain outside the new target's cohort. Eligible viability/intact information remains usable. Phase thresholds are unchanged, but fewer comparable mechanical observations can postpone phase progression.

The final slate is assembled after ordinary candidate generation and scoring. Protected roles, retests and rescues count toward slate/chemistry limits. Residual ordinary-origin quotas and existing similarity/diversity limits remain enforced; conflicts fail before a proposal is written. Group 10 retains its historical reference/anchor treatment. Group 11+ reserves neither a mechanical control nor an anchor.

Round ingestion recomputes the endpoint from the raw file and frozen settings. It rejects mismatched manual values and reused raw files, preserves incomplete/invalid statuses, and never replaces missing mechanics with zero. A missing loaded count permits total-force reporting but supplies no nominal per-needle training label.

The proposal writer no longer requires the removed preset specimen-count setting. The mechanical R² report also includes the previously missing `r2_score` import. These fixes were exercised in isolated workflow verification.

## Evidence and next decisions

The [explained model audit](audit/explained_comparison.md) compares Groups 3–8 at forward training cutoffs. It identifies source treatment as the largest concern; no model is automatically promoted. GP/noise/batch-adjustment/acquisition changes remain subject to the user's decision before the first full-mechanics proposal. The endpoint decision is already effective for Group 10.

Application viability and force thresholds remain unset. The force requirement must name its endpoint. No advance replicate count, separate base-medium setting, independent-preparation requirement or confirmation cadence is requested. Actual specimen count and raw paths come from the completed CSV; temperature/handling remain experimental protocol metadata.

See [verification.md](verification.md) for 105 passing tests, actual trace replay, data-hash checks and synthetic Group 9→10→11 verification. The [settings register](settings_register.md) lists resolved and remaining inputs. Follow [transition_group9_to_10.md](transition_group9_to_10.md) to incorporate actual Group 9 evidence and freeze the real Group 10 proposal.

## 2026-09-15 clarification

The user retains Group 10 as planned. No frozen or working campaign artifact was changed.
The reference is viability-only from Group 11; no recurring mechanical control or anchor
uses its four-test quota. The dynamic audit remains offline. See
[readiness and GP decisions](readiness_and_gp_decisions.md) for the architecture, staged-report
distinction, decision deadlines and current verification.


### Group 10 mechanical evidence clarification

The DMSO/sucrose row is a viability control only. Its valid Group 10 mechanical
result and measured same-batch viability/intact companions count as an ordinary
mechanical pair for phase progression, paired acquisition and observed trade-offs.
Control viability remains outside standalone viability-GP training/ordinary viability
metrics. Four valid Group 10 pairs plus four Group 9 pairs yield **8 pairs / 8
formulations / 2 batches**, meeting the unchanged hybrid gate for Group 11.
Group 10's planned tests stay unchanged; Group 11+ allocates no mechanical rank
to the recurring viability reference. Technical replicates still aggregate within
formulation/batch and do not inflate gate counts.

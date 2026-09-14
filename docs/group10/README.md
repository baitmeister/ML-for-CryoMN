# Group 10 workflow release — current contract

Released on **`main`** by merge commit `3e54384` (PR #10 from `codex/v2-methodology-group10`). Current policy/schema: **`group10_workflow_v4` / 4**. Updated 2026-09-14.

**The endpoint changes from Group 10. Changes to GP methodology require a user decision after the audit, before the first full-mechanics proposal.** The configured mechanical endpoint is force at **+0.8 mm after a sustained 1 N trigger**. Endpoint and GP methodology have separate activation controls.

## Configured versus live

The reviewed code and configuration are live on local and remote `main`; they reserve the reference and apply the new endpoint from Group 10. This does not retroactively change Group 9's frozen proposal. The current campaign data still contain the blank Group 9 worksheet, observations end at Group 8, and no live Group 10 proposal has been frozen. The software release is live, but the Group 10 wet-lab start is not ready until Group 9 results are validated and ingested.

Run the read-only readiness check from the reviewed checkout, pointing it at the campaign of interest:

```sh
python3 src/08_multi_objective/04_report_campaign/check_group10_readiness.py --campaign-root '/Users/doggonebastard/Antigravity/ML for CryoMN'
```

The result distinguishes configured code, the active worksheet, completed evidence, and whether Group 10 is frozen with the latest rule. `--require-ready` returns exit code 2 if a live Group 10 start is not ready. See [transition instructions](transition_group9_to_10.md).

The production path is also fail-closed. Stage 02 refuses to create any Group 10-or-later proposal unless the reviewed Group 10 configuration/protocol is complete, activation remains fixed at Group 10, and the immediately preceding round has validated observations. Stage 03 refuses to ingest Group 10-or-later results unless the frozen proposal contains a complete effective configuration matching the live reviewed configuration. These checks run before proposal artifacts or observation tables are written.

## What starts at Group 10

- One of twelve screen slots is the recurring **2.5% v/v DMSO + 100 mM sucrose** reference. It gets one mechanical position conditional on actual intact formation. Its narrowly scoped DMSO ceiling exception does not enlarge the optimizer's ordinary domain. Recipe conversion uses 1.10 g/mL, purity 1.0 and 78.13 g/mol, approximately 0.352 M DMSO.
- With that reference, the four primary mechanical formulations are **one reference, one screened hit and two fresh candidates**. Ordered actual-intact backups remain. Retests and first-mechanics follow-up have different roles. A prior maximum-force measurement does not count as characterization under the new terminal definition.
- Reference measurements are stored and monitored, excluded from production GP training and ordinary prediction metrics, and never used for automatic normalization.
- The new force extractor runs automatically on `instron_file` during round ingestion. Fresh paired viability/intact/mechanics remain attached to their own formulation/batch. The loaded count comes from the result row; replicate totals are counted from completed CSV records rather than requested in advance.
- Requirements remain reporting-only and unset. Neither thresholds nor acceptance status change acquisition, the Pareto ranking, or hypervolume reference. Confirmation scheduling remains removed.

## Endpoint meaning and model boundary

The mechanical GP keeps the existing compatibility output key `critical_axial_load_N_per_needle`. For Group 10, that value means **nominal terminal force per loaded needle**, tagged `terminal_force_08mm_after_1N_v1`. Total whole-patch force is stored separately. It does not mean fracture force. See the [exact endpoint SOP](mechanical_sop.md).

Training includes comparable new-definition mechanical labels and all otherwise eligible viability evidence, including screening observations. Old mechanical maxima remain stored but are excluded from the Group 10 target's training set. This can postpone an evidence-gated mechanical phase; it does not modify the gate thresholds. Unsupported zero-placeholder mechanical predictions are not scored as genuine forecasts.

GP architecture, Matérn kernel, scaling, source treatment, fixed observation noise, acquisition behavior and phase thresholds are unchanged. The [explained audit comparison](audit/explained_comparison.md) and [audit report](audit/audit_report.md) remain decision evidence. No candidate GP is automatically promoted when full mechanics begins. Review and choose the methodology before the first full-mechanics proposal.

## Data flow and ownership

1. `group10_config.py` validates settings; `group10_selection.py` applies roles and freezes policy, endpoint and training cutoff/hash with the proposal.
2. `terminal_force.py` reads the original acquired trajectory and computes the fixed terminal quantity. `mechanical_events.py` preserves the historical drop detector for old frozen contracts and provides JSON/plot export.
3. `feedback.py` dispatches by frozen endpoint settings, validates raw provenance, and appends measurements. `mechanics_execution.py` separates formulation allocation, attempted CSV specimen rows, interpretable force results and usable paired endpoints.
4. The production observation filter retains one mechanical definition and excludes the reference. No raw observation is edited to perform that filtering.
5. `evaluation_metrics.py` separates endpoint definitions in prospective pools and observed evidence; `group10_reporting.py` adds raw total-force records, acceptance, reference monitoring, replicate counts and role cohorts.
6. `check_group10_readiness.py` inspects configuration and live artifacts without writing. `tests/verify_group10_endpoint_workflow.py` exercises a complete synthetic campaign in an explicit empty output directory.

Details: [settings register](settings_register.md), [change ledger](changes_and_handoff.md), [verification](verification.md), [transition instructions](transition_group9_to_10.md). The retained [synthetic workflow verification](endpoint08/synthetic_workflow_verification.json) validates the current contract; it is not an experimental recommendation.

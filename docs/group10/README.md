# Group 10 workflow release — current contract

The Group 10 proposal remains frozen under **`group10_workflow_v4` / schema 4**.
The user confirmed on 2026-09-15 that its planned mechanical candidates must stay unchanged.
From **Group 11**, **`group10_workflow_v5` / schema 5** makes the recurring reference
strictly viability-only. See [the current plan and GP decision guide](readiness_and_gp_decisions.md).

Group 9 is ingested; the active worksheet is Group 10. The configured endpoint remains
force at **+0.8 mm after a sustained 1 N trigger**. No GP architecture/noise switch has been made.

Run the read-only Group 10 start check:

```sh
python3 src/08_multi_objective/04_report_campaign/check_group10_readiness.py --require-ready
```

The round loader preserves the exact Group 10 v4 effective configuration while applying v5
from Group 11. Selection and ingestion validate the appropriate round-specific settings
and frozen protocol. This start checker is not a full-mechanics readiness certificate.

## Reference allocation and endpoint workflow

- One of twelve screen slots is the recurring **2.5% v/v DMSO + 100 mM sucrose** reference. Its already-frozen Group 10 mechanical position remains conditional on actual intact formation. **From Group 11 it has no mechanical position or backup rank.** Its narrowly scoped DMSO ceiling exception does not enlarge the optimizer's ordinary domain. Recipe conversion uses 1.10 g/mL, purity 1.0 and 78.13 g/mol, approximately 0.352 M DMSO.
- Group 10 retains **one reference, one screened hit and two fresh candidates**. From Group 11 there is **no mechanical control**: the usual four positions are one eligible screened hit and three fresh candidates. Ordered actual-intact backups remain. Retests and first-mechanics follow-up have different roles. A prior maximum-force measurement does not count as characterization under the new terminal definition.
- Reference viability measurements are stored and monitored separately, excluded from viability GP training and ordinary viability metrics. Any already collected comparable mechanical labels are ordinary mechanical evidence, never a mechanical control or normalization basis.
- The new force extractor runs automatically on `instron_file` during round ingestion. Fresh paired viability/intact/mechanics remain attached to their own formulation/batch. The loaded count comes from the result row; replicate totals are counted from completed CSV records rather than requested in advance.
- Requirements remain reporting-only and unset. Neither thresholds nor acceptance status change acquisition, the Pareto ranking, or hypervolume reference. Confirmation scheduling remains removed.

## Endpoint meaning and model boundary

The mechanical GP keeps the existing compatibility output key `critical_axial_load_N_per_needle`. For Group 9 terminal ingestion and Group 10+, that value means **nominal terminal force per loaded needle**, tagged `terminal_force_08mm_after_1N_v1`. Total whole-patch force and apparent secant stiffness are stored separately. It does not mean fracture force. See the [exact endpoint SOP](mechanical_sop.md).

Training includes comparable new-definition mechanical labels and all otherwise eligible viability evidence, including screening observations. Old mechanical maxima remain stored but are excluded from the Group 10 target's training set. This can postpone an evidence-gated mechanical phase; it does not modify the gate thresholds. Unsupported zero-placeholder mechanical predictions are not scored as genuine forecasts.

GP architecture, Matérn kernel, scaling, source treatment, fixed observation noise, acquisition behavior and phase thresholds are unchanged. The [explained audit comparison](audit/explained_comparison.md) and [audit report](audit/audit_report.md) remain decision evidence. No candidate GP is automatically promoted when full mechanics begins. Review and choose the methodology before the first full-mechanics proposal.

## Data flow and ownership

1. `group10_config.py` validates settings; `group10_selection.py` applies roles and freezes policy, endpoint and training cutoff/hash with the proposal.
2. `terminal_force.py` reads the original acquired trajectory and computes the fixed terminal quantity. `mechanical_events.py` preserves the historical drop detector for old frozen contracts and provides JSON/plot export.
3. `feedback.py` dispatches by frozen endpoint settings, validates raw provenance, and appends measurements. `mechanics_execution.py` separates formulation allocation, attempted CSV specimen rows, interpretable force results and usable paired endpoints.
4. The production observation filter retains one mechanical definition and excludes the viability-control screening labels, while retaining any comparable mechanical labels already measured on that recipe. No raw observation is edited to perform that filtering.
5. `evaluation_metrics.py` separates endpoint definitions in prospective pools and observed evidence; `group10_reporting.py` adds raw total-force records, acceptance, reference monitoring, replicate counts and role cohorts.
6. `check_group10_readiness.py` inspects configuration and live artifacts without writing. `tests/verify_group10_endpoint_workflow.py` exercises a complete synthetic campaign in an explicit empty output directory.

Details: [settings register](settings_register.md), [change ledger](changes_and_handoff.md), [verification](verification.md), [transition instructions](transition_group9_to_10.md). The retained [synthetic workflow verification](endpoint08/synthetic_workflow_verification.json) records the earlier v4 endpoint verification; it is not an experimental recommendation.


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

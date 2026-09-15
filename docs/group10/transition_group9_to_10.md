# Group 9 to Group 10 transition

## Preserve the ongoing experiment

The authoritative checkout is `/Users/doggonebastard/Antigravity/ML for CryoMN`. The reviewed implementation was merged to local and remote `main` by commit `3e54384`; the former implementation branch was `codex/v2-methodology-group10`.

Group 9's original frozen proposal CSV and selection metadata remain unchanged. A separate `mechanical_endpoint_addendum.json` hashes both artifacts and activates `terminal_force_08mm_after_1N_v1` specifically at Group 9 validation/ingestion. Existing historical maxima remain unchanged; only the six newly staged Group 9 traces receive terminal-method labels.

## Before starting Group 10

1. Review the staged Group 9 worksheet: 48 viability samples, 12 formulation-level intact decisions and six mechanical traces.
2. Run `run_round.py --validate-only` without an output directory for in-memory parsing and console QC. Do not generate mechanical graphs or source/metadata bundles before ingestion. This exits before archival, canonical table updates, reports, selection or campaign advancement.
3. After explicit approval to progress the campaign, run the normal Group 9 update from the reviewed `main` checkout; this ingests the results and generates the completed-round reports, including mechanical graphs. Do not initialize or rebuild the campaign database.
4. Keep one authoritative campaign database in the reviewed `main` checkout. Do not run the transition from a stale secondary worktree whose default data paths differ.
5. Check the reference, primary roles, ordered backups and effective settings. Group 10 metadata must say `group10_workflow_v4`, proposal schema 4 and `terminal_force_08mm_after_1N_v1`. GP/noise revision flags remain false; endpoint revision is true.
6. Freeze the actual Group 10 proposal using reviewed `main`. Run `check_group10_readiness.py --campaign-root <that-checkout> --require-ready`. Start only when the intended active worksheet and frozen proposal agree. Stage 02 hard-stops before writing if Group 9 evidence or any required Group 10 setting/protocol is missing.

## During Group 10 and subsequent updates

Each compression specimen gets a CSV result row with the original Instron path and actual loaded-needle count when known. Replicate numbers are obtained from these rows; no advance count is configured. Include fresh paired viability/intact data for follow-up/control specimens as appropriate. Mark a lost attempt explicitly without inventing a force. Calculated compatibility fields may be blank or must reproduce from the raw trace.

The first qualifying run of >=1 N lasting >=0.100 s with >=5 samples sets the origin. Extract force at +0.800 mm by adjacent-sample interpolation. There is no fracture/drop alternative. The test must reach that displacement; a short trace does not become a low-force exact result.

The round update verifies that the frozen settings are complete and match the live reviewed configuration before it writes observations. It then ingests total and nominal force with provenance, reports attempted versus interpretable workloads, and generates the next proposal. The control is excluded from the production GP, no normalization occurs, and old-definition maxima cannot enter the new mechanical target.

## Before the first full-mechanics proposal

Review the existing model/noise/acquisition audit and explicitly choose any GP methodology change. Phase entry alone does not perform that switch. The endpoint decision has already been made for Group 10 and is independent of this later modeling decision. Mechanical phase gates count comparable paired evidence; retaining the old labels in the archive does not make them comparable to the new endpoint.


## Group 11 viability-only reference clarification (2026-09-15)

Group 10 remains unchanged at the user's request. From Group 11, workflow v5/schema 5
retains the reference for viability only and allocates no mechanical rank to it.
There is no mechanical control or reference-based force normalization. Already measured
comparable mechanics remain endpoint evidence. See [the current guide](readiness_and_gp_decisions.md).


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

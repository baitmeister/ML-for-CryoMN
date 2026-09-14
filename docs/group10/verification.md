# Endpoint v4 verification — 2026-09-14

Branch: `codex/v2-methodology-group10`. Endpoint: `terminal_force_08mm_after_1N_v1`.

## Current verification

**All 102 tests passed in 21.399 seconds in the separate branch worktree.** `git diff --check` passed. The original 90-test suite and 12 endpoint/readiness tests cover the new definition, historical dispatch, production cohort separation and data protection. All test outputs used isolated directories; no live artifact is an output of these tests.

Coverage includes sustained trigger equality/duration/sample count, pretrigger spikes and motion, no reset after a later drop, terminal interpolation, postterminal maxima/nonfinite values, incomplete windows, missing triggers, acquisition gaps, reversals, units rows, strain-column ambiguity, loaded counts, immutable settings, automatic ingestion, manual-value conflicts, duplicate raw files, Group 9 legacy parsing, definition-separated training/Pareto/metrics and configured-versus-live readiness. No live artifact is an output of these tests.

The optional BoTorch fused C++ acceleration cannot write its normal user cache in the sandbox. Shared-GP tests pass using BoTorch's supported pure-Python implementation. This is unrelated to the mechanical endpoint change; no acquisition methodology was activated or replaced.

## Six actual Instron traces

All six traces complete the +0.8 mm window, and extraction matches the prior exploratory values to within 1e-10 N:

| Raw file suffix | Formulation / replicate | Terminal total force, N |
|---|---|---:|
| 1 | #9 / 1 | 30.5224666667 |
| 2 | #12 / 1 | 21.0671666667 |
| 3 | #5 / only available | 21.5089500000 |
| 4 | #8 / only available | 15.1354250000 |
| 5 | #9 / 2 | 17.7165333333 |
| 6 | #12 / 2 | 12.1444666667 |

The files were found at their updated `Multi-objective/Group9/group_9-1.is_comp_Exports` location. Per-file hashes, exact settings, trigger/terminal coordinates and force values are preserved in [actual_curve_replay.json](endpoint08/actual_curve_replay.json). No loaded-needle count was invented for this replay. These are **Group 9 development data**, not a prospective evaluation of the chosen endpoint. The standalone CLI also produced an annotated figure, inspected visually, in an explicit temporary output directory.

## Complete isolated workflow

`tests/verify_group10_endpoint_workflow.py --output-dir <new-empty-directory>` is reproducible and exercises actual code paths with **synthetic** outcomes:

1. Copy the Group 9 frozen proposal and data into isolation; fill synthetic results and validate/ingest under the original Group 9 contract.
2. Generate/freeze Group 10 with the new definition, reference readiness and CSV-derived replicate accounting.
3. Verify twelve screens and four primary mechanical formulations: one reference, one screened hit, two fresh candidates.
4. Submit eight synthetic compression attempts across four formulations: six completed force results, one incomplete curve and one lost/no-raw attempt. Extract and ingest under frozen settings, archive, and generate round/campaign reports.
5. Verify control exclusion, separation from old maximum labels, placeholder-forecast exclusion, four usable formulation-batch observations, and a newly frozen Group 11 proposal.
6. Verify all 516 source data/results files in the isolated source checkout remain unchanged by the workflow.

The final complete run passed. See [synthetic_workflow_verification.json](endpoint08/synthetic_workflow_verification.json). Temporary run directory: `/private/tmp/cryomn-endpoint08-final-integration`. Synthetic results and slates are not copied into either live campaign database.

## Protection and live readiness

All **1050 protected file hashes** matched: data/results in both main and the development worktree, plus all six actual raw Instron files. [Verification record](endpoint08/protected_files.json). No raw data, Group 9 worksheet/proposal/predictions, archive, or legacy-lane file changed. The targeted preliminary cleanup and earlier audit remain documented in the historical ledger below.

The [read-only readiness result](endpoint08/live_readiness.json) shows main still has **ROUND_009**, blank result inputs, observed data through Group 8, no Group 10 code configuration and no frozen Group 10 proposal with the new endpoint. **Configured/tested branch policy is ready; starting the actual Group 10 experiment is not yet ready.** Complete the actual Group 9 update and use the reviewed branch for the actual Group 10 freeze.

## Limits

- This endpoint is a protocol-specific terminal force, not fracture strength, Young's modulus or demonstrated penetration capability.
- Unknown loaded count prevents a per-needle model label, not extraction of total patch force.
- Thermomechanical handling and specimen temperature are not recoverable from these CSV columns.
- Historical maximum-force labels are retained outside the new mechanical cohort. Until comparable new evidence accrues, mechanical phase progression can be delayed.
- GP methodology, adaptive noise, normalization/batch adjustment and shared multiple-GP production acquisition remain unchanged/deferred. The prior Rounds 3–8 audit remains valid as historical development evidence; it was not rerun or relabeled as a prospective result.

## Historical verification ledger

The entries below describe earlier workflow versions. Their pending-replicate, 1 mm and supplementary-only statements are superseded by the current SOP and settings register.


Verification date: 2026-09-08. Branch: `codex/v2-methodology-group10`.

## Automated regression

Command: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests`.
**88 tests passed**, including the 75 pre-existing tests and 13 additional methodology tests. Final run: 21.855 seconds. `git diff --check` passed.

The historical tests exercise frozen-proposal validation, ingestion, selection and metric contracts. The Group 9 archive remains byte-identical. New tests cover null requirements, incompatible definitions, activation gates, exact reference recipe/domain boundary and production exclusion, replicate independence, contact-relative endpoints, exactly 10% drops, persistence against spikes, incomplete/gapped/unloading curves, arrays, post-event maxima, and shared unequal-data GP acquisition. A supplementary ingestion test recomputes the raw curve, rejects altered results, stores the new operational value of 1 N alongside the unchanged legacy maximum of 4 N, and excludes its reference-role observations from production training.

BoTorch's optional fused C++ kernel could not use the sandboxed user cache. The actual shared-GP tests passed using BoTorch's supported pure-Python acquisition implementation. This affects acceleration, not acquisition identity.

## Isolated campaign

All writes used temporary copies/output roots. Starting with the Group 8 evidence snapshot, generated an explicit synthetic Group 10 preview with 2,000 candidates, froze its proposal, filled synthetic viability/intact/mechanical outcomes, validated and ingested them with next-generation skipped, produced campaign/round reports, then generated Group 11 from that isolated database. One primary formulation failed actual intact; ordered backup promotion supplied the four mechanical formulations. Synthetic observations are not scientific evidence and are absent from the repository data.

A second preview used explicitly synthetic reference reagent/base-medium/replicate settings. It reserved one reference, one screened hit, and two fresh primary mechanical formulations within the twelve-row slate. These synthetic settings were injected only for the preview; committed reference settings remain pending. Role tables and execution evidence are in `preview/`. They are demonstrations, not the actual post-Group-9 experimental recommendation.

The first short-pool preview exposed an infeasible origin quota. The selector now reserves available rescues before allocating remaining ordinary origins; unsatisfied quotas still fail before writing a proposal. The completed-mechanics end-to-end test exposed a pre-existing missing R² import, now fixed. Both scenarios were rerun successfully.

## Data protection

All **269** baseline data/frozen-artifact hashes matched in both the original checkout and the development worktree after testing. The original checkout is clean following the specifically requested preliminary cleanup. No raw data, completed worksheet, Group 9 proposal, or legacy lane was edited. See `baseline_manifest.json` for identities and dependencies.

## Scientific and rollout limitations

- No production GP, noise, acquisition, or mechanical-target revision was activated.
- The offline audit uses Rounds 3–8 only; no Group 9 outcome was inspected. Replay compares prediction, not counterfactual discovery policies.
- Reference replicate count and detector settings remain unresolved. They disable their dependent capabilities independently.
- Detector synthetic tests establish software behavior; instrument traces are still needed to choose contact, absolute noise, and persistence settings. No imaging validation is required or claimed.
- Repeated real reference groups and adequate mechanical outcomes do not yet exist for empirical batch-adjustment/new-target acquisition comparisons. These are explicitly pending in the audit; software smoke tests are not substitutes.
- The separate confirmation feature was subsequently removed at user request; acceptance reports measured requirements only.
- Incorporate actual completed Group 9 data and regenerate an isolated preview before any live Group 10 activation.

## Workflow v2 simplification verification

After removing the separate base-medium/preparation planning requirements and confirmation feature, the full suite passed **89 tests in 22.756 seconds**. The additional regression verifies that an ordinary reference replicate count alone completes reference readiness, invalid replicate counts fail validation, and the decision boundary is the beginning of full mechanics. All 269 protected hashes still match in both checkouts. The earlier synthetic previews above are historical workflow-v1 evidence; they were not regenerated or relabeled as workflow-v2 proposals.

## Workflow v3: no preset replicate counts

All 90 tests passed in 23.381 seconds. The new report test verifies endpoint-specific CSV-derived counts and repeated-ID deduplication. Reference readiness no longer depends on a supplied replicate count. Both checkouts retain all 269 protected hashes. The live proposal remains Group 9; no live Group 10 slate was generated.

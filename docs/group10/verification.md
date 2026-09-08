# Verification and limitations

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

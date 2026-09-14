# Endpoint v4 verification — 2026-09-14

Branch: `codex/v2-methodology-group10`. Endpoint: `terminal_force_08mm_after_1N_v1`.

## Current verification

**All 102 tests passed in 21.399 seconds in the separate branch worktree.** `git diff --check` passed. The suite covers historical workflow behavior and includes 12 endpoint/readiness tests for the new definition, historical dispatch, production cohort separation and data protection. All test outputs used isolated directories; no live artifact is an output of these tests.

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

The files were found at their updated `Multi-objective/Group9/group_9-1.is_comp_Exports` location. Per-file hashes, exact settings, trigger/terminal coordinates and force values are preserved in [actual_curve_replay.json](endpoint08/actual_curve_replay.json). No loaded-needle count was invented for this replay. These are **Group 9 development data**, not a prospective evaluation of the chosen endpoint. The standalone CLI also produced an annotated figure that was inspected visually. The replay JSON is the retained extraction evidence; temporary plots are not repository dependencies.

## Complete isolated workflow

`tests/verify_group10_endpoint_workflow.py --output-dir <new-empty-directory>` is reproducible and exercises actual code paths with **synthetic** outcomes:

1. Copy the Group 9 frozen proposal and data into isolation; fill synthetic results and validate/ingest under the original Group 9 contract.
2. Generate/freeze Group 10 with the new definition, reference readiness and CSV-derived replicate accounting.
3. Verify twelve screens and four primary mechanical formulations: one reference, one screened hit, two fresh candidates.
4. Submit eight synthetic compression attempts across four formulations: six completed force results, one incomplete curve and one lost/no-raw attempt. Extract and ingest under frozen settings, archive, and generate round/campaign reports.
5. Verify control exclusion, separation from old maximum labels, placeholder-forecast exclusion, four usable formulation-batch observations, and a newly frozen Group 11 proposal.
6. Verify all 516 source data/results files in the isolated source checkout remain unchanged by the workflow.

The final complete run passed. See [synthetic_workflow_verification.json](endpoint08/synthetic_workflow_verification.json). The verification script regenerates the complete synthetic campaign in a caller-specified empty directory; the summary is retained in the repository. Synthetic results and slates are not copied into either live campaign database.

## Protection and live readiness

All **1050 protected file hashes** matched: data/results in both main and the development worktree, plus all six actual raw Instron files. [Verification record](endpoint08/protected_files.json). No raw data, Group 9 worksheet/proposal/predictions, archive, or legacy-lane file changed. The original campaign baseline and dependency versions are retained in [baseline_manifest.json](baseline_manifest.json).

The [read-only readiness result](endpoint08/live_readiness.json) shows main still has **ROUND_009**, blank result inputs, observed data through Group 8, no Group 10 code configuration and no frozen Group 10 proposal with the new endpoint. **Configured/tested branch policy is ready; starting the actual Group 10 experiment is not yet ready.** Complete the actual Group 9 update and use the reviewed branch for the actual Group 10 freeze.

## Limits

- This endpoint is a protocol-specific terminal force, not fracture strength, Young's modulus or demonstrated penetration capability.
- Unknown loaded count prevents a per-needle model label, not extraction of total patch force.
- Thermomechanical handling and specimen temperature are not recoverable from these CSV columns.
- Historical maximum-force labels are retained outside the new mechanical cohort. Until comparable new evidence accrues, mechanical phase progression can be delayed.
- GP methodology, adaptive noise, normalization/batch adjustment and shared multiple-GP production acquisition remain unchanged/deferred. The prior Rounds 3–8 audit remains valid as historical development evidence; it was not rerun or relabeled as a prospective result.

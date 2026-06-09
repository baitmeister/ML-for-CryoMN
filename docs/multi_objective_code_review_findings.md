# Multi-Objective Code Review Findings

Environment validated during review:
- Homebrew Python `3.13.12`
- focused v2 pytest suite passing after the changes in this branch

This document consolidates the earlier architecture review with the newly
confirmed runtime and dependency findings.

## 1. Availability rules do not match the current campaign

Problem:
- The live selection config allowed `acetamide_M` and blocked `trehalose_M`,
  which was the opposite of the intended campaign state.

Evidence:
- `config_v2/availability.yaml`
- generated Stage 02 slate included acetamide-containing candidates while
  trehalose was excluded

Impact:
- The optimizer could recommend candidates that are not practically makeable
  for the current campaign.

Main recommendation:
- Treat `acetamide_M` as temporarily unavailable and `trehalose_M` as available
  in `config_v2/availability.yaml`.

Alternatives:
- Add CLI availability overrides for one-off runs.
- Support multiple campaign-specific availability profiles.

Tradeoff summary:
- The config update is the safest default and keeps the campaign state visible
  in one place. Profiles or CLI overrides are more flexible, but they raise the
  chance of silent operator mismatch between runs.

## 2. The old slate generator was viability-first, not truly multi-objective

Problem:
- The previous design prefiltered candidates using a viability-only shortlist,
  then applied mechanics logic inside that shortlist.

Evidence:
- `src/08_multi_objective/helper/selection.py`
- prior Stage 02 summary behavior

Impact:
- Mechanically promising candidates could be eliminated before they were ever
  compared on the joint objective.

Main recommendation:
- Score the full candidate pool with the active objective policy first, then
  build the 12-row wet-lab slate directly from that ranking.

Alternatives:
- Use a larger viability prescreen before mechanics-aware ranking.
- Reserve a fixed mechanics-bypass quota inside the old two-stage design.

Tradeoff summary:
- Full-pool scoring is the cleanest match to the intended optimization logic.
  The alternatives reduce code churn, but they preserve a structural viability
  bias.

## 3. Early rounds need explicit phase gating because mechanics arrives late

Problem:
- Real mechanical labels may not be supplied until roughly iterations 5 or 6,
  so early rounds do not have enough evidence for honest mechanics-aware
  acquisition.

Evidence:
- `src/08_multi_objective/helper/selection.py`
- `src/08_multi_objective/helper/models.py`
- workflow expectations from the project discussion

Impact:
- Using predicted mechanical load too early would create false confidence and
  could distort candidate ranking.

Main recommendation:
- Use an automatic data-driven phase selector:
  - `screening_only` for early rounds
  - `mechanics_enabled` once enough paired viability + critical load evidence
    exists across enough formulations and batches

Alternatives:
- Manual phase switching by the user.
- Weak mechanical prior in early rounds despite missing real labels.

Tradeoff summary:
- Automatic phase selection is more robust than iteration-based switching
  because data sufficiency, not round number, is the real transition signal.

## 4. Update, evaluate, and generate were not enforced in order

Problem:
- The previous workflow allowed Stage 02 generation to run even if the last
  round had not yet been ingested and evaluated.

Evidence:
- separate Stage 02, 03, and 04 entry points without orchestration

Impact:
- A user could accidentally generate the next slate from stale state.

Main recommendation:
- Add one supported orchestration command that advances a round in order:
  1. ingest
  2. evaluate
  3. validate readiness
  4. generate

Alternatives:
- Keep separate scripts and add a Stage 02 preflight guard.
- Improve docs and warnings only.

Tradeoff summary:
- Orchestration makes the intended lifecycle a system rule instead of a README
  rule. The alternatives are lighter, but they preserve more operator error
  risk.

## 5. Formulation-wide means hid batch variance and unstable observations

Problem:
- The old training frame collapsed observations to one formulation-wide mean per
  endpoint, hiding batch-to-batch disagreement.

Evidence:
- previous aggregation behavior in `src/08_multi_objective/helper/models.py`

Impact:
- The model could not distinguish stable evidence from unstable or off-trend
  batches, and it had no principled way to ask for confirmation.

Main recommendation:
- Preserve new validation rounds as separate batch-level observations and add a
  `retest_priority` policy for off-trend or conflicting batches.

Alternatives:
- Train on every raw replicate row.
- Keep formulation-level weighted aggregation with recency/variance-aware noise.

Tradeoff summary:
- Batch-level observations are the best fit for expensive experiments with few
  technical replicates. Full replicate-level training preserves maximum detail,
  while formulation-level aggregation is simpler but hides the exact effect we
  want the model to notice.

## 6. `PyYAML` was used but not declared

Problem:
- The v2 config loader imported `yaml`, but `requirements.txt` did not declare
  `PyYAML`.

Evidence:
- `src/08_multi_objective/helper/config.py`
- focused v2 tests initially failed during collection on a fresh Python 3.13
  environment

Impact:
- New environments could fail before the v2 workflow even started.

Main recommendation:
- Add `PyYAML>=6.0` to `requirements.txt`.

Alternatives:
- Pin the exact tested `PyYAML` version.
- Move YAML parsing behind an optional install group.

Tradeoff summary:
- A compatible minimum range is enough here. Exact pinning is more reproducible
  but adds unnecessary maintenance for a stable dependency.

## 7. Pandas 3 strict dtype assignment broke the Instron updater

Problem:
- `import_instron.py` wrote integers/floats into columns that had been
  initialized as strings, which fails under newer pandas string dtypes.

Evidence:
- `src/08_multi_objective/helper/instron.py`
- regression reproduced under Homebrew Python `3.13.12` with pandas 3.x

Impact:
- The helper for filling `next_round_candidates.csv` from Instron files could
  fail on a supported modern environment.

Main recommendation:
- Separate text and numeric wet-lab result columns, normalize them explicitly,
  and write numeric metrics into numeric-compatible dtypes.

Alternatives:
- Cast all writable columns to `object`.
- Temporarily cap `pandas<3`.

Tradeoff summary:
- Proper dtype normalization fixes the real bug and keeps modern pandas
  compatible. The alternatives are either less type-safe or just defer the
  problem.

## Dependency policy

Chosen policy:
- keep compatible minimum ranges
- avoid unnecessary upper bounds
- add focused regression tests when a newer interpreter or dependency major
  exposes stricter behavior

Rationale:
- This keeps the environment flexible enough for modern Python 3.13 installs
  without overcommitting to exact pins or prematurely capping working future
  versions.

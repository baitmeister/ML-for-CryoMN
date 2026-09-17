# GP comparison workflow: implementation and operating instructions

Implemented 2026-09-16. Group 10 remains frozen. No production GP has been selected
on the user's behalf and no Group 11 proposal has been generated.

## Evidence ready now

- [Historical forward comparison](audit/batch_gp_20260916/README.md): Groups 3–9,
  84 viability outcomes, each fitted using preceding data only.
- [Frozen Group 10 forecasts](../../results/multi_objective_v2/rounds/ROUND_010/gp_comparison/README.md):
  pre-outcome challenger predictions made after slate selection. The original proposal's
  predictions remain authoritative for what selected Group 10.

The Current GP and four Proposed GP comparisons include campaign-only, a matched fitted
residual-noise comparison, batch effects, and batch effects with separate control
calibration. The extra matched-noise comparison prevents attributing noise changes to the
batch structure. These are not four alternative experimental proposals.

Historical equal-Group MAE: Current GP 26.72 points; campaign-only 15.18; campaign-only
with fitted residual noise 14.89; batch model 15.63. Batch-model future coverage is 89.3%,
but the matched-noise model also reaches 89.3% with lower error. This is not evidence to
promote the batch model automatically. Control history in this snapshot is insufficient
for an independently supported calibration claim; its results currently match the batch
model on ordinary outcomes. The reports label this limitation.

## Statistical implementation

Formulation–batch means are the modeled labels. Technical replicates are not treated as
independent biological batches; no unverified s²/n reduction is applied. Input scaling,
formulation kernel (Matérn 2.5, unit scaled length) and target signal scale are matched
across campaign challengers. The additive batch model estimates batch SD in [0.01,50]
and total residual SD in [0.5,30] percentage points by bounded maximum likelihood.
Recorded noise is not added again when total residual noise is fitted. Batch offsets
share a zero-centered normal distribution. The separate control baseline has a normal
prior centered at the ordinary campaign mean with SD 50 points. Sparse calibration is
therefore explicitly prior-sensitive. Production control calibration requires at least
two preceding control batches; this is a minimum identifiability screen, not proof of
adequate calibration. No control is required for either non-calibrated Proposed GP.

This is empirical Bayes, not full Bayesian integration over model parameters. Intervals
condition on fitted parameters. Gaussian intervals can extend outside 0–100%; retain
those diagnostics rather than silently clipping. The broader logit sensitivity remains
in the earlier audit. All results require future-batch evaluation.

Current GP snapshots reproduce the fitted sklearn mean/covariance and transformations.
The Proposed GP acquisition bundle provides full cross-candidate formulation covariance
and independent objective outputs. Future batch offsets and residual noise are excluded
from acquisition outputs, but their effect on uncertainty about formulation response
remains. The mechanical model uses the Current GP's comparable endpoint posterior;
no viability noise floor is applied to force. Shared acquisition uses genuine BoTorch
finite-pool qLogNEHVI with cache_root=False. There is no heuristic fallback.

## Commands and state transitions

From the project root, the comparison entrypoint is:

```sh
python3 src/08_multi_objective/04_report_campaign/compare_gp_strategies.py audit --output /private/tmp/new-gp-audit
python3 src/08_multi_objective/04_report_campaign/compare_gp_strategies.py freeze --batch ROUND_010 --output /private/tmp/new-gp-freeze
python3 src/08_multi_objective/04_report_campaign/compare_gp_strategies.py evaluate --frozen results/multi_objective_v2/rounds/ROUND_010/gp_comparison --output /private/tmp/new-gp-evaluation
python3 src/08_multi_objective/04_report_campaign/compare_gp_strategies.py render --output docs/group10/audit/batch_gp_20260916
```

Audit/freeze/evaluate require new output directories. Do not repeat the Group 10 freeze
as a replacement for the already saved evidence. Rendering reads only saved evidence.

After Group 10 finishes, ingest with `--skip-generate`. The normal round workflow evaluates
its saved challenger models automatically and writes `reports/gp_comparison` inside the
round. Predictions are verified against immutable snapshots and before-batch forecasts.
Reference-conditioned evaluations reveal only the predeclared reference/retest subset;
these observations are excluded from the matched test set. Model parameters stay fixed.

Before generating Group 11, the user must choose a strategy and complete a versioned
JSON decision. Pass it with `--gp-strategy-decision PATH` to either round execution or
candidate selection. Alternatively, put it at `results/multi_objective_v2/gp_strategy_decision.json`.
No such active decision is created by this implementation.

Supported viability IDs: `current`, `campaign`, `campaign_noise`, `batch`, `control`.
Use acquisition `current_paired_qlognehvi` only with `current`; otherwise use
`shared_qlognehvi_finite_pool`. Mechanical configuration is `current_comparable_endpoint`.
The objective is `expected_formulation_response`. New selection acceptance thresholds
are not silently introduced: leave `acceptance_thresholds` null unless separately
implemented. Application acceptance criteria can be documented independently.

Example structure (intentionally not an approved decision):

```json
{
  "version": 1,
  "model_version": "batch_gp_v1",
  "decision_id": "",
  "chosen_by": "",
  "decided_at": "",
  "effective_round": 11,
  "viability_strategy": "batch",
  "mechanical_strategy": "current_comparable_endpoint",
  "acquisition": "shared_qlognehvi_finite_pool",
  "objective": "expected_formulation_response",
  "evidence": [],
  "acceptance_thresholds": null
}
```

Blank provenance is rejected. Archive the choice with each generated proposal. Reusing
a decision ID with changed content is rejected. Strategies do not change automatically.
Group 11 generation freezes alternative-model predictions for that one official slate.

## Reporting

Five new figures use the production transparent 300-dpi PNG + source CSV + metadata
contract: strategy summary, strategy by batch, reference-conditioning comparison,
batch-effect estimates, and official-recipe prediction comparison. Strategy and interval
labels are attached to existing production views; historical plots are not overwritten
with challenger predictions. Campaign report regeneration also redraws saved comparison
evidence without fitting. Observed trade-offs and mechanical figures retain raw values.

The summary weights Groups equally. Per-Group metrics include counts, MAE, RMSE, bias,
R², rank correlation, future coverage and interval width. Historical, pre-outcome challenger
and official production evidence remain separately labeled. No aggregate winner score
is computed. Existing frozen production reports remain separate from challenger reports.

## Remaining user-controlled steps

1. Complete Group 10 wet-lab work and supply its results/QC.
2. Ingest and review the frozen-prediction comparison.
3. Record the selected GP/acquisition strategy.
4. Generate and test the single official Group 11 proposal.
5. Review accumulated prospective evidence after Groups 11–12; do not switch merely
   because one model wins one round. Verify phase gates and application thresholds separately.

## Verification completed

The focused suite passed 59 tests (plus two subtests). An isolated synthetic Group 10
ingestion and Group 11 generation also passed with an explicit test-only batch-GP
decision: eight eligible pairs, hybrid phase, four mechanical primaries, viability
control unranked for mechanics, and one official test slate with frozen alternatives.
No synthetic outcomes or test decision were copied into live campaign tables.

BoTorch used its genuine Python qLogNEHVI implementation because the optional compiled
speed extension could not write to its cache. This is the same acquisition, not the
removed heuristic fallback.

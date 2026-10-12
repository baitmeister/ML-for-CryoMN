# Multi-objective campaign: round progression and method register

This register separates three decisions: **which GP supplies predictions**, **how candidate recipes are searched**, and **how the experimental slate is allocated**. A mechanics phase controls evidence requirements and allocation; it does not select a GP or activate continuous search.

## Evidence scope

Current campaign state after the 2026-10-12 Round 12 ingestion: 427 formulations and 1,153 observation rows through `ROUND_012`, with a frozen `ROUND_013` proposal and its editable next-round worksheet. Round 12 added 48 viability readings, 12 intact-patch outcomes (11 pass, one fail), and eight mechanical replicates across four formulations. The ingestion has no unresolved outcomes or mechanical execution violations. The 17 paired formulation–batch observations across 17 formulations and four batches satisfy the full mechanics gate, so Round 13 uses `mechanics_enabled` with the existing approved control-informed GP and finite-pool qLogNEHVI decision. See the [completed Round 12 worksheet](../results/multi_objective_v2/rounds/ROUND_012/completed/completed.csv) and [Round 13 selection metadata](../results/multi_objective_v2/rounds/ROUND_013/proposal/selection_metadata.json).

The campaign baseline is commit `0a0a7328b7d0248bd29e769ebeda3739a8295f02`: observations through `ROUND_011`, a prepared `ROUND_012` worksheet, 419 formulations and 1,061 observation rows. Frozen proposal metadata covers Rounds 2–12. Round 1 has reconstructed evaluation provenance and no frozen selection-metadata file in this baseline. Round 2 is a migration-frozen supplementary evaluation cohort; the formal prospective cohort starts at Round 3. The baseline excludes Round 12 ingestion outputs and a Round 13 proposal.

The implementation reference is `pre-full-phase-check` at commit `98d03e3`. Its continuous-search and full-phase allocation changes are described by activation conditions below. They do not establish that a frozen campaign proposal used those methods.

## Progression by proposal round

| Proposal round | Viability GP | Candidate search and acquisition | Rule introduced or carried forward |
| --- | --- | --- | --- |
| 1 | Exact executed model is not established by a frozen selection record. | Exact executed search is not established by a frozen selection record. | Reconstructed evaluation cohort; retain the evidence limitation. |
| 2 | Pooled legacy/campaign Matérn-5/2 GP, standardized inputs, per-row noise plus fixed `WhiteKernel(5.0)`. | Support-aware finite pool; screening utility, not qLogNEHVI. | `round2_candidate_feasibility_v1`; 40/35/25 generation targets, chemistry restrictions, support controls and at most one outside-support row. |
| 3 | Pooled Matérn-5/2 with per-observation `alpha` as the sole regression-noise mechanism; fixed WhiteKernel removed. | Finite-pool screening. | `round3_surrogate_uncertainty_v2`; history/pool similarity, marginal ingredient-frequency control and evidence-based retest policy. |
| 4 | Round 3 GP specification. | Finite-pool screening. | Intact technical replicates aggregate with the all-pass rule for each formulation–batch. |
| 5 | Round 3 GP specification. | Finite-pool screening. | `round5_solubility_viscosity_v2`: practical ingredient ceilings, CPA/sugar/nonpermeating totals, polymer-dependent limits and crystalline saturation burden. |
| 6 | Round 3 GP specification. | Finite-pool screening using viability UCB with chemistry/support deductions and bounded empirical intact deduction. | Exact-combination intact evidence and cold-ingredient graduation policies activate. The additive intact classifier is diagnostic. |
| 7 | Round 3 GP specification. | Round 6 screening method. | `round7_viability_prediction_labeling_v1`: unsupported predictions display as unknown; raw means/SDs remain available to acquisition and evaluation. |
| 8 | Round 3 GP specification. | Round 6 screening method. | Screening phase; similarity, feasibility, intact, cold-ingredient and labeling policies carry forward. |
| 9 | Pooled Matérn-5/2 specification. | Finite pool; bootstrap utility/diversity, without qLogNEHVI or continuous search. | Eight completed screening rounds permit mechanics bootstrap. A hash-bound endpoint addendum assigns terminal-force extraction to the Round 9 ingestion. |
| 10 | Pooled Matérn-5/2 specification; comparable terminal-force mechanical GP. | Finite pool; bootstrap utility/diversity. | `group10_workflow_v4`, schema 4: recurring control and screened-hit follow-up. Frozen mechanical primaries include one control, one follow-up and two fresh recipes. |
| 11 | **Campaign-only, control-informed batch-effect Matérn-5/2 GP** (`control`, `batch_gp_v1`). | Shared-posterior finite-pool qLogNEHVI; hybrid screening/mechanics score. | Decision `group11-control-informed-user-v1`. Workflow v5/schema 5 makes the reference viability-only; one historical control batch is explicitly permitted for Round 11. |
| 12 | Round 11 approved GP and decision. | Shared-posterior finite-pool qLogNEHVI; hybrid phase. | Two historical control batches satisfy the normal control-history requirement. Full-phase evidence gate remains incomplete in the frozen proposal. |
| 13 | Round 11 approved GP and decision. | Shared-posterior finite-pool qLogNEHVI; full mechanics phase. | Round 12 completion satisfies the full gate. Allocation is one control, one mechanics follow-up, one shared retest/rescue and nine ordinary candidates; four primary mechanical tests retain actual-intact rank promotion. |

Round 2 and Round 3 kernel definitions are recorded in source commits `ac1e387` and `a500ea4`; commit `6160fad` records the all-pass aggregation change used for Round 4. Round-indexed selection evidence is under `results/multi_objective_v2/rounds/ROUND_###/proposal/selection_metadata.json`. These records establish **finite-pool search for every frozen proposal from Round 2 through Round 13**. The `hybrid_finite_pool_fallback` label in Rounds 11–12 accompanies the explicit reason “Recorded strategy uses shared-model finite-pool qLogNEHVI”; it does not establish an attempted continuous solve or optimizer failure.

## GP and control definitions

| Model | Training evidence | Noise and control treatment |
| --- | --- | --- |
| Pooled Matérn-5/2 viability GP, used through Round 10 | Eligible legacy literature, legacy wet-lab and campaign viability labels. | Standardized inputs, fixed unit length scales, normalized targets. Recorded source noise defaults are 50, 5 and 1 percentage points respectively; Round 3 removes the additional fixed WhiteKernel. |
| Approved `control` viability GP, effective Round 11 | Campaign formulation–batch means, including viability-only screens; legacy evidence is excluded from this response model. | Separate formulation response, shared batch effect and uncertain control intercept. Batch SD and total residual SD are fitted by bounded maximum likelihood. Recorded row noise is not added a second time. Control-intercept prior SD is 50 percentage points. |
| `current_comparable_endpoint` mechanical GP | Eligible labels with the terminal-force definition; technical replicates aggregate within formulation–batch. | Separate Matérn-5/2 endpoint GP with its supplied force noise. Viability noise floors do not apply to force. |

The approved GP is an empirical-Bayes model: uncertainty conditions on fitted hyperparameters. Controls inform viability batch effects through a separate reference component; they are excluded from ordinary formulation-response labels and ordinary viability error metrics. They do not normalize force. The valid Round 10 control mechanical result and its measured viability companion count as an ordinary comparable mechanical pair.

The two optimization objectives are viability percentage and nominal terminal force per loaded needle. The mechanical quantity is force at +0.800 mm from a sustained 1 N trigger, tagged `terminal_force_08mm_after_1N_v1`; it is not a fracture-force measurement. Total patch force and apparent secant stiffness are stored separately, with stiffness serving a diagnostic role. See the [mechanical endpoint SOP](group10/mechanical_sop.md).

The acquisition uses the independent viability/mechanical **formulation-response posteriors**. Fresh assay noise and a fresh batch offset are excluded from the acquisition output; those terms belong to future-measurement uncertainty. `current` is a retained strategy identifier for the pooled sklearn baseline, not a deployment-status label.

The production record is [gp_strategy_decision.json](../results/multi_objective_v2/gp_strategy_decision.json). `production_model_revision: false` and `production_noise_revision: false` in the Group 10 endpoint contract describe that endpoint release; the separate GP decision controls the Round 11 model choice. See [the Group 11 decision](group10/group11_control_informed_decision.md), [model integration](../src/08_multi_objective/helper/gp_strategy.py) and [batch-GP implementation](../src/08_multi_objective/helper/batch_gp.py).

## Finite-pool generation rules

Finite-pool generation is **seeded random sampling under explicit rules**, followed by model scoring. It does not restrict concentrations to an integer or fixed-decimal grid. The configured seed is 42 and the ordinary generation target is 2,000 recipes.

| Generation stream | Target share | Sampling rule |
| --- | ---: | --- |
| `local_perturbation` | 40% | Choose a formulation from the formulation table and perturb its active concentrations multiplicatively with log-space SD 0.20. Use at most six active ingredients and clip to policy bounds. Anchors are not restricted to the highest-viability recipes. |
| `sparse_exploration` | 35% | Sample one or two available ingredients and concentrations across their permitted ranges using Beta(1.25, 1.25). This stream can fill a local-generation shortfall. |
| `boundary_probe` | 25% | Normally sample three or four ingredients at 50–95% of their permitted ranges, using the configured beta distribution. A generation shortfall stops the run. |

Every stream applies availability, feasibility, duplicate and active similarity checks. `boundary_probe` describes how a recipe was generated; measured support classification determines whether it consumes the single outside-support slot. The 40/35/25 shares describe pool generation, not the final 12 experimental positions.

Rescue candidates dilute intact-failing formulations with measured viability of at least 50%, using 0.25/0.50/0.75 concentration factors. Evidence-ranked retests target observed campaign instability. They enter as additional candidate types and do not change the ordinary pool-generation percentages. See [candidate generation](../src/08_multi_objective/helper/candidates.py) and [optimization settings](../config_v2/optimization.yaml).

## Evidence phases and experimental allocation

All phases select 12 viability/intact recipes. Mechanical capacity and priority are separate from these 12 screening positions.

| Phase | Automatic entry condition | Mechanical rule |
| --- | --- | --- |
| `screening_only` | Fewer than eight completed screening rounds. | No proposed mechanical tests. |
| `mechanics_bootstrap` | Screening entry satisfied; hybrid gate incomplete. | Four utility/diversity tests; no qLogNEHVI. Group 10 frozen role allocation includes its mechanical reference. |
| `mechanics_hybrid` | At least 8 paired observations, 6 distinct paired formulations and 2 paired batches; full gate incomplete. | Transition picker targets two acquisition positions, one local position and one coverage position, with recorded role fallbacks. The Group 10+ role layer prioritizes eligible reserved rows, so inspect the frozen mechanical ranks for execution. |
| `mechanics_enabled` | At least 16 paired observations, 12 distinct paired formulations and 3 paired batches. | One eligible mechanics follow-up has first priority; three other eligible recipes complete the primaries, followed by ordered backups. |

All three conditions within each paired-evidence gate must be satisfied. A pair is one formulation–batch with measured viability and comparable mechanics. Technical replicates do not increase the pair count. The phase resolver uses evidence, not the proposed round number.

Frozen proposal counts are:

| Proposal | Paired observations | Distinct paired formulations | Paired batches | Resolved phase |
| --- | ---: | ---: | ---: | --- |
| [Round 9](../results/multi_objective_v2/rounds/ROUND_009/proposal/selection_metadata.json) | 0 | 0 | 0 | Bootstrap |
| [Round 10](../results/multi_objective_v2/rounds/ROUND_010/proposal/selection_metadata.json) | 4 | 4 | 1 | Bootstrap |
| [Round 11](../results/multi_objective_v2/rounds/ROUND_011/proposal/selection_metadata.json) | 8 | 8 | 2 | Hybrid |
| [Round 12](../results/multi_objective_v2/rounds/ROUND_012/proposal/selection_metadata.json) | 13 | 13 | 3 | Hybrid |
| [Round 13](../results/multi_objective_v2/rounds/ROUND_013/proposal/selection_metadata.json) | 17 | 17 | 4 | Full mechanics |

### Source quotas and reserved roles

The Group 10 allocator in bootstrap/hybrid reserves its eligible control/follow-up, up to two retests and up to two rescues, subject to feasibility and slate constraints. It divides the residual ordinary positions proportionally to source counts in the preliminary slate, using floor allocation and largest fractional remainders. These quotas vary by run, but become hard counts within that run.

Frozen Rounds 10–12 each reserve five positions: one control, one follow-up, two retests and one rescue. Round 10 assigns the seven ordinary positions to four local and three sparse recipes; Rounds 11–12 assign all seven to local recipes. These source allocations are distinct from the hybrid **mechanical** 2/1/1 transition roles.

The repaired full-phase allocator uses policy `full_mechanics_1control_1followup_1retest_or_rescue_v1`:

| Screening role | Positions | Selection rule |
| --- | ---: | --- |
| Campaign control | 1 | Fixed 2.5% v/v DMSO + 0.100 M sucrose reference; viability-only. |
| Mechanics follow-up | 1 | Eligible screened hit with passing intact evidence and no comparable mechanical observation; ranked by observed viability, with deterministic ties. |
| Retest or rescue | 1 shared | Highest-priority admissible retest; admissible rescue ranked by mechanics acquisition if no retest can fill the slot. |
| Ordinary candidates | 9 | Rank all surviving origins together by mechanics acquisition, subject to the established constraints. |

There is **no finite/continuous origin quota** for the nine full-phase ordinary positions. Additional retest/rescue rows cannot occupy them. A missing required role or an underfilled feasible slate stops selection. Source counts are descriptive outcomes. This allocator applies whenever the repaired code resolves `mechanics_enabled`, for either finite-only or continuous-required search. Bootstrap/hybrid retain their phase-specific allocation.

## Continuous-search implementation and activation

| Decision format | Acquisition setting | Search behavior |
| --- | --- | --- |
| Version 1, `control` | `shared_qlognehvi_finite_pool` | Finite-only; this is the decision frozen for Rounds 11–12. |
| Version 2 | `shared_qlognehvi`, `search_mode: finite_pool` | Shared-posterior scoring of a finite pool. |
| Version 2 | `shared_qlognehvi`, `search_mode: continuous_required` | Finite pool plus continuous concentration refinement, in hybrid/full mechanics with continuous settings enabled. |

Setting `continuous_qlognehvi.enabled: true` or meeting the full-phase gate does not convert a version-1 finite decision into continuous search. Version 2 requires a complete, separately recorded decision with provenance. The approved GP can remain `control`; changing the search mode does not require changing its statistical model.

Continuous refinement uses one shared qLogNEHVI acquisition definition, the approved fitted GP posteriors, a comparable paired baseline, 128 Sobol samples and seed 42. Its mechanics score is:

```text
qLogNEHVI + log(max(empirical_intact_probability, 1e-9)) - chemistry penalty
```

1. Score eligible ingredient sets by their best feasible pool seed. Reserve the best-scoring set for each available ingredient count, then fill the set budget by score.
2. Refine concentrations within each selected fixed ingredient set. The configured budget is at most 16 sets, at most four ingredients per set, and a target of 24 accepted candidates. Every chosen set receives an optimization attempt prior to repeat attempts.
3. Use the shared posterior and existing chemistry constraints; fit no additional GP. Candidate generation is sequential with `q=1` and accepted proposals supplied as pending points.
4. Merge valid refinements with the finite pool, reapply candidate checks, and score the merged pool consistently for slate selection. A refined recipe has no guaranteed experimental position.

This is hybrid finite/continuous **search**, which is independent of the phase named `mechanics_hybrid`. It is not unrestricted continuous exploration of every ingredient combination or a simultaneous joint optimization of the 12-row slate. Required optimization failure or zero surviving refinements stops the run. No stock-volume, dispensing-resolution or concentration-rounding policy is added.

Implementation provenance: `98a3405` connects the approved posterior; `6b759c7` ranks ingredient sets and ensures mixture coverage; `95ecf9c` introduces full-phase 1/1/1/9 allocation; `98d03e3` removes comparison-runner dependencies on disposable outputs. Source: [continuous refinement](../src/08_multi_objective/helper/continuous_search.py), [final allocation](../src/08_multi_objective/helper/group10_selection.py), [Stage 02 instructions](../src/08_multi_objective/02_select_candidates/README.md).

## Constraints retained across the search methods

- Availability and round-effective chemistry limits apply to finite and refined candidates. Round 5 limits include total permeating CPA ≤2.50 M, sugar ≤0.50 M, nonpermeating solutes ≤0.50 M, polymer-dependent sugar/nonpermeating limits ≤0.30 M, at most one active polymer and saturation burden ≤0.80. The registry supplies ingredient-specific ceilings; the reference has a narrowly scoped DMSO exception.
- Round 3 similarity uses complete registry-bounds-normalized vectors: distance must exceed 0.05, and matching single-ingredient recipes must differ by at least 50% relative to the lower concentration. Intentional retests and reserved-role handling retain their documented exceptions.
- A slate permits at most one measured outside-support row, two occurrences of an exact two-ingredient set, one occurrence of an exact 3+-ingredient set, five occurrences of a shared ingredient pair and five rows containing any individual ingredient. Intentional retests are exempt from exact-set/shared-pair counts, but count toward ingredient frequency.
- Round 6 cold-ingredient policy uses fewer than three distinct campaign formulations with measured viability as its threshold and caps each cold ingredient at two ordinary rows. Full-phase ordinary positions compete by acquisition; the 1/1/1/9 rule adds no separate graduation reservation.
- Round 6 intact probability pools only identical complete active sets within normalized RMS radius 0.35, with a Beta(1,1) prior. Unseen sets receive probability 0.50. The classifier probability is diagnostic, not an additional selection gate.
- Mechanical execution follows ascending mechanical rank among recipes that actually pass intact. Unranked rows cannot supply mechanics or become backups. A shortage of passing eligible recipes is recorded explicitly.

## Interpreting model error across rounds

Round 11 combines a GP/source/noise change with entry into hybrid mechanical allocation. It does **not** introduce continuous candidate search. An MAE decrease across Rounds 11–12 therefore cannot be evidence that continuous search improved predictions. Training evidence, tested recipes and biological/assay variation also differ across rounds. Comparisons that hold the evaluated recipes and training cutoff fixed help distinguish model effects; cross-round MAE alone does not establish causation.

For each generated proposal, inspect `gp_strategy_decision`, `active_phase`, `phase_resolution`, `optimizer_mode`, `continuous_qlognehvi`, `group10.ordinary_origin_quotas`, the full-phase allocation policy, and the candidate/mechanical ranks. Model choice, search execution and allocation should each be evidenced independently.

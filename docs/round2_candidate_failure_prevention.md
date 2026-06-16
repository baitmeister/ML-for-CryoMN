# Round 2 Candidate-Failure Prevention

## Purpose and activation

This forward-only policy prevents future slates from repeating preparation
failures observed while making the first multi-objective slate:

- policy version: `round2_candidate_feasibility_v1`
- activation: `ROUND_002`
- affected data: generated or externally supplied candidate pools
- unaffected data: transferred formulations, transferred observations, legacy
  viability values, noise assignments, and the executed `ROUND_001` slate

Existing formulation and observation tables remain model evidence. They are
read to establish support and train models, but are never rewritten, filtered,
relabelled, or migrated by this policy.

## Why ROUND_001 produced difficult formulations

The first slate was not produced by continuous Bayesian optimization. Stage 02
generated 2,000 formulations by choosing ingredients and drawing each selected
concentration uniformly between zero and its registry upper bound. The GP then
scored this finite random pool.

For candidates 2, 5, 6, 7, 9, and 11, predictions reverted to the transferred
data mean (`59.83%`) with approximately `59.68%` uncertainty. UCB consequently
rose to approximately `80.72%`. Many unrelated formulations received nearly
identical scores, after which diversity selection favored chemically distant
points. The exact high concentrations therefore arose mainly from permissive
independent bounds, whole-box random sampling, out-of-support uncertainty, and
diversity tie-breaking. They were not concentration optima learned from
historical experiments.

Applying the new rules retrospectively for diagnosis, without changing the
executed slate, gives:

| ROUND_001 rank | Candidate | Main prospective rejection causes |
|---:|---|---|
| 2 | `cand_000754` | 25.25% total polymer, three polymers, 7.13% protein, 0.51 M sugars, 0.97 M non-permeating solutes |
| 5 | `cand_000440` | 24.21% total polymer and three polymers |
| 6 | `cand_001027` | 8.35% dextran plus HA and 0.89 M non-permeating solutes |
| 7 | `cand_001486` | 7.21% dextran plus HA, 1.12 M sugars, and 1.20 M non-permeating solutes |
| 9 | `cand_000585` | 15.28% FBS+HSA and 0.90 M non-permeating solutes with polymer |
| 11 | `cand_001920` | 16.69% total polymer, PVP+HA, 11.11% protein, and 0.91 M non-permeating solutes |

These are diagnostic classifications only. The stored ROUND_001 candidates are
not edited or relabelled.

## What the legacy optimizer did

The legacy single-objective code did not contain the new polymer, protein, or
combined-solids rules. It addressed preparation risk only indirectly:

- HA had a 1% feature bound and methylcellulose a 2% bound.
- Creatine had a 30 mM practical-solubility cap.
- The Bayesian optimizer used sparsity and distance-from-support penalties.
- A global explicit percentage cap rejected mixtures whose percentage-based
  ingredients summed above 100%.

It did not cap PVP at 10%, dextran at 5%, total polymer at 10%, serum/protein at
10%, polymer+protein at 15%, or polymer-associated sugar/osmolyte burden.
Consequently, legacy observations are useful evidence but are not treated as a
complete preparation-feasibility policy.

## Evidence provenance and limits

These defaults are conservative campaign guardrails, not universal physical
solubility constants. Polymer grade, molecular weight, temperature, shear,
solvent composition, and mixing protocol can materially change behavior.

| Rule | Default | Basis |
|---|---:|---|
| Active viscosity polymers | maximum 1 | No transferred wet-lab support for the failed polymer combinations |
| PVP | maximum 10% | Literature: 10% gave 69.7%, 20% gave 55.5%, and 40% gave 4.6%; wet-lab support reached 7.2% |
| Dextran | maximum 5% | Maximum transferred concentration; wet-lab support above 3.3% is absent |
| HA | maximum 1% | Existing exploratory ceiling; transferred evidence ends at 0.2% |
| Total polymer | maximum 10% | Prevents independent polymer limits from accumulating |
| FBS+HSA+human serum | maximum 10% | Conservative future protein/crowding burden |
| Polymer+serum/protein | maximum 15% | Allows 10%+5% or 5%+10%, but blocks simultaneous high loading |
| Listed sugars with polymer | maximum 0.50 M | Prevents several individually legal sugars accumulating into high solids |
| Non-permeating osmolytes with polymer | maximum 0.75 M | Allows moderate combinations while blocking approximately 1 M burdens seen among failures |

Transferred literature contained PVP at 1-40%, dextran at 2-5%, and HA at
0.1-0.2%. Transferred wet-lab data contained PVP to 7.2%, dextran to 3.3%, and
one HA row at 0.2%. No transferred wet-lab row combined PVP with dextran or
dextran with HA. ROUND_001 failures repeatedly combined polymers near their
independent maxima, often with concentrated sugars or proteins.

Exact transferred records used for these statements:

- PVP literature dose series: `legacy_lit_86` through `legacy_lit_90`.
  `legacy_lit_88` is 10% PVP with 69.7% viability,
  `legacy_lit_89` is 20% with 55.5%, and `legacy_lit_90` is 40% with 4.6%.
- Highest wet-lab PVP: `legacy_wetlab_EXP2103`,
  `legacy_wetlab_EXP8106`, and `legacy_wetlab_EXP9107`, each at 7.2%.
- Highest transferred dextran: `legacy_lit_289`, 5% dextran with 87.6 mM
  trehalose. Highest wet-lab dextran: `legacy_wetlab_EXP5206`, 3.3%.
- HA literature support: `legacy_lit_205` through `legacy_lit_208`, 0.1-0.2%.
  Wet-lab HA support: `legacy_wetlab_EXP3104`, 0.2%.
- The transferred table contains zero rows with more than one of PVP, dextran,
  and HA active, and no active methylcellulose observations.
- Legacy literature serum rows include `legacy_lit_97` and `legacy_lit_98`
  at 40% and 80% FBS with 10% PVP. Wet-lab records also reach high serum
  levels. These rows show that high-serum evidence exists; they do not establish
  preparation safety for the current microneedle campaign. The 10% future
  serum/protein limit is therefore explicitly conservative.

The source rows are transferred from
`data/processed/parsed_formulations.csv` and
`data/validation/validation_results.csv` into the read-only v2 evidence tables.

Evidence classification:

- **Empirical:** transferred concentrations and outcomes, the absence of
  transferred multi-polymer rows, and the reported ROUND_001 preparation
  failures.
- **Inferred:** accumulated polymer, protein, sugar, and osmolyte burdens are
  plausible contributors to the observed non-homogeneity and viscosity. The
  current data do not isolate a single causal ingredient for each failure.
- **Conservative policy:** the one-polymer rule and the 10%, 15%, 0.50 M, and
  0.75 M aggregate limits. These are campaign decisions chosen to prevent
  recurrence while new preparation labels accumulate.

Every candidate receives:

- `feasibility_pass` and `feasibility_reasons`
- polymer, serum/protein, sugar, and non-permeating-solute totals
- `estimated_small_solute_g_L`
- `nearest_support_distance` and `support_status`

Rejected generated attempts remain in `total_candidate_pool.csv` for audit but
are never scored or selected.

## Support-aware screening

During `screening_only`, the finite pool is generated as:

- 40% local perturbations around transferred or newly validated formulations
- 35% sparse single-ingredient and pairwise exploration
- 25% chemically feasible boundary-style exploration

Local work is capped at 40%. Duplicate or infeasible local shortfall is moved
to sparse exploration. The boundary-style quota is not silently reassigned:
generation stops with a diagnostic error if ingredient availability, duplicate
avoidance, and hard constraints make the configured 25% allocation impossible
within the bounded attempt budget. The quota is a sampling-mode allocation, not
a requirement that every boundary-style row be outside the observed support
radius.

The split preserves useful local learning without allowing narrow historical
support to dominate, produces interpretable sparse effects, and reserves
meaningful capacity for discovery.

High-viability formulations that fail intact-patch formation also seed a small
`rescue_dilution` set. These candidates scale down every active ingredient in
the failed formulation, then pass through the same campaign feasibility checks.
At most a configured number can enter a screening slate, so the workflow can
test concentration-reduction rescue hypotheses without overwhelming the
standard model-ranked slate. Rescue rows are ordered by strongest dilution
first, with predicted viability as a tie-breaker.

Representative configuration:

```yaml
formulation_feasibility:
  policy_version: round2_candidate_feasibility_v1
  start_round: 2
  max_active_polymers: 1
  max_total_polymer_pct: 10.0
  max_total_serum_protein_pct: 10.0
  max_total_polymer_serum_pct: 15.0
  max_sugar_M_with_polymer: 0.50
  max_nonpermeating_M_with_polymer: 0.75
  ingredient_caps:
    pvp_pct: 10.0
    dextran_pct: 5.0
    hyaluronic_acid_pct: 1.0

candidate_generation:
  local_fraction: 0.40
  sparse_fraction: 0.35
  boundary_fraction: 0.25
  rescue_min_viability_percent: 50.0
  rescue_scale_factors: [0.25, 0.50, 0.75]
  rescue_candidates_per_round: 2

support_policy:
  radius_percentile: 95.0
  radius_multiplier: 1.25
  uncertainty_cap_percentile: 90.0
  max_boundary_candidates_per_slate: 1
  diversity_weight: 0.05
```

Features are normalized by configured ranges. The support radius is the 95th
percentile of observed nearest-neighbor distances multiplied by 1.25. The
percentile retains almost all observed geometry while reducing sensitivity to
extreme outliers; the multiplier supplies controlled exploration slack.
The support set is derived from formulation IDs that have actual entries in
`observations.csv`, so unobserved candidate rows do not expand support merely
by being written into `formulations.csv`. Legacy literature, legacy wet-lab,
and new wet-lab observations remain support evidence regardless of outcome.
Failed round results still define where the campaign has evidence; feasibility
rules and endpoint models decide whether nearby lower-concentration candidates
are worth selecting.

Only one boundary candidate may enter a 12-row slate. Outside-support
uncertainty is capped at the 90th percentile of in-support uncertainty.
Outside-support candidates receive an explicit score penalty. Diversity
contributes at most 0.05 and is applied only inside a competitive utility band.
An unfitted intact model reports an unknown prior and has no ranking influence
from ROUND_002 onward.

## Optional preparation labels

ROUND_002+ sheets allow manual entry of:

- `preparation_feasibility_pass`
- `homogeneous_solution_pass`
- `fillability_pass`
- `preparation_failure_reason`

Accepted reasons are `insoluble_or_precipitated`, `phase_separated`,
`excessive_viscosity`, `incomplete_polymer_hydration`, and
`other_preparation_failure`.

Blank fields create no observations. Values are never inferred from slurry,
collapse, intact-patch results, or notes. Explicit preparation failure can be
recorded without viability and prevents mechanical data for that replicate.

A preparation classifier is fitted only after at least eight manual labels
contain both pass and fail classes. Eight is a startup threshold, not evidence
of a mature classifier. Deterministic chemistry gates remain authoritative.

## Constrained qLogNEHVI after screening

The existing mechanics transition remains:

- 8 paired viability/load observations
- 6 distinct paired formulations
- 2 paired batches

Once `mechanics_enabled`:

1. Probabilistic viability and critical-load models define the observed Pareto
   frontier.
2. Sparse ingredient masks are taken from the feasible pool.
3. BoTorch continuously optimizes qLogNEHVI over concentrations in each mask;
   inactive ingredients are fixed to zero.
4. Every point is rechecked against campaign feasibility.
5. Once fitted, preparation probability is an additional constraint.
6. Accepted points become pending points so later batch choices seek
   complementary experiments.
7. Support and one-boundary-candidate rules remain active.

qLogNEHVI estimates expected improvement in the objective-space hypervolume
dominated by the Pareto frontier. It can value a viability improvement, a
strength improvement, or a useful tradeoff without fixed objective weights.

If BoTorch is unavailable, fitting fails, or optimization produces too few
feasible points, selection uses the constrained support-aware finite pool.
Candidate origins and metadata identify the path used.

## Revising the policy

Review manual preparation labels, polymer grade and molecular weight,
solvent/media, temperature, mixing protocol, replication, and Pareto
performance before changing limits. Increment `policy_version` whenever any
limit or feature grouping changes so prior recommendations remain reproducible.

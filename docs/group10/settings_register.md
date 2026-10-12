# Settings register — Group 10 v4 / Group 11+ v5

Missing required Group 10 workflow or protocol settings stop proposal generation
and ingestion. Application thresholds are reporting-only. GP and search decisions
are separate from the endpoint contract; see the
[round progression register](../multi_objective_round_progression.md).

| Setting | Value/source | Consequence |
|---|---|---|
| Earliest endpoint evidence | Group 9 validation/ingestion, user | Hash-bound addendum preserves the original proposal CSV/metadata while applying the terminal method to new Group 9 results |
| Viability GP | `control`, `batch_gp_v1`; decision `group11-control-informed-user-v1`, effective Group 11 | Campaign-only, control-informed batch-effect Matérn-5/2 with fitted total residual noise |
| Candidate search | Version-1 decision: `shared_qlognehvi_finite_pool` | Finite-only for frozen Rounds 11–12; version-2 `continuous_required` explicitly requests shared-posterior refinement |
| Full-phase screening allocation | `full_mechanics_1control_1followup_1retest_or_rescue_v1`, repair commit `95ecf9c` | One control, one mechanics follow-up, one retest/rescue and nine ordinary positions ranked across origins; applies in `mechanics_enabled` |
| Mechanical endpoint revision | Group 9 addendum plus active Group 10 configuration, user | F at +0.8 mm after sustained 1 N |
| Apparent secant stiffness | `(F_terminal-F_trigger)/0.8 mm`, user | Stored in the compatibility stiffness field with terminal-method provenance |
| Endpoint definition | `terminal_force_08mm_after_1N_v1` | Different definitions cannot share a mechanical training cohort |
| Mechanical capacity | Four formulations, user | Replicate runs counted separately |
| Reference | 2.5% v/v DMSO + 100 mM sucrose, user | One viability screen slot; no mechanical rank from Group 11. Frozen Group 10 test retained |
| DMSO density/purity/MW | 1.10 g/mL / 100% / 78.13 g/mol | Approximately 0.352 M; exact recipe-only domain exception |
| Base medium | Common throughout campaign, user | No separate setup entry |
| Reference/candidate replicate numbers | Actual completed CSV records | No advance input required; aggregate means do not reveal hidden replicate counts |
| Minimum viability | **Pending user value**, percent | Acceptance stays pending |
| Minimum force and requirement definition | **Pending user value and named definition**, nominal N/loaded needle | Existing `minimum_fracture_force_N_per_needle` key retained; new value is not proven fracture strength |
| Trigger | >=1 N for >=5 samples spanning >=0.100 s | First sample of first sustained run defines displacement origin |
| Terminal displacement | +0.800 mm from that origin, user | Force interpolated at terminal, not maximum |
| Force drops | No endpoint role, user | No drop threshold, drop window or early termination setting needed |
| Baseline/smoothing | Instrument export / no extra subtraction / none | Software convention matching the reviewed traces |
| Force/displacement/time units | N/mm/s from CSV | Known units row automatically handled; no guessed conversion |
| Force sign/mode | Positive compression / whole patch array | Supported protocol checked |
| Sampling gap tolerance | 0.0375 s, software | Based on observed 40 Hz; slower acquisition requires protocol review |
| Displacement reversal tolerance | 0.0001 mm, software | One export resolution step; larger reversal is ambiguous |
| Floating comparison tolerance | 1e-9, software | Arithmetic boundary handling only |
| Nominal needle height | 1.6 mm, user | Descriptive geometry, not specimen strain |
| Loaded-needle count | Actual `needles_compressed` in result CSV | Positive integer; not inferred from intact count or a default array size; missing permits total force only |
| Compression speed | Approximately 0.75 mm/min in reviewed traces | Keep actual protocol consistent; no claim of constant temperature from CSV |
| Temperature, handling, zeroing, array loading convention | Actual protocol/notes | Cannot reliably fetch from these three measurement columns; no invented values |
| Optional replicate/test/cell-batch IDs | Actual result rows, when available | No independent-preparation requirement; unknown history stays unknown |
| Confirmation feature | Removed, user | No cadence or batch-count setup |
| Viability batch effects / mechanical normalization | Viability batch effects active under Group 11 `control`; mechanical normalization absent | Controls inform a separate viability reference/batch component; force remains unnormalized |
| Candidate/random seed | 42, software | Frozen reproducibility; does not alter raw endpoint calculation |
| Adaptive noise audit assumptions | Floor 1, fallback 5 viability points, shrinkage 4 degrees | Audit alternative; not the fitted-total-residual-noise model selected for Group 11 |

Density source: [Sigma-Aldrich DMSO 276855](https://www.sigmaaldrich.com/US/en/product/sial/276855), recorded in this project on 2026-09-08. The value is a literature preparation conversion, not a lot-specific measurement. Purity 100% is user supplied. Final-volume basis: 2.5 mL neat DMSO in 100 mL final formulation. Sucrose is an existing model feature.

Application thresholds require an explicit user value. The approved model/search
choice is recorded independently of those thresholds. Experimental metadata such
as raw file, loaded count and actual replicates comes from the completed round CSV.
Thermal history cannot be reconstructed from force/displacement/time alone.

Decision timing and the unchanged Group 10 mechanical list: [readiness and GP decisions](readiness_and_gp_decisions.md).


### Group 10 mechanical evidence clarification

The DMSO/sucrose row is a viability control only. Its valid Group 10 mechanical
result and measured same-batch viability/intact companions count as an ordinary
mechanical pair for phase progression, paired acquisition and observed trade-offs.
Control viability remains outside ordinary formulation-response labels and viability
metrics; the Group 11 `control` model uses it as reference/batch evidence.
Four valid Group 10 pairs plus four Group 9 pairs yield **8 pairs / 8
formulations / 2 batches**, meeting the unchanged hybrid gate for Group 11.
Group 10's planned tests stay unchanged; Group 11+ allocates no mechanical rank
to the recurring viability reference. Technical replicates still aggregate within
formulation/batch and do not inflate gate counts.

# Current settings register — workflow v4

Updated 2026-09-14. Missing required Group 10 workflow or protocol settings now hard-stop proposal generation and ingestion. Pending application thresholds remain reporting-only and do not disable the reviewed acquisition contract.

| Setting | Current value/source | Consequence |
|---|---|---|
| Earliest workflow/endpoint activation | Group 10, user | Frozen per proposal; no retroactive Group 9 edit |
| GP methodology / adaptive noise switch | Deferred, user | Review audit and decide before first full-mechanics proposal; no automatic switch |
| Mechanical endpoint revision | Active in Group 10 configuration, user | F at +0.8 mm after sustained 1 N |
| Endpoint definition | `terminal_force_08mm_after_1N_v1` | Different definitions cannot share a mechanical training cohort |
| Mechanical capacity | Four formulations, user | Replicate runs counted separately |
| Reference | 2.5% v/v DMSO + 100 mM sucrose, user | One screen slot; mechanics conditional on actual intact |
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
| Batch adjustment/control normalization | Inactive | Raw monitoring; evidence and later decision required |
| Candidate/random seed | 42, software | Frozen reproducibility; does not alter raw endpoint calculation |
| Adaptive noise audit assumptions | Floor 1, fallback 5 viability points, shrinkage 4 degrees | Offline only; unchanged production noise |

Density source: [Sigma-Aldrich DMSO 276855](https://www.sigmaaldrich.com/US/en/product/sial/276855), recorded in this project on 2026-09-08. The value is a literature preparation conversion, not a lot-specific measurement. Purity 100% is user supplied. Final-volume basis: 2.5 mL neat DMSO in 100 mL final formulation. Sucrose is an existing model feature.

The remaining decision inputs are the application thresholds and the future GP/noise/acquisition methodology. Experimental metadata such as raw file, loaded count and actual replicas comes from the completed round CSV. Thermal history cannot be reconstructed from force/displacement/time alone.

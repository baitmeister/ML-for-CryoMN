# Settings register

All pending settings below are intentional. Missing values disable only the dependent capability. No experimental settings were fabricated for live use.

| Setting | Value/status | Owner and consequence |
|---|---|---|
| Workflow activation | Group 10 | Locked user decision; later unstarted group if not ready |
| GP/noise/target revision | False | User decides after audit before the first full-mechanics proposal |
| Mechanical capacity | Four formulations | Locked; specimen runs counted separately |
| Reference composition | 2.5% v/v DMSO + 100 mM sucrose | Locked exact reference-only exception |
| DMSO density/purity | 1.10 g/mL / 100% | Literature density / user-specified purity; approximately 0.352 M at 2.5% v/v |
| DMSO molecular weight | 78.13 g/mol | Registry/reagent basis |
| Base medium | Common to all formulations | No separate reference setting |
| Reference replicates | Derived from completed CSV | No advance setting |
| Mechanical replicates/formulation | Derived from completed CSV | No advance setting |
| Minimum viability | Pending | User; acceptance pending |
| Minimum force and definition ID | Pending | User; acceptance pending |
| Operational stop | 1 mm from contact | Locked user decision |
| Relative drop | 10% | Locked operational rule |
| Contact force/baseline points | Pending | User/instrument trace review |
| Absolute drop/noise threshold | Pending | User/instrument trace review |
| Window/persistence | Pending | User/instrument trace review |
| Smoothing | None | Implementation default; no hidden filtering |
| Force/displacement units | N/mm | Required supported units |
| Force sign | Pending | User/instrument export |
| Protocol/test mode | Pending | User; detector disabled |
| Loaded-needle count | Per result, pending | User; positive count, one for single needle |
| Speed, temperature, geometry, handling | Record in protocol | User; comparability must be documented |
| Preparation/specimen/readout hierarchy | Per result | User; unknown historical independence remains unknown |
| Adaptive-noise floor/fallback/shrinkage | 1 point / 5 points / 4 degrees | Offline modeling assumptions, not production settings |
| Additive batch adjustment | Diagnostic only | More repeated controls and user decision needed |
| Confirmation feature | Removed | No cadence, batch-count requirement, or reserved slot |
| Acceptance claim | Measured requirements only | No separate confirmation classification |
| Seed | 42 | Software reproducibility |
| Learned audit kernel bounds | amplitude 0.1–10, shared length 0.1–10 | Bounded offline comparison, no automatic promotion |

The reference does not normalize results. Its first groups establish baseline variability. Testing replicates share one preparation; they do not estimate independent preparation variability. Assay/cell state, CPA exposure, freezing/thawing/storage, processing order and viability timing should be recorded consistently.

Density source: [Sigma-Aldrich DMSO 276855](https://www.sigmaaldrich.com/US/en/product/sial/276855), accessed 2026-09-08. This is a literature conversion value, not a measurement of the user’s reagent lot. Purity 100% is the user-specified preparation assumption. Final-volume convention: 2.5 mL neat DMSO per 100 mL final formulation. Replicate counts come from the completed CSV.

Transition decision: review the audit and select/freeze any model, noise or endpoint revision before generating the first full-mechanics proposal. Full-phase entry does not select a model automatically. Production flags stay false until that decision is implemented.

## Replicates from completed CSV (workflow v3)

No advance replicate counts are required. The completed round CSV supplies actual replicates; reports count ingested replicate IDs separately by formulation, batch and endpoint. An aggregate mean counts as one recorded measurement; underlying unrecorded replicates are not guessed. These are test replicates, not independent preparations. The reference is ready for new proposals from Group 10 on this development branch, with no replicate-count gate. No live Group 10 proposal has been generated; the live checkout still has Group 9 frozen. The reference receives one screen slot and a mechanical slot conditional on actual intact formation.

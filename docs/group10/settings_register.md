# Settings register

All pending settings below are intentional. Missing values disable only the dependent capability. No experimental settings were fabricated for live use.

| Setting | Value/status | Owner and consequence |
|---|---|---|
| Workflow activation | Group 10 | Locked user decision; later unstarted group if not ready |
| GP/noise/target revision | False | User decides after audit during full mechanics |
| Mechanical capacity | Four formulations | Locked; specimen runs counted separately |
| Reference composition | 2.5% v/v DMSO + 100 mM sucrose | Locked exact reference-only exception |
| DMSO density/purity | Pending | User/reagent record; exact conversion disabled |
| DMSO molecular weight | 78.13 g/mol | Registry/reagent basis |
| Base medium | Pending | User; reference readiness disabled |
| Reference preparations/specimens | Pending | User; reference readiness disabled |
| Candidate specimens/formulation | Pending | User; workload not invented |
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
| Confirmation cadence/batches | Pending, disabled | User; no confirmed-acceptable label |
| Future-batch reliability criterion | Pending | User; measured threshold pass only |
| Seed | 42 | Software reproducibility |
| Learned audit kernel bounds | amplitude 0.1–10, shared length 0.1–10 | Bounded offline comparison, no automatic promotion |

The reference does not normalize results. Its first groups establish baseline variability. Use independent preparations to distinguish preparation variability from repeated readout precision. Assay/cell state, CPA exposure, freezing/thawing/storage, processing order and viability timing should be recorded consistently.

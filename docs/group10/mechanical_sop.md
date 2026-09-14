# Whole-patch terminal-force SOP — Group 10 onward

**Frozen endpoint:** force at **+0.8 mm displacement after the first sustained 1 N trigger**. Definition ID: `terminal_force_08mm_after_1N_v1`. Protocol: `whole_patch_1N_08mm_v1`. Extraction version: `terminal_force_v1`.

This replaces the earlier proposed 1 mm/force-drop endpoint for new Group 10 proposals. It is the mechanical measurement used by the existing model when phase gates permit mechanical modeling. The GP architecture, kernel, scaling, source treatment, fixed noise and acquisition method remain unchanged. Any change to those methods is a separate decision before the first full-mechanics proposal.

## Exact extraction algorithm

1. Read the raw Instron export in its original acquisition order. Identify **Time**, **Displacement** and **Force**. Ignore the derived strain column. Recognize and remove only the explicit `(s)`, `(mm)`, `(N)` units row; do not discard malformed measurement rows. Supported export units are seconds, millimetres and newtons, with positive compressive force.
2. Use force as exported by the instrument. There is no new software baseline subtraction, absolute-value transformation or smoothing. Preserve/document the instrument's own zeroing convention. This matches the six Group 9 traces used to develop the rule.
3. Find the earliest uninterrupted run of readings **at or above 1.0 N**, containing at least **five samples** and spanning at least **0.100 s**. At the observed 40 Hz, five readings span 0.100 s. The displacement/time origin is the **first sample of that sustained run**, not the last confirmation sample. Equality at 1 N counts.
4. Ignore subthreshold disturbances and brief unsustained threshold spikes when finding this origin. Once established, **never reset the origin** because force subsequently drops below 1 N. Drops do not terminate the test or choose its endpoint.
5. Find the first acquisition sample reaching +0.800 mm relative to that origin. Use the two adjacent displacement/force readings that bracket +0.800 mm to linearly interpolate force. Exact terminal readings require no effective interpolation. Never extrapolate a short trace.
6. Store the **force at that displacement**, not the maximum force in the window and not the maximum anywhere in the file. Earlier peaks and later force increases do not replace it. Later unloading/data beyond the terminal bracket do not invalidate an otherwise valid completed window.
7. A trace without the sustained trigger or without +0.8 mm is `incomplete_test`. Gaps/nonfinite acquisition samples or invalid time order are `invalid_curve`. Meaningful reversal after trigger or a negative terminal load is `ambiguous_curve`. Unsupported/missing protocol inputs are `protocol_incomplete`. None supplies a mechanical training value. A complete total-force result remains measurable if the loaded-needle count is unknown; its nominal per-needle value and model label remain absent.

## Frozen numerical conventions

- Sampling-gap ceiling: **0.0375 s**, 1.5 times the 40 Hz interval. This detects a missing sample in these exports; changing acquisition rate to a slower rate requires a separately validated protocol version.
- Displacement reversal tolerance: **0.0001 mm**, one exported displacement resolution step. Larger reversal inside the analyzed interval is ambiguous. No sorting, resampling or stitching repairs it.
- Floating comparison tolerance: **1e-9** in the native numeric comparison units. This addresses floating arithmetic at exactly 0.100 s or 0.800 mm; it is not a biological tolerance.
- No automatic conversion from strain to displacement, no inferred loaded-needle count, and no fracture/drop threshold. Old 10%/20% and 0.1 s drop discussions are not the sustained-trigger definition.
- Test mode is whole patch/array. Nominal needle height is **1.6 mm**, supplied by the user. The +0.8 mm is 50% of nominal height, not a measured material strain: deformation can already have occurred before 1 N.

The sample-gap and reversal tolerances are software conventions tied to the available instrument export. They have synthetic regression coverage and were checked against all six Group 9 traces. A different acquisition format needs review rather than silent guessing.

## What is stored and modeled

| Record | Meaning |
|---|---|
| `terminal_force_08mm_status` | 1 for complete terminal force, 0 otherwise; full status/reason retained in provenance |
| `terminal_force_08mm_N_total` | Whole-patch total force at the defined displacement, N |
| `critical_axial_load_N_per_needle` | Compatibility field: terminal total force divided by actual `needles_compressed`, tagged with the new definition |
| `analysis_provenance` | Settings, trigger index/time/displacement, terminal bracket/time, total and nominal force, raw file identity/hash and interpretation |

The compatibility name **does not mean fracture force**. The per-needle quantity is **nominal N per loaded needle**, not the fracture strength of an individual tip. Raw total force is retained so the array-level result is always available.

The production training view uses only one comparable mechanical definition. From Group 10 proposals it excludes old maximum-force labels while retaining their viability/intact information. Historical observations are not deleted or rewritten. Existing mechanical phase gates use comparable new-definition paired measurements; thresholds and GP methods are unchanged. A smaller compatible mechanical evidence set can delay the mechanical phase. Control observations remain excluded from production training.

## Round CSV and automatic extraction

Use the Group 9 worksheet with its hash-bound endpoint addendum or a Group 10+ worksheet with frozen endpoint settings. For each compressed specimen, provide its `instron_file`, actual `needles_compressed` when known, an explicit `replicate_id`, and fresh viability/intact results as appropriate. The ordinary round-update command extracts the endpoint using the applicable contract. There is no requirement to enter replicate totals in advance.

Calculated compatibility fields may be left blank for automatic extraction. A supplied force must match the reproducible terminal value within 1e-6 N in its field's units. The existing stiffness column stores apparent secant stiffness `(F_terminal-F_trigger)/0.8 mm` per loaded needle and must also reproduce within 1e-6. A mismatched old maximum/slope is rejected. An unlabeled manual value without raw data is not accepted. Do not run the legacy maximum-force sheet-population helper for Group 9+ terminal-method evidence.

A nonnegative terminal force remains valid even when it is below the trigger-origin force. The signed secant is retained, with `stiffness_qc=negative_secant` in provenance and mechanical report tables. This flag does not remove either measured value from the observations or change the force endpoint's completion status.

A lost compression attempt can be represented by a row with `mechanical_test_attempted=true`, its replicate/test identity and no raw result. A short/incomplete raw trace remains recorded but supplies no force label. Four mechanical slots mean four formulations; attempted compression rows and interpretable outcomes are counted separately. Reusing the same raw-file hash for multiple specimens/formulations in one update is rejected.

For a standalone review, run from the branch root:

```sh
python3 src/08_multi_objective/04_report_campaign/analyze_mechanical.py /absolute/path/to/instron.csv --output-dir /absolute/path/to/new/analysis --needles 100 --proposal-metadata /absolute/path/to/round/proposal/selection_metadata.json
```

`100` is only a command example: use the actual loaded count, or omit `--needles` to review total force only. Column names and units are detected from the CSV; explicit column options resolve genuine ambiguity. This writes a JSON and annotated PNG to the specified output directory. `supplementary_analysis_file` is retained as a compatible optional field: if supplied, the round update rechecks its settings, hash and values. It is not required for automatic extraction.

## Scientific interpretation and reporting

Use **whole-patch terminal compression resistance at +0.8 mm after a sustained 1 N reference**. This is a protocol-specific force metric. It is not elastic modulus, isolated-needle stiffness, fracture strength or proof of skin penetration. No imaging validation is required for this force metric, and none is claimed.

Keep needle geometry, array loading/count convention, platen setup, zeroing, compression speed, thermal handling and timing consistent. At the observed approximately 0.75 mm/min, +0.8 mm takes about 64 s after the trigger; the CSV cannot establish specimen temperature or absence of melting. The shorter window limits additional travel/time but does not itself rule out damage or backing contribution. The 1 N seating reference is not first physical contact.

Group 9 supplied the first terminal-method evidence under a hash-bound validation/ingestion addendum. Its original proposal CSV and metadata remain unchanged. The six newly staged raw curves are labelled with the terminal definition and apparent-secant formula; earlier historical observations are not relabelled. Observed frontiers and prospective metric pools identify endpoint definitions and never pool old maximum-force/early-slope labels with terminal-method labels.

Mechanical report generation verifies every raw source against its recorded SHA-256 before writing any figure bundle. A changed source is rejected; existing reports remain intact. Graphs are rendered from the same byte snapshots whose hashes were verified.

# Literature concentration corrections — v2

This release fixes 42 confirmed concentration fields across 35 of the
198 parsed formulations. It replaces the active raw and parsed CSVs and saves
identical `_v2.csv` snapshots. Model retraining, manuscript recalculation,
experimental observations and the multi-objective pipeline are outside its scope.

## Error mechanism and provenance

The concentration extractor in commit `97efc14` lowercased input and used
case-insensitive matching, erasing the distinction between `M` and `mM`.
It inferred units from the regular-expression source: the string `[mM]` itself
contains `mM`, so molar concentrations were classified as millimolar and divided
by 1,000. The 198-row dataset in `8294f76` contains these errors.
The parser change in `88443af` removed the 500-character reference limit;
it did not introduce the concentration defect.

The corrected parser preserves unit case and uses the unit beside the matched
value. Ingredient names are case-insensitive. `M` passes through unchanged;
`mM` and `mmol/L` are divided by 1,000. Units elsewhere in a formulation, such
as a polymer's `wt%`, cannot determine a sugar's unit.

Raw record 102 contains a separate transcription error: `o.5M` (letter o)
instead of `0.5M`. Woods et al. describe 0.5, 1.0 and 1.5 M propylene glycol
([original paper](https://doi.org/10.1016/j.cryobiol.2009.06.005)). The defective
extractor matched the trailing `5M` and divided by 1,000, producing 0.005 M.
The correct value is 0.5 M. The corrected parser rejects that malformed numeric
token. All other raw CSV cells are unchanged; their stated molar units are
correct for the confirmed concentration entries.

## Files and traceability

- Active input: `data/raw/Cryopreservative Data 2026.csv`
- Raw v2 snapshot: `data/raw/Cryopreservative Data 2026_v2.csv`
- Active output: `data/processed/parsed_formulations.csv`
- Parsed v2 snapshot: `data/processed/parsed_formulations_v2.csv`
- Audit and review policy: `data/raw/concentration_review_v2.json`

The review file records every changed field, its baseline and corrected molar
values, contributing raw-record IDs, and verification sources. Formulation IDs
are retained identifiers from the raw table, not consecutive output row numbers.
The canonical CSVs and v2 snapshots are byte-identical. Reference order and all
unchanged cell text are preserved in the saved output.

## Unresolved concentrations

These five fields retain their legacy values under an explicit pending-review
hold. **The retained numbers are not scientifically validated concentrations.**
They are preserved to limit this release to confirmed corrections.

| Formulation ID | Field | Retained legacy value (M) | Raw text (M) | Unresolved issue |
|---|---|---:|---:|---|
| 237 | Sucrose | 0.0005 | 0.5 | Published abstract and Table 1 state 0.5 M; accessible author-manuscript Methods state 0.75 M. |
| 326 | Trehalose | 0.0002 | 0.2 | Source DOI appears misattributed; the relevant original table requires verification. |
| 332 | Ethylene glycol | 0.0015 | 1.5 | Same source-table verification gap. |
| 332 | Sucrose | 0.0001 | 0.1 | Same source-table verification gap. |
| 339 | Sucrose | 0.0003 | 0.3 | Concentration requires confirmation against the original Methods. |

Pure `parse_csv()` reads the units as written. The default script applies the
separately documented review holds before aggregation and emits a warning.
Its raw-file SHA-256 check prevents these row-specific holds from being silently
applied to changed or reordered input. Input changes require a corresponding
review update for output generation. The unresolved holds prevent classification
of v2 as a fully validated training release.

Formulation 323's 0.1 M sucrose value is supported by Balci & Can
([DOI 10.2174/1574888X11308010008](https://doi.org/10.2174/1574888X11308010008)).
Its recorded CSV source DOI appears misattributed; that source string and the
other unresolved provenance issues are unchanged in this concentration-only
release. The audit records the paper used to confirm the concentration.
Conflicts between dedicated DMSO columns and formulation prose are also outside
this M/mM correction; the DMSO-column precedence is unchanged.

## Matching compositions and preserved targets

Corrected units give two pairs of equal feature vectors:

| Formulation IDs | Shared composition | Recorded viability targets (%) |
|---|---|---|
| 2 and 386 | 0.25 M trehalose | 12.5 and 77.0 |
| 41 and 372 | 0.2 M sucrose | 51.5 and 22.5 |

Unrestricted duplicate aggregation would merge these pairs and re-average the
contributing raw measurements to 55.5% and 37.0%, respectively. That would change
target data in addition to correcting concentrations. The review file preserves
the four baseline aggregation groups. All other duplicate handling is unchanged.
Combining these cross-study observations requires a separate decision;
v2 retains all 198 records.

## Verification and model status

Twelve regression tests cover M/mM and mmol/L conversion, ingredient case,
malformed concentrations, mixed-unit formulations, percentages and mass units,
full-length references, the 42 confirmed corrections, the five pending holds,
unchanged targets for equal profiles, regeneration from raw input,
and rejection of mismatched review/input combinations.

```bash
python -m unittest discover -s tests -p 'test_formulation_concentration_units.py' -v
python src/01_data_parsing/parse_formulations.py
```

The parser reproduces all saved numeric values and complete reference content;
reference ordering within a merged string can vary because the aggregation uses
a set. Targets, formulation IDs, source strings and all non-corrected data cells
match the baseline dataset. Model pickle hashes match the baseline files.
Saved models contain the uncorrected training concentrations and are not
validated against v2. This correction includes no model retraining.

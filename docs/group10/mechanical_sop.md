# Supplementary axial-load endpoint SOP

**Definition:** maximum supported force before the first qualifying force-drop event, within 1 mm displacement relative to detected needle–platen contact. This is a protocol-defined load capacity, not ultimate fracture strength.

## Required configuration

Use `config_v2/group10.yaml: mechanical_endpoint`. The stop is 1.0 mm and the relative drop is 0.10. Supply protocol ID, single_needle/array mode, force sign (+1 or −1), baseline point count, contact force threshold (N above baseline), absolute drop threshold (N), drop window (mm), and persistence point count. This first detector supports N, mm and no smoothing; unsupported unit/smoothing settings return protocol_incomplete rather than being guessed. Choose settings from actual instrument traces and document them. External imaging is not required for this operational definition.

Baseline is the mean of the configured initial pre-contact samples. Contact is the first later sample crossing the baseline-corrected threshold. The stop is relative to that contact sample. Sampling resolution therefore affects the contact/event precision. Settings must be appropriate to the recorded approach segment.

A local peak qualifies when at least the configured number of consecutive subsequent samples remain below it by both 10% and the absolute threshold, within the displacement window. Select the earliest qualifying peak. The endpoint is the maximum load up to that peak; later loading is excluded. With no event, the curve must reach 1 mm. Interpolate force at 1 mm if the first crossing sample lies beyond it; exclude post-window peaks.

Reject invalid/nonfinite samples rather than stitching gaps. Flag unloading within the window as ambiguous. A short test without an event is incomplete, not a lower-valued complete test. A qualifying event can terminate a usable result before 1 mm. No event means **no qualifying drop detected**, never no failure. Ordinary regression can eventually model this operational value for both complete branches; a censored-fracture interpretation is not being implemented.

For an array, divide by documented loaded-needle count and label nominal N per loaded needle. Backing/structural contributions can affect a 1 mm test. Do not claim isolated needle strength. Review curves for invalid contact/backing artifacts; a trace alone cannot identify all physical causes.

## Analysis and ingestion

Run the stage-4 `analyze_mechanical.py` CLI with source file, explicit output directory, exact force/displacement columns, loaded needle count and `--proposal-metadata` pointing to the frozen proposal JSON. It produces `mechanical_analysis.json` and an annotated PNG. Without frozen metadata it can generate exploratory analysis from current settings, but ingestion requires exact agreement with frozen settings.

Enter the JSON path in `supplementary_analysis_file` and the matching raw path in `instron_file`. Ingestion checks the raw-file hash and frozen settings. Results are appended under distinct supplementary endpoint names, including a status record and the operational load when usable. Original maximum-force processing remains unchanged. Keep raw files available and preserve analysis provenance.

Group 9 reanalysis is allowed only as a separate derived artifact when the same contact-referenced protocol is documented. Do not overwrite Group 9 observations or use supplementary values in the production mechanical GP before the later user decision.

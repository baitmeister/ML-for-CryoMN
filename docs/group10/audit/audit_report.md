# Offline model audit

Development evidence: frozen candidates from Groups 3–8. No policy or model is promoted.
Group 9 outcomes are not used. This is not a replay of alternative experimental selections.

| Variant | MAE (points) | Bias | R² | 95% coverage | Mean width |
|---|---:|---:|---:|---:|---:|
| campaign_bounds_learned | 14.989 | 9.739 | -0.111 | 0.722 | 46.749 |
| campaign_existing_adaptive | 15.594 | 11.246 | -0.178 | 0.833 | 55.338 |
| legacy_correction | 15.739 | 10.537 | -0.204 | 0.986 | 98.206 |
| campaign_existing | 15.854 | 11.147 | -0.193 | 0.792 | 53.300 |
| campaign_median | 16.302 | 9.525 | -0.227 | 0.931 | 67.171 |
| campaign_mean | 16.612 | 10.790 | -0.280 | 0.931 | 67.171 |
| current_mixed | 27.456 | 25.224 | -2.303 | 0.667 | 69.912 |

Review the simplest campaign-only alternatives first; the observed prior/source mismatch is material. Learned-kernel and correction variants require fresh prospective validation.
Adaptive noise assumes independent preparations only when explicitly identified. Historical unknown independence uses a conservative fallback.
The correction uncertainty uses an independent-component approximation; it is diagnostic, not validated calibration.
No repeated configured controls or supplementary mechanical endpoints exist in this snapshot: batch adjustment and mechanical-target comparison are not estimable.
The separate ModelListGP implementation is tested on synthetic unequal endpoint datasets; it has not replaced production acquisition.
Failed fits: 0. Details and convergence warnings are retained in CSV/JSON.
User decision during full mechanics is required before any production switch.

For model definitions, metric interpretation, per-group qualifications, and uncertainty/noise limitations, read [the explained comparison](explained_comparison.md).

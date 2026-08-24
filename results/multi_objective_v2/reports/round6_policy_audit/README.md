# Round 3-6 Forward-Policy Audit

This is a counterfactual audit only. Rounds 1-5 and their frozen proposals were not changed.
The audit identifies positions the new rules would have constrained; it does not claim that unknown replacements would have performed better.

| Round | Cold ingredients under reconstructed prior evidence | Ordinary cold-cap violations | Rows with intact deduction | Largest deduction |
|---|---|---|---:|---:|
| ROUND_003 | glycerol_M, propylene_glycol_M, raffinose_M, taurine_M, myo_inositol_M, acetamide_M, methylcellulose_pct | none | 2 | 0.0667 |
| ROUND_004 | propylene_glycol_M, raffinose_M, taurine_M, myo_inositol_M, acetamide_M, methylcellulose_pct | raffinose_M=5 | 0 | 0.0000 |
| ROUND_005 | propylene_glycol_M, taurine_M, myo_inositol_M, acetamide_M, methylcellulose_pct | acetamide_M=5 | 2 | 0.0813 |
| ROUND_006 | propylene_glycol_M, taurine_M, myo_inositol_M, methylcellulose_pct | none | 1 | 0.0667 |

- ROUND_004: 5 raffinose-containing rows averaged 16.56% observed viability; the other 7 averaged 44.20%. The independent cap would retain at most two ordinary raffinose rows and free three positions.

- ROUND_005: 5 acetamide-containing rows averaged 21.54% observed viability; the other 7 averaged 30.43%. The independent cap would retain at most two ordinary acetamide rows and free three positions.

- ROUND_003 intact counterfactual: ordinary candidate `cand_000293` would have received a 0.021 screening deduction and subsequently failed the intact gate.

- ROUND_005 intact counterfactual: candidate `cand_000731` had closer/stronger weighted pass than failure evidence, would have received no deduction, and subsequently passed intact.

The detailed CSV contains every proposal row, reconstructed empirical intact evidence, cold-start status, and observed outcome where available.

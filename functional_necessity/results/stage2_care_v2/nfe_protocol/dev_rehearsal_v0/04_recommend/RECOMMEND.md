# NFE recommendation — DEV REHEARSAL

Screening metrics on selected observations. Event-enriched rates are stress-test rates, not deployment rates.

| model | strategy | G=1 | rec. NFE | NFE | status | arm med ratio | arm P95 ratio | grip dev / floor (comparisons) | severe dev / floor | reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| pi05 | random | 19 | 2 | 1 | reject | 1.22 [1.13, 1.35] | 0.94 [0.65, 1.11] | 5/240 vs 2/120 | 0.000 vs 0.000 | arm median > floor median; gripper deviation rate > floor rate |
| pi05 | random | 19 | 2 | 2 | pass | 0.89 [0.83, 1.00] | 0.60 [0.41, 0.70] | 2/240 vs 2/120 | 0.000 vs 0.000 | — |
| pi05 | random | 19 | 2 | 3 | not evaluable | | | | | no probe output for this NFE |
| pi05 | alpha | 34 | 2 | 1 | reject | 1.17 [1.07, 1.25] | 0.84 [0.57, 1.05] | 7/240 vs 7/120 | 0.000 vs 0.000 | arm median > floor median |
| pi05 | alpha | 34 | 2 | 2 | pass | 0.82 [0.75, 0.88] | 0.46 [0.30, 0.56] | 4/240 vs 7/120 | 0.000 vs 0.000 | — |
| pi05 | alpha | 34 | 2 | 3 | not evaluable | | | | | no probe output for this NFE |
| pi05 | g_only | 55 | 2 | 1 | reject | 1.20 [1.09, 1.30] | 0.88 [0.63, 1.18] | 10/240 vs 8/120 | 0.000 vs 0.000 | arm median > floor median |
| pi05 | g_only | 55 | 2 | 2 | pass | 0.88 [0.80, 0.94] | 0.53 [0.35, 0.74] | 6/240 vs 8/120 | 0.000 vs 0.000 | — |
| pi05 | g_only | 55 | 2 | 3 | not evaluable | | | | | no probe output for this NFE |
| pi05 | alpha_event | 55 | 2 | 1 | reject | 1.16 [1.07, 1.27] | 0.90 [0.65, 1.11] | 10/240 vs 8/120 | 0.000 vs 0.000 | arm median > floor median |
| pi05 | alpha_event | 55 | 2 | 2 | pass | 0.85 [0.79, 0.90] | 0.49 [0.31, 0.59] | 6/240 vs 8/120 | 0.000 vs 0.000 | — |
| pi05 | alpha_event | 55 | 2 | 3 | not evaluable | | | | | no probe output for this NFE |
| smolvla | random | 19 | 2 | 1 | reject | 1.18 [1.10, 1.28] | 0.92 [0.82, 1.20] | 8/240 vs 4/120 | 0.000 vs 0.000 | arm median > floor median |
| smolvla | random | 19 | 2 | 2 | pass | 0.72 [0.68, 0.77] | 0.54 [0.48, 0.71] | 2/240 vs 4/120 | 0.000 vs 0.000 | — |
| smolvla | random | 19 | 2 | 3 | pass | 0.53 [0.51, 0.58] | 0.37 [0.33, 0.50] | 1/240 vs 4/120 | 0.000 vs 0.000 | — |
| smolvla | alpha | 34 | 2 | 1 | reject | 1.13 [1.07, 1.24] | 0.99 [0.87, 1.07] | 11/240 vs 7/120 | 0.000 vs 0.000 | arm median > floor median |
| smolvla | alpha | 34 | 2 | 2 | pass | 0.69 [0.65, 0.75] | 0.55 [0.49, 0.62] | 2/240 vs 7/120 | 0.000 vs 0.000 | — |
| smolvla | alpha | 34 | 2 | 3 | pass | 0.50 [0.47, 0.55] | 0.39 [0.33, 0.42] | 1/240 vs 7/120 | 0.000 vs 0.000 | — |
| smolvla | g_only | 55 | 2 | 1 | reject | 1.24 [1.16, 1.30] | 0.99 [0.80, 1.19] | 13/240 vs 8/120 | 0.000 vs 0.000 | arm median > floor median |
| smolvla | g_only | 55 | 2 | 2 | pass | 0.75 [0.72, 0.79] | 0.57 [0.47, 0.74] | 3/240 vs 8/120 | 0.000 vs 0.000 | — |
| smolvla | g_only | 55 | 2 | 3 | pass | 0.56 [0.53, 0.59] | 0.40 [0.32, 0.50] | 2/240 vs 8/120 | 0.000 vs 0.000 | — |
| smolvla | alpha_event | 55 | 2 | 1 | reject | 1.16 [1.07, 1.24] | 0.98 [0.87, 1.09] | 14/240 vs 8/120 | 0.000 vs 0.000 | arm median > floor median |
| smolvla | alpha_event | 55 | 2 | 2 | pass | 0.72 [0.67, 0.76] | 0.55 [0.50, 0.63] | 3/240 vs 8/120 | 0.000 vs 0.000 | — |
| smolvla | alpha_event | 55 | 2 | 3 | pass | 0.52 [0.48, 0.55] | 0.39 [0.34, 0.43] | 2/240 vs 8/120 | 0.000 vs 0.000 | — |
| molmoact2 | — | — | — | — | not evaluated | | | | | no fixed-noise probe yet; no Molmo dev outputs; thresholds not borrowed from Pi/Smol |

Rule: candidates [1, 2, 3], reference NFE 10, fallback 10; arm ratio ≤ 1.0 (median) and ≤ 1.0 (P95); gripper rate diff ≤ 0.0; severe (≥3 steps) diff ≤ 0.0.

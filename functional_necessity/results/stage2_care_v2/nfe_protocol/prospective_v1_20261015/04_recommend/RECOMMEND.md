# NFE recommendation — PROSPECTIVE

Development evidence. Primary `recommended_nfe` = abs E_N≤0.085. Three rules reported independently.

| model | strategy | G=1 | abs≤0.085 | abs improve | rel improve | joint (supp.) | primary |
|---|---|---|---|---|---|---|---|
| pi05 | random | 17 | 1 | 1 | 1 | 1 | 1 |
| pi05 | alpha | 35 | 1 | 1 | 1 | 1 | 1 |
| pi05 | g_only | 48 | 1 | 1 | 1 | 1 | 1 |
| pi05 | alpha_event | 48 | 1 | 1 | 1 | 1 | 1 |
| smolvla | random | 17 | 2 | 2 | 2 | 2 | 2 |
| smolvla | alpha | 35 | 2 | 2 | 2 | 2 | 2 |
| smolvla | g_only | 48 | 2 | 2 | 2 | 2 | 2 |
| smolvla | alpha_event | 48 | 2 | 2 | 2 | 2 | 2 |
| molmoact2 | random | 17 | 1 | 1 | 1 | 1 | 1 |
| molmoact2 | alpha | 35 | 1 | 1 | 1 | 1 | 1 |
| molmoact2 | g_only | 48 | 1 | 1 | 1 | 1 | 1 |
| molmoact2 | alpha_event | 48 | 1 | 1 | 1 | 1 | 1 |

Candidates [1, 2, 3], reference NFE 10, fallback 10. Metrics: `nfe_en_metrics` (LIBERO norm, H=10, 7-d, mean over obs×noise).

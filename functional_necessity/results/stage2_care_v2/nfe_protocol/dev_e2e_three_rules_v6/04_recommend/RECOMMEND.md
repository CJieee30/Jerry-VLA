# NFE recommendation — DEV REHEARSAL

Development evidence. Primary `recommended_nfe` = abs E_N≤0.085. Three rules reported independently.

| model | strategy | G=1 | abs≤0.085 | abs improve | rel improve | joint (supp.) | primary |
|---|---|---|---|---|---|---|---|
| pi05 | random | 19 | 1 | 1 | 1 | 1 | 1 |
| pi05 | alpha | 34 | 1 | 1 | 1 | 1 | 1 |
| pi05 | g_only | 55 | 1 | 1 | 1 | 1 | 1 |
| pi05 | alpha_event | 55 | 1 | 1 | 1 | 1 | 1 |
| smolvla | random | 19 | 2 | 2 | 2 | 2 | 2 |
| smolvla | alpha | 34 | 2 | 2 | 2 | 2 | 2 |
| smolvla | g_only | 55 | 2 | 2 | 2 | 2 | 2 |
| smolvla | alpha_event | 55 | 2 | 2 | 2 | 2 | 2 |
| molmoact2 | random | 19 | 1 | 1 | 1 | 1 | 1 |
| molmoact2 | alpha | 34 | 1 | 1 | 1 | 1 | 1 |
| molmoact2 | g_only | 55 | 1 | 1 | 1 | 1 | 1 |
| molmoact2 | alpha_event | 55 | 1 | 1 | 1 | 1 | 1 |

Candidates [1, 2, 3], reference NFE 10, fallback 10. Metrics: `nfe_en_metrics` (LIBERO norm, H=10, 7-d, mean over obs×noise).

# NFE recommendation — PROSPECTIVE

Development evidence. Primary `recommended_nfe` = abs E_N≤0.085. Three rules reported independently.

| model | strategy | G=1 | abs≤0.085 | abs improve | rel improve | joint (supp.) | primary |
|---|---|---|---|---|---|---|---|
| pi05 | random | 17 | 1 | 2 | 2 | 2 | 1 |
| pi05 | alpha | 35 | 1 | 2 | 2 | 2 | 1 |
| pi05 | g_only | 48 | 1 | 2 | 2 | 2 | 1 |
| pi05 | alpha_event | 48 | 1 | 2 | 2 | 2 | 1 |

Candidates [1, 2, 3], reference NFE 10, fallback 10. Metrics: `nfe_en_metrics` (LIBERO norm, H=10, 7-d, mean over obs×noise).

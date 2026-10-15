# Validation (holdout) — PROSPECTIVE

Per-rule holdout pass/fail/NA/reference for **calibration-chosen** NFE (not holdout re-pick).
Diagnostic holdout re-pick is in validation.json only.

| model | strat | rule | screening NFE | holdout | closed loop |
|---|---|---|---|---|---|
| pi05 | random | abs_0.085 | 1 | fail | fail |
| pi05 | random | abs_improve | 2 | pass | pass |
| pi05 | random | rel_improve | 2 | pass | pass |
| pi05 | alpha | abs_0.085 | 1 | fail | fail |
| pi05 | alpha | abs_improve | 2 | pass | pass |
| pi05 | alpha | rel_improve | 2 | pass | pass |
| pi05 | g_only | abs_0.085 | 1 | fail | fail |
| pi05 | g_only | abs_improve | 2 | pass | pass |
| pi05 | g_only | rel_improve | 2 | pass | pass |
| pi05 | alpha_event | abs_0.085 | 1 | fail | fail |
| pi05 | alpha_event | abs_improve | 2 | pass | pass |
| pi05 | alpha_event | rel_improve | 2 | pass | pass |

# OpenVLA-OFT suite-aware CosSim / Fisher

Each suite uses its own 128 calibration samples. Global uses all 512.
High score = keep. Drop-k is the k lowest scores.

| Method | Scope | Drop-4 | Drop-8 | vs global Drop-4 | vs global Drop-8 |
|---|---|---|---|---|---|
| cossim | global | `[3, 4, 5, 6]` | `[3, 4, 5, 6, 19, 20, 23, 28]` | same | same |
| cossim | spatial | `[3, 4, 5, 6]` | `[3, 4, 5, 6, 19, 23, 27, 28]` | same | different |
| cossim | object | `[3, 4, 5, 6]` | `[3, 4, 5, 6, 19, 21, 23, 28]` | same | different |
| cossim | goal | `[3, 4, 5, 6]` | `[2, 3, 4, 5, 6, 19, 20, 23]` | same | different |
| cossim | long | `[3, 4, 5, 6]` | `[3, 4, 5, 6, 19, 23, 27, 28]` | same | different |
| fisher | global | `[0, 3, 4, 30]` | `[0, 3, 4, 5, 28, 29, 30, 31]` | same | same |
| fisher | spatial | `[0, 3, 30, 31]` | `[0, 3, 4, 5, 28, 29, 30, 31]` | different | same |
| fisher | object | `[0, 3, 4, 30]` | `[0, 3, 4, 5, 28, 29, 30, 31]` | same | same |
| fisher | goal | `[0, 3, 30, 31]` | `[0, 2, 3, 4, 5, 29, 30, 31]` | different | different |
| fisher | long | `[0, 3, 4, 30]` | `[0, 3, 4, 5, 28, 29, 30, 31]` | same | same |

# Last-layer open-loop identity

Keep-KV skip of the last VLM layer vs the full-model action chunk.
32 calibration frames, NFE=1 and NFE=10, seed offset 1000.
Not a LIBERO eval. Action expert is not dropped.

| stack | last layer | NFE | mean max-abs | max max-abs | identity (≤1e-4) |
|---|---:|---:|---:|---:|---|
| pi05 | 17 | 1 | 0.000e+00 | 0.000e+00 | yes |
| pi05 | 17 | 10 | 0.000e+00 | 0.000e+00 | yes |
| molmoact2 | 35 | 1 | 1.621e-03 | 7.813e-03 | no |
| molmoact2 | 35 | 10 | 3.719e-03 | 6.836e-03 | no |
| smolvla | 15 | 1 | 0.000e+00 | 0.000e+00 | yes |
| smolvla | 15 | 10 | 0.000e+00 | 0.000e+00 | yes |


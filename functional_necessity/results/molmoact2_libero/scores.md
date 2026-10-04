# MolmoAct2 LIBERO redundancy scores

Checkpoint: `allenai/MolmoAct2-LIBERO-LeRobot`.
VLM layers only. A dropped layer keeps K/V for the paired action expert and does not update the residual.
Loss: continuous flow-matching, one frozen (noise, time) per sample. Greedy-PPL is not included.
Calibration samples: 32.

| Method | Drop-4 | Drop-8 |
|---|---|---|
| magnitude | `[1, 2, 3, 21]` | `[1, 2, 3, 18, 19, 20, 21, 22]` |
| cossim | `[1, 2, 3, 4]` | `[1, 2, 3, 4, 5, 6, 7, 8]` |
| cka | `[29, 30, 31, 32]` | `[2, 3, 28, 29, 30, 31, 32, 33]` |
| ppl | `[4, 6, 7, 23]` | `[3, 4, 6, 7, 23, 26, 28, 34]` |
| taylor | `[32, 33, 34, 35]` | `[28, 29, 30, 31, 32, 33, 34, 35]` |
| fisher | `[30, 32, 33, 35]` | `[28, 29, 30, 31, 32, 33, 34, 35]` |
| igia | `[32, 33, 34, 35]` | `[28, 29, 30, 31, 32, 33, 34, 35]` |
| gateprobe | `[30, 31, 33, 35]` | `[28, 29, 30, 31, 32, 33, 34, 35]` |
| hessian | out of memory | out of memory |

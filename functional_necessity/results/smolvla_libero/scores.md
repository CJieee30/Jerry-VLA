# SmolVLA LIBERO redundancy scores

Checkpoint: `lerobot/smolvla_libero`.
VLM layers only (16). A dropped layer keeps K/V for the paired action expert and does not update the residual.
Loss: continuous flow-matching, one frozen (noise, time) per sample. Greedy-PPL is not included.
Formulas match MolmoAct2 / pi0.5. Calibration: 32 frames from `lerobot/libero`, seed=42.
Calibration samples: 32.

| Method | Drop-4 | Drop-8 |
|---|---|---|
| magnitude | `[0, 3, 7, 8]` | `[0, 2, 3, 4, 6, 7, 8, 9]` |
| cossim | `[2, 4, 5, 6]` | `[1, 2, 4, 5, 6, 7, 8, 10]` |
| cka | `[1, 2, 4, 10]` | `[1, 2, 4, 5, 7, 10, 12, 15]` |
| ppl | `[11, 13, 14, 15]` | `[7, 9, 10, 11, 12, 13, 14, 15]` |
| taylor | `[12, 13, 14, 15]` | `[7, 8, 10, 11, 12, 13, 14, 15]` |
| fisher | `[10, 11, 12, 14]` | `[4, 6, 8, 10, 11, 12, 13, 14]` |
| igia | `[10, 12, 14, 15]` | `[7, 8, 10, 11, 12, 13, 14, 15]` |
| gateprobe | `[7, 13, 14, 15]` | `[7, 8, 10, 11, 12, 13, 14, 15]` |
| hessian | `[8, 10, 12, 14]` | `[6, 7, 8, 10, 11, 12, 13, 14]` |

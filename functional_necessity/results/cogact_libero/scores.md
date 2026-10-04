# DB-CogACT LIBERO redundancy scores

Checkpoint: `Dexmal/libero-db-cogact` (Qwen2.5-7B, 28 layers, DiT-B action head).
A dropped layer is an identity block. The DiT reads the last unmasked hidden state, not per-layer K/V.
Loss: diffusion noise MSE, one frozen (noise, timestep) per sample, `repeated_diffusion_steps=1`.
Calibration: 32 frames from `lerobot/libero`, seed 42, chunk 16, min-max norm from the checkpoint.
High score = keep. Formulas match pi0.5 / Molmo (Fisher / IGIA square the per-example gradient).

| Method | Drop-4 | Drop-8 | Drop-12 |
|---|---|---|---|
| magnitude | `[0, 1, 2, 9]` | `[0, 1, 2, 3, 4, 5, 6, 9]` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14]` |
| cossim | `[4, 7, 8, 9]` | `[2, 4, 6, 7, 8, 9, 10, 19]` | `[2, 4, 6, 7, 8, 9, 10, 11, 14, 18, 19, 20]` |
| cka | `[6, 7, 8, 17]` | `[4, 6, 7, 8, 9, 10, 17, 22]` | `[4, 6, 7, 8, 9, 10, 15, 16, 17, 19, 21, 22]` |
| ppl | `[1, 2, 4, 5]` | `[1, 2, 4, 5, 6, 9, 10, 20]` | `[1, 2, 4, 5, 6, 7, 8, 9, 10, 20, 21, 22]` |
| taylor | `[1, 2, 4, 5]` | `[0, 1, 2, 3, 4, 5, 6, 9]` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 27]` |
| fisher | `[0, 1, 2, 4]` | `[0, 1, 2, 3, 4, 6, 7, 9]` | `[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 20, 25]` |
| igia | `[0, 1, 2, 4]` | `[0, 1, 2, 3, 4, 6, 7, 10]` | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]` |
| gateprobe | `[3, 4, 9, 19]` | `[1, 2, 3, 4, 9, 19, 21, 22]` | `[0, 1, 2, 3, 4, 6, 9, 19, 21, 22, 23, 24]` |
| hessian | out of memory | out of memory | out of memory |

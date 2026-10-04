# pi05_base vs DTR Table 9

DTR scored pretrained π0.5 on LIBERO calibration (64×8, seed 42), then reported keep-sets.
Exact match = same keep set. Jaccard is |A∩B|/|A∪B| over keep indices.

| Method | k | Ours keep | DTR keep | Match | Jaccard |
|---|---|---|---|---|---|
| gateprobe | Drop-9 | [0, 1, 2, 3, 4, 5, 6, 8, 11] | [0, 1, 2, 3, 4, 5, 6, 8, 9] | no | 0.80 |
| gateprobe | Drop-12 | [0, 1, 2, 4, 5, 6] | [0, 1, 2, 5, 6, 8] | no | 0.71 |
| gateprobe | Drop-16 | [0, 5] | [0, 5] | yes | 1.00 |
| gateprobe | Drop-17 | [0] | [0] | yes | 1.00 |
| taylor | Drop-9 | [1, 2, 3, 4, 5, 6, 7, 8, 9] | [0, 1, 3, 4, 5, 6, 7, 8, 9] | no | 0.80 |
| taylor | Drop-12 | [3, 4, 5, 6, 7, 8] | [0, 3, 4, 5, 6, 8] | no | 0.71 |
| taylor | Drop-16 | [4, 5] | [5, 6] | no | 0.33 |
| taylor | Drop-17 | [5] | [5] | yes | 1.00 |
| igia | Drop-9 | [0, 1, 3, 4, 5, 6, 7, 8, 9] | [0, 1, 3, 4, 5, 6, 7, 8, 9] | yes | 1.00 |
| igia | Drop-12 | [0, 3, 4, 6, 7, 9] | [0, 1, 3, 4, 6, 7] | no | 0.71 |
| igia | Drop-16 | [0, 6] | [0, 7] | no | 0.33 |
| igia | Drop-17 | [0] | [0] | yes | 1.00 |
| fisher | Drop-9 | [0, 1, 2, 4, 5, 6, 7, 8, 10] | [0, 1, 2, 3, 4, 5, 6, 7, 8] | no | 0.80 |
| fisher | Drop-12 | [0, 1, 2, 5, 7, 10] | [0, 1, 5, 6, 7, 8] | no | 0.50 |
| fisher | Drop-16 | [0, 7] | [0, 5] | no | 0.33 |
| fisher | Drop-17 | [0] | [0] | yes | 1.00 |
| hessian | Drop-9 | [1, 2, 7, 8, 9, 10, 11, 12, 14] | [0, 1, 2, 3, 5, 8, 9, 11, 13] | no | 0.38 |
| hessian | Drop-12 | [1, 8, 9, 10, 11, 14] | [0, 1, 3, 8, 9, 13] | no | 0.33 |
| hessian | Drop-16 | [8, 10] | [0, 1] | no | 0.00 |
| hessian | Drop-17 | [10] | [0] | no | 0.00 |
| ppl | Drop-9 | [0, 2, 5, 6, 13, 14, 15, 16, 17] | [0, 1, 4, 7, 8, 10, 11, 13, 14] | no | 0.20 |
| ppl | Drop-12 | [0, 2, 5, 13, 14, 15] | [0, 1, 4, 7, 10, 13] | no | 0.20 |
| ppl | Drop-16 | [0, 13] | [0, 13] | yes | 1.00 |
| ppl | Drop-17 | [0] | [0] | yes | 1.00 |
| magnitude | Drop-9 | [2, 3, 6, 12, 13, 14, 15, 16, 17] | [2, 3, 6, 12, 13, 14, 15, 16, 17] | yes | 1.00 |
| magnitude | Drop-12 | [2, 13, 14, 15, 16, 17] | [2, 13, 14, 15, 16, 17] | yes | 1.00 |
| magnitude | Drop-16 | [16, 17] | [16, 17] | yes | 1.00 |
| magnitude | Drop-17 | [17] | [17] | yes | 1.00 |
| cossim | Drop-9 | [0, 1, 2, 12, 13, 14, 15, 16, 17] | [0, 1, 2, 12, 13, 14, 15, 16, 17] | yes | 1.00 |
| cossim | Drop-12 | [0, 1, 13, 15, 16, 17] | [0, 1, 14, 15, 16, 17] | no | 0.71 |
| cossim | Drop-16 | [16, 17] | [16, 17] | yes | 1.00 |
| cossim | Drop-17 | [17] | [17] | yes | 1.00 |

**Exact keep-set matches: 15/32**

Magnitude is data-independent: an exact match means LeRobot `pi05_base` language-layer
weights agree with the OpenPI checkpoint DTR used. Gradient methods can differ slightly
because of BF16, preprocessor, and microbatch reconstruction.


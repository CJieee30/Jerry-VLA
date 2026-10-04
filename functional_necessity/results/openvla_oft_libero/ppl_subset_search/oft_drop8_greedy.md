# OpenVLA-OFT Drop-8 greedy PPL

Same rule as pi0.5 `search_ppl_drop9.py`: each step adds the remaining layer with the smallest action-L1 increase.
Baseline L1: **0.037880** over 512 calibration samples.

- Greedy Drop-8: `[3, 4, 5, 19, 23, 26, 28, 29]`
- Pick order: `[3, 5, 29, 4, 23, 26, 19, 28]`
- Keep: `[0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 24, 25, 27, 30, 31]`
- Loss: 0.092482 (Δ +0.054602)
- Independent PPL Drop-8: `[2, 3, 4, 5, 6, 23, 26, 29]`
- Same set: **False**

- Step 1: pick L3 → `[3]` loss=0.039432 Δ=+0.001552
- Step 2: pick L5 → `[3, 5]` loss=0.043098 Δ=+0.005218
- Step 3: pick L29 → `[3, 5, 29]` loss=0.050211 Δ=+0.012331
- Step 4: pick L4 → `[3, 4, 5, 29]` loss=0.057423 Δ=+0.019543
- Step 5: pick L23 → `[3, 4, 5, 23, 29]` loss=0.065175 Δ=+0.027295
- Step 6: pick L26 → `[3, 4, 5, 23, 26, 29]` loss=0.073844 Δ=+0.035963
- Step 7: pick L19 → `[3, 4, 5, 19, 23, 26, 29]` loss=0.082965 Δ=+0.045084
- Step 8: pick L28 → `[3, 4, 5, 19, 23, 26, 28, 29]` loss=0.092482 Δ=+0.054602


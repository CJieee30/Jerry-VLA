# Stage-2 calibration weights (action discontinuity)

Definition: mean adjacent L2 on **loss-normalized** actions, window H=10.
Weights are then divided by their mean so a uniform field is exactly 1.
GPU scoring uses the 7-d weights if Spearman(7d, arm-6d) ≥ 0.9.

| Stack | Tensor | n | ESS (7d) | Spearman 7d vs 6d | GPU |
|---|---|---:|---:|---:|---|
| pi05 | 50-step normalized chunk already stored with the loss; window = first 10 valid steps | 512 | 386.80 | 0.9375 | 7d |
| smolvla | 50-step normalized chunk already stored with the loss; window = first 10 valid steps | 32 | 22.70 | 0.9824 | 7d |
| molmoact2 | loss keeps the stored 1-step (1,1,32) action; weights use the same preprocessor (masked normalizer + clamp, no PackInputs) on the next 10 frames of the same episode | 32 | 23.30 | 0.7005 | 7d and 6d |

## Per-stack 7-d raw discontinuity

### pi05

n=512 filled=5 ESS=386.80 cv=0.569 raw p0/25/50/75/90/100=0.0201/0.1998/0.3040/0.4414/0.6031/1.3351

50-step normalized chunk already stored with the loss; window = first 10 valid steps

### smolvla

n=32 filled=0 ESS=22.70 cv=0.640 raw p0/25/50/75/90/100=0.0390/0.2101/0.2719/0.4151/0.5598/1.1353

50-step normalized chunk already stored with the loss; window = first 10 valid steps

### molmoact2

n=32 filled=0 ESS=23.30 cv=0.611 raw p0/25/50/75/90/100=0.0152/0.0835/0.0954/0.1602/0.2509/0.3938

loss keeps the stored 1-step (1,1,32) action; weights use the same preprocessor (masked normalizer + clamp, no PackInputs) on the next 10 frames of the same episode
First subsequent step vs stored loss action: max abs diff 0.000e+00 (must match the same normalizer).
Windows with <2 steps (median-filled): 0/32.

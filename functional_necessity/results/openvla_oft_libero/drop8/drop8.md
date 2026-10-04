# OpenVLA-OFT Drop-8

Checkpoint: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`.
Skip: whole Llama decoder layer (no K/V). Loss for scores: action L1.
Closed loop: seed 7, 50 trials, center crop. Anchor is this machine's seed-7 run (95.95%).

| Method | Drop-8 | Spatial | Object | Goal | Long | Avg |
|---|---|---:|---:|---:|---:|---:|
| magnitude | `0 1 2 3 4 6 7 8` | 0.0% | 0.0% | 0.0% | 0.0% | 0.00% |
| taylor | `0 1 2 3 4 5 30 31` | 0.0% | 0.0% | 0.0% | 0.0% | 0.00% |
| fisher (igia) | `0 3 4 5 28 29 30 31` | 0.0% | 0.0% | 0.0% | 0.0% | 0.00% |
| hessian | `24 25 26 27 28 29 30 31` | 3.8% | 0.0% | 0.4% | 1.0% | 1.30% |
| gateprobe | `0 3 4 5 6 23 26 27` | 0.0% | 0.0% | 0.0% | 0.0% | 0.00% |
| cossim | `3 4 5 6 19 20 23 28` | 69.0% | 96.6% | 39.0% | 63.0% | 66.90% |
| cka | `0 2 15 17 18 19 23 24` | 0.0% | 0.0% | 0.0% | 0.0% | 0.00% |
| ppl | `2 3 4 5 6 23 26 29` | 60.8% | 72.8% | 8.6% | 25.6% | 41.95% |

Anchor seed-7: Spatial 93.2%, Object 99.2%, Goal 97.2%, Long 94.2%, avg 95.95%.

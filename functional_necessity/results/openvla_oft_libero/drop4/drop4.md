# OpenVLA-OFT Drop-4

Checkpoint: `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`.
Skip: whole Llama decoder layer (no K/V). Scores are the same nine files as Drop-8, cut at k=4.
Closed loop: seed 7, 50 trials, center crop. Anchor is this machine's seed-7 run (95.95%).

| Method | Drop-4 | Spatial | Object | Goal | Long | Avg |
|---|---|---:|---:|---:|---:|---:|
| magnitude | `0 1 3 6` | 0.0% | 0.0% | 0.0% | 0.0% | 0.00% |
| taylor | `0 1 2 3` | 0.0% | 0.0% | 0.0% | 0.0% | 0.00% |
| fisher | `0 3 4 30` | 0.0% | 0.0% | 0.0% | 0.0% | 0.00% |
| igia | `0 29 30 31` | 0.0% | 0.0% | 0.0% | 0.0% | 0.00% |
| hessian | `28 29 30 31` | 11.6% | 0.6% | 5.6% | 7.2% | 6.25% |
| gateprobe | `0 3 4 5` | 0.0% | 0.0% | 0.0% | 0.0% | 0.00% |
| cossim (ppl) | `3 4 5 6` | 85.2% | 99.2% | 73.0% | 89.0% | 86.60% |
| cka | `0 2 18 19` | 0.0% | 0.0% | 0.0% |  |  |

Anchor seed-7: Spatial 93.2%, Object 99.2%, Goal 97.2%, Long 94.2%, avg 95.95%.

Some suites are still running.

# OpenVLA-OFT Drop-4, layer 0 excluded

Four lowest-scoring layers after removing layer 0 from the redundancy order.
Protocol matches Drop-4: seed 7, 50 trials, center crop, whole-layer identity.
GateProbe uses the finished CosSim/PPL run because the drop set is identical.

| Method | Drop-4 | Spatial | Object | Goal | Long | Avg |
|---|---|---:|---:|---:|---:|---:|
| fisher | `3 4 30 31` | 25.8% | 11.8% | 7.0% | 15.4% | 15.00% |
| igia | `3 29 30 31` | 15.2% | 1.4% | 6.4% | 7.6% | 7.65% |
| gateprobe (same as cossim/ppl) | `3 4 5 6` | 85.2% | 99.2% | 73.0% | 89.0% | 86.60% |
| taylor | `1 2 3 4` | 0.0% | 0.0% |  |  |  |

Anchor seed-7: Spatial 93.2%, Object 99.2%, Goal 97.2%, Long 94.2%, avg 95.95%.

Some suites are still running.

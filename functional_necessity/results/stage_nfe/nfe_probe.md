# NFE probe (this machine, not a paper table)

RTX 3090, `compile_model=false`, CUDA events warmup=5 / repeats=20, one calibration frame, `FlopCounterMode`. Same observation and the same inference noise for the offline check. Fisher Drop-6 is `[1, 13–17]`. Taylor Drop-16 is `[1–4, 24–35]`. Closed-loop is not in this file.

Source: `results/stage_nfe/{pi05,molmoact2}.json`. Script: `scripts/stage_nfe/measure_nfe.py`.

## Offline

NFE=10 against itself is exact (`max_abs=0`). NFE=1 uses the same noise and moves the first 10 executed steps.

| Setting | replay max abs | NFE=1 vs NFE=10 L2 | NFE=10 action L2 |
|---|---:|---:|---:|
| π0.5 full | 0 | 0.662 | 7.916 |
| π0.5 Fisher Drop-6 | 0 | 1.110 | 7.717 |
| MolmoAct2 full | 0 | 0.103 | 3.377 |
| MolmoAct2 Taylor Drop-16 | 0 | 0.677 | 3.866 |

## `predict_action_chunk`

| Setting | NFE | Time (ms) | vs own NFE=10 | FLOPs (G) | vs own NFE=10 |
|---|---:|---:|---:|---:|---:|
| π0.5 full | 10 | 548.5 | — | 5024.4 | — |
| π0.5 full | 1 | 186.7 | −66.0% | 4674.4 | −7.0% |
| π0.5 Fisher Drop-6 | 10 | 525.5 | −4.2% vs full NFE=10 | 3711.6 | −26.1% vs full |
| π0.5 Fisher Drop-6 | 1 | 160.1 | −70.8% vs full NFE=10 | 3361.6 | −33.1% vs full |
| MolmoAct2 full | 10 | 1166.4 | — | 5250.6 | — |
| MolmoAct2 full | 1 | 254.0 | −78.2% | 5176.7 | −1.4% |
| MolmoAct2 Taylor Drop-16 | 10 | 1165.2 | −0.1% vs full NFE=10 | 4043.3 | −23.0% vs full |
| MolmoAct2 Taylor Drop-16 | 1 | 246.9 | −78.8% vs full NFE=10 | 3969.3 | −24.4% vs full |

Drop-6 NFE=10 FLOPs 3711.6 G and Drop-16 NFE=10 FLOPs 4043.3 G match `efficientcy_analysis.md`. Expert-only time: π0.5 403.1 → 40.2 ms (−90.0%, FLOPs 388.9 → 38.9 G); MolmoAct2 1022.8 → 103.7 ms (−89.9%, FLOPs 82.1 → 8.2 G).

# SmolVLA，无恢复

零恢复，keep-KV，NFE=10，每任务 10 集（400）。排序不去掉 L0。

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | — | 78.0 | 89.0 | 89.0 | 76.0 | 83.00 |
| Drop-4 (4/16) | Taylor | 72.0 | 88.0 | 84.0 | 70.0 | 78.50 |
|  | Fisher | 69.0 | 86.0 | 72.0 | 50.0 | 69.25 |
|  | GateProbe | 72.0 | 86.0 | 72.0 | 58.0 | 72.00 |
|  | CosSim | 39.0 | 49.0 | 7.0 | 27.0 | 30.50 |
|  | CKA | 30.0 | 40.0 | 6.0 | 15.0 | 22.75 |

来源：`functional_necessity/main_results/eval_smolvla_libero.md`。

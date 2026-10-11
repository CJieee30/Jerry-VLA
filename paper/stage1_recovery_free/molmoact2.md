# MolmoAct2，无恢复

零恢复，keep-KV，NFE=10，每任务 50 集（2000）。Drop-16 的排序已去掉 L0。Drop-8 的 Taylor、Fisher、GateProbe 是同一层集，该集合不含 L0。Drop-8 CosSim 的 Long 未跑完，未完成的集记 0。Drop-16 CosSim 在 Goal 任务 2 被终止，已完成的 22 个任务全是 0，其余记 0。

| Setting | Metric | Spatial | Object | Goal | Long | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | — | 97.4 | 99.6 | 96.8 | 96.2 | 97.50 |
| Drop-8 (8/36) | Taylor | 97.4 | 100.0 | 97.2 | 96.4 | 97.75 |
|  | Fisher | 97.4 | 100.0 | 97.2 | 96.4 | 97.75 |
|  | GateProbe | 97.4 | 100.0 | 97.2 | 96.4 | 97.75 |
|  | CosSim | 1.4 | 0.0 | 0.0 | 0.0 | 0.35 |
| Drop-16 (16/36) | Taylor | 98.2 | 99.4 | 98.6 | 96.2 | 98.10 |
|  | Fisher | 91.2 | 79.8 | 51.0 | 69.0 | 72.75 |
|  | GateProbe | 91.6 | 96.6 | 77.2 | 82.6 | 87.00 |
|  | CosSim | 0.0 | 0.0 | 0.0 | 0.0 | 0.00 |
|  | Last-16 | 74.4 | 72.2 | 46.4 | 40.0 | 58.25 |

来源：`functional_necessity/main_results/eval_molmoact2_libero.md`。

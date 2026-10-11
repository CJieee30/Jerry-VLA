# 论文三阶段结果

主表数字仍以 `functional_necessity/main_results/` 为准。这里只收论文三阶段要用的表和后续新结果，不覆盖、不复制已有的 `eval_info.json`。

| 阶段 | 目录 | 内容 |
| --- | --- | --- |
| 1 | `stage1_recovery_free/` | 无恢复深度剪枝。baseline 为 NFE=10。 |
| 2 | `stage2_care/` | CARE：用动作不连续给校准帧加权。 |
| 3 | `stage3_ledge/` | LEDGE：当前模型在当前基准上可删到哪一档。 |

NFE=1 不作为第 1 阶段的 baseline。

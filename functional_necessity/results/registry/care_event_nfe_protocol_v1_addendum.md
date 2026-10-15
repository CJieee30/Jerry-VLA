# CARE_event NFE 协议 v1 增补（与 v0 草案并存）

实现文件：`scripts/stage2_care_v2/nfe_protocol_v1.py`（不替换 `nfe_protocol.py`）。

## v1 变更摘要

| 项 | v0 草案 | v1 |
|---|---|---|
| Ledger 阶段 | 可重复 append 同名 stage | 非重复 stage 二次写入报错（`--force-restage` 例外） |
| 哈希 | freeze 时记录 config+code | 新增 `verify-ledger` 全量复核 |
| 闭环 | 仅 config 内嵌 results | `closed-loop-import` 从独立 JSON 导入，与探针误差解耦 |
| 校准帧 | `[start, stop-2]` | `[start, stop-H]`，H=10 完整窗口 |
| Molmo 夹爪 | 易误用 libero mean/std | config 标明 quantile+mask=0+物理 g>0 |

## 开发池三模型指标（CPU，非协议冻结）

- 全池：`results/stage2_care_v2/dev_three_model_screening/pool_observation_level.csv`
- 按 NFE 表：`REPORT.md`、`strategy_nfe_table.csv`
- 阈值候选（开发）：`THRESHOLD_CANDIDATES.md` — **非已验证通用规则**
- 待决：`FREEZE_REMAINING.md`
- 冻结前收口：`results/stage2_care_v2/dev_nfe_rules_cpu_v2/PRE_FREEZE_REPORT.md`（`nfe_en_metrics` 统一口径）
- E2E 演练：`nfe_protocol/dev_e2e_three_rules_v8/`（开发证据）

```bash
$PY scripts/stage2_care_v2/nfe_protocol_v1.py --config $C verify-ledger
$PY scripts/stage2_care_v2/nfe_protocol_v1.py --config $C closed-loop-import --closed-loop-json path/to/cl.json
```

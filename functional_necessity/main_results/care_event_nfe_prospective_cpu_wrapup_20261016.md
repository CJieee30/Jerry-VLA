# CPU 收口修改摘要（2026-10-16）

## 协议运行 `prospective_v1_20261015`

1. **`build_historical_closed_loop_import.py`** → `05_validation/historical_closed_loop_import.json`（`success_rate_unit=percent`，附 `eval_info.json` 路径与成功数）。
2. **`closed-loop-import`** → `closed_loop_import.json`（ledger 已记录；标明历史复用、早于 freeze）。
3. **`validate`** → `06_validate/validation.json`、`VALIDATE.md`（holdout + 闭环列均为 pass）。
4. **`verify-ledger`** → 通过。

说明：曾误跑过一次无闭环的 `validate`，已从 ledger 删除该条后重新 `validate`。

## 主报告

- 更新 `care_event_action_expert_nfe.md`：§1–3 方法、§4–§8 开发标注、新增 §9 正式前瞻、重写 §10–12、扩展 §13 链接。
- 未启动 GPU；未 commit/push。

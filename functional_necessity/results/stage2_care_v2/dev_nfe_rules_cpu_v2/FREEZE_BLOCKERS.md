# FREEZE_BLOCKERS（分阶段）

## 冻结前（文档/顾问）

- 主规则 E_N≤0.085 与改善规则**尚未证明**能预测闭环 SR（仅开发一致）
- 顾问需确认：主规则单一冻结 vs 保留两改善规则为对照
- 前瞻 `prospective_prefreeze_v1` 尚未执行 `freeze`（本轮故意未做）

## 推理前（prospective GPU）

- 新校准/验证集 `split` + 各模型 `probe-import`（未揭示数据上的探针封存）
- Molmo 官方后处理器批量对照：`molmo_postprocessor_check.py`（可选 CPU；重载 policy 时耗显存）

## 验证阶段（前瞻完成后）

- `closed-loop-import` + 独立验证集 `validation-import`
- 闭环 JSON 与推荐 NFE 解耦消费（机制已接，数据未产生）

**不列为冻结阻塞**：历史 Molmo holdout 缺失、Pi 无 NFE3 旧探针、尚未产生的未来闭环数字（不要求重跑旧 GPU）。

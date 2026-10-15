# 冻结前剩余事项（开发池 480，2026-10-15）

## 已完成（本轮 CPU）

- 三模型筛查按 **model×strategy×NFE** 拆分；旧跨 NFE 汇总已归档至 `_archive_pre_nfe_split/`
- 全池 `pool_observation_level.csv` + `conversion_manifest.json` / `conversion_check.json`
- 20× Random + 20× G-only 阈值稳定性（`THRESHOLD_CANDIDATES.md`，开发候选）
- Molmo 全池探针 DONE；协议配置夹爪映射修正为 quantile+物理 g>0
- 主协议实现另存 `nfe_protocol_v1.py`（ledger 不可重复追加、校验、闭环独立导入、校准帧 H=10）

## 仍 pending（不伪造）

- Molmo **holdout/前瞻验证**探针输出（`dev_validation_probe`）
- 前瞻 `prospective` split/freeze 与独立验证集 seal-import
- 闭环成功率导入与离线 RMSE 推荐**解耦**后的正式 validate（需 `closed-loop-import` + 可比 episode 协议）
- 阈值规则**冻结**（当前均为开发候选，未保证 Pi=1 / Smol=2 / Molmo=1）

## 可比性提醒

- Pi/Smol 闭环标签：400 ep；Molmo：2000 ep — 仅作标签，不可直接比绝对成功率
- 模块精度：探针 bf16（见各 `correctness.json`）；与 fp32 训练闭环可能有微小差异

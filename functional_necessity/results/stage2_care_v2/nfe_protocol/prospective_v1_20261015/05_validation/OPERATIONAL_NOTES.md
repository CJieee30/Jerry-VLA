# 验证探针运行说明

## MolmoAct2（GPU2）

- 仓库内冻结的 `probe_molmoact2_nfe.py`（freeze 哈希 `d71bde17…`）在同卡存在 `eval_drop.py` 时会直接退出。
- 验证阶段使用 **运行目录副本** `molmoact2/probe_molmoact2_nfe_validation.py`（增加 `--ignore-eval-drop`），由 `RUN_MOLMO_VALIDATION.sh` 调用；**不等于**冻结字节与运行时一致。
- 数据：`MOLMO_PROBE_DATA_ROOT=molmo_root`；验证 480 个 `global_index` 在补全分片后已 100% 覆盖（校验脚本见日志 `validation_plan.log` 同期检查）。

## 数据分片（校准阶段延续）

- 校准探针前曾补下载 Molmo parquet；验证集额外需要分片 `file-379/380/381` 等，当前 `molmo_root` 对 480 gidx 无缺失。

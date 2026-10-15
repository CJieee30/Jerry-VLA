# 第二主线独立验证 — 开环结论（prospective_v1_20261015）

## 范围

- 已执行：`validation-plan` → 三模型验证探针 → `validation-import` → CPU 开环汇总（**未** ledger `validate`，闭环待导入）。
- 校准推荐未改：`04_recommend/recommendations.json`（π0.5/Molmo **NFE=1**，SmolVLA **NFE=2**）。
- 验证集：160 episode × 3 窗 = **480** obs，H=10，双噪声，NFE {1,2,3,10}。

## 算力与产物

| 模型 | GPU | 速度场求值 | 封存 | 日志 |
|---|---|---:|---|---|
| π0.5 | 1 | 15360 | `05_validation/pi05/sealed_validation.npz` | `logs/pi05_validation_probe.log` |
| SmolVLA | 2 | 15360 | `05_validation/smolvla/sealed_validation.npz` | `logs/smolvla_validation_probe.log` |
| MolmoAct2 | 2（与 eval_drop 共卡） | 15360 | `05_validation/molmoact2/sealed_validation.npz` | `logs/molmo_validation_probe.log` |

队列脚本：`logs/run_validation_probes.sh`。Molmo 因冻结探针拒绝同卡 `eval_drop`，使用运行目录副本 `molmoact2/probe_molmoact2_nfe_validation.py` + `RUN_MOLMO_VALIDATION.sh`（见 `OPERATIONAL_NOTES.md`）。

## 开环规则检验（校准选定 NFE，阈值未调）

| 模型 | N* | abs≤0.085 | abs improve | rel improve |
|---|---:|---|---|---|
| π0.5 | 1 | **pass** (E₁=0.0629) | **pass** | **pass** |
| SmolVLA | 2 | **pass** (E₂=0.0566) | **pass** | **pass** |
| MolmoAct2 | 1 | **pass** (E₁=0.0358) | **pass** | **pass** |

四条选样策略的 screening NFE 相同，holdout 结论一致。**开环规则通过 ≠ 闭环已验证**；闭环 SR 待 `closed-loop-import` 后跑正式 `validate`。

## 验证集 E_N（诊断）

| 模型 | E₁ | E₂ | E₃ |
|---|---:|---:|---:|
| π0.5 | 0.0629 | 0.0428 | 0.0321 |
| SmolVLA | 0.0971 | 0.0566 | 0.0413 |
| MolmoAct2 | 0.0358 | 0.0249 | 0.0193 |

## 推荐 NFE 下夹爪／手臂诊断（相对 NFE=10，无新增门槛）

| 模型 | N* | 夹爪不一致率 | 手臂 median ratio | 手臂 P95 ratio |
|---|---:|---:|---:|---:|
| π0.5 | 1 | 0.0156 | 1.154 | 0.884 |
| SmolVLA | 2 | 0.0052 | 0.697 | 0.531 |
| MolmoAct2 | 1 | 0.0042 | 1.852 | 1.016 |

验证集上若按 E 重新选步（`holdout_diagnostic_repick_only`）仍指向与校准相同或更低 NFE；**仅作诊断**，见 `open_loop_validation.json`。

## 异常说明

- Molmo 首轮在 GPU2 因冻结脚本遇 `eval_drop` 失败，已用运行目录补丁探针重跑并成功 import。
- Molmo 数据：480/480 `global_index` 在 `molmo_root` 可解析（校验时缺 3 个分片文件但不缺帧）。

详细表：`05_validation/OPEN_LOOP_VALIDATE.md`、`05_validation/open_loop_validation.json`。汇总脚本：`05_validation/open_loop_validate_cpu.py`。

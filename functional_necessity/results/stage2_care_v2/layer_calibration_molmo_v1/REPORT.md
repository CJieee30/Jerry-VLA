# MolmoAct2 前瞻校准池层剪枝选样（v1 rev2）

**开发诊断** · 子集选样后**等权**层分数（非旧 α 加权实验）。

协议：`layer_calibration_molmo_v1_rev2` — 见 `PROTOCOL_REV2.md`。

## 当前状态：**scores_pending_gpu**

GPU2 空闲约 **14.1 GiB**，**不够**完成本轮打分。已按协议停止，不再反复 OOM。

### GPU2 实测（2026-10-16 18:55–19:00）

| 项 | 值 |
|----|----|
| 卡上占用 | `plus_fisher6_noise_10` PID 3565007 ≈ 9.43 GiB |
| 启动前空闲 | 14 880 669 696 B ≈ 13.86 GiB（nvidia-smi 曾显示 ~14.1 GiB） |
| checkpointing | **有效**：`policy.config` / transformer / vision_backbone 均为 True，且 `policy.training` |
| 冻结 | VLM 288 tensors 可训；其余 1006 冻结；lm_head 在 CPU |
| 载入后分配 | ≈ 11.25 GiB allocated |
| 反传峰值 | ≈ **14.75 GiB** allocated / 进程约 **14.1 GiB** |

两次单样本（固定校准第 1 帧）均 OOM：

1. **三指标 + GateProbe `retain_grad`**：差 96 MiB。完整堆栈：`score_smoke_verify_gate_oom.json`
2. **仅 IGIA/Fisher**（不再保留隐状态）：差 48 MiB。完整堆栈：`score_smoke_verify_igia_oom.json`

结论：共享这块卡时，Molmo 单步反传峰值与 14 GiB 空闲几乎齐平，碎片再要 48–96 MiB 就会炸。需要 **评测进程释放后的整卡**，或空闲大约 **≥ 16 GiB**，再跑 `RUN_LAYER_CALIB_MOLMO_V1.sh <id> gpu`。

对照原实现的分数误差 **尚未测到**（反传未完成）。CPU 自检仍通过。

## 待回答问题（打分后才能写数值）

1. **Fisher / IGIA**：CARE_event 相对 Random 是否改善选层？
2. **α-only / G-only** 能否复现 CARE_event 的 drop？
3. **原 Taylor**（b=80 四组）相对 S1 Taylor-16 是否稳定？
4. **历史复用**：Taylor-16 / Fisher-16 / IGIA-16 / GateProbe-16 可引用；未知层集最多 2 个闭环候选，且不自动判低性能。

未启动闭环，未做效率测量。

## 历史可复用（Drop-16）

| 标签 | SR% | 说明 |
|------|-----|------|
| S1 Taylor-16 | 98.10 | 层集可比性已核实 |
| S1 Fisher-16 / IGIA-16 | 72.75 | 同层集悬崖 |
| S1 GateProbe-16 | 87.00 | 独立历史点 |

## 文件

- `PROTOCOL_REV2.md`、`selftest.log`、`gpu_check.log`
- `score_smoke_verify_gate_oom.json`、`score_smoke_verify_igia_oom.json`
- 待生成：`scores/*.pt`、`taylor_groups/*.pt`、`details.csv`、`drop_frequency.csv`、`selections/`

# MolmoAct2 CARE 开发探针状态

## 正确性（本地分片，3 帧，GPU3）

- 时间：2026-10-15（`correctness.json`）
- NFE 调用数：1/2/3/10 → 1/2/3/10 ✓
- NFE10 探针路径 vs 官方 `predict_action_chunk`：max_abs=0 ✓
- reset 后 NFE10 重复 ✓；同 seed 下 NFE1 重复 ✓
- 初始噪声：`noise_sha256_fp32` 记录在 `correctness.json`；形状 `[1,10,32]`，dtype float32（与 action expert 权重一致）
- 归一化：手臂 QUANTILES（q01/q99）；夹爪 mask=0，开合判定为反量化后 LIBERO 物理 g>0
- CUDA graph：关闭（`load_policy`）

## 数据

- `fetch_molmo.py`：169 个 parquet 分片就绪（13 硬链 + 156 下载），`molmo_root/READY`
- 全池 480 观测探针：**进行中**（见 `run.log`、`DONE`）

## 开发成本（估算）

| 项 | 数量 |
|---|---|
| 观测 | 480 |
| 噪声 | 2 |
| NFE | 1,2,3,10（参考 10） |
| 逻辑速度场求值 / 观测 | 32（(1+2+3+10)×2） |
| 合计速度场求值 | 15 360 |
| 分片下载 | ~21 GB（一次性） |

## 完成后 CPU

```bash
python dev_three_model_screening.py   # 三模型四策略表 + observation_level.csv
```

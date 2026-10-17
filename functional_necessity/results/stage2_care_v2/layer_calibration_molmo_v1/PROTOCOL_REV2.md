# layer_calibration_molmo_v1 — 协议修订 rev2（2026-10-16）

## 冻结不变

- 校准张量 `calib_molmoact2.pt`（400 帧）、`calib_manifest.json`、`pool_identity.json`
- 前瞻池 `prospective_v1_20261015` 与 CARE v2 `shared.pt` episode **重叠 0**
- 每样本固定 `noise` / `timesteps`；NFE 闭环协议未改

## 评分口径

| 指标 | 口径 |
|------|------|
| IGIA | 原 Molmo：逐参数 Σg²，按层求和 |
| Fisher | 原 Molmo：逐参数 Σ(g·θ)² |
| GateProbe | 复用原实现：\|grad_h·(h_out−h_in)\| |
| **Taylor** | **原公式**：选中帧上逐参数累积**带符号** g·θ，再 Σ_w \|Σ_i wg_i\| / N |
| Taylor_absmean | 旧 V2：逐帧先 \|g·θ\| 再平均 — **不与 Taylor 混用** |

## 实验范围

- **逐帧 GPU**：400×36，每帧一次反传同时写出 IGIA / Fisher / GateProbe
- **Taylor 分组**：仅 **budget=80**、四策略（Random、α-only、G-only、CARE_event）；Random/G-only **seed=0**；不生成 400×36 Taylor 矩阵
- 选样：Random/G-only 种子 0–19（逐帧指标）；主预算 80，补充 40/120
- 汇总：选样后等权平均；Drop-16、排除 L0

## 本轮交付状态

- CPU：`selftest_layer_calibration_molmo_v1.py` 通过（合成分数 + 闭环去重）
- `analyze_layer_calibration_molmo_v1.py` 在缺 GPU 分数时 **exit 1**
- GPU：与 LIBERO-plus 评测共享卡时单样本反传 **OOM**（见 `score_smoke_verify.json` / `score_smoke.log`）→ **正式打分 pending**

## 运行

```bash
cd functional_necessity/scripts/stage2_care_v2
bash RUN_LAYER_CALIB_MOLMO_V1.sh cpu          # 自检 + dump
CUDA_VISIBLE_DEVICES=<free_gpu> bash RUN_LAYER_CALIB_MOLMO_V1.sh <id> gpu
```

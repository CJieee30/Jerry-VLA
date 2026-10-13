# CROP-Lite × NFE=10 诊断（不进论文效率表）

`measurement_mode=diagnostic`。共享 GPU1（RTX 3090），sunxy 仍在。标注：**共享GPU诊断，仅供参考，待独占复核**。未写 `main_results` 正式效率表。未跑 400 回合。

Source:

- `offline_check_lite_role_nfe10.json`
- `offline_check_lite_shared_mix_nfe10.json`
- `efficiency_diagnostic_lite_nfe10_20261013T054204Z.json`

代码：`crop_vla.py` selector=`lite`（greedy FPS 仍保留）。head 见 json `code_version`。独占：否。

协议：π0.5 v044，Fisher Drop-6 keep-KV，**NFE=10**（`predict_action_chunk(..., num_steps=10)`；官方 `denoise_step` 与 CROP `_body` 均为 10 次）。`compile_model=false`，batch=1。16 帧 × 预热 5 × 重复 3，三组按帧轮换。

## 离线检查

keep-all vs 同一 Drop-6 NFE=10 基线：`max_abs=0`。prefix/KV 968→712。scout `[1,1,1,0…]`。role 腕 96+32 / 全局 32+96；shared_mix 两路 64+64。无重复索引。动作有限。旧 `offline_check.json`（greedy NFE=1）未覆盖。

## 完整 chunk（profile=False）

| 配置 | CUDA median / P95 (ms) | vs baseline median | prefix/KV |
|---|---:|---:|---|
| baseline Drop-6 NFE=10 | 590.71 / 618.40 | — | 968 |
| role-lite | 585.70 / 600.01 | **−5.01 ms（−0.85%）** | 712 |
| shared-lite | 586.56 / 605.55 | **−4.15 ms（−0.70%）** | 712 |

墙钟与 CUDA 差 <0.3 ms。velocity_calls=10。

## 分段（独立 profile=True）

| 段 | role-lite | shared-lite |
|---|---:|---:|
| embed | 85.0 | 84.9 |
| scout | 16.8 | 17.4 |
| select | **0.49** | **0.49** |
| prefix | 64.0 | 66.5 |
| expert（10 步） | 417.1 | 416.3 |

选择器从 greedy 诊断的 ~70 ms 降到 0.5 ms。NFE=10 下专家约占 70%，净加速只有约 1%。role 与 shared_mix 差 <1 ms，不能区分。

## 闭环

未跑。GPU 未独占。点估计 <1% 的诊断加速不够当成论文效率。待空闲卡：独占复测；若净加速仍在，再跑 role-lite / shared-lite 各 400 回合（NFE=10）。Baseline SR 复用 Drop-6 NFE=10 的 97.75，不复用 NFE=1。旧 greedy CROP SR 不能当 CROP-Lite 的结果。

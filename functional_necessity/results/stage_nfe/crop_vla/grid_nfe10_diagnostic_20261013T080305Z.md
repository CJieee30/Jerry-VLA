# uniform-grid × NFE=10 诊断（不进论文效率表）

`measurement_mode=diagnostic`。共享 GPU1（NVIDIA GeForce RTX 3090）。sunxy ts2audio 全程占用约 1.7 GiB。标注：**共享 GPU，仅供诊断，不进入论文正式效率表**。本轮未跑 400 回合，未做 NFE=1 联合评估，未写 `main_results` 正式效率表。

Source:

- `offline_check_grid_nfe10.json`
- `efficiency_diagnostic_grid_nfe10_20261013T080305Z.json`

代码：git `03c0519`（工作区未提交）。修改文件 SHA-256：

| 文件 | sha256 |
|---|---|
| `crop_vla.py` | `65b27bacfa470a13167d0dbd5bda11dd408abfbcaf8e91ea2021709f3b38cc82` |
| `check_crop_vla.py` | `5b474b647611308541f2bd4ba589538cde34665b694a706cc2ceb1cb06b948f2` |
| `measure_crop_efficiency.py` | `753ae41068b1b550a23f82493c63bf3bf1caf726fcc3f85df66b6bb35fa52fb5` |
| `eval_drop.py` | `d97cc10ab53ea0ec8b12be4caca9722bb64fc0e460b273a932042070527f9bf0` |

设备：物理 GPU1，逻辑 `cuda:0`。独占：否。开始时 GPU1 已用 1722 MiB / 利用率瞬时 85%；结束后本进程约 9750 MiB，sunxy 仍在。GPU0/2 上另有本机 Plus Sensor Noise 评估，未触碰。

协议：π0.5 v044，Fisher Drop-6 `[1,13,14,15,16,17]` keep-KV，**NFE=10**（`predict_action_chunk(..., num_steps=10)`）。`compile_model=false`，batch=1，`n_action_steps=10`。16 帧 × 预热 5 × 重复 3，三组按帧轮换。`profile=False` 测完整 `predict_action_chunk`（视觉编码到动作输出）。分段为独立 `profile=True` 一轮，**用均值分析，不加各段中位数，不与完整计时相减**。

三组实际速度场次数均为 10。

## 离线检查

grid keep-all vs 同一 Drop-6 NFE=10 官方基线：`max_abs=0`（门限 2e-2 未放宽）。prefix/KV **968→712**。两个真实相机各保留 128 个无重复、按原索引排序、`(row+col)%2==0` 的 token；空相机 256。原始 RoPE / action offset 保留（例：offset 560，压缩后 pad-sum 304，不是用 712 当 offset）。grid scout 全 0。动作有限。greedy / Lite 实现与历史 json 未覆盖。

## 完整 chunk（profile=False）

n=48（16 帧 × 3 次）。墙钟与 CUDA 差 <2 ms。

| 配置 | CUDA median / mean / P95 (ms) | vs baseline median | vs baseline mean | prefix/KV | velocity |
|---|---:|---:|---:|---|---|
| baseline | 627.27 / 617.93 / 669.09 | — | — | 968 | 10 |
| uniform-grid | 620.53 / 606.29 / 640.03 | **−6.74 ms（−1.07%）** | −11.64 ms（−1.88%） | 712 | 10 |
| role-lite | 629.98 / 613.88 / 658.24 | +2.71 ms（+0.43%） | −4.05 ms（−0.66%） | 712 | 10 |

同协议上次 Lite 诊断（`efficiency_diagnostic_lite_nfe10_20261013T054204Z.json`）baseline CUDA median 为 590.71 ms。两次共享卡 baseline 相差约 36 ms，大于本次 grid 相对 baseline 的 7 ms。因此 **不能凭本次点估计宣布加速成立**。

## 分段（独立 profile=True，均值）

| 段 | baseline | uniform-grid | role-lite |
|---|---:|---:|---:|
| embed | 88.80 | 85.87 | 87.64 |
| scout | — | —（路径上为 0） | 18.67 |
| select | — | — | 1.17（median 0.49） |
| prefix（grid/lite 含打包） | 93.79 | 65.52 | 64.28 |
| expert（10 步） | 449.46 | 439.16 | 439.37 |
| profile 合计 | 632.04 | 590.55 | 611.12 |

grid 相对 baseline（同一次分段、均值）：prefix **−28.27 ms**（该段约 −30%），expert **−10.29 ms**（该段约 −2.3%），embed 差 3 ms 量级、视为噪声。专家仍约占分段合计的 70%。

role-lite 相对 grid：scout+select ≈ 19.8 ms，prefix/expert 几乎相同。差距主要来自 scout，不是 128 token 预算本身。

分段合计与完整 chunk 不是同一轮，禁止相减或把各段中位数加总。

## 四个问题

1. **grid 相对 baseline 实际节省多少？** 完整 chunk CUDA median **−6.74 ms（−1.07%）**，mean −11.64 ms（−1.88%）。共享卡、且两次诊断 baseline 波动远大于该差值，**不进入论文效率表，不宣布加速成立**。
2. **prefix 和 expert 各自节省多少？** 分段均值：prefix 约 **28 ms**，expert 约 **10 ms**。NFE=10 下专家仍主导，所以完整 chunk 只剩约 1%。
3. **role-lite 与 grid 的差距是否主要来自 scout？** **是。** 同预算 712 token 时 prefix/expert 对齐；lite 多出 scout ~19 ms 和 select ~1 ms。
4. **下一步？** 不要现在开 400 回合。先等目标 GPU 完全空闲做正式复测。NFE=10 收益小 **不能** 推断 NFE=1 无价值——NFE=1 时专家变短，prefix 的 28 ms 相对权重会上升。中途剪枝主要服务仍要 scout 的路径，grid 已经是「编码一次 → 打包 → 一次 prefix → expert」。若独占 NFE=10 仍只有约 1%，论文主表若停在 NFE=10，视觉模块可暂缓大规模投入；但应保留一次空闲卡上的 **NFE=1 grid 诊断**，再决定是否闭环。

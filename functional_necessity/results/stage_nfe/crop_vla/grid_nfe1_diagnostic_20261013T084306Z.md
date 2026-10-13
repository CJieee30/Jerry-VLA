# NFE=1 三组效率诊断（修正轮换后；不进论文效率表）

`measurement_mode=diagnostic`。本机 RTX 3090，物理 GPU1。sunxy ts2audio 全程约 1.7 GiB。标注：**共享 GPU，仅供诊断，不进入论文正式效率表**。未跑 400 回合。此前约 13%/5% 只是测量动机，不是目标，也不用来判断代码对错。

Source:

- `offline_check_grid_nfe1.json`
- `offline_check_lite_role_nfe1.json`
- `efficiency_diagnostic_grid_nfe1_20261013T084306Z.json`

NFE=10 旧文件未覆盖。

代码：git `4fa6dac`（工作区未提交）。SHA-256：

| 文件 | sha256 |
|---|---|
| `crop_vla.py` | `731f228446e08e90dcca2b74779eef7f058a2fa1a1eaf341f87d3bbf3ae7070d` |
| `check_crop_vla.py` | `fcdcc4b61f66e0baa5503a6e044e902c9bb911c8665815699db9e3d047bb1228` |
| `measure_crop_efficiency.py` | `e031f128822e110fa9d4104152a6b012fe9797336d14eefba2e493fc655f9c14` |
| `eval_drop.py` | `d97cc10ab53ea0ec8b12be4caca9722bb64fc0e460b273a932042070527f9bf0` |

设备：GPU1，逻辑 `cuda:0`。独占：否。开始时 GPU1 已用 1722 MiB、利用率瞬时 84%；结束后本进程约 9750 MiB，sunxy 仍在。GPU0/2 上另有本机 Plus 评估，未触碰。

协议：π0.5 v044，Fisher Drop-6 `[1,13,14,15,16,17]` keep-KV，**NFE=1**（实际 `denoise_step` / 速度场 1 次）。`compile_model=false`，batch=1，`n_action_steps=10`。16 帧、每组预热 5、每帧重复 3。完整计时在每个 `(frame, repeat)` 内轮换三组；分段在每帧内轮换三组、每组一次。完整 `profile=False`，分段独立 `profile=True`。基准分段改为包裹原 `embed_prefix` / `denoise_step` / `sample_actions`（只加 CUDA event）。同输入同噪声动作 `max_abs=0`，`attn=eager`，参数 `bf16`，动作 `fp32`。

## 离线检查（NFE=1）

| 项 | grid | role-lite |
|---|---|---|
| 实际速度场 | 1 | 1 |
| keep-all vs 官方 Drop-6 NFE=1 | `max_abs=0` | `max_abs=0` |
| prefix/KV | 968→712 | 968→712 |
| 真实相机 | 各 128，无重复 | 各 128，无重复 |
| action offset 例 | 560（压缩 pad-sum 304） | 同左 |
| scout | 全 0 | `[1,1,1,0…]`（仅 0–2 层） |
| 基准 profiling 对齐 | `max_abs=0` | `max_abs=0` |

## 完整 chunk（profile=False）

n=48。墙钟与 CUDA 差约 1–2 ms。

| 配置 | CUDA median / mean / P95 (ms) | vs baseline median | vs baseline mean | velocity |
|---|---:|---:|---:|---|
| baseline | 230.36 / 225.81 / 266.65 | — | — | 1 |
| uniform-grid | 211.72 / 202.71 / 241.63 | **−18.64 ms（−8.09%）** | −23.10 ms（−10.23%） | 1 |
| role-lite | 223.54 / 218.23 / 260.52 | **−6.82 ms（−2.96%）** | −7.58 ms（−3.36%） | 1 |

此前 NFE=1 greedy 诊断 baseline median 约 208 ms，与本次 230 ms 相差约 22 ms，大于 role-lite 的 7 ms 差。共享卡点估计带不确定性，**不进论文表**。

## 分段（独立 profile=True，均值）

| 段 | baseline | uniform-grid | role-lite |
|---|---:|---:|---:|
| embed | 86.39 | 90.26 | 91.18 |
| scout | — | — | 17.47 |
| select | — | — | 1.10 |
| prefix（含打包） | 94.99 | 66.52 | 64.22 |
| expert（1 步） | 42.70 | 43.68 | 43.83 |
| profile 合计 | 224.09 | 200.46 | 217.80 |

同一次分段、均值：grid 相对 baseline，prefix **−28.47 ms**，expert **+0.98 ms**（噪声）。role-lite 的 prefix 同样少约 31 ms，但 scout+select 约 **18.6 ms**。

完整 mean 与分段合计相差约 1–2 ms（baseline 225.8 vs 224.1；grid 202.7 vs 200.5；lite 218.2 vs 217.8），方向一致。禁止把分段中位数加总，也不把两轮直接相减当精确分解。

## 四个问题

1. **NFE=1 下各节省多少？** 完整 CUDA median：grid **−18.64 ms（−8.09%）**，role-lite **−6.82 ms（−2.96%）**。mean 分别为 −10.23% 与 −3.36%。共享卡，保留不确定性。
2. **收益主要来自 prefix 还是 expert？** **prefix。** NFE=1 时专家只约 43 ms，压缩 KV 几乎不缩短 expert；prefix 从约 95 ms 降到约 65–66 ms。
3. **scout 是否抵消 role-lite？** **大部分抵消。** prefix 省约 31 ms，scout+select 吃掉约 19 ms，所以完整 chunk 只剩约 3%，明显小于无 scout 的 grid。
4. **完整 vs 分段是否不一致？下一步是否闭环？** 两轮均值只差 1–2 ms，没有明显打架。效率方向对 grid 比 NFE=10 更清楚，但仍是共享卡。下一步应先等目标 GPU 空闲做正式复测；**现在不要开 400 回合**。若独占后 grid 仍有约数个百分点且 SR 未知，再考虑闭环；role-lite 在 NFE=1 上被 scout 拖住，闭环优先级低于 grid。

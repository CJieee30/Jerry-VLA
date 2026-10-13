# CROP-VLA closed-loop LIBERO (this machine)

CROP-VLA：training-free，按相机角色剪视觉 token。叠在 π0.5 Fisher Drop-6 keep-KV 上，不改权重、不 drop action expert、不做 recovery。论文数字只用本机。不要把闭环 `eval_s` 当效率（MuJoCo + 共享卡主导墙钟）。正式 chunk 延迟还没有独占 GPU，本文不写时间/FLOPs。

本文两块协议不要混表：

- **§1–Lab**：greedy CROP，**NFE=1**，空相机保留，prefix 968→712。对照是 Drop-6 × NFE=1 **96.50**。
- **§4 起**：prefix-trim ± uniform-grid，**NFE=10**，空相机在视觉编码前丢掉。对照是 Drop-6 × NFE=10 **97.75**；剪真实视觉的对照是 trim-only，不是 greedy `eval_crop/`。

局部锚点：同一协议、不剪视觉的 Drop-6 × NFE=1 是 **96.50**（`eval_nfe_libero.md`，`results/stage_nfe/pi05_k6_fisher_nfe1/`，386/400）。400 集在 96–97% 附近的二项 SE 约 0.8–1.0 分。0.25–0.75 Avg. 是噪声。

通过规则（与 NFE 文同一条）：相对**同一协议**的 96.50，Avg. 差在 2 分以内，且没有套件塌掉。

---

## Protocol

单进程四套件（Spatial / Object / Goal / Long=LIBERO-10），`seed=1000`，`batch_size=1`，`compile_model=false`，`n_action_steps=10`，`num_inference_steps=1`，skip = DTR Appendix B keep prefix K/V，删层 `[1, 13, 14, 15, 16, 17]`，无 recovery。权重 `lerobot/pi05_libero_finetuned_v044`。每任务 10 集（400）。

视觉：LIBERO `agentview_image` → `observation.images.image`（global），`robot0_eye_in_hand_image` → `observation.images.image2`（wrist）。v044 `empty_cameras=1` 的 padding 槽不剪。每路真实相机 256 token（16×16），各留 128；空相机仍 256；语言 200。正式 prefix / KV **968 → 712**。RoPE / action offset 用未剪的 `sum(pad)`，不是 712。

Scout：视觉编码一次；只跑语言层 0..2，在第 2 层（仍执行 attention）取语言→视觉 attention；丢掉 scout KV；按原始顺序 gather 原始 embedding，再跑完整 18 层 keep-KV prefix + 动作专家。

Baseline **不重跑**。两组 CROP 各 400 集，新目录，不覆盖 `pi05_k6_fisher_nfe1`。

| 配置 | 视觉 token | GPU | 闭环 |
|---|---|---|---|
| Drop-6 × NFE=1（不剪视觉） | prefix 968 | 当时 GPU1 | 复用 `results/stage_nfe/pi05_k6_fisher_nfe1/` |
| CROP **role** | wrist 96 相关 + 32 多样；global 32 相关 + 96 多样 | GPU1 | `results/stage_nfe/crop_vla/eval_crop/` |
| CROP **shared_mix** | 两路都是 64 相关 + 64 多样 | GPU3 | `results/stage_nfe/crop_vla/eval_crop_shared_mix/` |

Source:

- 闭环：上表三个目录的 `{eval_info.json,drop_meta.json}`
- 离线：`results/stage_nfe/crop_vla/{offline_check.json,offline_check_shared_mix.json}`
- 脚本：`scripts/pi05_libero/{crop_vla.py,check_crop_vla.py,run_eval_crop.sh,eval_drop.py}`
- 日志：`results/logs/{eval_crop_vla.log,eval_crop_vla_shared_mix.log}`

---

## 1. Closed-loop success rate (%)

三行都是 400 集、本机、同一 init / seed / 动作协议。NFE=1 的不剪视觉行不是这次重跑。

| Setting | Visual | Spatial | Object | Goal | Long | Avg. | bits |
|---|---|---:|---:|---:|---:|---:|---:|
| Fisher Drop-6 × NFE=1 | 不剪 | 97.0 | 100.0 | 93.0 | 96.0 | 96.50 | 386/400 |
| CROP role | 96+32 / 32+96 | 95.0 | 100.0 | 93.0 | 95.0 | **95.75** | 383/400 |
| CROP shared_mix | 64+64 / 64+64 | 97.0 | 100.0 | 91.0 | 96.0 | **96.00** | 384/400 |

Δ Avg. vs Drop-6 × NFE=1（96.50）：

- role：**−0.75**（383/400）。Spatial −2.0，Long −1.0，Object / Goal 不动。
- shared_mix：**−0.50**（384/400）。Goal −2.0，其余三套件与 baseline 相同。

role vs shared_mix：**−0.25**（1 集）。角色配比没有好于统一 64+64。

没有套件塌掉。Object 三行都是 100。最大单套件移动是 2 分。Avg. 都在 2 分闸门内。点估计略低于 96.50，幅度与 400 集噪声同量级，不能读成「剪视觉 token 明显伤 SR」，也不能读成「角色配比有效」。

---

## 2. Token mix

每路真实相机 keep=128，序列长度相同。只改相关/多样切分。

| 相机 | role | shared_mix |
|---|---|---|
| wrist（`image2`） | 96 相关 + 32 多样 | 64 + 64 |
| global（`image`） | 32 相关 + 96 多样 | 64 + 64 |
| empty | 256，不剪 | 256，不剪 |

相关性：scout 第 2 层语言 query → 该相机视觉 key，跨 head 与有效语言 query 平均。多样性：从相关集合出发，在 L2 归一化视觉特征上按余弦最近邻贪心补最不相似的 token。选出后按原始空间顺序 gather，不是置零。

---

## 3. Offline check（eval 前）

两组都先跑 `check_crop_vla.py`，失败则不进 400 集。keep-all 路径（每相机 256）与同一噪声的 Drop-6 × NFE=1 动作 `max_abs=0`。剪枝后 prefix=KV=712，scout 层调用 `[1, 1, 1, 0…0]`（第 3 层及更深未执行）。空相机 256 不动。

| Check | role (GPU1) | shared_mix (GPU3) |
|---|---|---|
| json | `offline_check.json` | `offline_check_shared_mix.json` |
| identity max_abs | 0 | 0 |
| full / short / kv | 968 / 712 / 712 | 968 / 712 / 712 |
| global keep | 128 = 32+96 | 128 = 64+64 |
| wrist keep | 128 = 96+32 | 128 = 64+64 |
| viz | `crop_vla/viz/` | `crop_vla/viz_shared_mix/` |

相机拼图（左 global / 右 wrist）：`results/stage_nfe/crop_vla/cameras/`。

---

## Checks

`eval_info.json` 均为 `n_episodes=400`（40 任务 × 10）。`pc_success` 与 `per_group`、逐回合 `successes` 一致。`drop_meta.json`：`compile_model=false`，`n_action_steps=10`，`num_inference_steps=1`，`seed=1000`，`keep_kv=true`，`crop=true`。shared_mix 另有 `crop_budget=shared_mix`。GPU1 的 `drop_meta` 无该字段（当时默认 role，与离线 json 的 96+32 / 32+96 一致）。

| Cell | GPU | tmux | bits | eval_s | 日志 |
|---|---|---|---|---:|---|
| `pi05_k6_fisher_nfe1`（复用） | 1 | — | 386/400 | 1.20 h | 历史 |
| `eval_crop` role | 1 | `crop_vla_eval` | 383/400 | 1.81 h | `eval_crop_vla.log`；`eval_info` 03:01 |
| `eval_crop_shared_mix` | 3 | `crop_vla_shared_mix` | 384/400 | 1.64 h | `EXIT:0` 03:09 |

`eval_s` 含仿真，且两张卡当时都有 sunxy 进程，**不是** CROP 开销。效率测量要等独占 GPU，走 `measure_crop_efficiency.py`，禁止 `CROP_ALLOW_OCCUPIED`。

GPU1 包装脚本在 python 写完 `eval_info.json` 之后打印了 `olicy.num_inference_steps=1: command not found`（当时改 `run_eval_crop.sh` 接 shared_mix，GPU1 的 bash 还停在那条 python 续行上）。以 json 为准，不把这条当评测失败。

未改 `eval_nfe_libero.md` / `eval_pi05_libero_v044.md` 的已有行，也未写效率表。

---

## Lab：与 baseline 不一致的任务（/10）

只列出三行不全相同的任务。其余 28 个任务三行都是 10/10 或同一分数。

| Suite | task_id | Drop-6 NFE=1 | role | shared_mix |
|---|---:|---:|---:|---:|
| Spatial | 4 | 10 | 9 | 9 |
| Spatial | 5 | 10 | 9 | 10 |
| Spatial | 8 | 7 | 7 | 8 |
| Goal | 1 | 10 | 9 | 10 |
| Goal | 3 | 7 | 7 | 5 |
| Goal | 6 | 10 | 9 | 10 |
| Goal | 9 | 7 | 9 | 7 |
| Long | 2 | 9 | 10 | 10 |
| Long | 3 | 10 | 9 | 10 |
| Long | 4 | 10 | 9 | 9 |
| Long | 6 | 9 | 9 | 7 |
| Long | 8 | 10 | 9 | 10 |
| Long | 9 | 8 | 9 | 10 |

相对不剪视觉：role 净少 3 集（10 集独败 / 7 集独成），shared_mix 净少 2 集（7 / 5）。role 与 shared_mix 互有 6–7 集对倒，净差 1 集。

---

## 4. Prefix-trim + uniform-grid（NFE=10）

空相机在 SigLIP 前丢掉，语言按 attention mask 右裁。trim-only 两路真实相机各留 256 token。trim-grid 再棋盘留下各 128，无 scout。速度场调用离线实测为 10，不是只改配置记录。`eval_crop/` / `eval_crop_shared_mix/` 是 greedy × NFE=1，本块不引用它们的 SR。

| 配置 | 视觉 | GPU | 闭环 |
|---|---|---|---|
| Drop-6 × NFE=10（不 trim、不剪真实视觉） | prefix 968 | 历史 | 复用 `results/eval_drop/k6_fisher/` |
| **trim-only** | 空相机去掉，真实相机 256+256 | 1 共享 | `results/stage_nfe/crop_vla/eval_trim_only_nfe10/` |
| **trim-grid** | 同上再 grid 128+128，无 scout | 3 共享 | `results/stage_nfe/crop_vla/eval_trim_grid_nfe10/` |

Source:

- 闭环：上表 `{eval_info.json,drop_meta.json}`
- 离线：`results/stage_nfe/crop_vla/offline_check_trim_grid_nfe10.json`（未覆盖 `*_nfe1.json`）
- 脚本：`scripts/pi05_libero/{run_eval_trim.sh,eval_drop.py,check_crop_vla.py,prefix_trim.py,crop_vla.py}`
- 日志：`results/logs/eval_trim_{only,grid}_nfe10_20261013T115942.log`

### 4.1 Closed-loop success rate (%)

三行都是 400 集、`seed=1000`、`EXIT:0`、每任务 10 个有效回合。NFE=10 不剪视觉行不是这次重跑。

| Setting | Visual | Spatial | Object | Goal | Long | Avg. | bits |
|---|---|---:|---:|---:|---:|---:|---:|
| Fisher Drop-6 × NFE=10 | 不 trim | 97.0 | 100.0 | 96.0 | 98.0 | 97.75 | 391/400 |
| trim-only | 去空相机 + 语言右裁 | 97.0 | 99.0 | 96.0 | 94.0 | **96.50** | 386/400 |
| trim-grid | trim + 128/相机 checkerboard | 99.0 | 99.0 | 89.0 | 92.0 | **94.75** | 379/400 |

Δ Avg. vs Drop-6 × NFE=10（97.75）：

- trim-only：**−1.25**（386/400）。Long −4.0，Object −1.0，Spatial / Goal 不动。
- trim-grid：**−3.00**（379/400）。Goal −7.0，Long −6.0，Object −1.0，Spatial +2.0。

Δ trim-grid vs trim-only（同一 trim、同一 NFE=10）：**−1.75**（净少 7 集）。Goal −7.0，Long −2.0，Spatial +2.0，Object 0。

相对不剪视觉的 2 分闸门：trim-only 的 Avg. 过了（−1.25），Long −4 是最大套件缝，没有塌到不可用。trim-grid 的 Avg. **没过**（−3.00）；Goal 89 是本块最大套件洞（task 7 = 5/10）。剪真实视觉的对照必须用 trim-only，不要拿 greedy NFE=1 的 95.75 / 96.00 比。

`drop_meta.json`：`checkpoint=lerobot/pi05_libero_finetuned_v044`，`drop_layers=[1,13,14,15,16,17]`，`keep_kv=true`，`num_inference_steps=actual_nfe=10`，`n_action_steps=10`，`compile_model=false`，`seed=1000`，`prefix_trim=true`。trim-only `crop=false`；trim-grid `crop=true`、`crop_selector=grid`。代码 `e957c92`。

### 4.2 Offline check（eval 前）

`check_crop_vla.py --gpu 1 --allow-shared --prefix-trim --selector grid --nfe 10`。`all_ok=true`。阈值 `IDENTITY_MAX_ABS=2e-2` 未放宽。剪枝后的 grid vs trim 只记录，不套 keep-all 阈值。

| Check | 结果 |
|---|---|
| official / trim-only / grid 速度场调用 | 均为 **10** |
| trim-only vs raw max-abs | 0.00734 |
| keep-all vs trim-only max-abs | **0** |
| pruned grid vs trim max-abs / rel-L2 | 0.357 / 0.131（分析） |
| 非有限动作 | 无 |
| 真实相机 | 各 128 个不同 token；无空相机 |
| expert offset | = 剪视觉前有效 prefix（560 / 569 / 563 / 554） |
| 18 层 K/V（含 Drop-6） | raw 968；trim-only 554–569；grid 298–313 |

### 4.3 Checks

`eval_info.json` 均为 `n_episodes=400`（40 任务 × 10）。`pc_success` 与 `per_group`、逐回合 `successes` 一致。共享评估：`--allow-shared --gpu N`，启动前 ≥12 GiB 空闲、同卡没有第二个本项目 `eval_drop`。未杀 sunxy。LIBERO 上限是仿真步（280 / 300 / 520），没有墙钟超时。

| Cell | GPU | tmux | bits | eval_s | 日志 |
|---|---|---|---|---:|---|
| `k6_fisher`（复用） | — | — | 391/400 | 1.79 h | 历史 NFE=10 |
| `eval_trim_only_nfe10` | 1 | `eval_trim_only_nfe10` | 386/400 | 2.02 h | `EXIT:0` 22:04 CST |
| `eval_trim_grid_nfe10` | 3 | `eval_trim_grid_nfe10` | 379/400 | 2.16 h | `EXIT:0` 22:12 CST |

启动占用：GPU1 sunxy 1712 MiB；GPU3 sunxy 1866+1712 MiB。闭环墙钟含共享卡与仿真，**不是**效率。正式 raw / trim-only / trim-grid × NFE=10 仍等独占 3090：`bash run_measure_trim.sh --nfe 10`。

### 4.4 Lab：与 Drop-6 × NFE=10 不一致的任务（/10）

只列出三行不全相同的任务。其余 21 个任务三行分数相同。

| Suite | task_id | Drop-6 NFE=10 | trim-only | trim-grid |
|---|---:|---:|---:|---:|
| Spatial | 3 | 10 | 9 | 10 |
| Spatial | 5 | 10 | 9 | 10 |
| Spatial | 8 | 8 | 9 | 9 |
| Spatial | 9 | 9 | 10 | 10 |
| Object | 3 | 10 | 9 | 10 |
| Object | 7 | 10 | 10 | 9 |
| Goal | 0 | 10 | 10 | 8 |
| Goal | 3 | 7 | 7 | 8 |
| Goal | 5 | 10 | 10 | 9 |
| Goal | 7 | 10 | 10 | **5** |
| Goal | 8 | 10 | 10 | 9 |
| Goal | 9 | 9 | 9 | 10 |
| Long | 2 | 9 | 8 | 10 |
| Long | 3 | 10 | 9 | 8 |
| Long | 4 | 10 | 9 | 9 |
| Long | 6 | 9 | 10 | 9 |
| Long | 7 | 10 | 10 | 8 |
| Long | 8 | 10 | 10 | 9 |
| Long | 9 | 10 | 8 | 9 |

相对 Drop-6 × NFE=10：trim-only 净少 5 集（独败 8 / 独成 3）；trim-grid 净少 12 集（17 / 5）。trim-grid 相对 trim-only 净少 7 集（独败 15 / 独成 8），Goal task 7 的 10→5 占了其中一半。

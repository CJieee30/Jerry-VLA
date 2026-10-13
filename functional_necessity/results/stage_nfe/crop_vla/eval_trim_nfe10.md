# Prefix-trim + grid，NFE=10（Jerry-VLA）

本轮只做 **Drop-6 keep-KV + NFE=10** 的 prefix-trim。NFE=1 检查文件与目录未覆盖。闭环在共享 3090 上跑，墙钟**不是**论文效率。正式 chunk 延迟仍等独占卡。

协议：`lerobot/pi05_libero_finetuned_v044`，Fisher Drop-6 `[1,13,14,15,16,17]` keep-KV，`n_action_steps=10`，`num_inference_steps=10`（速度场调用已实测为 10），batch=1，compile / gradient checkpointing 关，seed=1000，relative，四套 LIBERO × 40 任务 × 10 回合。prefix trim：视觉编码前丢掉 mask 全 false 的空相机，按语言 attention mask 裁掉右侧 padding。trim-grid 无 scout。

代码：`e957c92`。入口：`scripts/pi05_libero/run_eval_trim.sh`、`run_measure_trim.sh`。

---

## 1. 离线检查（已通过）

新文件（未动 NFE=1）：

`offline_check_trim_grid_nfe10.json`（本目录）

历史仍在：`offline_check_trim_grid_nfe1.json`。

| 项 | 结果 |
|---|---|
| official / trim-only / grid 速度场调用 | 均为 **10** |
| trim-only vs raw max-abs | 0.00734（阈值 0.02） |
| keep-all vs trim-only max-abs | **0** |
| 剪枝后 grid vs trim max-abs / rel-L2 | 0.357 / 0.131（分析，不套 keep-all 阈值） |
| 非有限动作 | 无（会直接报错） |
| 真实相机 | 各 128 个不同 token；无空相机；语言有效位全留 |
| expert offset | = 剪视觉前有效 prefix（560/569/563/554） |
| 18 层 K/V（含 Drop-6） | raw 968；trim-only 560–569；grid 298–313 |

GPU1 共享检查：`--gpu 1 --allow-shared --prefix-trim --selector grid --nfe 10`。

---

## 2. 闭环 400 回合（进行中，尚未出 SR）

先前无完整 `eval_trim_*_nfe10`。已启动、配置已写入 `drop_meta.json`，**eval_info.json 要等 400 回合结束才写**。OOM/崩溃不会把残局记进成功率。

| 组 | 目录 | GPU | 配置 | 进程 / 显存 |
|---|---|---|---|---|
| A trim-only | `eval_trim_only_nfe10/` | 1 | trim，每路真实相机 256 token，不剪视觉 | pid 830215，~10129 MiB |
| B trim-grid | `eval_trim_grid_nfe10/` | 3 | trim + `--crop --crop-selector grid`，每路 128，无 scout | pid 830223，~10061 MiB |

tmux：`eval_trim_only_nfe10`、`eval_trim_grid_nfe10`。  
日志：`results/logs/eval_trim_only_nfe10_20261013T115942.log`、`eval_trim_grid_nfe10_20261013T115942.log`。

`drop_meta`：`actual_nfe=10`，`prefix_trim=true`，A 的 `crop=false`，B 的 `crop=true` / `crop_selector=grid`，`skip_mode=keep_kv`，`seed=1000`。

启动时共享卡占用（未杀任何人）：

- GPU1：sunxy pid 1011148，1712 MiB；空闲 22405 MiB。
- GPU3：sunxy pid 1997700 + 1009712，1866+1712 MiB；空闲 20534 MiB。
- GPU0/2 上本项目 Plus `eval_drop` 未动。同一 GPU 不超过一个本项目 LIBERO 评估。

LIBERO 上限是 **仿真步**（Spatial/Object 280，Goal 300，Long 520），`lerobot_eval` 没有墙钟超时。NFE=10 只让每个 action chunk 更慢，不改 `max_episode_steps`。进度条已是 `Running rollout with at most 280 steps`。

闭环墙钟（含共享卡）不当正式效率。

---

## 3. 正式效率（未跑，待独占 3090）

三组都是 Drop-6 + NFE=10：raw（不清理、不剪真实视觉）/ trim-only / trim-grid。16 帧、预热 5、每帧重复 3、固定输入与噪声、轮换；完整计时 `profile=False`，分段另测。raw 只做效率对照，不新增 400 回合。

当前四张卡都有计算进程，**没有独占 3090**。不得与上面两组闭环同卡。FLOPs 未测。

完整卡空出来之后：

```bash
cd /home/guorj/Skill_Efficient_VLA/functional_necessity/scripts/pi05_libero
bash run_measure_trim.sh --nfe 10
```

可加 `--gpu N`，但仍走独占门禁；共享卡会直接拒绝。

NFE=1 诊断（共享卡，非正式）：raw 236.12 → trim-only 183.33 → trim-grid 158.24 ms。NFE=10 的专家占比更高，不能套用该百分比。

---

## 4. 本轮未做

- 不改其它预算 / 选择器 / NFE。
- 不覆盖 `eval_crop/`、`offline_check_trim_grid_nfe1.json`、NFE=1 闭环目录。
- 闭环 SR 与正式效率数字：前者等 400 回合写完 `eval_info.json`，后者等独占卡。

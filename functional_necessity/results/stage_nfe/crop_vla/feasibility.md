# CROP-VLA 最小可行性筛查（π0.5 LIBERO）

Training-free。只比两格：Fisher Drop-6 keep-KV + NFE=1，有/无按相机角色剪视觉 token。本文件写到 2026-10-12。闭环与效率**尚未跑**——四张 GPU 都被占用，脚本已拒绝启动，下面没有编造的成功率或延迟。

## 状态

| 项 | 状态 |
|---|---|
| 实现（scout / 选择 / 缩短 prefix / 原位置 RoPE） | 完成，未改权重 |
| 相机角色核对 + 示例图 | 完成（CPU） |
| 选择逻辑 CPU 自测 | 通过 |
| keep-all 动作一致、剪枝后 KV 缩短、token 可视化 | **等待空闲 GPU** |
| CROP 400 回合闭环 | **等待空闲 GPU** |
| 完整 chunk 延迟（含 scout） | **等待独占 GPU** |
| Baseline 400 回合 | 复用已有协议匹配结果，不重跑 |

当前占用（不要去抢、不要在共享负载下测延迟）：

- GPU0：本机 Plus Sensor Noise spatial full + sunxy ts2audio
- GPU1：本机 Plus Background Long full + sunxy + yany
- GPU2：本机 Plus Sensor Noise spatial Drop-6 + sunxy
- GPU3：本机 Plus Background Long Drop-6 + sunxy

`run_eval_crop.sh` 已探测并退出 2：`WAIT GPU not idle`。

## 两个配置

其他设置相同：`n_action_steps=10`，`num_inference_steps=1`，`compile_model=false`，`seed=1000`，四套件 × 10 任务 × 10 回合，`batch_size=1`，keep-KV `[1,13,14,15,16,17]`。CROP 不改空相机槽。

| 配置 | 视觉 token | 闭环 |
|---|---|---|
| Baseline | 不剪。prefix **968** = 3×256 + 200 | 复用 `results/stage_nfe/pi05_k6_fisher_nfe1/`（`drop_meta.json`：Fisher Drop-6、NFE=1、seed=1000、`n_episodes=10`、`compile=false`、skip=`dtr_appendix_b_prefix_kv`） |
| CROP | 两路真实相机各留 128 | **未跑**。目录预留 `results/stage_nfe/crop_vla/eval_crop/` |

Baseline 已有成功率（400 回合，不是这次的结果）：

| Spatial | Object | Goal | Long | Avg. |
|---:|---:|---:|---:|---:|
| 97.0 | 100.0 | 93.0 | 96.0 | **96.50** |

400 回合在 96–97% 附近的二项 SE 约 0.8 分。CROP 出来以后只报点估计不够，至少要看套件是否塌、以及和 96.50 的差是否大于噪声。

## 相机对应（已核对）

LIBERO `lerobot/envs/libero.py`：`agentview_image` → `observation.images.image`（全局），`robot0_eye_in_hand_image` → `observation.images.image2`（腕部）。v044 `empty_cameras=1`，第三路是 padding，CROP 不剪。

校准集只有 `image` / `image2` 两个 key，空相机在 `_preprocess_images` 里补全且 mask=False。保存的拼图左=全局第三人称桌面，右=腕部夹爪近景（夹爪出现在画面底部）：

- `results/stage_nfe/crop_vla/cameras/task00_pair_global_left_wrist_right.png`（番茄酱→篮子）
- `.../task01_pair_global_left_wrist_right.png`（黑碗→盘子）
- `.../task02_pair_global_left_wrist_right.png`（沙拉酱→篮子）
- 映射 JSON：`cameras/camera_map.json`

每路真实相机原来 256 token（16×16）。保留 128：

- 腕部：96 attention + 32 多样性
- 全局：32 attention + 96 多样性

相关性：第 2 层（0-index，Drop-6 仍执行 attention）语言 query → 视觉 key，跨 head 与有效语言 query 取平均，两相机分别排序。多样性：从相关性集合出发，在 L2 归一化视觉特征上按余弦最近邻贪心补最不相似的 token。选出后按**原始顺序** gather，不是置零。

预期正式 prefix / KV：**968 → 712**（128+128+256 空相机+200 语言）。动作 RoPE 仍用**未剪**的 `sum(pad)`，不是 712。

## 实现要点

核心：`scripts/pi05_libero/crop_vla.py`

1. `embed_prefix` 只跑一次（视觉编码一次）。
2. Scout 只跑语言层 0..2（Drop-6 的 layer 1 仍走现有 keep-KV 包装），hook `layers[2].self_attn` 取 attention 后丢弃临时 cache。第 3 层及更深层在 scout 中不执行；正式压缩 prefix 再跑完整 18 层（含 keep-KV）。
3. 按相机 gather 原始 embeddings / pad / 原始 `position_ids`。
4. 再跑缩短后的 prefix，得到更短 KV；专家 denoise 用原始 action offset。

接入：`eval_drop.py --crop` / `--crop-keep-all`。默认输出在 `results/stage_nfe/crop_vla/`，不会覆盖 `eval_drop/k6_fisher` 或 `pi05_k6_fisher_nfe1`。

## 离线检查

CPU 已通过：`select_one_camera` 腕部/全局预算共 128、相关性与多样性不重叠、`keep_all` 返回全部 256、多样性不回取 seed、增量多样性与朴素循环一致、timer 相邻段 pairing、scout 假层只执行 0..2。

GPU 项（`check_crop_vla.py`，需空闲卡；`run_eval_crop.sh` 失败即不进 400 回合）：

- 同一噪声下，keep-all（每相机 256）动作 vs Baseline；非有限输出直接失败。阈值 `max_abs>2e-2` 中止（keep-all 应对齐同一 Drop-6 prefix；2e-2 是 O(1) 动作上 bf16 累积的上限，不放宽）。
- 剪枝后 `short_len == full_len - 128×n_live` 且 `kv_len == short_len`；腕部 96+32、全局 32+96、空相机 256；动作 RoPE offset 仍为未剪 `sum(pad)`。
- Scout 调用计数：layer 2 ≥1，layer ≥3 为 0。
- 可视化：红框=相关性，蓝框=多样性，变暗=丢掉。将写到 `results/stage_nfe/crop_vla/viz/`。

尚未得到动作误差、实际 KV 长度、可视化。identity 若失败，先修正确性，不进入 400 回合。

## 延迟（未测）

要求：同一张**独占** GPU，多帧观测，充分预热，完整 `predict_action_chunk`（scout + 选择 + 第二次 prefix + 动作专家），报中位数和 P95。脚本 `measure_crop_efficiency.py` 会先查占用，卡不空闲就退出。

没有独占测量数字。定性瓶颈（供空闲后对照，不是结果）：

- Scout 是一次 **968** token、只到第 2 层的 prefix，不再跑第 3–17 层。
- 正式 prefix 省 256 token（约 −26% 序列），仍要付 scout（短栈）+ 第二次压缩 prefix。
- 现有 Drop-6 × NFE=1 里，NFE=1 已经把专家压到约 40 ms，prefix+视觉仍占 chunk 的大部分。CROP 是否还有净加速，取决于 scout 的额外 prefix 是否小于第二次缩短省下的时间。这是本筛查要回答的效率问题。
- 空相机 256 token 仍留在序列里（按需求不改）。

## 是否值得继续

现在**不能**下结论。实现和相机角色已经对齐，但缺三块硬证据：keep-all 是否动作一致、400 回合 SR 相对 96.50 是否只在噪声内、独占延迟是否下降。

建议空闲后按这个顺序，不要加预算/模型/混合实验：

1. `check_crop_vla.py`。identity 或长度不对就停。
2. `bash run_eval_crop.sh`（tmux 名 `crop_vla_eval`）。只跑 CROP 400 回合。
3. 同一张空卡 `measure_crop_efficiency.py`。
4. 用真实数字填本文件。若某套件塌、Avg. 掉出约 2 分、或中位延迟不降/变慢，把 scout 记为主要瓶颈，而不是宣称无损加速。

## 运行命令

```bash
export PYTHONPATH=/home/guorj/lerobot/src
export HF_ENDPOINT=https://hf-mirror.com
cd /home/guorj/Skill_Efficient_VLA/functional_necessity/scripts/pi05_libero
PY=/home/guorj/.miniforge3/envs/lerobot/bin/python

# 已跑：相机示例 + CPU 选择自测
$PY check_crop_vla.py --cameras-only

# GPU 空闲后：identity / 长度 / 可视化（脚本会拒绝占用中的卡）
$PY check_crop_vla.py --gpu $GPU

# GPU 空闲后：先离线检查，通过后才 400 回合（不重跑 baseline）
bash run_eval_crop.sh $GPU

# 独占 GPU 后：完整 chunk 中位数 / P95 + scout/select/prefix/expert 分解
$PY measure_crop_efficiency.py --gpu $GPU
```

## 路径

| 内容 | 路径 |
|---|---|
| 本报告 | `results/stage_nfe/crop_vla/feasibility.md` |
| 相机图 | `results/stage_nfe/crop_vla/cameras/` |
| 离线检查 JSON / viz | `results/stage_nfe/crop_vla/offline_check.json`, `viz/`（待 GPU） |
| CROP 闭环 | `results/stage_nfe/crop_vla/eval_crop/`（待 GPU） |
| 效率 | `results/stage_nfe/crop_vla/efficiency.json`（待独占 GPU） |
| Baseline 闭环 | `results/stage_nfe/pi05_k6_fisher_nfe1/{eval_info.json,drop_meta.json}` |
| 代码 | `scripts/pi05_libero/{crop_vla.py,check_crop_vla.py,measure_crop_efficiency.py,run_eval_crop.sh,eval_drop.py}` |

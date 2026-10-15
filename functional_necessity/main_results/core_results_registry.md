# 核心结果清单（可追溯索引）

更新：2026-10-15。服务器路径前缀：`/home/guorj/Skill_Efficient_VLA/functional_necessity/`。Git `HEAD`：`9c36e96`（与顾问审阅一致）。

**证据等级**

| 标记 | 含义 |
| --- | --- |
| 已核实 | 原始 `eval_info.json` / `drop_meta.json` / `drop.json` / 探针 CSV 与文档数字一致，或脚本当场复算一致 |
| 推断 | 由日志时间、文件修改时间、提交历史等间接支持，**无**运行当时源码哈希 |
| 待核实 | 缺闭环、算子与源码对应不明、或仅单帧效率检查 |

**可比性**：仅当检查点、skip 算子、NFE、`n_episodes`、seed、运行时（eager/compile/graph）、`num_steps_wait` 一致时可横向比较 SR。不同会话的毫秒/FLOPs **不可**相减拼接。

---

## 1. 层剪枝主配置（NFE=10，LIBERO 四套件）

| 模型 | 配置 | 层集 (0-based) | L0 | 判据 | n | SR Avg | bits | Skip 算子（记录） | 证据 | 原始路径 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| π0.5 | Full | — | — | — | 400 | 97.25 | 389/400 | 无 skip | 已核实 | `results/eval_drop/baseline/eval_info.json` |
| π0.5 | Fisher Drop-6 | `[1,13,14,15,16,17]` | 未删 L0 | Fisher | 400 | 97.75 | 391/400 | DTR Appendix B prefix keep-KV | 已核实 | `results/eval_drop/k6_fisher/eval_info.json` |
| π0.5 | 同上 | 同上 | 同上 | Fisher | 2000 | 96.80→97.10 | 1936/1942 | 同上 | 已核实 | `results/stage_nfe/pi05_full_nfe10_2000/`, `pi05_k6_fisher_nfe10_2000/` |
| MolmoAct2 | Full | — | — | — | 2000 | 97.50 | 1950/2000 | 无 skip | 已核实 | `results/molmoact2_libero/baseline_eval/eval_info.json` |
| MolmoAct2 | Taylor Drop-16 no-L0 | `[1–4,24–35]` | 排除 L0 后选层 | Taylor | 2000 | 98.10 | 1962/2000 | `keep_kv_identity_residual` | 已核实 SR；算子与源码见 §3 | `results/molmoact2_libero/drop16_taylor_nol0/eval_info.json` |
| SmolVLA | Full | — | — | — | 400 | 83.00 | 332/400 | 无 skip | 已核实 | `results/smolvla_libero/lerobot_official_nas10/eval_info.json` |
| SmolVLA | Taylor Drop-4 | `[12,13,14,15]` | 未排除 L0 | Taylor | 400 | 78.50 | 314/400 | keep-KV（VLM 联合 forward） | 已核实 | `results/smolvla_libero/drop4_taylor/eval_info.json` |

**核心对照（节选，400 或 2000 见各 `eval_*_libero.md`）**

| 对照 | n | 结果要点 | 状态 |
| --- | ---: | --- | --- |
| π0.5 CosSim/CKA Drop-6 | 400 | 0.75% / 2.00% | 已核实 |
| π0.5 Last-6 vs Fisher Drop-6 | 400 | 73.50% vs 97.75% | 已核实 |
| π0.5 K/V 写零 vs keep-KV Drop-6 | 400 | 26.25% vs 97.75% | 已核实 |
| Molmo Last-16 vs Taylor Drop-16 | 2000 | 58.25% vs 98.10% | 已核实 |
| Molmo Fisher vs Taylor Drop-16 | 2000 | 72.75% vs 98.10% | 已核实 |
| Molmo CosSim/PPL Drop-16 | — | 提前停止，无完整 `eval_info.json` | 已核实（stopped） |
| SmolVLA 全预算 Drop-4 | 400 | 无配置 ≥83% 锚点 | 已核实 |

层评分脚本：`scripts/pi05_libero/profiling_*.py`、`scripts/molmoact2_libero/score.py`、`scripts/smolvla_libero/score.py` → `rank`/`scores.md`。闭环：`eval_drop.py` / `run_drop*.py`。

---

## 2. 少步与联合闭环（本机）

| 模型 | 配置 | NFE | n | SR Avg | bits | seed | 可比锚点 | 证据 | 原始路径 |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- |
| π0.5 | Full | 1 | 400 | 97.00 | 388/400 | 1000 | Full NFE=10 400 | 已核实 | `results/stage_nfe/pi05_full_nfe1/` |
| π0.5 | Drop-6 | 1 | 400 | 96.50 | 386/400 | 1000 | Drop-6 NFE=10 400 | 已核实 | `results/stage_nfe/pi05_k6_fisher_nfe1/` |
| π0.5 | Drop-6 | 1 | 2000 | 95.70 | 1914/2000 | 1000 | 同上 2000 | 已核实 | `pi05_k6_fisher_nfe1_2000/` |
| π0.5 | Drop-6 | 1 | 2000 | 96.25 | 1925/2000 | 1000 | 同配置复跑 | 已核实 | `pi05_k6_fisher_nfe1_2000_r2/` |
| π0.5 | Full | 1 | 2000 | — | — | — | **缺目录** | 待核实 | （无 `pi05_full_nfe1_2000`） |
| Molmo | Full | 1 | 400 | 98.25 | 393/400 | 1000 | 与 2000 协议不同 | 已核实 | `molmoact2_full_nfe1/` |
| Molmo | Drop-16 | 1 | 400 | 99.25 | 397/400 | 1000 | 与 2000 协议不同 | 已核实 | `molmoact2_drop16_taylor_nfe1/` |
| Molmo | Full | 1 | 2000 | 97.80 | 1956/2000 | 1000 | Full NFE=10 2000 | 已核实 SR；skip 见 §3 | `molmoact2_full_nfe1_2000/` |
| Molmo | Drop-16 | 1 | 2000 | 97.60 | 1952/2000 | 1000 | Drop-16 NFE=10 2000 | 已核实 SR；skip 见 §3 | `molmoact2_drop16_taylor_nfe1_2000/` |
| SmolVLA | Full | 1/2/3 | 400 | 79.25/82.00/81.00 | 见 JSON | 1000 | Full 83.00 | 已核实 | `results/stage_nfe/smolvla_full_nfe{1,2,3}/` |
| SmolVLA | Drop-4 | 1/2 | 400 | 73.00/77.00 | 见 JSON | 1000 | 边界案例 | 已核实 | `smolvla_drop4_taylor_nfe{1,2}/` |

汇总文档：`main_results/eval_nfe_libero.md`（§7.2 已补 Molmo 2000 NFE=1）。

---

## 3. Skip 算子与源码版本（勿混用）

| 栈 | 名称 | 行为摘要 | 典型 SR 路径 | 典型效率路径 | 源码 |
| --- | --- | --- | --- | --- | --- |
| π0.5 | DTR Appendix B keep-KV | 语言层 identity 残差 + 写 K/V | 全部 π Drop-6 闭环 | `efficiency_fisher_k6.json`, §4 `sota_efficiency_pi05.json` | `scripts/pi05_libero/layer_skip.py` |
| π0.5 | no-KV 写零 | 对照实验 | Drop-6 26.25% | 非主路径 | 同上 `keep_kv=False` |
| Molmo/Smol | 历史 keep-KV | 残差不变；Molmo 曾仍跑完整 attention 后丢弃输出 | 多数 `molmoact2_libero` 2000 ep | `efficiency_taylor_drop16.json`, §4 旧行 | git `5682389` 的 `layer_skip.py`（**推断**对应关系） |
| Molmo/Smol | K/V-only | 仅 K/V+缓存；跳过 Q/attn/O/MLP | **待核实**是否与旧 SR 等价 | §3/§5 `sota_efficiency_*_kvonly.json` | 当前 `layer_skip.py`（Molmo 自 2026-10-09 11:41 修改，提交 `0824277` 2026-10-11） |
| 三栈 | Materialized | 释放 Q/MLP 权重；前向对齐 K/V-only | 代表帧 max-abs=0 | `deployment_*.json` | `scripts/stage_nfe/materialize.py` |

**Molmo 2000×NFE=1（2026-10-09）算子—源码对应（仅推断，非已确认）**

| 事实（已核实） | 内容 |
| --- | --- |
| 启动 | 日志 `molmoact2_*_nfe1_2000.log.gz`：`START 2026-10-09T01:31:29Z`，命令 `run_molmo_nfe1.py` |
| 元数据 | `drop.json`：`operator=keep_kv_identity_residual`, `num_inference_steps=1`, 50 ep/task |
| 结束 | `eval_info.json` 写入约 15:28–15:34 UTC |
| 磁盘 | 当日 `molmoact2_libero/layer_skip.py` 修改时间 **11:41 UTC**（在启动之后） |
| 提交 | K/V-only 重构进入 git **`0824277`（2026-10-11）**；此前 `5682389` 为“仍调用 `self_attn`”实现 |

**推断（待哈希核实）**：01:31 启动的进程在导入时更可能加载 **11:41 之前**磁盘上的 skip 实现（与 `5682389` 行为同类），**不能**写成“已确认旧 keep-KV”，也不能写成“已确认当前 K/V-only”。Full 运行 `MOLMO_DROP=` 为空，skip 集合不影响 Full 前向。

---

## 4. 效率测量（单输入诊断，非正式主表）

| 会话 / 文件 | 内容 | 算子 | 证据 |
| --- | --- | --- | --- |
| `sota_efficiency_pi05.json` | §4 四格 ms/FLOPs | π DTR keep-KV | 已核实（单帧） |
| `sota_efficiency_molmoact2.json` | §4 Molmo 四格 | 旧 keep-KV | 已核实（单帧）；与 kvonly **不可**相减 |
| `sota_efficiency_*_kvonly.json` | §3/§5 语言模块与部分整 chunk | K/V-only | 已核实（单帧）；Molmo bf16 差 ~0.008 **推断**未做全量闭环 |
| `deployment_molmoact2.json` | Expert CUDA graph + materialized | Graph 动作 max-abs=0（代表帧） | 已核实（单帧） |
| 正式多输入、独占卡 median/P95 | — | — | **待核实** |

脚本：`scripts/stage_nfe/measure_sota_efficiency.py`, `measure_deployment.py`。

---

## 5. CARE_event 校准池（480 帧，CPU 回放）

| 项目 | 值 | 证据 |
| --- | --- | --- |
| 池规模 | 40 任务 × 12 episode × 1 观测 = 480 | `results/stage2_care_v2/shared.pt` |
| H, λ | 10, 1 | `dump_shared.py`, `config.json` |
| 固定均值 | `mean(d)=2.047233…`, `mean(α+G)=1.116667…` | 已核实（`shared.pt` 复算） |
| 探针 | π0.5 NFE {1,2,10}；Smol {1,2,3,10}；2 噪声 | `nfe_probe_pi05/`, `nfe_probe_smolvla/` |
| 等预算四组 + 20 种子 | Random/α/G-only/α_event；种子见下 | `nfe_probe_event_care/budget_compare/` |
| Holdout | 120 观测，seed 4242，240 次比较/NFE | `nfe_probe_care_event_holdout/` |

**四组对照（合并 noise0+1，2026-10-15 复现通过）** — 详见 `budget_compare/REPORT.md`：

| 模型 | Random | α | G-only (seed42) | α_event |
| --- | --- | --- | --- | --- |
| Smol 夹爪 NFE1 | 8/15 | 11/15 | 13/15 | 14/15 |
| Smol 尾部 NFE1 | 26/73 | 36/73 | 38/73 | 44/73 |
| Pi 夹爪 NFE1 | 5/10 | 7/10 | 10/10 | 10/10 |
| Pi 尾部 NFE1 | 23/71 | 38/71 | 32/71 | 38/71 |

选样索引：`budget_compare/selections/*.json`。前瞻接受规则：**未制定**（本轮未做）。

---

## 6. LIBERO-plus（π0.5，旁支）

| 状态 | 说明 |
| --- | --- |
| 已完成六维 Full/Drop-6 NFE=10 宏平均 | 85.89 vs 81.65（−4.24 pp），见 `eval_libero_plus.md` |
| Drop-6 NFE=1 六维 | Language/Init/Camera/Layout/Light/Background 已写入工作区文档；**引用前应对各维 `eval_info.json` 核对** |
| Sensor Noise | Goal 部分 JSON 已有；Long NFE=10 评测进行中（勿杀进程） |
| 历史笔误 | 分布内 Drop-6 NFE=1 曾误写 **76.00**（实为 Smol Drop-4 NFE=3）；已改为 π0.5 **96.50 / 95.70–96.25** |

---

## 7. 明确缺口（本轮不补 GPU）

- 冻结前瞻 NFE 推荐协议与 holdout 外新 episode 验证  
- Molmo CARE 探针  
- K/V-only / materialized 与历史 SR 的闭环对齐（除代表帧外）  
- 正式效率：多输入、独占 3090、与最终实现配对  
- π0.5 Full NFE=1 @ 2000 ep  

---

## 8. 相关文档

| 文档 | 用途 |
| --- | --- |
| `eval_pi05_libero_v044.md` | π 层剪枝全表 |
| `eval_molmoact2_libero.md` | Molmo 层剪枝 |
| `eval_smolvla_libero.md` | Smol 边界 |
| `eval_nfe_libero.md` | NFE 闭环 |
| `efficiency_analysis.md` | 效率分节 |
| `care_event_action_expert_nfe.md` | CARE 方法与 holdout |
| `eval_libero_plus.md` | OOD 旁支 |

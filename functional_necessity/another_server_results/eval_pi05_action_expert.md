# Action expert: fewer flow-matching steps (NFE) on π0.5-LIBERO, with and without LLM layer drop

π0.5-LIBERO v044 (`lerobot/pi05_libero_finetuned_v044`, local copy `refs/official_ckpts/pi05_libero_finetuned`, `config.json` identical to the Hub snapshot). Inference only: no recovery fine-tune, no retraining of the action expert. NFE = `policy.num_inference_steps` (Euler steps, default 10). LLM drop is the same DTR Appendix B runtime skip and the same Fisher masks as `eval_pi05_libero_v044.md`.

**Protocol.** Four suites × 10 tasks × 10 episodes (400 episodes), `n_action_steps=10`, `compile_model=false`, `seed=1000`, `batch_size=1`, Gaussian source N(0, I) unless noted. Suites run as four separate processes (`scripts/run_drop_nfe_libero.sh`, `scripts/run_nfe_source_libero.sh`) so a π0.5 eval fits on a shared 3090; each suite restarts the RNG at seed 1000. The NFE=10 anchors in `eval_pi05_libero_v044.md` ran all four suites in one process, so their random streams differ. Under the split protocol the full model at NFE=10 scored 95.00, vs 97.25 single-process; treat ~2 points as protocol/seed noise (binomial SE ≈ 0.8 pt at 97%, ≈ 2.1 pt at 77%, per 400 episodes).

Source: `results/nfe_source/libero_v044/{cell}/{eval_info.json,drop_meta.json,libero_*/}`. Logs: `results/logs/{k6_fisher_nfe2,k6_fisher_nfe1,k6_fisher_nfe1_run2,k6_fisher_nfe10,k6_fisher_nfe1_trim,k6_fisher_nfe1_trim_run2,k9_fisher_nfe2,k6_igia_nfe1,k6_gateprobe_nfe1,nfe_source_libero*}.log`. Latency probes: `results/nfe_source/libero_v044/probes/`. Split-protocol reruns use the same `seed=1000` (`*_run2`); they measure CUDA / shared-GPU nondeterminism, not a new RNG seed.

## Fisher Drop-k × NFE

Table: LIBERO success (%), no recovery. Single-process NFE=10 rows are from `eval_pi05_libero_v044.md` (run 1 / run 2). Split-protocol Δ is vs the same-protocol NFE=10 cell of that mask; single-process rows have no split NFE=10 twin.

| LLM drop | NFE | Spatial | Object | Goal | Long | Avg. | Δ vs same-protocol NFE=10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| None (18/18), single-process | 10 | 97.0 | 100.0 | 95.0 | 97.0 | 97.25 | — |
| None, split protocol | 10 | 96.0 | 97.0 | 94.0 | 93.0 | 95.00 | — |
| None, split protocol | 2 | 98.0 | 100.0 | 97.0 | 98.0 | 98.25 | +3.25 |
| Fisher Drop-6 (12/18), single-process | 10 | 97.0 / 99.0 | 100.0 / 100.0 | 96.0 / 95.0 | 98.0 / 96.0 | 97.75 / 97.50 | — |
| Fisher Drop-6, split protocol | 10 | 96.0 | 100.0 | 96.0 | 91.0 | 95.75 | — |
| Fisher Drop-6, split protocol | 2 | 95.0 | 100.0 | 93.0 | 94.0 | 95.50 | −0.25 |
| Fisher Drop-6, split protocol | 1 | 94.0 / 93.0 | 100.0 / 100.0 | 93.0 / 93.0 | 97.0 / 95.0 | 96.00 / 95.25 | +0.25 / −0.50 |
| Fisher Drop-9 (9/18), single-process | 10 | 89.0 / 87.0 | 87.0 / 93.0 | 53.0 / 57.0 | 78.0 / 81.0 | 76.75 / 79.50 | — |
| Fisher Drop-9, split protocol | 2 | 88.0 | 93.0 | 54.0 | 74.0 | 77.25 | n/a |

Masks: Drop-6 `[1, 13, 14, 15, 16, 17]`; Drop-9 `[1, 2, 3, 10, 13, 14, 15, 16, 17]`.

- **Drop-6 split NFE=10 = 95.75**, 2 points under the single-process 97.75 / 97.50, matching the full-model protocol gap (97.25 → 95.00). Drop-6 NFE=2 is 95.50 (−0.25). Drop-6 NFE=1 is **96.00 / 95.25** on two same-seed reruns (+0.25 / −0.50). The ~2-point gap vs the single-process 97.75 was protocol noise, not an NFE loss. Same-protocol rerun spread is 0.75 pt.
- **Drop-9 × NFE=2 = 77.25**, inside the two single-process NFE=10 runs (76.75 / 79.50). Cutting Euler steps from 10 to 2 does not amplify the Drop-9 loss. Goal stays the weak suite (54 vs 53 / 57). No split-protocol Drop-9 NFE=10 was run.
- NFE=1 is a single Euler step from t=1, so the expert outputs roughly its conditional-mean action. Drop-6 still holds 96.00 / 95.25 there, straddling its own split NFE=10 (95.75).
- Working read: on LIBERO, LLM drop and NFE reduction are close to orthogonal for success. Drop depth sets the level (≈96 at k=6 under split protocol, ≈77 at k=9); NFE in {1, 2, 10} moves it by no more than same-protocol rerun noise.

## Drop-6 masks from three methods × NFE=1

Same protocol, NFE=1, no recovery. NFE=10 rows are the single-process runs in `results/eval_drop/k6_{method}{,_run2}` (run 1 / run 2).

| Method (Drop-6 mask) | NFE | Spatial | Object | Goal | Long | Avg. | Δ vs NFE=10 (run 1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fisher `[1, 13, 14, 15, 16, 17]` | 10 | 97.0 / 99.0 | 100.0 / 100.0 | 96.0 / 95.0 | 98.0 / 96.0 | 97.75 / 97.50 | — |
| Fisher, split protocol | 10 | 96.0 | 100.0 | 96.0 | 91.0 | 95.75 | −2.00 |
| Fisher | 1 | 94.0 / 93.0 | 100.0 / 100.0 | 93.0 / 93.0 | 97.0 / 95.0 | 96.00 / 95.25 | −1.75 / −2.50 |
| IGIA `[2, 13, 14, 15, 16, 17]` | 10 | 97.0 / 98.0 | 99.0 / 99.0 | 94.0 / 96.0 | 98.0 / 92.0 | 97.00 / 96.25 | — |
| IGIA | 1 | 97.0 | 99.0 | 91.0 | 97.0 | 96.00 | −1.00 |
| GateProbe `[2, 4, 14, 15, 16, 17]` | 10 | 97.0 / 98.0 | 93.0 / 96.0 | 94.0 / 96.0 | 96.0 / 96.0 | 95.00 / 96.50 | — |
| GateProbe | 1 | 98.0 | 94.0 | 93.0 | 95.0 | 95.00 | 0.00 |

- All three NFE=1 cells sit inside or within ~1 point of their own two NFE=10 runs, and the gaps (−1.75 / −1.00 / 0.00) are no larger than the ~2-point split-protocol offset. Fisher's split-protocol NFE=10 (95.75) makes that explicit: NFE=1 is +0.25 / −0.50 on two same-seed reruns. NFE=1 does not hurt any of the three Drop-6 masks.
- At NFE=1 the method order stays the same as at NFE=10: Fisher ≈ IGIA (96.00 / 95.25) > GateProbe (95.00). The spread (1 point) is within rerun noise.
- Each method keeps its weak suite: IGIA Goal 91 (94 / 96 at NFE=10), GateProbe Object 94 (93 / 96). These shortfalls come from the mask, not from NFE.
- Swapping in layer 2 (IGIA) or layers 2 and 4 (GateProbe) for Fisher's layer 1 does not change sensitivity to NFE. So on LIBERO the drop × NFE orthogonality holds across masks, not only for Fisher.

## Prefix trim (Fisher Drop-6, NFE=1, split protocol)

Inference only: drop the all −1 `empty_camera_0` image stream (256 tokens, mask already false) and right-truncate language from 200 to the last valid token (LIBERO prompts keep ~56–58). Does not change weights. Closed-loop uses `--prefix-trim` in `eval_drop.py` (`scripts/prefix_trim.py`). Smoke: one calib-frame L2 vs untrimmed Drop-6 NFE=1 was 0.044 (chunk norm 9.16); spatial 10 episodes (1 per task) scored 90.00.

| Setting | NFE | Spatial | Object | Goal | Long | Avg. | Δ vs untrimmed NFE=1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fisher Drop-6, split, no trim | 10 | 96.0 | 100.0 | 96.0 | 91.0 | 95.75 | — |
| Fisher Drop-6, split, no trim | 1 | 94.0 / 93.0 | 100.0 / 100.0 | 93.0 / 93.0 | 97.0 / 95.0 | 96.00 / 95.25 | — |
| Fisher Drop-6, split, prefix trim | 1 | 95.0 / 96.0 | 100.0 / 100.0 | 92.0 / 95.0 | 90.0 / 91.0 | 94.25 / 95.50 | −1.75 / +0.25 |

- Two same-seed reruns: untrimmed NFE=1 is **96.00 / 95.25** (spread 0.75 pt); trim is **94.25 / 95.50** (spread 1.25 pt). The two bands overlap. Trim vs untrimmed is −1.75 on run 1 and **+0.25** on run 2, so the first-run −1.75 is not a stable trim tax.
- Both trim runs sit inside the ~2-point protocol band around split NFE=10 (95.75). Object stays 100 on all four cells. Goal / Long move by 1–3 points between reruns, not a suite collapse.
- Closed-loop logs print `prefix_trim=True` and `images 3->2 keep=[0, 1]`, language 200→57–58. Pre-eval action L2 vs untrimmed Gaussian NFE=10 is 2.56, matching untrimmed NFE=1 (2.53), so trim is not a second NFE change.
- Prefix trim is therefore usable on LIBERO at Drop-6 × NFE=1: the ~400 placeholder tokens are not functionally necessary for this success level, and any remaining gap is within same-protocol rerun noise.

## Inference-only structured source (full model, no drop)

x₁ ~ N(0, Σ_β), Σ_β = 0.5 Σ_action + 0.5 I, Σ_action estimated in padded [50, 32] MEAN_STD space from `results/calib/suite_64x8_seed123` (2048 chunks). Split protocol, 400 episodes per cell.

| Source | NFE | Spatial | Object | Goal | Long | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 10 | 96.0 | 97.0 | 94.0 | 93.0 | 95.00 |
| Gaussian | 2 | 98.0 | 100.0 | 97.0 | 98.0 | 98.25 |
| Structured | 10 | 99.0 | 98.0 | 97.0 | 97.0 | 97.75 |
| Structured | 2 | 99.0 | 98.0 | 97.0 | 97.0 | 97.75 |

Structured NFE=2 does not beat Gaussian NFE=2 (97.75 vs 98.25), so a structured source gives no free low-NFE gain to a Gaussian-trained expert. The two Structured rows share suite percentages by chance; 9 tasks differ and the files are separate runs. Known issues with this operator: on the 7 real action dims Σ_β has diagonal ≈ 0.99 (trace 346 vs 350 for I), so the expected path length is unchanged; 625 of the 629 units of removed variance sit on the 25 zero-padded dims (variance 0.5 instead of 1); 15.8% of calib timesteps are `action_is_pad` copies of the last frame. With an ideal Gaussian-trained velocity field, one Euler step from t=1 returns E[a | o] for any source.

## Action-generation latency

`predict_action_chunk`, batch 1, one process, configs shuffled each round, 12 rounds, median ms. Measured on a shared 3090 (other jobs on the card), so absolute values run high; use ratios. Source: `probes/nfe_unified_timing.json`, `probes/prefix_trim_probe.json`.

| LLM drop | NFE=10 | NFE=4 | NFE=2 | NFE=1 |
| --- | ---: | ---: | ---: | ---: |
| None | 610.5 | 426.2 | 364.2 | 334.1 |
| Fisher Drop-6 | 549.3 | 365.2 | 307.2 | 276.9 |
| Fisher Drop-9 | 519.1 | 340.3 | 275.2 | 245.6 |

Linear fit per row: one Euler step costs ≈ 30.3–30.7 ms regardless of drop; the fixed part (vision + language prefix) is 303 / 246 / 216 ms. At NFE=10 the prefix is ~50% of the call; at NFE=2 it is ~83%, so LLM drop is a larger share of what remains.

| Cell (closed-loop success) | Latency | vs full NFE=10 |
| --- | ---: | ---: |
| None, NFE=10 (97.25) | 610.5 | — |
| None, NFE=2 (98.25, split) | 364.2 | −40% |
| Fisher Drop-6, NFE=2 (95.50) | 307.2 | −50% |
| Fisher Drop-6, NFE=1 (96.00 / 95.25) | 276.9 | −55% |
| Fisher Drop-6, NFE=1 + prefix trim (94.25 / 95.50) | 187.6 | −69% |
| Fisher Drop-9, NFE=2 (77.25) | 275.2 | −55% |

Structured vs Gaussian source has no latency effect (NFE=10: 604.8 vs 610.5 ms; NFE=2: 364.2 vs 364.2 ms; drawing the coloured noise costs 0.04 ms). The earlier per-cell timings (Gaussian 598 / 321 ms vs Structured 380 / 192 ms) were taken while another job shared the GPU for the Gaussian cells only.

The prefix also carries ~400 masked tokens: 256 from the all −1 `empty_camera_0` image and ~150 padding tokens (LIBERO prompts have 39–62 valid tokens of 200). Dropping them changes the action chunk by L2 0.03–0.07 (chunk norm ≈ 15) and cuts full NFE=10 from 596 to 489 ms and Drop-6 NFE=1 from 275 to 188 ms (probe median). Closed-loop Fisher Drop-6 × NFE=1 + trim scored 94.25 / 95.50 (table above). The eval-time `action_gen` mean on a contended GPU was 215 ms (no skip hook in that timer; skip is on for the rollouts).

## Checks

All seven drop × NFE cells plus two same-seed reruns (`k6_fisher_nfe10`, `k6_fisher_nfe2`, `k6_fisher_nfe1`, `k6_fisher_nfe1_run2`, `k6_fisher_nfe1_trim`, `k6_fisher_nfe1_trim_run2`, `k9_fisher_nfe2`, `k6_igia_nfe1`, `k6_gateprobe_nfe1`) wrote four suite `eval_info.json` files plus a merged one: 40 tasks × 10 episodes, `pc_success` = success bits / 400, `compile_model=false`, `n_action_steps=10`, `seed=1000`, `recovery=false`, `noise_type=gaussian`. `drop_meta.json` records the drop list and `num_inference_steps`. The trim cells also record `prefix_trim=true`, `images 3→2 keep=[0, 1]`, language 200→58. Pre-eval sanity on one calib frame, before the skip hook is installed: action L2 vs Gaussian NFE=10 is 0.00 at NFE=10, 1.25 at NFE=2, 2.53 at NFE=1, and 2.56 at NFE=1+trim, so the NFE flag takes effect and trim is not a second NFE change. Closed-loop logs print `Drop fisher k=6 layers=[1, 13, 14, 15, 16, 17]` / `k=9 layers=[1, 2, 3, 10, 13, 14, 15, 16, 17]`, `Drop igia k=6 layers=[2, 13, 14, 15, 16, 17]`, and `Drop gateprobe k=6 layers=[2, 4, 14, 15, 16, 17]`, matching `ranks.json`. Spatial 10-episode trim smoke scored 90.00 and is not mixed into the 400-ep cell. Same-seed rerun spread: NFE=1 0.75 pt, NFE=1+trim 1.25 pt.

Missing for a clean NFE comparison: IGIA / GateProbe Drop-6 and Fisher Drop-9 at NFE=10 under the split protocol. Fisher Drop-6 split NFE=10 is done (`k6_fisher_nfe10`, 95.75).

## RoboTwin Easy, Drop-2 × NFE {10, 2, 1} (100 episodes)

LIBERO NFE=1 is not a free lunch on RoboTwin. Same model family, Appendix B skip, `n_action_steps=50`, `seed=1000`, 8 tasks × 100 episodes, Fisher Drop-2 `[16, 17]`. Full table: `eval_pi05_robotwin.md`. Cells: `results/robotwin/eval/{n100/anchor,n100_k2_fisher_task,n100_nfe1/anchor,n100_nfe1/k2_fisher_task,n100_nfe2/anchor,n100_nfe2/k2_fisher_task}/`.

Full-model NFE=1 `beat_block_hammer` / `click_bell` were first OOM'd on a contended GPU2, then completed on GPU0 (64 and 68). All NFE=2 tasks finished `exit=0`.

| Setting | NFE=10 | NFE=2 | NFE=1 |
| --- | ---: | ---: | ---: |
| Full model | 69.63 | 69.75 | 53.75 |
| Fisher Drop-2 | 71.75 | 71.13 | 57.00 |

NFE=2 matches NFE=10. NFE=1 loses **−15.88 / −14.75**. Drop-2 and the full model stay within ~2 points at every NFE: the cliff is the Euler-step cut, not `[16, 17]`. `click_bell` is the NFE=2 exception (91→66 / 93→78). Opposite of LIBERO Drop-6 × NFE=1 (96.00 / 95.25).

An offline source gate on the same eight tasks (14-dim \(\Sigma\), pad dropped, 96 paired forwards) gives median \(\|a_{G1}-a_{S1}\|/\|a_{G10}\|=0.009\). Structured NFE=1 is source-independent here too; no 20-episode smoke.

A follow-up offline mid-state gate (one Euler jump from \(z_t=(1-t)a_{\mathrm{warm}}+t\varepsilon\), still 1 NFE) gives median \(\|a-a_{G10}\|/\|a_{G10}\|\) of 0.0585 for Gaussian NFE=1 vs 0.0325 / 0.0326 for oracle / causal GT at \(t=0.5\). Replacing GT with the model's own G1: \(\Delta\approx 1\) stays closer (0.0342) but \(\Delta=50\) hold-last is farther (0.1581, 93/96 pairs). Closed-loop `click_bell` NFE=1 mid-state scored 55 vs Gaussian 68. Details: `eval_pi05_robotwin.md`.

## Visual tokens on Fisher Drop-6 × NFE=1

EfficientVLA §3.3 selection, not their layer metric and not the DiT cache. Base is Fisher Drop-6 `[1, 13, 14, 15, 16, 17]`, NFE=1, prefix trim (empty camera and language padding removed). Score layer is the first non-dropped layer at depth ≥ 2, which is layer 2. Each real camera keeps `K_key=4` plus α=0.5 task tokens and the rest by cosine distance to that key set. Layers 0–2 still see the full trimmed prefix; layers 3–17 and every layer's KV are gathered to the same keep set. Expert RoPE still starts at the original trimmed prefix length. Relative L2 is on the first 10 action steps versus the same-seed trim reference. 32 calib frames (4 suites × 8) from `suite_64x8_seed123`. Closed-loop 2×2 is below. Source: `results/nfe_source/libero_v044/probes/visual_token_phase_a.json`.

Phase 0 on one frame: trim 3→2 images, language 200→56, prefix 568→168 at 56 tokens/camera (→280 at 112). Layer inputs 0–2 stay 568; layers 3–17 match the keep length. All 18 KV lengths match. Action shape `[1, 50, 7]`.

| Keep / camera | Prefix tokens (example) | Median r guided | Median r random | Mean r guided | Mean r random | Time (ms) | FLOPs (G) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 (trim only) | 568 | — | — | — | — | 96.2 | 2016.0 |
| 56 guided | 168 | 0.213 | 0.227 | 0.266 | 0.251 | 88.0 | 1104.0 |
| 112 guided | 280 | 0.096 | 0.122 | 0.165 | 0.148 | 89.3 | 1356.7 |

Random uses the same per-camera budget. Guided beats random on 20/32 frames at 56 tokens and 18/32 at 112. The median edge is real and small. The mean does not favor guided, and LIBERO-10's guided median is worse than random at both budgets (56: 0.352 vs 0.264; 112: 0.213 vs 0.190). One Long frame reaches r=1.15 at 56 tokens and r=1.03 at 112.

FLOPs fall 2016→1104 G at 56 tokens (1.83×) and to 1357 G at 112 (1.49×). Wall time only moves 96→88–89 ms, because layers 0–2, the vision tower, and the NFE=1 expert still run.

### Closed-loop 2×2 (112/56 × guided/random)

Same split protocol as the trim cells: 4 suites × 10 tasks × 10 episodes, `seed=1000`, `compile_model=false`, Fisher Drop-6, NFE=1, prefix trim on. Selector is EfficientVLA §3.3 (`K_key=4`, α=0.5, score layer 2); random keeps the same per-camera count. Cells: `k6_fisher_nfe1_trim_tok{112,56}{,_random}`. All four merged `eval_info.json` files exist (`n_episodes=400`). Logs: `==== OK` at 2026-10-07 01:35:22 (112 guided), 02:11:40 (112 random), 02:20:05 (56 random), 02:29:01 (56 guided). Hold bar was ~2 points vs trim **94.25 / 95.50**.

| Keep / camera | Mode | Spatial | Object | Goal | Long | Avg. | Δ vs trim 94.25 / 95.50 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 (trim only) | — | 95.0 / 96.0 | 100.0 / 100.0 | 92.0 / 95.0 | 90.0 / 91.0 | 94.25 / 95.50 | — |
| 112 | guided | 97.0 | 98.0 | 84.0 | 89.0 | **92.00** | −2.25 / −3.50 |
| 112 | random | 93.0 | 100.0 | 89.0 | 89.0 | **92.75** | −1.50 / −2.75 |
| 56 | guided | 73.0 | 80.0 | 61.0 | 59.0 | **68.25** | −26.00 / −27.25 |
| 56 | random | 86.0 | 87.0 | 83.0 | 79.0 | **83.75** | −10.50 / −11.75 |

- **Budget, not the selector, sets the level.** 112 tokens sit 2–3 points under the trim band. 56 tokens fail the hold bar even as random (−10.5 / −11.75). Guided at 56 collapses to 68.25.
- **Guided does not beat random.** At 112 the gap is −0.75 (92.00 vs 92.75), inside binomial noise. At 56 guided is **−15.5** vs random (68.25 vs 83.75): the scorer is actively harmful once the budget is tight.
- **112 guided is a Goal leak, not a uniform tax.** Spatial 97 / Object 98 / Long 89 sit on or inside trim; Goal 84 is −8 / −11 vs trim 92 / 95. Random 112 spreads the loss (Goal 89, Spatial 93, Object 100, Long 89).
- Phase A already pointed here: Long guided median relative L2 was worse than random at both budgets, and the mean did not favor guided. Closed-loop confirms that the written median gate was the wrong go/no-go.
- Eval wall time is not a latency win: 112 guided 16.1 s/ep vs 56 guided 22.3 s/ep, because failed rollouts run toward the 520-step cap. Probe FLOPs still fall 2016→1357 G at 112 and →1104 G at 56; chunk time stays ~88–89 ms.
- This stack is layer-drop + token prune without MLP prune, closer to EfficientVLA Table 5 Ex3 (layer+token, −9.9 on pick-coke-can) and Table 7 π0 LIBERO (tokens slightly hurt) than to Table 5 Ex1 (token-only, +4.3). It does not unlock extra layer drop.

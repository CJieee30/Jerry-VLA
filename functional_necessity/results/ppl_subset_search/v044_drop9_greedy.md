# v044 Drop-9 greedy PPL search

Checkpoint: `lerobot/pi05_libero_finetuned_v044`. Protocol matches `profiling_ppl.py` (64×8, frozen ξ, Appendix B skip).
Baseline loss: **0.050213**.

- Greedy Drop-9: `[2, 4, 7, 10, 11, 14, 15, 16, 17]`
- Keep: `[0, 1, 3, 5, 6, 8, 9, 12, 13]`
- Loss: 0.502188 (Δ +0.451975)
- Matches independent PPL `[2, 4, 7, 9, 10, 14, 15, 16, 17]`: **False**

- Step 1: pick L17 → drop `[17]` loss=0.050213
- Step 2: pick L16 → drop `[16, 17]` loss=0.051116
- Step 3: pick L14 → drop `[14, 16, 17]` loss=0.060956
- Step 4: pick L15 → drop `[14, 15, 16, 17]` loss=0.070931
- Step 5: pick L2 → drop `[2, 14, 15, 16, 17]` loss=0.089423
- Step 6: pick L11 → drop `[2, 11, 14, 15, 16, 17]` loss=0.140407
- Step 7: pick L7 → drop `[2, 7, 11, 14, 15, 16, 17]` loss=0.232014
- Step 8: pick L10 → drop `[2, 7, 10, 11, 14, 15, 16, 17]` loss=0.363048
- Step 9: pick L4 → drop `[2, 4, 7, 10, 11, 14, 15, 16, 17]` loss=0.502188

JSON: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/ppl_subset_search/v044_drop9_greedy.json`


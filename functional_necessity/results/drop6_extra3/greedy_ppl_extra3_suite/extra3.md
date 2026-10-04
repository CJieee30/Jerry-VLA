# v044 Drop-6 + extra-3 (greedy_ppl, suite pool)

Checkpoint: `lerobot/pi05_libero_finetuned_v044`. Universal skip `U=[1, 13, 14, 15, 16, 17]`. Candidates `[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`. Greedy PPL on skip(U), frozen ξ, extra-k=3.

- Pooling: **concat 10×16×8 per-task dumps inside each suite**
- Unique extra-3 triples: **2**
- Match global Fisher `{2,3,10}`: **0/40**
- extra-3 contains L0: **0/40**
- extra-3 contains L7: **30/40**

| Triple | Count |
| --- | ---: |
| `[2, 7, 11]` | 30 |
| `[2, 8, 11]` | 10 |

## Pools

| Pool | n_batches | extra-3 | union | Δloss | seconds |
| --- | ---: | --- | --- | ---: | ---: |
| `Spatial` | 160 | `[2, 7, 11]` | `[1, 2, 7, 11, 13, 14, 15, 16, 17]` | 0.344755 | 2541.1 |
| `Object` | 160 | `[2, 8, 11]` | `[1, 2, 8, 11, 13, 14, 15, 16, 17]` | 0.219511 | 2544.3 |
| `Goal` | 160 | `[2, 7, 11]` | `[1, 2, 7, 11, 13, 14, 15, 16, 17]` | 0.298302 | 2544.5 |
| `Long` | 160 | `[2, 7, 11]` | `[1, 2, 7, 11, 13, 14, 15, 16, 17]` | 0.260306 | 2545.1 |

## Per-task extra-3

- `Spa0` (libero_spatial 0): `[2, 7, 11]`
- `Spa1` (libero_spatial 1): `[2, 7, 11]`
- `Spa2` (libero_spatial 2): `[2, 7, 11]`
- `Spa3` (libero_spatial 3): `[2, 7, 11]`
- `Spa4` (libero_spatial 4): `[2, 7, 11]`
- `Spa5` (libero_spatial 5): `[2, 7, 11]`
- `Spa6` (libero_spatial 6): `[2, 7, 11]`
- `Spa7` (libero_spatial 7): `[2, 7, 11]`
- `Spa8` (libero_spatial 8): `[2, 7, 11]`
- `Spa9` (libero_spatial 9): `[2, 7, 11]`
- `Obj0` (libero_object 0): `[2, 8, 11]`
- `Obj1` (libero_object 1): `[2, 8, 11]`
- `Obj2` (libero_object 2): `[2, 8, 11]`
- `Obj3` (libero_object 3): `[2, 8, 11]`
- `Obj4` (libero_object 4): `[2, 8, 11]`
- `Obj5` (libero_object 5): `[2, 8, 11]`
- `Obj6` (libero_object 6): `[2, 8, 11]`
- `Obj7` (libero_object 7): `[2, 8, 11]`
- `Obj8` (libero_object 8): `[2, 8, 11]`
- `Obj9` (libero_object 9): `[2, 8, 11]`
- `Goal0` (libero_goal 0): `[2, 7, 11]`
- `Goal1` (libero_goal 1): `[2, 7, 11]`
- `Goal2` (libero_goal 2): `[2, 7, 11]`
- `Goal3` (libero_goal 3): `[2, 7, 11]`
- `Goal4` (libero_goal 4): `[2, 7, 11]`
- `Goal5` (libero_goal 5): `[2, 7, 11]`
- `Goal6` (libero_goal 6): `[2, 7, 11]`
- `Goal7` (libero_goal 7): `[2, 7, 11]`
- `Goal8` (libero_goal 8): `[2, 7, 11]`
- `Goal9` (libero_goal 9): `[2, 7, 11]`
- `Long0` (libero_10 0): `[2, 7, 11]`
- `Long1` (libero_10 1): `[2, 7, 11]`
- `Long2` (libero_10 2): `[2, 7, 11]`
- `Long3` (libero_10 3): `[2, 7, 11]`
- `Long4` (libero_10 4): `[2, 7, 11]`
- `Long5` (libero_10 5): `[2, 7, 11]`
- `Long6` (libero_10 6): `[2, 7, 11]`
- `Long7` (libero_10 7): `[2, 7, 11]`
- `Long8` (libero_10 8): `[2, 7, 11]`
- `Long9` (libero_10 9): `[2, 7, 11]`

JSON: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/drop6_extra3/greedy_ppl_extra3_suite/extra3.json`


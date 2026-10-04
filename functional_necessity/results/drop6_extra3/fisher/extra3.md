# v044 Drop-6 + extra-3 (fisher)

Checkpoint: `lerobot/pi05_libero_finetuned_v044`. Universal skip `U=[1, 13, 14, 15, 16, 17]`. Candidates `[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`. Per-task calib 16×8, seed 42.

- Unique extra-3 triples: **6**
- Match global Fisher `{2,3,10}`: **0/40**
- Symmetric difference ≥1 vs `{2,3,10}`: **40/40**
- extra-3 contains L0: **0/40**
- extra-3 contains L7: **3/40**
- Closed-loop gate: **True**

| Triple | Count |
| --- | ---: |
| `[3, 4, 12]` | 31 |
| `[3, 4, 7]` | 3 |
| `[4, 10, 12]` | 3 |
| `[3, 10, 12]` | 1 |
| `[4, 9, 12]` | 1 |
| `[3, 4, 9]` | 1 |

## Per-task extra-3

- `Spa0` (libero_spatial 0): `[3, 4, 12]`
- `Spa1` (libero_spatial 1): `[3, 4, 12]`
- `Spa2` (libero_spatial 2): `[3, 4, 12]`
- `Spa3` (libero_spatial 3): `[3, 4, 12]`
- `Spa4` (libero_spatial 4): `[3, 4, 12]`
- `Spa5` (libero_spatial 5): `[3, 4, 7]`
- `Spa6` (libero_spatial 6): `[3, 4, 12]`
- `Spa7` (libero_spatial 7): `[3, 4, 7]`
- `Spa8` (libero_spatial 8): `[3, 4, 12]`
- `Spa9` (libero_spatial 9): `[3, 4, 12]`
- `Obj0` (libero_object 0): `[4, 10, 12]`
- `Obj1` (libero_object 1): `[3, 4, 12]`
- `Obj2` (libero_object 2): `[3, 4, 12]`
- `Obj3` (libero_object 3): `[3, 10, 12]`
- `Obj4` (libero_object 4): `[3, 4, 12]`
- `Obj5` (libero_object 5): `[3, 4, 12]`
- `Obj6` (libero_object 6): `[3, 4, 12]`
- `Obj7` (libero_object 7): `[3, 4, 12]`
- `Obj8` (libero_object 8): `[3, 4, 12]`
- `Obj9` (libero_object 9): `[3, 4, 12]`
- `Goal0` (libero_goal 0): `[4, 9, 12]`
- `Goal1` (libero_goal 1): `[3, 4, 12]`
- `Goal2` (libero_goal 2): `[4, 10, 12]`
- `Goal3` (libero_goal 3): `[3, 4, 12]`
- `Goal4` (libero_goal 4): `[3, 4, 9]`
- `Goal5` (libero_goal 5): `[3, 4, 12]`
- `Goal6` (libero_goal 6): `[3, 4, 12]`
- `Goal7` (libero_goal 7): `[3, 4, 7]`
- `Goal8` (libero_goal 8): `[3, 4, 12]`
- `Goal9` (libero_goal 9): `[3, 4, 12]`
- `Long0` (libero_10 0): `[3, 4, 12]`
- `Long1` (libero_10 1): `[3, 4, 12]`
- `Long2` (libero_10 2): `[3, 4, 12]`
- `Long3` (libero_10 3): `[3, 4, 12]`
- `Long4` (libero_10 4): `[3, 4, 12]`
- `Long5` (libero_10 5): `[3, 4, 12]`
- `Long6` (libero_10 6): `[3, 4, 12]`
- `Long7` (libero_10 7): `[3, 4, 12]`
- `Long8` (libero_10 8): `[4, 10, 12]`
- `Long9` (libero_10 9): `[3, 4, 12]`

JSON: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/drop6_extra3/fisher/extra3.json`


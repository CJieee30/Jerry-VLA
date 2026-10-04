# Suite×skill core-2 held-out matched/swap Δloss (Step 2)

Protocol: skip(U=[1, 13, 14, 15, 16, 17]), frozen ξ captured on held-out 32×8, episode-disjoint from seed42/43 dumps. Val seeds [44, 45]. GO if matched Δloss + 0.01 ≤ swapped on **both** val seeds. No closed-loop, no extra-3 search.

- Turn GO: **False**
- Place GO: **False**

## Turn  `{2,7}` vs `{2,9}`

| Seed | Goal matched `{2,7}` | Goal swap `{2,9}` | adv | Long matched `{2,9}` | Long swap `{2,7}` | adv | diagonal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 44 | 0.2164 | 0.1979 | -0.0185 | 0.1651 | 0.1901 | 0.0250 | **False** |
| 45 | 0.2258 | 0.2088 | -0.0170 | 0.1483 | 0.1678 | 0.0194 | **False** |

## Place  Object `{2,8}` vs Spatial/Long `{2,7}`

| Seed | Obj `{2,8}` | Obj swap `{2,7}` | adv | Spa `{2,7}` | Spa swap `{2,8}` | adv | Long `{2,7}` | Long swap `{2,8}` | adv | diagonal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 44 | 0.1487 | 0.1504 | 0.0017 | 0.2055 | 0.2146 | 0.0092 | 0.1587 | 0.1857 | 0.0270 | **False** |
| 45 | 0.1352 | 0.1404 | 0.0051 | 0.2210 | 0.2362 | 0.0152 | 0.1755 | 0.1956 | 0.0200 | **False** |

## Per-cell masks

| Cell | seed | U | L2 | `{2,7}` | `{2,8}`/`{2,9}` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Goal/turn` | 44 | 0.0000 | 0.0970 | 0.2164 | 0.1979 |
| `Long/turn` | 44 | 0.0000 | 0.0810 | 0.1901 | 0.1651 |
| `Object/place` | 44 | 0.0000 | 0.0813 | 0.1504 | 0.1487 |
| `Spatial/place` | 44 | 0.0000 | 0.1029 | 0.2055 | 0.2146 |
| `Long/place` | 44 | 0.0000 | 0.0744 | 0.1587 | 0.1857 |
| `Goal/turn` | 45 | 0.0000 | 0.0985 | 0.2258 | 0.2088 |
| `Long/turn` | 45 | 0.0000 | 0.0720 | 0.1678 | 0.1483 |
| `Object/place` | 45 | 0.0000 | 0.0717 | 0.1404 | 0.1352 |
| `Spatial/place` | 45 | 0.0000 | 0.1151 | 0.2210 | 0.2362 |
| `Long/place` | 45 | 0.0000 | 0.0738 | 0.1755 | 0.1956 |

## Verdict

- Turn: **STOP**. No stable matched>swapped preference at the pre-registered 0.01 gap.
- Place: **STOP**. Object L8 is not a held-out place-core, or Spatial/Long do not prefer `{2,7}` over `{2,8}`.
- Do not run extra-3 closed-loop or Drop-8 from this file.

JSON: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/drop6_extra3/suite_skill_core2_heldout/compare_heldout.json`


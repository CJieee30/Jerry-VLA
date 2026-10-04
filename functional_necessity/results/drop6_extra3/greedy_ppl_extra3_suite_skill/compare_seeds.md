# Suite×skill extra-3: seed 42 vs seed 43

- Global greedy control: `[2, 7, 11]`
- Goal-pick excludes Goal5 (push mislabeled as pick).
- This is calibration-only; no closed-loop.

## `pick`

- Unique extra-3 seed42: **3** [[2, 7, 9], [2, 7, 11], [2, 10, 11]]
- Unique extra-3 seed43: **4** [[2, 4, 9], [2, 7, 10], [2, 9, 10], [2, 10, 11]]
- Cross-suite split both seeds: **True**
- Suites that differ from global on both seeds: `['Spatial', 'Object']`

| Suite | seed42 | seed43 | same across seeds | both ≠ `{2,7,11}` |
| --- | --- | --- | --- | --- |
| `Spatial` | `[2, 7, 9]` | `[2, 4, 9]` | False | True |
| `Object` | `[2, 10, 11]` | `[2, 9, 10]` | False | True |
| `Goal` | `[2, 7, 11]` | `[2, 10, 11]` | False | False |
| `Long` | `[2, 7, 11]` | `[2, 7, 10]` | False | False |

| Pair | seed42 same | seed43 same | stable different |
| --- | --- | --- | --- |
| `Spatial` vs `Object` | False | False | **True** |
| `Spatial` vs `Goal` | False | False | **True** |
| `Spatial` vs `Long` | False | False | **True** |
| `Object` vs `Goal` | False | False | **True** |
| `Object` vs `Long` | False | False | **True** |
| `Goal` vs `Long` | True | False | **False** |

## `place`

- Unique extra-3 seed42: **4** [[2, 4, 11], [2, 5, 8], [2, 7, 8], [2, 7, 10]]
- Unique extra-3 seed43: **2** [[2, 7, 11], [2, 8, 11]]
- Cross-suite split both seeds: **True**
- Suites that differ from global on both seeds: `['Object']`

| Suite | seed42 | seed43 | same across seeds | both ≠ `{2,7,11}` |
| --- | --- | --- | --- | --- |
| `Spatial` | `[2, 7, 8]` | `[2, 7, 11]` | False | False |
| `Object` | `[2, 5, 8]` | `[2, 8, 11]` | False | True |
| `Goal` | `[2, 4, 11]` | `[2, 7, 11]` | False | False |
| `Long` | `[2, 7, 10]` | `[2, 7, 11]` | False | False |

| Pair | seed42 same | seed43 same | stable different |
| --- | --- | --- | --- |
| `Spatial` vs `Object` | False | False | **True** |
| `Spatial` vs `Goal` | False | True | **False** |
| `Spatial` vs `Long` | False | True | **False** |
| `Object` vs `Goal` | False | False | **True** |
| `Object` vs `Long` | False | False | **True** |
| `Goal` vs `Long` | False | True | **False** |

## `turn`

- Unique extra-3 seed42: **2** [[2, 7, 9], [2, 7, 11]]
- Unique extra-3 seed43: **2** [[2, 7, 10], [2, 9, 11]]
- Cross-suite split both seeds: **True**
- Suites that differ from global on both seeds: `['Long']`

| Suite | seed42 | seed43 | same across seeds | both ≠ `{2,7,11}` |
| --- | --- | --- | --- | --- |
| `Goal` | `[2, 7, 11]` | `[2, 7, 10]` | False | False |
| `Long` | `[2, 7, 9]` | `[2, 9, 11]` | False | True |

| Pair | seed42 same | seed43 same | stable different |
| --- | --- | --- | --- |
| `Goal` vs `Long` | False | False | **True** |

## Readout (pre-registered)

- Stable Object-pick ≠ other-suite pick, with Object using L8 not L7: suite/scene, not verb.
- Stable Goal-turn ≠ Long-turn: same as the already-known turn transfer failure.
- All suites collapse to `{2,7,11}`: verb pooling was not hiding a suite×skill split.

JSON: `/home/guorj/Skill_Efficient_VLA/functional_necessity/results/drop6_extra3/greedy_ppl_extra3_suite_skill/compare_seeds.json`


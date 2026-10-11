# Task × Layer extra Drop-1 heatmap (Drop-9 recovered π0.5)

Cell = Δ successes vs recovered baseline (out of 10). Source: `results/eval_drop1/l{0,1,2,3,4,5,6,8,9}` and `results/eval_drop3/drop9_recovered`. Seed=7.

- Recovered baseline: **95.00%** (380/400)
- Global extra L2: **87.00%** (348/400)
- Task-oracle extra k=1 (leaky): **94.75%** (379/400)
- Tasks that leave L2: **14/40**  hist={2: 26, 1: 7, 5: 3, 9: 1, 6: 1, 4: 1, 3: 1}

Oracle mapping is leaky (selected on the same 10 trials). Use `map_oracle_k1.json` only as an upper-bound / wiring test. Fair E4 must pick extra_drop from calib loss, not this SR.

| Task | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L8 | L9 | extra | vs L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spa0 | -9 | +0 | +1 | +1 | +1 | +1 | +1 | -1 | -1 | L2 | +0 |
| Spa1 | -10 | +0 | +0 | -5 | -2 | -2 | -2 | -8 | -4 | L2 | +0 |
| Spa2 | -10 | +0 | +0 | +0 | +0 | -1 | +0 | +0 | +0 | L2 | +0 |
| Spa3 | -10 | +0 | +0 | -1 | +0 | -1 | -1 | -2 | -2 | L2 | +0 |
| Spa4 | -10 | -2 | +0 | -10 | -8 | -6 | -5 | -4 | -5 | L2 | +0 |
| Spa5 | -8 | +1 | -4 | +0 | +1 | -5 | -3 | -7 | -4 | L1 | +5 |
| Spa6 | -10 | +0 | +0 | +0 | -1 | +0 | -2 | -1 | -3 | L2 | +0 |
| Spa7 | -10 | -1 | +0 | -5 | -7 | -1 | -2 | -2 | -1 | L2 | +0 |
| Spa8 | -9 | +1 | +1 | -1 | -3 | -2 | -4 | -3 | +0 | L2 | +0 |
| Spa9 | -10 | -5 | -1 | -7 | -4 | -3 | -4 | -8 | -10 | L2 | +0 |
| Obj0 | -10 | +0 | +0 | +0 | +0 | -1 | -1 | -2 | -2 | L2 | +0 |
| Obj1 | -10 | +0 | -1 | -4 | -8 | +0 | -1 | -4 | -1 | L1 | +1 |
| Obj2 | -10 | -4 | -1 | -8 | -4 | +0 | -1 | +0 | +0 | L5 | +1 |
| Obj3 | -10 | +0 | +0 | -4 | -1 | +0 | +0 | -1 | -1 | L2 | +0 |
| Obj4 | -10 | +0 | +0 | -1 | +0 | +0 | -3 | -2 | -3 | L2 | +0 |
| Obj5 | -10 | -1 | -1 | +0 | -2 | +0 | +0 | -1 | +0 | L5 | +1 |
| Obj6 | -10 | -1 | +0 | -1 | -3 | -1 | -2 | -1 | -1 | L2 | +0 |
| Obj7 | -10 | +0 | +0 | +0 | +0 | -1 | +0 | +0 | -2 | L2 | +0 |
| Obj8 | -10 | +0 | +0 | +0 | -2 | +0 | +0 | +0 | +0 | L2 | +0 |
| Obj9 | -10 | +0 | -1 | -1 | -2 | +0 | +0 | -3 | -1 | L1 | +1 |
| Goal0 | -10 | -4 | -9 | -5 | -5 | -2 | -8 | -5 | -1 | L9 | +8 |
| Goal1 | -10 | +0 | +0 | -1 | -2 | +0 | +0 | -4 | -2 | L2 | +0 |
| Goal2 | -10 | -2 | -1 | -7 | -3 | -3 | -2 | -9 | -7 | L2 | +0 |
| Goal3 | -6 | -1 | +3 | -4 | -6 | +1 | -6 | -3 | -4 | L2 | +0 |
| Goal4 | -10 | +0 | +0 | -10 | -8 | -1 | +0 | -2 | +0 | L2 | +0 |
| Goal5 | -9 | -7 | +1 | -9 | -9 | -3 | -5 | -4 | -5 | L2 | +0 |
| Goal6 | -10 | +0 | -1 | -2 | -10 | +0 | -1 | -3 | -3 | L1 | +1 |
| Goal7 | -10 | -1 | -4 | -10 | -3 | -10 | -10 | -10 | -7 | L1 | +3 |
| Goal8 | -10 | +0 | +0 | +0 | +0 | +0 | +0 | +0 | +0 | L2 | +0 |
| Goal9 | -6 | +2 | +1 | -4 | +1 | -1 | -1 | -4 | -6 | L1 | +1 |
| Long0 | -10 | +0 | +0 | -1 | -7 | +0 | -5 | -7 | -1 | L2 | +0 |
| Long1 | -10 | +0 | +0 | +0 | -1 | -1 | -1 | -3 | -5 | L2 | +0 |
| Long2 | -10 | +0 | -1 | -4 | +0 | -1 | -4 | -4 | -4 | L1 | +1 |
| Long3 | -10 | -3 | -1 | -3 | -1 | +0 | -6 | -1 | -4 | L5 | +1 |
| Long4 | -10 | -2 | -2 | -7 | -5 | -6 | -5 | -8 | -9 | L2 | +0 |
| Long5 | -10 | -4 | -1 | -8 | -4 | -3 | +0 | -8 | -2 | L6 | +1 |
| Long6 | -9 | -5 | -5 | -7 | +0 | -5 | -8 | -7 | -8 | L4 | +5 |
| Long7 | -10 | -7 | -1 | +0 | -4 | -3 | -3 | -2 | -6 | L3 | +1 |
| Long8 | -5 | -5 | -1 | -3 | -1 | -5 | -4 | -5 | -2 | L2 | +0 |
| Long9 | -9 | -9 | -3 | -7 | -6 | -9 | -8 | -9 | -4 | L2 | +0 |

## Switches vs global L2

| id | extra | oracle | L2 | base | task |
| --- | ---: | ---: | ---: | ---: | --- |
| Spa5 | 1 | 9 | 4 | 8 | pick up the black bowl on the ramekin and place it on the plate |
| Obj1 | 1 | 10 | 9 | 10 | pick up the cream cheese and place it in the basket |
| Obj2 | 5 | 10 | 9 | 10 | pick up the salad dressing and place it in the basket |
| Obj5 | 5 | 10 | 9 | 10 | pick up the tomato sauce and place it in the basket |
| Obj9 | 1 | 10 | 9 | 10 | pick up the orange juice and place it in the basket |
| Goal0 | 9 | 9 | 1 | 10 | open the middle drawer of the cabinet |
| Goal6 | 1 | 10 | 9 | 10 | put the cream cheese in the bowl |
| Goal7 | 1 | 9 | 6 | 10 | turn on the stove |
| Goal9 | 1 | 8 | 7 | 6 | put the wine bottle on the rack |
| Long2 | 1 | 10 | 9 | 10 | turn on the stove and put the moka pot on it |
| Long3 | 5 | 10 | 9 | 10 | put the black bowl in the bottom drawer of the cabinet and close it |
| Long5 | 6 | 10 | 9 | 10 | pick up the book and place it in the back compartment of the caddy |
| Long6 | 4 | 9 | 4 | 9 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| Long7 | 3 | 10 | 9 | 10 | put both the alphabet soup and the cream cheese box in the basket |

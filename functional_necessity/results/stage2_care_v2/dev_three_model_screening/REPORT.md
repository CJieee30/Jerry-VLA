# 三模型开发筛查（统一 LIBERO 口径 + 原生指标）

## pi05
转换核对: `{"model": "pi05", "n": 8, "native_rmse_recompute_max_delta": 7.450580680190555e-09, "libero_grip_demo_frac_positive": 0.4751796390762986, "unified_norm": "libero meta mean/std on physical actions"}`
| 策略 | G=1 | cos(mean) | rmse(mean) | arm P95(mean) | grip NFE1 | 尾部捕获 |
|---|---|---|---|---|---|---|
| random | 19 | 0.9974 | 0.0513 | 0.0959 | 0.020833333333333332 | 15/47 |
| alpha | 34 | 0.9969 | 0.0640 | 0.1176 | 0.029166666666666667 | 25/47 |
| g_only | 55 | 0.9966 | 0.0556 | 0.0954 | 0.041666666666666664 | 21/47 |
| alpha_event | 55 | 0.9961 | 0.0637 | 0.1113 | 0.041666666666666664 | 25/47 |

## smolvla
转换核对: `{"model": "smolvla", "n": 8, "native_rmse_recompute_max_delta": 7.450580666312767e-09, "libero_grip_demo_frac_positive": 0.4751796390762986, "unified_norm": "libero meta mean/std on physical actions"}`
| 策略 | G=1 | cos(mean) | rmse(mean) | arm P95(mean) | grip NFE1 | 尾部捕获 |
|---|---|---|---|---|---|---|
| random | 19 | 0.9963 | 0.0654 | 0.1128 | 0.03333333333333333 | 19/51 |
| alpha | 34 | 0.9959 | 0.0755 | 0.1301 | 0.04583333333333333 | 26/51 |
| g_only | 55 | 0.9954 | 0.0686 | 0.1145 | 0.05416666666666667 | 27/51 |
| alpha_event | 55 | 0.9953 | 0.0752 | 0.1261 | 0.058333333333333334 | 32/51 |

## molmoact2
转换核对: `{"model": "molmoact2", "n": 8, "native_rmse_recompute_max_delta": 1.862645211681002e-09, "libero_grip_demo_frac_positive": 0.4751796390762986, "unified_norm": "libero meta mean/std on physical actions"}`
| 策略 | G=1 | cos(mean) | rmse(mean) | arm P95(mean) | grip NFE1 | 尾部捕获 |
|---|---|---|---|---|---|---|
| random | 19 | 0.9992 | 0.0279 | 0.0524 | 0.0 | 18/40 |
| alpha | 34 | 0.9992 | 0.0334 | 0.0649 | 0.004166666666666667 | 22/40 |
| g_only | 55 | 0.9991 | 0.0284 | 0.0522 | 0.004166666666666667 | 12/40 |
| alpha_event | 55 | 0.9991 | 0.0324 | 0.0608 | 0.004166666666666667 | 22/40 |


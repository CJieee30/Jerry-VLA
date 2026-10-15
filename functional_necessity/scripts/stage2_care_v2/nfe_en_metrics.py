"""Unified EN (E_N) metrics and three independent NFE selection rules — shared by reports and protocol."""

from __future__ import annotations

from typing import Any

import numpy as np

from dev_metrics_common import compare_chunk, libero_mean_std, native_to_physical, to_libero_normalized
from protocol import ACTION_DIM, HORIZON

REF_NFE = 10
ABS_MAIN = 0.085
ABS_SENS = (0.08, 0.09)
ABS_IMP_MAX = 0.03
REL_IMP_MAX = 0.375
NOISE_IDS = (0, 1)


def rmse7_libero_batch(model: str, pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """RMSE7 in LIBERO norm; pred/ref share leading dims, last two are (H, ACTION_DIM)."""
    mean, std = libero_mean_std()
    lp = to_libero_normalized(native_to_physical(model, pred), mean, std)
    lr = to_libero_normalized(native_to_physical(model, ref), mean, std)
    return np.sqrt(((lp - lr) ** 2).mean(axis=(-2, -1)))


def per_comparison_rmse7(model: str, actions: dict[str, np.ndarray], nfe: int, ref_nfe: int = REF_NFE) -> np.ndarray:
    """Shape (n_obs, n_noise) RMSE vs reference for each obs×noise."""
    a = actions[f"nfe{nfe}"].astype(np.float64)
    r = actions[f"nfe{ref_nfe}"].astype(np.float64)
    if a.shape != r.shape:
        raise ValueError(f"shape mismatch nfe{nfe} {a.shape} vs nfe{ref_nfe} {r.shape}")
    n_obs, n_noise = a.shape[0], a.shape[1]
    out = np.empty((n_obs, n_noise), dtype=np.float64)
    for si in range(n_obs):
        for nid in range(n_noise):
            out[si, nid] = rmse7_libero_batch(model, a[si, nid], r[si, nid])
    return out


def E_N(model: str, actions: dict[str, np.ndarray], nfe: int, idx: np.ndarray | None = None, ref_nfe: int = REF_NFE) -> float:
    mat = per_comparison_rmse7(model, actions, nfe, ref_nfe)
    if idx is not None:
        mat = mat[idx]
    return float(mat.mean())


def E_table(model: str, actions: dict[str, np.ndarray], candidates: list[int], idx: np.ndarray | None = None) -> dict[int, float]:
    return {n: E_N(model, actions, n, idx) for n in candidates}


def _adjacent_delta(E: dict[int, float], n: int, candidates: list[int]) -> tuple[str, float | None]:
    nxt = n + 1
    if nxt not in candidates:
        return "NA", None
    d = E[n] - E[nxt]
    return "ok", float(d)


def passes_abs(E: dict[int, float], n: int, tau: float = ABS_MAIN) -> tuple[bool, str]:
    return E[n] <= tau, f"E_{n}={E[n]:.4f}"


def passes_abs_improve(E: dict[int, float], n: int, candidates: list[int]) -> tuple[bool, str]:
    tag, d = _adjacent_delta(E, n, candidates)
    if tag == "NA":
        return False, "adjacent_NA"
    assert d is not None
    if d < 0:
        return False, f"non_monotonic delta={d:.4f}"
    if d > ABS_IMP_MAX:
        return False, f"delta={d:.4f}>{ABS_IMP_MAX}"
    return True, f"delta={d:.4f}"


def passes_rel_improve(E: dict[int, float], n: int, candidates: list[int]) -> tuple[bool, str]:
    tag, d = _adjacent_delta(E, n, candidates)
    if tag == "NA":
        return False, "adjacent_NA"
    assert d is not None
    if d < 0:
        return False, f"non_monotonic delta={d:.4f}"
    if E[n] == 0.0 and d == 0.0:
        return True, "rel=0"
    if E[n] == 0.0:
        return False, "E_n=0 with positive delta"
    rel = d / E[n]
    if rel > REL_IMP_MAX:
        return False, f"rel={rel:.4f}>{REL_IMP_MAX}"
    return True, f"rel={rel:.4f}"


def pick_lowest(rule: str, E: dict[int, float], candidates: list[int], fallback: int = REF_NFE) -> dict[str, Any]:
    notes: list[str] = []
    for n in sorted(candidates):
        if rule == "abs":
            ok, why = passes_abs(E, n, ABS_MAIN)
        elif rule == "abs_improve":
            ok, why = passes_abs_improve(E, n, candidates)
        elif rule == "rel_improve":
            ok, why = passes_rel_improve(E, n, candidates)
        elif rule == "joint":
            ok1, w1 = passes_abs(E, n, ABS_MAIN)
            ok2, w2 = passes_abs_improve(E, n, candidates)
            ok3, w3 = passes_rel_improve(E, n, candidates)
            ok, why = ok1 and ok2 and ok3, f"{w1};{w2};{w3}"
        else:
            raise ValueError(rule)
        if not ok:
            notes.append(f"n={n}:{why}")
            continue
        return {"recommended_nfe": n, "fallback": False, "rule": rule, "E": {str(k): E[k] for k in candidates}, "notes": notes}
    return {"recommended_nfe": fallback, "fallback": True, "rule": rule, "E": {str(k): E[k] for k in candidates}, "notes": notes + ["fallback_reference"]}


def recommend_three_rules(
    model: str, actions: dict[str, np.ndarray], candidates: list[int], idx: np.ndarray, fallback: int = REF_NFE
) -> dict[str, Any]:
    E = E_table(model, actions, candidates, idx)
    return {
        "E_on_selection": E,
        "abs_0.085": pick_lowest("abs", E, candidates, fallback),
        "abs_improve": pick_lowest("abs_improve", E, candidates, fallback),
        "rel_improve": pick_lowest("rel_improve", E, candidates, fallback),
        "joint_supplementary": pick_lowest("joint", E, candidates, fallback),
        "sensitivity_abs": {str(t): pick_lowest_abs_tau(E, candidates, t, fallback) for t in ABS_SENS},
    }


def pick_lowest_abs_tau(E: dict[int, float], candidates: list[int], tau: float, fallback: int) -> dict[str, Any]:
    for n in sorted(candidates):
        if E[n] <= tau:
            return {"recommended_nfe": n, "fallback": False, "tau": tau}
    return {"recommended_nfe": fallback, "fallback": True, "tau": tau}


def holdout_status_for_screening_nfe(
    model: str,
    actions: dict[str, np.ndarray],
    n_star: int,
    candidates: list[int],
    idx: np.ndarray,
    fallback: int,
    rule: str,
) -> dict[str, Any]:
    """Evaluate whether calibration-chosen n_star passes rule on holdout (no re-pick)."""
    E = E_table(model, actions, candidates, idx)
    if n_star == fallback:
        return {"status": "reference", "nfe": n_star, "E": E, "detail": "fallback NFE10"}
    if n_star not in candidates:
        return {"status": "NA", "nfe": n_star, "E": E, "detail": "nfe not in candidates"}
    if rule == "abs":
        ok, detail = passes_abs(E, n_star)
    elif rule == "abs_improve":
        ok, detail = passes_abs_improve(E, n_star, candidates)
    elif rule == "rel_improve":
        ok, detail = passes_rel_improve(E, n_star, candidates)
    else:
        raise ValueError(rule)
    return {
        "status": "pass" if ok else "fail",
        "nfe": n_star,
        "E_n": float(E[n_star]),
        "E": {str(k): float(v) for k, v in E.items()},
        "detail": detail,
    }


def validate_action_arrays(actions: dict[str, np.ndarray], nfes: list[int], n_expected: int, n_noise: int = 2) -> None:
    for nfe in nfes:
        key = f"nfe{nfe}"
        if key not in actions:
            raise ValueError(f"missing {key}")
        arr = actions[key]
        if arr.shape[0] != n_expected:
            raise ValueError(f"{key} n_obs {arr.shape[0]} != {n_expected}")
        if arr.shape[1] != n_noise:
            raise ValueError(f"{key} n_noise {arr.shape[1]} != {n_noise}")
        if arr.shape[2] != HORIZON or arr.shape[3] != ACTION_DIM:
            raise ValueError(f"{key} window {arr.shape[2:]} != ({HORIZON},{ACTION_DIM})")
        if not np.isfinite(arr).all():
            raise ValueError(f"{key} non-finite values")

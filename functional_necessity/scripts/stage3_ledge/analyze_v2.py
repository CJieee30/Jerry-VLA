#!/usr/bin/env python3
"""Offline LEDGE v2 analysis: r(D), sigma_self, validation against closed-loop SR, k_safe per order."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import torch

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
RES = FN / "results" / "stage3_ledge"
AW = FN / "results" / "stage2_action_weight"
sys.path.insert(0, str(FN / "scripts" / "stage3_ledge"))
from run_ledge_v2 import CURVES  # noqa: E402

PROTOCOL = json.loads((RES / "v2_protocol.json").read_text())
C = float(PROTOCOL["c"])
MARGIN = 5.0
N_BOOT = 2000


def load_store(stack: str) -> dict | None:
    shards = sorted((RES / "v2").glob(f"{stack}_shard*.pt"))
    if not shards:
        return None
    merged = None
    for path in shards:
        s = torch.load(path, map_location="cpu", weights_only=False)
        if merged is None:
            merged = {"full": s["full"], "sets": dict(s["sets"]), "n_seeds": s["n_seeds"]}
            continue
        for a, b in zip(merged["full"], s["full"]):
            if float((a - b).abs().max()) > 1e-3:
                raise SystemExit(f"{stack}: shards disagree on the full model")
        merged["sets"].update(s["sets"])
    return merged


def per_frame(a: torch.Tensor, b: torch.Tensor, dims=None) -> torch.Tensor:
    if dims is not None:
        a, b = a[..., dims], b[..., dims]
    return (a - b).flatten(1).norm(dim=-1)


def wmean(v: torch.Tensor, w: torch.Tensor) -> float:
    return float((v * w).sum() / w.sum())


def boot_ci(v: torch.Tensor, w: torch.Tensor) -> tuple[float, float]:
    gen = torch.Generator().manual_seed(0)
    idx = torch.randint(0, v.numel(), (N_BOOT, v.numel()), generator=gen)
    vals = (v[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return float(vals.quantile(0.025)), float(vals.quantile(0.975))


def auc(fail: list[float], ok: list[float]) -> float | None:
    if not fail or not ok:
        return None
    wins = sum((f > o) + 0.5 * (f == o) for f in fail for o in ok)
    return wins / (len(fail) * len(ok))


def analyze(stack: str, targets: dict) -> dict | None:
    store = load_store(stack)
    if store is None:
        return None
    alpha = torch.load(AW / f"weights_{stack}.pt", map_location="cpu", weights_only=False)["weights7"].float().reshape(-1)
    full = store["full"]
    if full[0].shape[0] != alpha.numel():
        raise SystemExit(f"{stack}: {full[0].shape[0]} frames vs {alpha.numel()} weights")
    dim = full[0].shape[-1]
    pairs = list(itertools.combinations(range(len(full)), 2))
    self_frames = torch.stack([per_frame(full[i], full[j]) for i, j in pairs]).mean(0)
    sigma = wmean(self_frames, alpha)
    sigma_ci = boot_ci(self_frames, alpha)
    sigma_arm = wmean(torch.stack([per_frame(full[i], full[j], slice(0, 6)) for i, j in pairs]).mean(0), alpha)

    def drift(key: str) -> dict:
        acts = store["sets"][key]
        frames = torch.stack([per_frame(a, f) for a, f in zip(acts, full)]).mean(0)
        arm = torch.stack([per_frame(a, f, slice(0, 6)) for a, f in zip(acts, full)]).mean(0)
        per_seed = [wmean(per_frame(a, f), alpha) for a, f in zip(acts, full)]
        r = wmean(frames, alpha)
        lo, hi = boot_ci(frames, alpha)
        return {"r": r, "r_ci": [lo, hi], "r_per_seed": per_seed, "r_arm": wmean(arm, alpha), "ratio": r / sigma}

    drifts = {key: drift(key) for key in store["sets"]}
    anchor = float(targets["anchor"])
    rows = []
    for row in targets["sets"]:
        key = ",".join(str(i) for i in row["drop"])
        if not row["drop"] or key not in drifts:
            continue
        d = drifts[key]
        rows.append(
            {
                **row,
                **d,
                "has_l0": 0 in row["drop"],
                "pass": row["sr"] >= anchor - MARGIN,
                "pred_safe": d["r"] <= C * sigma,
            }
        )
    rows.sort(key=lambda x: x["r"])
    valid = [x for x in rows if not x["has_l0"]]
    fail = [x["r"] for x in valid if not x["pass"]]
    ok = [x["r"] for x in valid if x["pass"]]
    rule_acc = sum(x["pass"] == x["pred_safe"] for x in valid) / max(1, len(valid))
    sep = None
    if ok and fail and max(ok) < min(fail):
        sep = [max(ok) / sigma, min(fail) / sigma]

    curves = {}
    sr_by_key = {}
    for row in targets["sets"]:
        sr_by_key.setdefault(",".join(str(i) for i in row["drop"]), []).append(row["sr"])
    for name in CURVES[stack]:
        order = targets["orders"][name]
        rk, known = [], {}
        for k in range(1, len(order) + 1):
            key = ",".join(str(i) for i in sorted(order[:k]))
            if key not in drifts:
                break
            rk.append(drifts[key]["r"])
            if key in sr_by_key:
                known[k] = sum(sr_by_key[key]) / len(sr_by_key[key])
        cross = next((k for k, r in enumerate(rk, start=1) if r > C * sigma), None)
        pred = (cross - 1) if cross is not None else None
        obs_safe, obs_fail = None, None
        for k in sorted(known):
            if known[k] >= anchor - MARGIN and obs_fail is None:
                obs_safe = k
            elif known[k] < anchor - MARGIN and obs_fail is None:
                obs_fail = k
        curves[name] = {
            "order": order,
            "r": rk,
            "ratio": [r / sigma for r in rk],
            "k_safe_pred": pred,
            "k_cap": len(rk),
            "known_sr": known,
            "obs_last_pass": obs_safe,
            "obs_first_fail": obs_fail,
        }
    return {
        "stack": stack,
        "n_seeds": store["n_seeds"],
        "n_frames": int(alpha.numel()),
        "action_dim": int(dim),
        "sigma_self": sigma,
        "sigma_self_ci": list(sigma_ci),
        "sigma_self_arm": sigma_arm,
        "anchor": anchor,
        "pass_threshold": anchor - MARGIN,
        "auc_no_l0": auc(fail, ok),
        "n_pass": len(ok),
        "n_fail": len(fail),
        "rule_accuracy_no_l0": rule_acc,
        "perfect_separation_ratio_range": sep,
        "sets": rows,
        "curves": curves,
        "drifts": drifts,
    }


def main() -> None:
    targets = json.loads((RES / "v2_targets.json").read_text())
    out = {}
    for stack in ("pi05", "molmoact2", "smolvla"):
        res = analyze(stack, targets[stack])
        if res is None:
            print(f"== {stack}: no data yet")
            continue
        out[stack] = res
        print(
            f"== {stack}: sigma_self={res['sigma_self']:.4f} CI={res['sigma_self_ci'][0]:.4f}-{res['sigma_self_ci'][1]:.4f}"
            f" | AUC={res['auc_no_l0']} (pass {res['n_pass']}, fail {res['n_fail']})"
            f" | rule acc={res['rule_accuracy_no_l0']:.2f} | separable ratio range={res['perfect_separation_ratio_range']}"
        )
        for x in res["sets"]:
            mark = "ok " if x["pass"] else "BAD"
            pred = "safe" if x["pred_safe"] else "unsafe"
            l0 = " L0" if x["has_l0"] else ""
            print(
                f"   {mark} sr={x['sr']:6.2f} r={x['r']:.4f} [{x['r_ci'][0]:.4f},{x['r_ci'][1]:.4f}]"
                f" r/sigma={x['ratio']:6.2f} {pred:6s} {x['tag']}{l0}"
            )
        for name, c in res["curves"].items():
            ratio = " ".join(f"{v:.2f}" for v in c["ratio"])
            print(
                f"   curve {name}: pred k_safe={c['k_safe_pred']} (cap {c['k_cap']}) | obs last pass={c['obs_last_pass']}"
                f" first fail={c['obs_first_fail']} | known={c['known_sr']}"
            )
            print(f"      r/sigma: {ratio}")
    (RES / "v2_analysis.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

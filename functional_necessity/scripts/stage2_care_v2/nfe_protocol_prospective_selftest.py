#!/usr/bin/env python3
"""CPU self-test for prospective protocol import / validate guards (no GPU probes)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FN_ROOT = HERE.parents[1]
PY = sys.executable
PROTO = HERE / "nfe_protocol_v1.py"
CFG_TEMPLATE = HERE / "nfe_protocol_config.prospective_prefreeze.json"


def run_stage(cfg: Path, stage: str, *extra: str) -> None:
    cmd = [PY, str(PROTO), "--config", str(cfg), stage, *extra]
    subprocess.run(cmd, check=True, cwd=FN_ROOT)


def synth_actions(n: int, nfes: list[int], e_by_nfe: dict[int, float]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 0.02, (n, 2, 10, 7))
    out = {}
    for nfe in nfes:
        scale = e_by_nfe.get(nfe, 0.05)
        out[f"nfe{nfe}"] = ref + rng.normal(0, scale, ref.shape)
    out["nfe10"] = ref.copy()
    return out


def write_probe_dir(tmp: Path, name: str, records: list[dict], acts: dict[str, np.ndarray]) -> Path:
    d = tmp / name
    d.mkdir(parents=True)
    np.savez_compressed(d / "actions.npz", shared_index=np.arange(len(records)), **acts)
    (d / "DONE").write_text("ok\n")
    import pandas as pd

    rows = []
    for i, r in enumerate(records):
        for nid in (0, 1):
            rows.append({"shared_index": i, "global_index": r["global_index"], "noise_id": nid, "noise_sha256": f"n{nid}"})
    pd.DataFrame(rows).to_csv(d / "frames.csv", index=False)
    return d


def main() -> None:
    failures: list[str] = []
    tmp_base = Path(tempfile.mkdtemp(prefix="nfe_prospective_selftest_"))
    cfg_path = tmp_base / "config.json"
    cfg = json.loads(CFG_TEMPLATE.read_text())
    cfg["run_name"] = f"prospective_selftest_cpu_{int(time.time())}"
    cfg["mode"] = "prospective"
    cfg["models"] = {"pi05": cfg["models"]["pi05"]}
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    try:
        run_stage(cfg_path, "freeze")
        run_stage(cfg_path, "split")
        run_stage(cfg_path, "select")
        run_stage(cfg_path, "probe-plan")

        run_dir = FN_ROOT / "results" / "stage2_care_v2" / "nfe_protocol" / cfg["run_name"]
        req = __import__("torch").load(run_dir / "03_probe" / "probe_request.pt", map_location="cpu", weights_only=False)
        records = req["records"]
        n = len(records)
        nfes = [1, 2, 3, 10]
        cal_acts = synth_actions(n, nfes, {1: 0.01, 2: 0.04, 3: 0.03})

        probe_dir = write_probe_dir(tmp_base, "cal_probe", records, cal_acts)
        run_stage(cfg_path, "probe-import", "--model", "pi05", "--probe-dir", str(probe_dir))
        run_stage(cfg_path, "recommend")
        run_stage(cfg_path, "validation-plan")

        vreq = __import__("torch").load(run_dir / "05_validation" / "probe_request.pt", map_location="cpu", weights_only=False)
        vrecords = vreq["records"]
        val_acts = synth_actions(len(vrecords), nfes, {1: 0.35, 2: 0.04, 3: 0.03})
        vdir = write_probe_dir(tmp_base, "val_probe", vrecords, val_acts)
        run_stage(cfg_path, "validation-import", "--model", "pi05", "--probe-dir", str(vdir))

        cl = {
            "success_rate_unit": "fraction",
            "results": {
                "pi05": {
                    "10": {
                        "suites": {"spatial": 0.97, "object": 0.97, "goal": 0.97, "long": 0.97},
                        "meta": {"n_episodes": 10, "seed": 1000, "eval_protocol": "libero"},
                    },
                    "1": {
                        "suites": {"spatial": 0.50, "object": 0.97, "goal": 0.97, "long": 0.97},
                        "meta": {"n_episodes": 10, "seed": 1000, "eval_protocol": "libero"},
                    },
                    "2": {
                        "suites": {"spatial": 0.96, "object": 0.96, "goal": 0.96, "long": 0.96},
                        "meta": {"n_episodes": 10, "seed": 1000, "eval_protocol": "libero"},
                    },
                }
            },
        }
        cl_path = tmp_base / "closed_loop.json"
        cl_path.write_text(json.dumps(cl))
        run_stage(cfg_path, "closed-loop-import", "--closed-loop-json", str(cl_path))
        run_stage(cfg_path, "validate")
        run_stage(cfg_path, "verify-ledger")

        val = json.loads((run_dir / "06_validate" / "validation.json").read_text())
        hs = val["models"]["pi05"]["strategies"]["random"]["holdout_rule_status"]["abs_0.085"]
        if hs["status"] != "fail":
            failures.append(f"expected abs holdout fail for E1~0.20, got {hs}")
        hr = val["models"]["pi05"]["strategies"]["random"]["holdout_rule_status"]
        cl_by = val["models"]["pi05"]["strategies"]["random"]["closed_loop_by_rule"]
        if hr["abs_0.085"]["nfe"] != 1 or hr["abs_improve"]["nfe"] != 2:
            failures.append(f"expected screening abs→1 abs_improve→2, got {hr}")
        if len({cl_by[k]["screening_nfe"] for k in cl_by}) < 2:
            failures.append("closed_loop_by_rule should attach multiple distinct NFE")
        cl1 = val["models"]["pi05"]["strategies"]["random"]["closed_loop_by_rule"]["abs_0.085"]
        if cl1["status"] != "fail" or cl1["suite_delta_pp"]["spatial"] != -47.0:
            failures.append(f"expected closed-loop spatial -47pp, got {cl1}")

        # duplicate import rejected
        try:
            run_stage(cfg_path, "probe-import", "--model", "pi05", "--probe-dir", str(probe_dir))
            failures.append("duplicate probe-import should fail")
        except subprocess.CalledProcessError:
            pass

        # bad shape rejected
        bad = synth_actions(n, nfes, {1: 0.1})
        bad["nfe1"] = bad["nfe1"][:, :, :5, :]
        bad_dir = write_probe_dir(tmp_base, "bad_shape", records, bad)
        cfg2 = tmp_base / "config2.json"
        c2 = json.loads(cfg_path.read_text())
        c2["run_name"] = f"prospective_selftest_bad_{int(time.time())}"
        cfg2.write_text(json.dumps(c2, indent=2))
        run_stage(cfg2, "freeze")
        run_stage(cfg2, "split")
        run_stage(cfg2, "select")
        try:
            run_stage(cfg2, "probe-import", "--model", "pi05", "--probe-dir", str(bad_dir))
            failures.append("bad action shape should fail import")
        except subprocess.CalledProcessError:
            pass

        # closed-loop missing unit
        cl_bad = {"results": cl["results"]}
        cl_bad_path = tmp_base / "cl_bad.json"
        cl_bad_path.write_text(json.dumps(cl_bad))
        cfg3 = tmp_base / "config3.json"
        c3 = json.loads(cfg_path.read_text())
        c3["run_name"] = f"prospective_selftest_clbad_{int(time.time())}"
        cfg3.write_text(json.dumps(c3, indent=2))
        run_stage(cfg3, "freeze")
        try:
            run_stage(cfg3, "closed-loop-import", "--closed-loop-json", str(cl_bad_path))
            failures.append("missing success_rate_unit should fail")
        except subprocess.CalledProcessError:
            pass

        # molmo official check
        molmo_script = HERE / "molmo_postprocessor_check.py"
        r = subprocess.run([PY, str(molmo_script)], cwd=FN_ROOT)
        if r.returncode != 0:
            failures.append("molmo_postprocessor_check failed")

        # dev table unchanged row count
        bud = FN_ROOT / "results" / "stage2_care_v2" / "dev_nfe_rules_cpu_v2" / "three_rule_budget_table.csv"
        if sum(1 for _ in bud.open()) != 379:
            failures.append(f"three_rule_budget_table.csv line count != 378+header ({bud})")

    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)

    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print(" -", f)
        raise SystemExit(1)
    print("SELFTEST OK")


if __name__ == "__main__":
    main()

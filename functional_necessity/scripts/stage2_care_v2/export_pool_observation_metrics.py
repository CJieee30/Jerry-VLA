#!/usr/bin/env python3
"""Export full 480-pool per-(obs,noise,nfe) metrics in common LIBERO space."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dev_metrics_common import (
    MODEL_DIRS,
    POOL_CSV,
    SCREENING_DIR,
    build_conversion_check,
    export_pool_rows,
    write_conversion_manifest,
)
from protocol import OUT_DIR


def main() -> None:
    SCREENING_DIR.mkdir(parents=True, exist_ok=True)
    records = torch.load(OUT_DIR / "shared.pt", map_location="cpu", weights_only=False)["records"]
    all_rows = []
    status = {}
    for model, mdir in MODEL_DIRS.items():
        npz = mdir / "actions.npz"
        if not npz.is_file():
            status[model] = "missing_probe"
            continue
        z = np.load(npz)
        rows = export_pool_rows(model, z, records)
        all_rows.extend(rows)
        status[model] = {"n_rows": len(rows), "nfes": sorted({r["nfe"] for r in rows})}
    df = pd.DataFrame(all_rows)
    df.to_csv(POOL_CSV, index=False)
    manifest = write_conversion_manifest()
    check = build_conversion_check()
    meta = {
        "n_observations_pool": len(records),
        "n_rows": len(df),
        "models": status,
        "manifest": "conversion_manifest.json",
        "check": "conversion_check.json",
    }
    (SCREENING_DIR / "pool_export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {POOL_CSV} ({len(df)} rows)")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

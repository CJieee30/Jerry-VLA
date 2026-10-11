#!/usr/bin/env python3
"""Build a local MolmoAct2 dataset root with only the CARE v2 parquet files.

Hard-links files already in the Hub snapshot; downloads the rest from hf-mirror
in one parallel aria2c job. Does not write into the Hub cache.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol import COMPOSE_DIR, MOLMO_REPO, MOLMO_ROOT, OUT_DIR, hub_snapshot, molmo_parquet_rels

MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


def _copy_meta(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    meta_dst = dest / "meta"
    if not meta_dst.exists():
        shutil.copytree(src / "meta", meta_dst)
    (dest / "data").mkdir(parents=True, exist_ok=True)


def _rels() -> list[str]:
    import torch

    records = []
    for path in (
        OUT_DIR / "shared.pt",
        COMPOSE_DIR / "shared_disc.pt",
        COMPOSE_DIR / "shared_smooth.pt",
    ):
        if not path.is_file():
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        records.extend(blob["records"])
    files: set[str] = set()
    if records:
        max_file = max(int(r["file_index"]) for r in records) + 1
        files.update(molmo_parquet_rels(records, max_file_index=max_file))
    files_txt = OUT_DIR / "molmo_files.txt"
    if files_txt.is_file():
        files.update(ln.strip() for ln in files_txt.read_text().splitlines() if ln.strip())
    if not files:
        raise SystemExit("run dump_shared.py first")
    return sorted(files)


def _is_regular_file(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    except OSError:
        return False


def _clear_dest(dest: Path) -> None:
    if dest.is_symlink() or dest.exists():
        dest.unlink()


def _link_or_copy(src_file: Path, dest: Path) -> None:
    real = src_file.resolve()
    try:
        os.link(real, dest)
    except FileExistsError:
        _clear_dest(dest)
        os.link(real, dest)
    except OSError as exc:
        if getattr(exc, "errno", None) == 18:  # EXDEV
            shutil.copy2(real, dest)
        else:
            raise


def main() -> None:
    rels = _rels()
    (OUT_DIR / "molmo_files.txt").write_text("\n".join(rels) + "\n")
    src = hub_snapshot(MOLMO_REPO)
    _copy_meta(src, MOLMO_ROOT)
    counts = {"have": 0, "link": 0, "download": 0}
    missing: list[str] = []
    for rel in rels:
        dest = MOLMO_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        src_f = src / rel
        if _is_regular_file(dest):
            counts["have"] += 1
            continue
        _clear_dest(dest)
        if src_f.exists() and src_f.resolve().is_file() and src_f.resolve().stat().st_size > 0:
            _link_or_copy(src_f, dest)
            counts["link"] += 1
            continue
        missing.append(rel)
        counts["download"] += 1
    print("plan", counts, "missing", len(missing), flush=True)
    if missing:
        url_file = OUT_DIR / "molmo_urls.txt"
        lines = []
        for rel in missing:
            dest = MOLMO_ROOT / rel
            url = f"{MIRROR}/datasets/{MOLMO_REPO}/resolve/main/{rel}"
            lines.append(url)
            lines.append(f"  dir={dest.parent}")
            lines.append(f"  out={dest.name}")
        url_file.write_text("\n".join(lines) + "\n")
        cmd = [
            "aria2c",
            "-j",
            "8",
            "-x",
            "8",
            "-s",
            "8",
            "-c",
            "-i",
            str(url_file),
        ]
        print("aria2c", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        bad = [rel for rel in missing if not _is_regular_file(MOLMO_ROOT / rel)]
        if bad:
            raise SystemExit(f"download failed: {bad[:5]}")
    bad = [rel for rel in rels if not _is_regular_file(MOLMO_ROOT / rel)]
    if bad:
        raise SystemExit(f"molmo parquet not a regular file: {bad[:5]}")
    marker = MOLMO_ROOT / "READY"
    marker.write_text(json.dumps(counts, indent=2) + "\n")
    print("molmo root ready", counts, flush=True)


if __name__ == "__main__":
    main()

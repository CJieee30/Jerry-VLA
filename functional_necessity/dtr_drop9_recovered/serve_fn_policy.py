#!/usr/bin/env python3
"""Serve DTR Drop-9 recovered π0.5 with an explicit LLM drop list (Stage-1 ∪ extra)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

OPENPI = Path("/home/guorj/VLADrop/models/pi0.5/openpi")
sys.path[:0] = [
    str(OPENPI / "src"),
    str(OPENPI / "packages" / "openpi-client" / "src"),
    str(OPENPI / "scripts"),
]

# Serving only needs OpenPI transforms/norm stats, not the LeRobot dataset loader.
import types

for name in (
    "lerobot",
    "lerobot.common",
    "lerobot.common.datasets",
    "lerobot.common.datasets.lerobot_dataset",
):
    sys.modules.setdefault(name, types.ModuleType(name))


def parse_ints(raw: str) -> list[int]:
    text = str(raw).strip().strip("[]")
    if not text:
        return []
    return [int(part.strip()) for part in text.replace(" ", ",").split(",") if part.strip()]


def ensure_paligemma_tokenizer() -> None:
    """OpenPI PaligemmaTokenizer fetches gs://big_vision/... via gcsfs.

    This machine has no gcsfs and GCS python access is unnecessary: the same
    file is available over HTTPS. Seed ~/.cache/openpi so maybe_download
    short-circuits.
    """
    cache = Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"
    if cache.is_file() and cache.stat().st_size > 1_000_000:
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
    logging.info("Seeding Paligemma tokenizer via HTTPS -> %s", cache)
    import urllib.request

    tmp = cache.with_suffix(".model.partial")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(cache)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_libero_dropped")
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--llm-drop",
        default=None,
        help="Comma-separated base LLM drop (Stage-1). Default 7,10-17 when --task-extra-map is set.",
    )
    parser.add_argument(
        "--task-extra-map",
        type=Path,
        default=None,
        help="JSON map extra_drop=f(task). Serve Stage-1 drop and add remaining layers per prompt.",
    )
    args = parser.parse_args()

    from backend import STAGE1_DROP

    if args.llm_drop is None:
        if args.task_extra_map is None:
            raise SystemExit("need --llm-drop or --task-extra-map")
        drop = list(STAGE1_DROP)
    else:
        drop = parse_ints(args.llm_drop)
    if not drop:
        raise SystemExit("empty --llm-drop")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_paligemma_tokenizer()
    logging.info(
        "Serving %s on port %s with llm_drop=%s task_extra_map=%s",
        args.dir,
        args.port,
        drop,
        args.task_extra_map,
    )

    import socket

    from serve_policy_batch_drop import Args, create_policy
    from openpi.serving import websocket_policy_server

    policy = create_policy(
        Args(
            config=args.config,
            dir=str(args.dir),
            port=args.port,
            llm_drop_attn_list=drop,
            llm_drop_mlp_list=drop,
        )
    )
    if args.task_extra_map is not None:
        from task_extra_map import wrap_policy_with_task_extra

        policy = wrap_policy_with_task_extra(policy, args.task_extra_map, stage1_drop=drop)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

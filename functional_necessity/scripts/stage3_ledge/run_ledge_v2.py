#!/usr/bin/env python3
"""LEDGE v2: executed actions of the full model and of every target drop set under several frozen noise seeds.

Only raw actions are stored; r(D), sigma_self and the decisions are computed offline by analyze_v2.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
HERE = FN / "scripts" / "stage3_ledge"
OUT = FN / "results" / "stage3_ledge" / "v2"
TARGETS = FN / "results" / "stage3_ledge" / "v2_targets.json"

_spec = importlib.util.spec_from_file_location("ledge_v1", HERE / "run_ledge.py")
v1 = importlib.util.module_from_spec(_spec)
sys.modules["ledge_v1"] = v1
_spec.loader.exec_module(v1)

EXEC_STEPS = v1.EXEC_STEPS
DENOISE_STEPS = v1.DENOISE_STEPS

CURVES = {
    "pi05": ("fisher_care", "fisher_uniform", "gateprobe_uniform", "gateprobe_care", "igia_uniform", "taylor_care"),
    "molmoact2": (
        "taylor_care",
        "taylor_care_arm6",
        "fisher_uniform",
        "gateprobe_uniform",
        "gateprobe_care",
        "gateprobe_care_arm6",
        "igia_uniform",
    ),
    "smolvla": ("taylor_care", "fisher_uniform", "fisher_care", "gateprobe_uniform", "gateprobe_care", "igia_uniform"),
}
K_CAP = {"pi05": 12, "molmoact2": 24, "smolvla": 15}
SEEDS = {"pi05": 2, "molmoact2": 4, "smolvla": 4}
EXTRA = {"molmoact2": ([1, 2, 3, 4, 6, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],)}


def target_sets(stack: str) -> list[list[int]]:
    t = json.loads(TARGETS.read_text())[stack]
    sets = {tuple(r["drop"]) for r in t["sets"] if r["drop"]}
    for name in CURVES[stack]:
        order = t["orders"][name]
        for k in range(1, min(K_CAP[stack], len(order)) + 1):
            sets.add(tuple(sorted(order[:k])))
    for extra in EXTRA.get(stack, ()):
        sets.add(tuple(sorted(extra)))
    return [list(s) for s in sorted(sets, key=lambda s: (len(s), s))]


class Pi05:
    def __init__(self, device):
        common = v1._load("pi05_common", FN / "scripts" / "pi05_libero" / "common.py")
        layer_skip = v1._load("pi05_layer_skip", FN / "scripts" / "pi05_libero" / "layer_skip.py")
        self.skip = layer_skip.skip_llm_layers
        self.move = common.move_to_device
        self.device = device
        self.calib = common.load_calib_data(FN / "results" / "calib" / "pi05_v044_libero_64x8_seed42.pt")
        policy = common.load_policy("lerobot/pi05_libero_finetuned_v044", device)
        policy.config.n_action_steps = EXEC_STEPS
        policy.config.num_inference_steps = DENOISE_STEPS
        policy.config.compile_model = False
        policy.eval()
        self.policy = policy

    def noises(self, seed_index: int):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(1000 * (seed_index + 1))
        cfg = self.policy.config
        return [
            torch.randn(int(b["action"].shape[0]), int(cfg.chunk_size), int(cfg.max_action_dim), generator=gen)
            for b in self.calib
        ]

    def actions(self, drop, noises) -> torch.Tensor:
        ctx = self.skip(self.policy, drop) if drop else nullcontext()
        out = []
        with torch.no_grad(), ctx:
            for batch, noise in zip(self.calib, noises):
                batch = self.move(batch, self.device)
                self.policy.reset()
                pred = self.policy.predict_action_chunk(batch, noise=noise.to(self.device))
                out.append(pred[:, :EXEC_STEPS].detach().float().cpu())
        return torch.cat(out)


class Smol:
    def __init__(self, device):
        layer_skip = v1._load("smol_layer_skip", FN / "scripts" / "smolvla_libero" / "layer_skip.py")
        load_mod = v1._load("smol_load_policy", FN / "scripts" / "smolvla_libero" / "load_policy.py")
        score = v1._load("smol_score", FN / "scripts" / "smolvla_libero" / "score.py")
        self.skip = layer_skip.skip_vlm_layers
        self.device = device
        policy = load_mod.load_policy(str(device))
        policy.config.n_action_steps = EXEC_STEPS
        if hasattr(policy.config, "num_steps"):
            policy.config.num_steps = DENOISE_STEPS
        self.policy = policy
        self.samples = score.build_calib(policy)

    def noises(self, seed_index: int):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(1000 * (seed_index + 1))
        cfg = self.policy.config
        return [torch.randn(1, int(cfg.chunk_size), int(cfg.max_action_dim), generator=gen) for _ in self.samples]

    def actions(self, drop, noises) -> torch.Tensor:
        out = []
        with torch.no_grad(), self.skip(drop):
            for sample, noise in zip(self.samples, noises):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in sample["batch"].items()}
                self.policy.reset()
                pred = self.policy.predict_action_chunk(batch, noise=noise.to(self.device))
                out.append(pred[:, :EXEC_STEPS].detach().float().cpu())
        return torch.cat(out)


class Molmo:
    def __init__(self, device):
        layer_skip = v1._load("molmo_layer_skip", FN / "scripts" / "molmoact2_libero" / "layer_skip.py")
        load_mod = v1._load("molmo_load_policy", FN / "scripts" / "molmoact2_libero" / "load_policy.py")
        score = v1._load("molmo_score", FN / "scripts" / "molmoact2_libero" / "score.py")
        self.skip = layer_skip.skip_vlm_layers
        self.device = device
        policy = load_mod.load_policy(str(device))
        policy.config.gradient_checkpointing = False
        policy.config.enable_inference_cuda_graph = False
        policy.config.n_action_steps = EXEC_STEPS
        if hasattr(policy.config, "num_inference_steps"):
            policy.config.num_inference_steps = DENOISE_STEPS
        policy.eval()
        self.policy = policy
        self.samples = score.build_calib(policy)

    def noises(self, seed_index: int):
        return [10_000 * (seed_index + 1) + i for i in range(len(self.samples))]

    def actions(self, drop, noises) -> torch.Tensor:
        out = []
        with torch.no_grad(), self.skip(drop):
            for sample, seed in zip(self.samples, noises):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in sample["batch"].items()}
                gen = torch.Generator(device=self.device)
                gen.manual_seed(int(seed))
                if hasattr(self.policy, "reset"):
                    self.policy.reset()
                pred = self.policy.predict_action_chunk(
                    batch, generator=gen, num_steps=DENOISE_STEPS, inference_action_mode="continuous"
                )
                out.append(pred[:, :EXEC_STEPS].detach().float().cpu())
        torch.cuda.empty_cache()
        return torch.cat(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stack", choices=("pi05", "smolvla", "molmoact2"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    sets = target_sets(args.stack)
    mine = [s for i, s in enumerate(sets) if i % args.nshards == args.shard]
    print(f"{args.stack}: {len(sets)} target sets, shard {args.shard}/{args.nshards} has {len(mine)}", flush=True)
    if args.dry_run:
        return

    device = torch.device(args.device)
    runner = {"pi05": Pi05, "smolvla": Smol, "molmoact2": Molmo}[args.stack](device)
    n_seeds = SEEDS[args.stack]
    noises = [runner.noises(j) for j in range(n_seeds)]

    out = OUT / f"{args.stack}_shard{args.shard}of{args.nshards}.pt"
    OUT.mkdir(parents=True, exist_ok=True)
    if out.exists():
        store = torch.load(out, map_location="cpu", weights_only=False)
    else:
        t0 = time.time()
        full = [runner.actions([], noises[j]) for j in range(n_seeds)]
        again = runner.actions([], noises[0])
        err = float((again - full[0]).abs().max())
        if err > 1e-3:
            raise SystemExit(f"{args.stack} replay not deterministic: {err}")
        store = {"stack": args.stack, "n_seeds": n_seeds, "full": full, "replay_err": err, "sets": {}}
        torch.save(store, out)
        print(f"full model x{n_seeds} ok, replay err {err:.2e}, shape {tuple(full[0].shape)}, {time.time() - t0:.0f}s", flush=True)

    for drop in mine:
        key = ",".join(str(i) for i in drop)
        if key in store["sets"]:
            continue
        t0 = time.time()
        acts = [runner.actions(drop, noises[j]) for j in range(n_seeds)]
        store["sets"][key] = acts
        torch.save(store, out)
        r = sum(float((a - f).flatten(1).norm(dim=-1).mean()) for a, f in zip(acts, store["full"])) / n_seeds
        print(f"[{len(store['sets'])}/{len(mine)}] drop={drop} r_unweighted={r:.4f} {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

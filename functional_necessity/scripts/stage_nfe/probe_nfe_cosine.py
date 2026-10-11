#!/usr/bin/env python3
"""π0.5 action-expert NFE cosine probe (baseline, no layer drop).

Records three things on the same frozen noise:
1. ODE states x_t along a 10-step Euler path (t=1 → 0), including x_1=noise.
2. Velocities v_t at those 10 steps (what denoise_step actually returns).
3. Final actions of independent NFE ∈ {1,2,3,5,10} schedules (the real operator).

Cosine is on the first 10 executed steps × 7 unpadded LIBERO dims.
This is not EfficientVLA Fig 1(c) (MLP/attention feature cache).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
sys.path.insert(0, str(FN / "scripts" / "pi05_libero"))

from common import (  # noqa: E402
    DEFAULT_CALIB_FILE,
    DEFAULT_CHECKPOINT,
    load_calib_data,
    load_policy,
    move_to_device,
    slice_batch,
)
from measure_module_efficiency import prepare_inputs  # noqa: E402

OUT = FN / "results" / "stage_nfe" / "nfe_cosine_pi05.json"
N_FRAMES = 64
EXEC = 10
DIMS = 7
NFE_PATH = 10
NFES = (1, 2, 3, 5, 10)
SEED = 1000


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x[..., :EXEC, :DIMS].reshape(x.shape[0], -1)


def _cos_mat(tokens: torch.Tensor) -> torch.Tensor:
    """tokens: [K, B, D] → mean over B of K×K cosine."""
    k, b, _ = tokens.shape
    x = F.normalize(tokens.reshape(k * b, -1), dim=-1).reshape(k, b, -1)
    out = torch.zeros(k, k, device=tokens.device)
    for i in range(k):
        for j in range(k):
            out[i, j] = (x[i] * x[j]).sum(dim=-1).mean()
    return out


def euler_record(denoise_fn, noise: torch.Tensor, num_steps: int):
    bsize = noise.shape[0]
    device = noise.device
    dt = -1.0 / num_steps
    x_t = noise.clone()
    xs = [x_t.clone()]
    vs = []
    hs = []
    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
        v_t, hidden = denoise_fn(x_t, time_tensor)
        vs.append(v_t.clone())
        hs.append(hidden.clone())
        x_t = x_t + dt * v_t
        xs.append(x_t.clone())
    return torch.stack(xs, 0), torch.stack(vs, 0), torch.stack(hs, 0)


def main() -> None:
    device = torch.device("cuda")
    policy = load_policy(DEFAULT_CHECKPOINT, device)
    policy.config.n_action_steps = 10
    policy.config.compile_model = False
    policy.config.num_inference_steps = 10
    policy.eval()
    model = policy.model
    calib = load_calib_data(DEFAULT_CALIB_FILE)
    horizon = int(model.config.chunk_size)
    adim = int(model.config.max_action_dim)

    x_path = []
    v_path = []
    h_path = []
    finals = {n: [] for n in NFES}

    n_done = 0
    with torch.no_grad():
        for batch_i, raw in enumerate(calib):
            if n_done >= N_FRAMES:
                break
            take = min(raw[next(iter(raw))].shape[0], N_FRAMES - n_done)
            for row in range(take):
                batch = move_to_device(slice_batch(raw, row, row + 1), device)
                images, img_masks, tokens, masks, states, state_masks = prepare_inputs(policy, batch)
                gen = torch.Generator(device="cpu")
                gen.manual_seed(SEED + n_done)
                noise = torch.randn(1, horizon, adim, generator=gen).to(device)

                prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
                    images, img_masks, tokens, masks, states, state_masks
                )
                from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d

                attn = prepare_attention_masks_4d(make_att_2d_masks(prefix_pad, prefix_att))
                pos = torch.cumsum(prefix_pad, dim=1) - 1
                model._paligemma_language_model_core().config._attn_implementation = "eager"
                _, past = model.paligemma_with_expert.forward(
                    attention_mask=attn,
                    position_ids=pos,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                )

                def denoise_fn(x_cur, time_tensor, *, _past=past, _pad=prefix_pad):
                    from lerobot.policies.pi05.modeling_pi05 import clone_past_key_values

                    suffix_embs, suffix_pad, suffix_att, adarms = model.embed_suffix(x_cur, time_tensor)
                    suffix_len = suffix_pad.shape[1]
                    bsize = _pad.shape[0]
                    prefix_len = _pad.shape[1]
                    prefix_pad_2d = _pad[:, None, :].expand(bsize, suffix_len, prefix_len)
                    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d

                    suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
                    full_att = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
                    prefix_offsets = torch.sum(_pad, dim=-1)[:, None]
                    position_ids = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1
                    full_att_4d = prepare_attention_masks_4d(full_att)
                    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
                    outputs_embeds, _ = model.paligemma_with_expert.forward(
                        attention_mask=full_att_4d,
                        position_ids=position_ids,
                        past_key_values=clone_past_key_values(_past),
                        inputs_embeds=[None, suffix_embs],
                        use_cache=False,
                        adarms_cond=[None, adarms],
                    )
                    suffix_out = outputs_embeds[1][:, -horizon:].to(dtype=torch.float32)
                    v = model.action_out_proj(suffix_out)
                    return v, suffix_out

                xs, vs, hs = euler_record(denoise_fn, noise, NFE_PATH)
                x_path.append(_flat(xs.squeeze(1)))  # [11, D]
                v_path.append(_flat(vs.squeeze(1)))  # [10, D]
                h_path.append(hs.squeeze(1).reshape(NFE_PATH, -1))  # [10, H]

                for nfe in NFES:
                    policy.config.num_inference_steps = nfe
                    policy.reset()
                    act = policy.predict_action_chunk(batch, noise=noise.clone(), num_steps=nfe)
                    if act.ndim == 3:
                        chunk = act
                    else:
                        chunk = act
                    # predict_action_chunk returns [B, n_action_steps, action_dim]
                    finals[nfe].append(_flat(chunk))
                n_done += 1
                if n_done % 8 == 0:
                    print(f"frame {n_done}/{N_FRAMES}", flush=True)

    x_tok = torch.stack(x_path, 1).to(device)  # [11, B, D]
    v_tok = torch.stack(v_path, 1).to(device)
    h_tok = torch.stack(h_path, 1).to(device)
    fin = {n: torch.cat(finals[n], 0).to(device) for n in NFES}  # [B, D]

    cos_x = _cos_mat(x_tok).cpu()
    cos_v = _cos_mat(v_tok).cpu()
    cos_h = _cos_mat(h_tok).cpu()
    a10 = fin[10]
    vs_nfe10 = F.normalize(a10, dim=-1)
    nfe_vs_10 = {}
    nfe_unit = {}
    for nfe in NFES:
        a = F.normalize(fin[nfe], dim=-1)
        nfe_unit[nfe] = a
        nfe_vs_10[str(nfe)] = float((a * vs_nfe10).sum(dim=-1).mean())
    n_nfe = len(NFES)
    cos_act = torch.zeros(n_nfe, n_nfe)
    for i, ni in enumerate(NFES):
        for j, nj in enumerate(NFES):
            cos_act[i, j] = (nfe_unit[ni] * nfe_unit[nj]).sum(dim=-1).mean()
    cos_act = cos_act.cpu()

    # path state after k Euler steps vs final x_10 (same 10-step schedule, early x)
    x_vs_final = []
    x_final = F.normalize(x_tok[-1], dim=-1)
    for k in range(x_tok.shape[0]):
        xk = F.normalize(x_tok[k], dim=-1)
        x_vs_final.append(float((xk * x_final).sum(dim=-1).mean()))

    payload = {
        "checkpoint": DEFAULT_CHECKPOINT,
        "n_frames": n_done,
        "exec_steps": EXEC,
        "action_dims": DIMS,
        "seed": SEED,
        "note": (
            "x_t is the Euler ODE state (noise at k=0, action at k=10). "
            "v_t is denoise_step output (velocity), not an action chunk. "
            "independent NFE uses dt=-1/NFE, which is not early-stopping the 10-step path."
        ),
        "cosine_x_path": cos_x.tolist(),
        "cosine_v_path": cos_v.tolist(),
        "cosine_expert_suffix": cos_h.tolist(),
        "x_k_vs_x10": x_vs_final,
        "independent_nfe_vs_nfe10": nfe_vs_10,
        "nfe_order": list(NFES),
        "cosine_action_nfe": cos_act.tolist(),
        "mean_offdiag_v": float(
            (cos_v.sum() - torch.diagonal(cos_v).sum()) / (cos_v.numel() - cos_v.shape[0])
        ),
        "min_v": float(cos_v.min()),
        "mean_offdiag_action": float(
            (cos_act.sum() - torch.diagonal(cos_act).sum()) / (cos_act.numel() - cos_act.shape[0])
        ),
        "min_action": float(cos_act.min()),
        "mean_adjacent_x": float(torch.diagonal(cos_x, offset=1).mean()),
        "mean_adjacent_v": float(torch.diagonal(cos_v, offset=1).mean()),
        "mean_adjacent_h": float(torch.diagonal(cos_h, offset=1).mean()),
        "v0_vs_v9": float(cos_v[0, -1]),
        "x1_vs_x10": float(cos_x[1, -1]),
        "noise_vs_x10": float(cos_x[0, -1]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", OUT, flush=True)
    print("adjacent x", payload["mean_adjacent_x"], "v", payload["mean_adjacent_v"], "h", payload["mean_adjacent_h"])
    print("nfe vs 10", nfe_vs_10)
    print("action pairwise min/mean_offdiag", payload["min_action"], payload["mean_offdiag_action"])
    print("x_k vs x10", [round(v, 4) for v in x_vs_final])


if __name__ == "__main__":
    main()

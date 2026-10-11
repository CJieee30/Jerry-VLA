#!/usr/bin/env python3
"""Compare K/V-only skip against the previous full-attention keep-KV skip.

One calibration frame. Prints max abs action and cache diffs. Exits 1 if the
dropped layer still calls Q, or if actions differ by more than the tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

FN = Path("/home/guorj/Skill_Efficient_VLA/functional_necessity")
MOLMO_DROP = [1, 2, 3, 4, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
SMOL_DROP = [12, 13, 14, 15]
ATOL = 1e-4


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _cache_pairs(past):
    layers = getattr(past, "layers", None)
    if layers is None:
        return []
    out = []
    for i, layer in enumerate(layers):
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
        if key is None:
            key = getattr(layer, "key_cache", None)
            value = getattr(layer, "value_cache", None)
        if torch.is_tensor(key):
            out.append((i, key, value))
    return out


def check_smol(device: torch.device) -> None:
    sys.path.insert(0, str(FN / "scripts" / "smolvla_libero"))
    import layer_skip
    from layer_skip import skip_vlm_layers
    from load_policy import CALIB_PATH, load_policy
    from score import _ensure_policy_batch

    policy = load_policy(str(device))
    policy.config.compile_model = False
    policy.config.n_action_steps = 10
    policy.config.num_steps = 1
    policy.eval()
    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    sample = blob["samples"][0]
    batch = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in _ensure_policy_batch(sample["batch"]).items()
    }
    gen_noise = torch.Generator(device="cpu")
    gen_noise.manual_seed(1000)
    noise = torch.randn(
        batch["observation.state"].shape[0],
        int(policy.config.chunk_size),
        int(policy.config.max_action_dim),
        generator=gen_noise,
    ).to(device)

    new_forward = layer_skip._forward_keep_kv
    legacy = _smol_legacy_forward(layer_skip)

    def run(fn):
        layer_skip._forward_keep_kv = fn
        with torch.no_grad(), skip_vlm_layers(SMOL_DROP):
            policy.reset()
            action = policy.predict_action_chunk(batch, noise=noise.clone())
            loss, _ = policy.forward(
                batch, noise=sample["noise"].to(device), time=sample["time"].to(device)
            )
        return action, loss

    q_calls = _count_module_calls(
        [policy.model.vlm_with_expert.get_vlm_model().text_model.layers[i].self_attn.q_proj for i in SMOL_DROP]
    )
    action_new, loss_new = run(new_forward)
    q_new = q_calls()
    action_old, loss_old = run(legacy)
    q_old = q_calls()
    layer_skip._forward_keep_kv = new_forward
    print(
        f"smolvla action max_abs={_max_abs(action_new, action_old):.3e} "
        f"loss {float(loss_old):.6f} -> {float(loss_new):.6f} "
        f"q_proj calls legacy={q_old} kvonly={q_new}"
    )
    if q_new != 0:
        raise SystemExit("SmolVLA dropped layers still called q_proj")
    if _max_abs(action_new, action_old) > ATOL or abs(float(loss_new - loss_old)) > 1e-5:
        raise SystemExit("SmolVLA K/V-only does not match the previous skip")


def _count_module_calls(modules):
    counts = [0]

    def hook(_mod, _inp, _out):
        counts[0] += 1

    handles = [mod.register_forward_hook(hook) for mod in modules]

    def read():
        value = counts[0]
        counts[0] = 0
        return value

    read.handles = handles
    return read


def _smol_legacy_forward(layer_skip):
    """Previous keep-KV: full QKV+attention, then drop VLM o_proj/MLP."""
    from transformers.cache_utils import DynamicCache

    def forward(
        self, attention_mask=None, position_ids=None, past_key_values=None,
        inputs_embeds=None, use_cache=None,
    ):
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)
        for hidden_states in inputs_embeds:
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]
        fill_kv_cache = use_cache and past_key_values is None
        if fill_kv_cache:
            past_key_values = DynamicCache()
        num_layers = self.num_vlm_layers
        head_dim = self.vlm.config.text_config.head_dim
        for layer_idx in range(num_layers):
            if (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            ):
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers, inputs_embeds, layer_idx, position_ids, attention_mask,
                    batch_size, head_dim, use_cache=use_cache, past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers, inputs_embeds, layer_idx, position_ids, attention_mask,
                    batch_size, head_dim, use_cache=use_cache, past_key_values=past_key_values,
                )
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                if hidden_states is None:
                    outputs_embeds.append(None)
                    continue
                if layer is None:
                    outputs_embeds.append(hidden_states)
                    continue
                end = start + hidden_states.shape[1]
                if i == 0 and layer_idx in layer_skip._DROP:
                    outputs_embeds.append(hidden_states)
                    start = end if len(att_outputs) == 1 else 0
                    continue
                if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                out_emb = layer.self_attn.o_proj(att_output[:, start:end])
                out_emb += hidden_states
                after = out_emb.clone()
                out_emb = layer.post_attention_layernorm(out_emb)
                out_emb = layer.mlp(out_emb)
                out_emb += after
                outputs_embeds.append(out_emb)
                start = end if len(att_outputs) == 1 else 0
            inputs_embeds = outputs_embeds
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            outputs_embeds.append(models[i].norm(hidden_states) if hidden_states is not None else None)
        return outputs_embeds, past_key_values

    return forward


def check_molmo(device: torch.device) -> None:
    sys.path.insert(0, str(FN / "scripts" / "molmoact2_libero"))
    import layer_skip
    from layer_skip import skip_vlm_layers
    from load_policy import CALIB_PATH, load_policy
    from lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 import (
        MolmoAct2DecoderLayer,
        MolmoAct2PostNormDecoderLayer,
    )

    policy = load_policy(str(device))
    policy.eval()
    policy.config.compile_model = False
    policy.config.enable_inference_cuda_graph = False
    backbone = policy._backbone()
    if getattr(backbone, "action_cuda_graph_manager", None) is not None:
        backbone.action_cuda_graph_manager = None
    blob = torch.load(CALIB_PATH, map_location="cpu", weights_only=False)
    sample = blob["samples"][0]
    batch = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in sample["batch"].items()
    }
    blocks = backbone.transformer.blocks
    watched = [blocks[i].self_attn.att_proj for i in MOLMO_DROP]
    calls = _count_module_calls(watched)

    def bind(pre, post):
        MolmoAct2DecoderLayer.forward = pre
        MolmoAct2PostNormDecoderLayer.forward = post

    def run(pre, post):
        generator = torch.Generator(device=device)
        generator.manual_seed(1000)
        with torch.no_grad(), skip_vlm_layers(MOLMO_DROP), policy._autocast_context():
            # install() inside the context would put the K/V-only forward back.
            # Bind after that so this call uses the requested implementation.
            bind(pre, post)
            policy.reset()
            action = policy.predict_action_chunk(batch, num_steps=1, generator=generator)
            model_inputs = policy._model_inputs(batch)
            loss, _ = policy._compute_flow_matching_loss_joint_per_layer(
                batch=batch,
                model_inputs=model_inputs,
                timesteps=sample["timesteps"].to(device),
                noise=sample["noise"].to(device),
            )
        return action, loss

    legacy_pre, legacy_post = _molmo_legacy(layer_skip)
    action_old, loss_old = run(legacy_pre, legacy_post)
    q_old = calls()
    action_new, loss_new = run(layer_skip._skip_prenorm, layer_skip._skip_postnorm)
    q_new = calls()
    print(
        f"molmo action max_abs={_max_abs(action_new, action_old):.3e} "
        f"loss {float(loss_old):.6f} -> {float(loss_new):.6f} "
        f"att_proj calls legacy={q_old} kvonly={q_new}"
    )
    if q_new != 0:
        raise SystemExit("Molmo dropped layers still called att_proj")
    # Fused QKV vs K/V-only GEMMs differ by a few bf16 ulps. qwen3 K-norm
    # turns that into ~1e-2 on a few key elements. Actions stay within 1e-2.
    if _max_abs(action_new, action_old) > 1e-2 or abs(float(loss_new - loss_old)) > 1e-2:
        raise SystemExit("Molmo K/V-only drifted past bf16 noise")


def _molmo_legacy(layer_skip):
    def pre(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None,
            past_key_values=None, output_attentions=False, use_cache=False, cache_position=None, **kwargs):
        if int(self.self_attn.layer_idx) not in layer_skip._DROP:
            return layer_skip._ORIGINAL(
                self, hidden_states, position_embeddings, attention_mask, position_ids,
                past_key_values, output_attentions, use_cache, cache_position, **kwargs,
            )
        collect = bool(kwargs.pop("collect_layer_kv_states", False))
        normed = self.attn_norm(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=normed, position_embeddings=position_embeddings, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values, output_attentions=output_attentions,
            use_cache=use_cache, cache_position=cache_position, collect_layer_kv_states=True, **kwargs,
        )
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if collect:
            outputs += (attn_outputs[2], attn_outputs[3])
        return outputs

    def post(self, hidden_states, position_embeddings, attention_mask=None, position_ids=None,
             past_key_values=None, output_attentions=False, use_cache=False, cache_position=None, **kwargs):
        if int(self.self_attn.layer_idx) not in layer_skip._DROP:
            return layer_skip._ORIGINAL_POST(
                self, hidden_states, position_embeddings, attention_mask, position_ids,
                past_key_values, output_attentions, use_cache, cache_position, **kwargs,
            )
        collect = bool(kwargs.pop("collect_layer_kv_states", False))
        attn_outputs = self.self_attn(
            hidden_states=hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values, output_attentions=output_attentions,
            use_cache=use_cache, cache_position=cache_position, collect_layer_kv_states=True, **kwargs,
        )
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)
        if collect:
            outputs += (attn_outputs[2], attn_outputs[3])
        return outputs

    return pre, post


def main() -> None:
    stack = sys.argv[1]
    device = torch.device("cuda")
    if stack == "smolvla":
        check_smol(device)
    elif stack == "molmoact2":
        check_molmo(device)
    else:
        raise SystemExit("usage: check_kvonly_actions.py smolvla|molmoact2")


if __name__ == "__main__":
    main()

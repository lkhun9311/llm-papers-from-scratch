"""References for check.py: small transformers models (Llama, Mixtral, DeepSeek-V2) with their weights copied
into impl, and vLLM's CPU-runnable RoPE and router functions.

Nothing in transformers or vLLM is changed.
"""
from __future__ import annotations

import torch
import transformers
import vllm
from transformers import (DeepseekV2Config, DeepseekV2ForCausalLM, DynamicCache, LlamaConfig, LlamaForCausalLM,
                          MixtralConfig, MixtralForCausalLM)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.deepseek_v2 import modeling_deepseek_v2 as hf_ds
from transformers.models.llama import modeling_llama as hf_llama
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.router import cpu_router
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import DeepseekScalingRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.llama3_rope import Llama3RotaryEmbedding

import impl

DT = torch.float64
TRANSFORMERS_VERSION = "5.22.2"
VLLM_VERSION = ("0.34.0", "g62d4bc3c7")


def check_version() -> dict:
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"transformers {transformers.__version__}, expected {TRANSFORMERS_VERSION}")
    import vllm._version

    found = (vllm.__version__, vllm._version.__commit_id__)
    if found != VLLM_VERSION:
        raise RuntimeError(f"vLLM {found}, expected {VLLM_VERSION}")
    return {"transformers": transformers.__version__, "vllm": f"{found[0]} ({found[1]})"}


# ================================================================================================
# RoPE
# ================================================================================================


def hf_rope_parameters(r: impl.Rope) -> dict:
    p = {"rope_type": r.kind, "rope_theta": r.base}
    if r.kind == "llama3":
        p.update(factor=r.factor, low_freq_factor=r.low_freq_factor, high_freq_factor=r.high_freq_factor,
                 original_max_position_embeddings=r.orig_max)
    elif r.kind == "yarn":
        p.update(factor=r.factor, original_max_position_embeddings=r.orig_max, beta_fast=r.beta_fast,
                 beta_slow=r.beta_slow, mscale=r.mscale, mscale_all_dim=r.mscale_all_dim)
    return p


def hf_inv_freq(r: impl.Rope) -> tuple[torch.Tensor, float]:
    """transformers' RoPE init for a Llama config with head_dim = r.dim (float32, as transformers computes it)."""
    cfg = LlamaConfig(hidden_size=r.dim, num_attention_heads=1, head_dim=r.dim, rope_parameters=hf_rope_parameters(r),
                      max_position_embeddings=max(r.orig_max, 1) * int(max(r.factor, 1)))
    if r.kind == "default":
        return hf_llama.LlamaRotaryEmbedding.compute_default_rope_parameters(cfg)
    return ROPE_INIT_FUNCTIONS[r.kind](cfg)


def hf_apply_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """transformers' Llama apply_rotary_pos_emb with impl's cos and sin. x [B, h, T, d]; cos, sin [T, d/2]."""
    c = torch.cat((cos, cos), dim=-1)[None]
    s = torch.cat((sin, sin), dim=-1)[None]
    q, _ = hf_llama.apply_rotary_pos_emb(x, x, c, s)
    return q


def hf_apply_gptj(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """transformers' DeepSeek-V2 apply_rotary_emb (it computes in float32). x [B, h, T, d]."""
    freqs_cis = torch.complex(cos, sin)[None]
    q, _ = hf_ds.apply_rotary_emb(x, x, freqs_cis)
    return q


def _vllm_rope(r: impl.Rope, max_pos: int):
    neox = r.style == "neox"
    with set_current_vllm_config(VllmConfig()):
        if r.kind == "default":
            return RotaryEmbedding(r.dim, r.dim, max_pos, r.base, neox, DT)
        if r.kind == "llama3":
            return Llama3RotaryEmbedding(r.dim, r.dim, max_pos, r.base, neox, DT, r.factor, r.low_freq_factor,
                                         r.high_freq_factor, r.orig_max)
        if r.kind == "yarn":
            # DeepSeek's model code passes the original context length as max_position_embeddings.
            return DeepseekScalingRotaryEmbedding(r.dim, r.dim, r.orig_max, r.base, neox, r.factor, DT,
                                                  beta_fast=int(r.beta_fast), beta_slow=int(r.beta_slow),
                                                  mscale=r.mscale, mscale_all_dim=r.mscale_all_dim)
    raise ValueError(r.kind)


def vllm_inv_freq(r: impl.Rope) -> tuple[torch.Tensor, float]:
    """vLLM's frequencies and cos/sin factor for the same RoPE."""
    rope = _vllm_rope(r, max(r.orig_max, 64))
    if r.kind == "yarn":
        return rope._compute_inv_freq(r.factor), rope.mscale
    return rope._compute_inv_freq(r.base), 1.0


def vllm_apply(r: impl.Rope, x: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    """vLLM's forward_static with impl's cos/sin cache: x [T, h, d] at `positions`."""
    cache = torch.cat((cos, sin), dim=-1)  # row p holds position p
    T, h, d = x.shape
    q, _ = RotaryEmbedding.forward_static(positions, x.reshape(T, h * d).clone(), x[:, :1].reshape(T, d).clone(), d,
                                          d, cache, r.style == "neox")
    return q.view(T, h, d)


# ================================================================================================
# Routers
# ================================================================================================


def vllm_softmax_topk(logits: torch.Tensor, k: int, renormalize: bool):
    return cpu_router._softmax_topk(logits, k, renormalize)


def vllm_grouped_topk(logits: torch.Tensor, k: int, renormalize: bool, n_group: int, topk_group: int,
                      scaling: float):
    return cpu_router._grouped_topk(logits, k, renormalize, n_group, topk_group, "softmax", scaling, None)


def hf_mixtral_route(weight: torch.Tensor, k: int, x: torch.Tensor):
    cfg = MixtralConfig(hidden_size=weight.shape[1], num_attention_heads=1, num_key_value_heads=1,
                        num_local_experts=weight.shape[0], num_experts_per_tok=k)
    r = hf_mixtral_router_cls()(cfg).to(DT)
    with torch.no_grad():
        r.weight.copy_(weight)
        _, w, ids = r(x)
    return w, ids


def hf_mixtral_router_cls():
    from transformers.models.mixtral.modeling_mixtral import MixtralTopKRouter
    return MixtralTopKRouter


def hf_deepseek_route(weight: torch.Tensor, k: int, x: torch.Tensor, method: str, n_group: int, topk_group: int,
                      scaling: float):
    cfg = DeepseekV2Config(hidden_size=weight.shape[1], num_attention_heads=1, num_key_value_heads=1,
                           n_routed_experts=weight.shape[0], num_experts_per_tok=k,
                           topk_method=method, n_group=n_group, topk_group=topk_group, routed_scaling_factor=scaling)
    r = hf_ds.DeepseekV2TopkRouter(cfg).to(DT)
    with torch.no_grad():
        r.weight.copy_(weight)
        _, w, ids = r(x)
    return w, ids


# ================================================================================================
# Small models
# ================================================================================================


def _randomize(model: torch.nn.Module, seed: int, router_scale: float = 3.0) -> None:
    """Weights of a useful size: normal / sqrt(fan_in) for matrices, 1 + 0.1 normal for norms."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1:
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g, dtype=DT))
            else:
                scale = (router_scale if name.endswith("mlp.gate.weight") else 1.0) / p.shape[-1] ** 0.5
                p.copy_(scale * torch.randn(p.shape, generator=g, dtype=DT))


def _common(vocab: int, hidden: int, layers: int, heads: int, rope: impl.Rope | None = None) -> dict:
    # For YaRN, transformers expects max_position_embeddings = factor * the original length.
    max_pos = int(rope.orig_max * rope.factor) if rope is not None and rope.kind == "yarn" else 4096
    return dict(vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
                max_position_embeddings=max_pos, rms_norm_eps=1e-6, tie_word_embeddings=False, use_cache=True)


def tiny_llama(seed: int, heads: int, kv_heads: int, head_dim: int = 8, rope: impl.Rope | None = None,
               vocab: int = 40, hidden: int = 32, layers: int = 2) -> LlamaForCausalLM:
    rope = rope or impl.Rope(head_dim, 10000.0)
    cfg = LlamaConfig(**_common(vocab, hidden, layers, heads), num_key_value_heads=kv_heads, head_dim=head_dim,
                      intermediate_size=48, rope_parameters=hf_rope_parameters(rope), attn_implementation="eager",
                           experts_implementation="eager")
    m = LlamaForCausalLM(cfg).to(DT).eval()
    _randomize(m, seed)
    return m


def tiny_mixtral(seed: int, heads: int = 4, kv_heads: int = 2, experts: int = 8, top_k: int = 2,
                 vocab: int = 40, hidden: int = 32, layers: int = 2) -> MixtralForCausalLM:
    cfg = MixtralConfig(**_common(vocab, hidden, layers, heads), num_key_value_heads=kv_heads, head_dim=8,
                        intermediate_size=24, num_local_experts=experts, num_experts_per_tok=top_k,
                        rope_parameters={"rope_type": "default", "rope_theta": 10000.0}, attn_implementation="eager",
                           experts_implementation="eager")
    m = MixtralForCausalLM(cfg).to(DT).eval()
    _randomize(m, seed)
    return m


def tiny_deepseek(seed: int, q_lora_rank: int | None = 16, method: str = "group_limited_greedy", experts: int = 8,
                  top_k: int = 2, n_group: int = 4, topk_group: int = 2, rope: impl.Rope | None = None,
                  vocab: int = 40, hidden: int = 32, layers: int = 2) -> DeepseekV2ForCausalLM:
    rope = rope or impl.Rope(8, 10000.0, style="gptj")
    cfg = DeepseekV2Config(**_common(vocab, hidden, layers, 4, rope), num_key_value_heads=4, intermediate_size=48,
                           moe_intermediate_size=16, first_k_dense_replace=1, n_routed_experts=experts,
                           n_shared_experts=2, num_experts_per_tok=top_k, topk_method=method, n_group=n_group,
                           topk_group=topk_group, routed_scaling_factor=1.5, kv_lora_rank=12, q_lora_rank=q_lora_rank,
                           qk_nope_head_dim=8, qk_rope_head_dim=rope.dim, v_head_dim=6,
                           rope_parameters=hf_rope_parameters(rope), attn_implementation="eager",
                           experts_implementation="eager")
    m = DeepseekV2ForCausalLM(cfg).to(DT).eval()
    _randomize(m, seed)
    return m


def _rope_from_hf(cfg, dim: int, style: str) -> impl.Rope:
    p = cfg.rope_parameters
    r = impl.Rope(dim, float(p["rope_theta"]), style=style, kind=p["rope_type"])
    if r.kind == "llama3":
        r.factor, r.low_freq_factor, r.high_freq_factor = p["factor"], p["low_freq_factor"], p["high_freq_factor"]
        r.orig_max = p["original_max_position_embeddings"]
    elif r.kind == "yarn":
        r.factor, r.orig_max = p["factor"], p["original_max_position_embeddings"]
        r.beta_fast, r.beta_slow = p.get("beta_fast") or 32, p.get("beta_slow") or 1
        r.mscale, r.mscale_all_dim = p.get("mscale") or 0.0, p.get("mscale_all_dim") or 0.0
    return r


def _mlp(m) -> impl.MLP:
    return impl.MLP(m.gate_proj.weight.detach().clone(), m.up_proj.weight.detach().clone(),
                    m.down_proj.weight.detach().clone())


def _experts(e) -> list[impl.MLP]:
    out = []
    for i in range(e.gate_up_proj.shape[0]):
        gate, up = e.gate_up_proj[i].detach().clone().chunk(2, dim=0)
        out.append(impl.MLP(gate, up, e.down_proj[i].detach().clone()))
    return out


def to_impl(model) -> impl.Model:
    """Copy a transformers Llama, Mixtral or DeepSeek-V2 model into impl."""
    cfg = model.config
    layers = []
    for i, l in enumerate(model.model.layers):
        a = l.self_attn
        if isinstance(model, DeepseekV2ForCausalLM):
            rope = _rope_from_hf(cfg, cfg.qk_rope_head_dim, "gptj")
            lora = cfg.q_lora_rank is not None
            attn = impl.MLA(
                wq=None if lora else a.q_proj.weight.detach().clone(),
                wq_a=a.q_a_proj.weight.detach().clone() if lora else None,
                q_a_norm=a.q_a_layernorm.weight.detach().clone() if lora else None,
                wq_b=a.q_b_proj.weight.detach().clone() if lora else None,
                wkv_a=a.kv_a_proj_with_mqa.weight.detach().clone(), kv_a_norm=a.kv_a_layernorm.weight.detach().clone(),
                wkv_b=a.kv_b_proj.weight.detach().clone(), wo=a.o_proj.weight.detach().clone(),
                n_heads=cfg.num_attention_heads, d_c=cfg.kv_lora_rank, d_nope=cfg.qk_nope_head_dim,
                d_rope=cfg.qk_rope_head_dim, d_v=cfg.v_head_dim, rope=rope, eps=a.kv_a_layernorm.variance_epsilon)
        else:
            attn = impl.GQA(a.q_proj.weight.detach().clone(), a.k_proj.weight.detach().clone(),
                            a.v_proj.weight.detach().clone(), a.o_proj.weight.detach().clone(),
                            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
                            _rope_from_hf(cfg, cfg.head_dim, "neox"))
        mlp = l.mlp
        if isinstance(model, MixtralForCausalLM):
            ffn = impl.MoE(impl.Router(mlp.gate.weight.detach().clone(), cfg.num_experts_per_tok, True),
                           _experts(mlp.experts))
        elif isinstance(model, DeepseekV2ForCausalLM) and i >= cfg.first_k_dense_replace:
            grouped = cfg.topk_method == "group_limited_greedy"
            router = impl.Router(mlp.gate.weight.detach().clone(), cfg.num_experts_per_tok, False,
                                 cfg.routed_scaling_factor, cfg.n_group if grouped else 1,
                                 cfg.topk_group if grouped else 1)
            ffn = impl.MoE(router, _experts(mlp.experts), _mlp(mlp.shared_experts))
        else:
            ffn = _mlp(mlp)
        layers.append(impl.Layer(l.input_layernorm.weight.detach().clone(), attn,
                                 l.post_attention_layernorm.weight.detach().clone(), ffn))
    return impl.Model(model.model.embed_tokens.weight.detach().clone(), layers, model.model.norm.weight.detach().clone(),
                      model.lm_head.weight.detach().clone(), cfg.rms_norm_eps)


def hf_logits(model, tokens: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return model(torch.tensor([tokens])).logits[0]


def hf_incremental(model, chunks: list[list[int]]) -> tuple[torch.Tensor, DynamicCache]:
    """Feed the chunks one after another through a DynamicCache; logits of every token, and the cache."""
    cache = DynamicCache(config=model.config)
    outs = []
    with torch.no_grad():
        for chunk in chunks:
            out = model(torch.tensor([chunk]), past_key_values=cache, use_cache=True)
            outs.append(out.logits[0])
    return torch.cat(outs), cache


def hf_cache_elements(cache: DynamicCache) -> int:
    return sum(layer.keys.numel() + layer.values.numel() for layer in cache.layers)

"""Transformer variants from scratch: RoPE and its long-context scalings, grouped-query attention,
mixture-of-experts routing, and multi-head latent attention (MLA), with KV caches for incremental decode.

Float64, torch only; no transformers or vLLM import.

Notation:
    T = new tokens in this call, S = tokens in the cache after it, H = hidden size
    Hq = query heads, Hkv = key/value heads (GQA: Hq % Hkv == 0; MHA: Hkv = Hq; MQA: Hkv = 1), d = head dim
    E = routed experts, k = experts per token
    MLA (DeepSeek-V2 section 2.1): d_c = kv_lora_rank (the cached latent), d_R = qk_rope_head_dim (the one
    shared rotary key), d_nope = qk_nope_head_dim, d_v = v_head_dim
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

DT = torch.float64


# ================================================================================================
# RoPE
# ================================================================================================


@dataclass
class Rope:
    dim: int  # rotated dimensions (the whole head for Llama, d_R for MLA)
    base: float
    style: str = "neox"  # "neox": rotate (x_i, x_{i+dim/2}); "gptj": rotate (x_{2i}, x_{2i+1})
    kind: str = "default"  # "default", "llama3", "yarn"
    factor: float = 1.0  # context extension factor
    orig_max: int = 0  # context length the model was trained at
    low_freq_factor: float = 1.0  # llama3
    high_freq_factor: float = 4.0  # llama3
    beta_fast: float = 32.0  # yarn
    beta_slow: float = 1.0  # yarn
    mscale: float = 1.0  # yarn
    mscale_all_dim: float = 0.0  # yarn


def base_inv_freq(dim: int, base: float) -> torch.Tensor:
    """theta_i = base^(-2i/dim), i = 0 .. dim/2 - 1 (RoFormer section 3.2.2)."""
    return 1.0 / base ** (torch.arange(0, dim, 2, dtype=DT) / dim)


def llama3_inv_freq(r: Rope) -> torch.Tensor:
    """Llama 3.1: keep short wavelengths, divide long ones by `factor`, blend linearly in between."""
    inv = base_inv_freq(r.dim, r.base)
    wavelen = 2 * math.pi / inv
    low_wavelen = r.orig_max / r.low_freq_factor  # longer than this: fully interpolated
    high_wavelen = r.orig_max / r.high_freq_factor  # shorter than this: untouched
    smooth = (r.orig_max / wavelen - r.low_freq_factor) / (r.high_freq_factor - r.low_freq_factor)
    blended = (1 - smooth) * inv / r.factor + smooth * inv
    return torch.where(wavelen < high_wavelen, inv, torch.where(wavelen > low_wavelen, inv / r.factor, blended))


def yarn_mscale(factor: float, mscale: float = 1.0) -> float:
    return 1.0 if factor <= 1 else 0.1 * mscale * math.log(factor) + 1.0


def yarn_inv_freq(r: Rope) -> tuple[torch.Tensor, float]:
    """YaRN: dimensions that turn fewer than beta_slow times over the trained context are interpolated
    (divided by factor), those that turn more than beta_fast times are kept, with a linear ramp between.
    Returns the frequencies and the factor applied to cos and sin."""
    inv = base_inv_freq(r.dim, r.base)

    def dim_for(rotations: float) -> float:  # the dimension index that turns `rotations` times over orig_max
        return r.dim * math.log(r.orig_max / (rotations * 2 * math.pi)) / (2 * math.log(r.base))

    low = max(math.floor(dim_for(r.beta_fast)), 0)
    high = min(math.ceil(dim_for(r.beta_slow)), r.dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(r.dim // 2, dtype=DT) - low) / (high - low)).clamp(0, 1)  # 0: keep, 1: interpolate
    out = inv * (1 - ramp) + inv / r.factor * ramp
    if r.mscale and r.mscale_all_dim:
        attention_factor = yarn_mscale(r.factor, r.mscale) / yarn_mscale(r.factor, r.mscale_all_dim)
    else:
        attention_factor = yarn_mscale(r.factor)
    return out, attention_factor


def rope_inv_freq(r: Rope) -> tuple[torch.Tensor, float]:
    """Frequencies and the factor applied to cos and sin."""
    if r.kind == "default":
        return base_inv_freq(r.dim, r.base), 1.0
    if r.kind == "llama3":
        return llama3_inv_freq(r), 1.0
    if r.kind == "yarn":
        return yarn_inv_freq(r)
    raise ValueError(f"unknown rope kind {r.kind}")


def rope_cos_sin(r: Rope, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of position * theta_i, shape [T, dim/2], times the attention factor."""
    inv, factor = rope_inv_freq(r)
    angles = positions.to(DT)[:, None] * inv[None, :]
    return angles.cos() * factor, angles.sin() * factor


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, style: str) -> torch.Tensor:
    """Rotate each pair of x[..., T, dim] by its angle. cos, sin: [T, dim/2]."""
    half = x.shape[-1] // 2
    if style == "neox":
        a, b = x[..., :half], x[..., half:]
        return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)
    if style == "gptj":
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * cos - b * sin, b * cos + a * sin), dim=-1).flatten(-2)
    raise ValueError(f"unknown rope style {style}")


def neox_to_gptj(n: int) -> torch.Tensor:
    """The permutation P with gptj(x[P]) = neox(x)[P]: neox pairs (i, i + n/2), gptj pairs (2i, 2i + 1)."""
    return torch.stack((torch.arange(n // 2), torch.arange(n // 2) + n // 2), dim=-1).flatten()


# ================================================================================================
# Shared pieces
# ================================================================================================


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


@dataclass
class MLP:
    gate: torch.Tensor  # [I, H]
    up: torch.Tensor  # [I, H]
    down: torch.Tensor  # [H, I]


def swiglu(m: MLP, x: torch.Tensor) -> torch.Tensor:
    return (torch.nn.functional.silu(x @ m.gate.T) * (x @ m.up.T)) @ m.down.T


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_pos: torch.Tensor, k_pos: torch.Tensor,
           scale: float) -> torch.Tensor:
    """Causal attention. q [Hq, T, dk], k [Hkv, S, dk], v [Hkv, S, dv] -> [Hq, T, dv].

    Query head h reads key/value head h // (Hq / Hkv): consecutive query heads share a group.
    """
    group = q.shape[0] // k.shape[0]
    k = k.repeat_interleave(group, dim=0)
    v = v.repeat_interleave(group, dim=0)
    scores = (q @ k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(k_pos[None, None, :] > q_pos[None, :, None], float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


# ================================================================================================
# Grouped-query attention
# ================================================================================================


@dataclass
class GQA:
    wq: torch.Tensor  # [Hq * d, H]
    wk: torch.Tensor  # [Hkv * d, H]
    wv: torch.Tensor  # [Hkv * d, H]
    wo: torch.Tensor  # [H, Hq * d]
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope: Rope


@dataclass
class KVCache:
    k: torch.Tensor | None = None  # [Hkv, S, d], already rotated
    v: torch.Tensor | None = None  # [Hkv, S, d]

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[1]

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.k = k if self.k is None else torch.cat((self.k, k), dim=1)
        self.v = v if self.v is None else torch.cat((self.v, v), dim=1)
        return self.k, self.v

    def elements(self) -> int:
        return 0 if self.k is None else self.k.numel() + self.v.numel()


def gqa_forward(a: GQA, x: torch.Tensor, cache: KVCache) -> torch.Tensor:
    """x [T, H] at positions cache.length .. cache.length + T - 1 -> [T, H]; appends to cache."""
    T = x.shape[0]
    pos = torch.arange(cache.length, cache.length + T)
    q = (x @ a.wq.T).view(T, a.n_heads, a.head_dim).transpose(0, 1)
    k = (x @ a.wk.T).view(T, a.n_kv_heads, a.head_dim).transpose(0, 1)
    v = (x @ a.wv.T).view(T, a.n_kv_heads, a.head_dim).transpose(0, 1)
    cos, sin = rope_cos_sin(a.rope, pos)
    q, k = apply_rope(q, cos, sin, a.rope.style), apply_rope(k, cos, sin, a.rope.style)
    k_all, v_all = cache.append(k, v)
    out = attend(q, k_all, v_all, pos, torch.arange(k_all.shape[1]), a.head_dim ** -0.5)
    return out.transpose(0, 1).reshape(T, -1) @ a.wo.T


def mha_to_gqa(a: GQA, n_kv_heads: int) -> GQA:
    """GQA paper section 2.2: build each group's key and value head by mean-pooling the MHA heads in it."""
    if a.n_kv_heads != a.n_heads or a.n_heads % n_kv_heads:
        raise ValueError("needs an MHA layer and a group count that divides the heads")
    H = a.wk.shape[1]

    def pool(w: torch.Tensor) -> torch.Tensor:
        return w.view(n_kv_heads, a.n_heads // n_kv_heads, a.head_dim, H).mean(dim=1).reshape(-1, H)

    return GQA(a.wq, pool(a.wk), pool(a.wv), a.wo, a.n_heads, n_kv_heads, a.head_dim, a.rope)


# ================================================================================================
# Mixture of experts
# ================================================================================================


@dataclass
class Router:
    weight: torch.Tensor  # [E, H]
    top_k: int
    renormalize: bool  # Mixtral: yes; DeepSeek-V2: no
    scaling: float = 1.0  # DeepSeek-V2's routed_scaling_factor
    n_group: int = 1  # DeepSeek-V2 group-limited routing: experts in n_group groups,
    topk_group: int = 1  # tokens may only use experts in the topk_group groups with the best expert


def route(r: Router, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate weights [T, k] and expert ids [T, k] (ids in descending score order)."""
    scores = torch.softmax(x @ r.weight.T, dim=-1)
    E = scores.shape[-1]
    if r.n_group > 1:
        if E % r.n_group or r.topk_group * (E // r.n_group) < r.top_k:
            raise ValueError("groups must split the experts and the kept groups must hold top_k experts")
        best = scores.view(-1, r.n_group, E // r.n_group).amax(dim=-1)  # [T, n_group]
        keep = torch.zeros_like(best, dtype=torch.bool)
        keep.scatter_(1, best.topk(r.topk_group, dim=-1).indices, True)
        # -inf, not 0: a softmax that underflows to 0 must not let an expert outside the kept groups tie in.
        scores = scores.masked_fill(~keep.repeat_interleave(E // r.n_group, dim=1), float("-inf"))
    weights, ids = scores.topk(r.top_k, dim=-1)
    if r.renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * r.scaling, ids


@dataclass
class MoE:
    router: Router
    experts: list[MLP]
    shared: MLP | None = None  # DeepSeek-V2: always-on shared experts, merged into one MLP


def moe_forward(m: MoE, x: torch.Tensor) -> torch.Tensor:
    """Each token: sum over its k experts of gate weight * expert(x), plus the shared experts."""
    weights, ids = route(m.router, x)
    out = torch.zeros_like(x)
    for e, expert in enumerate(m.experts):
        token, slot = torch.where(ids == e)
        if token.numel():
            out.index_add_(0, token, weights[token, slot, None] * swiglu(expert, x[token]))
    if m.shared is not None:
        out = out + swiglu(m.shared, x)
    return out


def expert_load(ids: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Tokens sent to each expert."""
    return torch.bincount(ids.flatten(), minlength=n_experts)


# ================================================================================================
# Multi-head latent attention (DeepSeek-V2)
# ================================================================================================


@dataclass
class MLA:
    wq: torch.Tensor | None  # [Hn * (d_nope + d_R), H] when q_lora_rank is None
    wq_a: torch.Tensor | None  # [q_lora_rank, H]
    q_a_norm: torch.Tensor | None  # [q_lora_rank]
    wq_b: torch.Tensor | None  # [Hn * (d_nope + d_R), q_lora_rank]
    wkv_a: torch.Tensor  # [d_c + d_R, H]: the latent and the shared rotary key
    kv_a_norm: torch.Tensor  # [d_c]
    wkv_b: torch.Tensor  # [Hn * (d_nope + d_v), d_c]: per head, W_UK rows then W_UV rows
    wo: torch.Tensor  # [H, Hn * d_v]
    n_heads: int
    d_c: int
    d_nope: int
    d_rope: int
    d_v: int
    rope: Rope
    eps: float = 1e-6  # the latent norms


@dataclass
class LatentCache:
    """What MLA caches per token: the normed latent c (d_c) and the rotated shared key k_R (d_R)."""

    c: torch.Tensor | None = None  # [S, d_c]
    k_rope: torch.Tensor | None = None  # [S, d_R]

    @property
    def length(self) -> int:
        return 0 if self.c is None else self.c.shape[0]

    def append(self, c: torch.Tensor, k_rope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.c = c if self.c is None else torch.cat((self.c, c), dim=0)
        self.k_rope = k_rope if self.k_rope is None else torch.cat((self.k_rope, k_rope), dim=0)
        return self.c, self.k_rope

    def elements(self) -> int:
        return 0 if self.c is None else self.c.numel() + self.k_rope.numel()


def mla_scale(a: MLA) -> float:
    """Softmax scale: 1/sqrt(d_nope + d_R), times mscale_all_dim's YaRN factor squared when YaRN is on."""
    s = (a.d_nope + a.d_rope) ** -0.5
    if a.rope.kind == "yarn" and a.rope.mscale_all_dim:
        s *= yarn_mscale(a.rope.factor, a.rope.mscale_all_dim) ** 2
    return s


def mla_split_kv_b(a: MLA) -> tuple[torch.Tensor, torch.Tensor]:
    """W_UK [Hn, d_nope, d_c] and W_UV [Hn, d_v, d_c] from kv_b_proj."""
    w = a.wkv_b.view(a.n_heads, a.d_nope + a.d_v, a.d_c)
    return w[:, : a.d_nope], w[:, a.d_nope :]


def _mla_project(a: MLA, x: torch.Tensor, cache: LatentCache):
    T = x.shape[0]
    pos = torch.arange(cache.length, cache.length + T)
    if a.wq is not None:
        q = x @ a.wq.T
    else:
        q = rms_norm(x @ a.wq_a.T, a.q_a_norm, a.eps) @ a.wq_b.T
    q = q.view(T, a.n_heads, a.d_nope + a.d_rope).transpose(0, 1)  # [Hn, T, d_nope + d_R]
    q_nope, q_rope = q[..., : a.d_nope], q[..., a.d_nope :]
    ckv = x @ a.wkv_a.T
    c = rms_norm(ckv[:, : a.d_c], a.kv_a_norm, a.eps)
    k_rope = ckv[:, a.d_c :]
    cos, sin = rope_cos_sin(a.rope, pos)
    q_rope = apply_rope(q_rope, cos, sin, a.rope.style)
    k_rope = apply_rope(k_rope, cos, sin, a.rope.style)
    c_all, k_rope_all = cache.append(c, k_rope)
    return pos, q_nope, q_rope, c_all, k_rope_all


def mla_forward(a: MLA, x: torch.Tensor, cache: LatentCache) -> torch.Tensor:
    """Naive MLA: expand the cached latents into per-head keys and values, then ordinary attention."""
    T = x.shape[0]
    pos, q_nope, q_rope, c, k_rope = _mla_project(a, x, cache)
    S = c.shape[0]
    w_uk, w_uv = mla_split_kv_b(a)
    k_nope = torch.einsum("sc,hnc->hsn", c, w_uk)  # [Hn, S, d_nope]
    v = torch.einsum("sc,hvc->hsv", c, w_uv)  # [Hn, S, d_v]
    k = torch.cat((k_nope, k_rope[None].expand(a.n_heads, S, a.d_rope)), dim=-1)
    q = torch.cat((q_nope, q_rope), dim=-1)
    out = attend(q, k, v, pos, torch.arange(S), mla_scale(a))
    return out.transpose(0, 1).reshape(T, -1) @ a.wo.T


def mla_forward_absorbed(a: MLA, x: torch.Tensor, cache: LatentCache) -> torch.Tensor:
    """MLA as decode kernels run it: W_UK folds into the query and W_UV into the output, so attention reads
    the latent cache directly and per-head keys and values never exist (DeepSeek-V2 section 2.1.2).

        q_nope . (W_UK c) = (W_UK^T q_nope) . c      and      sum_s p_s (W_UV c_s) = W_UV (sum_s p_s c_s)
    """
    T = x.shape[0]
    pos, q_nope, q_rope, c, k_rope = _mla_project(a, x, cache)
    S = c.shape[0]
    w_uk, w_uv = mla_split_kv_b(a)
    q_lat = torch.einsum("htn,hnc->htc", q_nope, w_uk)  # [Hn, T, d_c]
    scores = (q_lat @ c.T + q_rope @ k_rope.T) * mla_scale(a)  # [Hn, T, S]
    scores = scores.masked_fill(torch.arange(S)[None, None, :] > pos[None, :, None], float("-inf"))
    o_lat = torch.softmax(scores, dim=-1) @ c  # [Hn, T, d_c]
    out = torch.einsum("htc,hvc->htv", o_lat, w_uv)  # [Hn, T, d_v]
    return out.transpose(0, 1).reshape(T, -1) @ a.wo.T


# ================================================================================================
# KV cache size
# ================================================================================================


def kv_elements_per_token(layers: int, *, n_kv_heads: int = 0, head_dim: int = 0, v_head_dim: int | None = None,
                          d_c: int = 0, d_rope: int = 0) -> int:
    """Cached numbers per token over all layers. Attention: Hkv * (d + d_v); MLA: d_c + d_R."""
    if d_c:
        return layers * (d_c + d_rope)
    return layers * n_kv_heads * (head_dim + (head_dim if v_head_dim is None else v_head_dim))


# ================================================================================================
# A decoder-only model
# ================================================================================================


@dataclass
class Layer:
    attn_norm: torch.Tensor
    attn: GQA | MLA
    ffn_norm: torch.Tensor
    ffn: MLP | MoE


@dataclass
class Model:
    embed: torch.Tensor  # [V, H]
    layers: list[Layer]
    norm: torch.Tensor
    head: torch.Tensor  # [V, H]
    eps: float
    absorbed: bool = False  # MLA layers: run the absorbed form

    def new_cache(self) -> list[KVCache | LatentCache]:
        return [LatentCache() if isinstance(l.attn, MLA) else KVCache() for l in self.layers]


def model_forward(m: Model, tokens: list[int], cache: list) -> torch.Tensor:
    """Logits [T, V] for `tokens`, which follow whatever the cache already holds."""
    if len(cache) != len(m.layers):
        raise ValueError(f"{len(cache)} caches for {len(m.layers)} layers")
    x = m.embed[torch.tensor(tokens)]
    for layer, c in zip(m.layers, cache):
        h = rms_norm(x, layer.attn_norm, m.eps)
        if isinstance(layer.attn, MLA):
            h = (mla_forward_absorbed if m.absorbed else mla_forward)(layer.attn, h, c)
        else:
            h = gqa_forward(layer.attn, h, c)
        x = x + h
        h = rms_norm(x, layer.ffn_norm, m.eps)
        x = x + (moe_forward(layer.ffn, h) if isinstance(layer.ffn, MoE) else swiglu(layer.ffn, h))
    return rms_norm(x, m.norm, m.eps) @ m.head.T


def cache_elements(cache: list) -> int:
    return sum(c.elements() for c in cache)

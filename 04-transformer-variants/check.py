"""Check impl against transformers and vLLM, and against identities that hold exactly in float64.

Reference comparisons:
- rope_freq:     impl's RoPE frequencies and cos/sin factor (default, Llama 3.1, YaRN) vs transformers' RoPE
                 init functions and vLLM's rotary classes, on small and real-model settings.
- rope_apply:    impl.apply_rope vs transformers' Llama apply_rotary_pos_emb and vLLM's forward_static (both
                 given impl's cos and sin, so float64 throughout), and vs DeepSeek-V2's apply_rotary_emb.
- routing:       impl.route vs transformers' Mixtral and DeepSeek-V2 routers and vLLM's CPU routers
                 (_softmax_topk, _grouped_topk): the same experts per token and the same gate weights.
- model_hf:      impl's whole decoder vs transformers' LlamaForCausalLM (MHA, GQA, MQA; default and Llama 3.1
                 RoPE), MixtralForCausalLM and DeepseekV2ForCausalLM (MLA, with and without the query
                 low-rank path, greedy and group-limited routing, default RoPE and YaRN): logits of every token.
- model_decode:  the same models fed in chunks through their caches (impl's and transformers' DynamicCache):
                 logits equal the one-shot logits, and the cache holds the predicted number of elements.

Identities inside impl (float64):
- rope_identity: the score between a rotated query and key depends only on their distance; rotation keeps
                 norms; the "gptj" pairing is the "neox" pairing under a fixed permutation of dimensions.
- gqa_identity:  a GQA layer equals an MHA layer whose key/value heads are repeated per group; mean-pooled
                 conversion of an MHA layer whose heads agree within each group changes nothing.
- moe_identity:  softmax-then-top-k-then-renormalize equals top-k-of-logits-then-softmax; the sparse MoE equals
                 the dense sum with zero weights for unchosen experts; group-limited routing picks only experts
                 in the chosen groups.
- mla (inside model_decode): absorbed MLA equals naive MLA, prefill and decode.
- kv_size:       impl.kv_elements_per_token agrees with the caches above and gives the README's numbers.

`--mutate NAME` plants one bug in impl; `--self-test` runs every mutation in a subprocess and passes only
if each is reported as a mismatch, after an unmutated run passes.

Usage:  python check.py [--mutate NAME] [--out results/check.json]
        python check.py --self-test [--out results/self_test.json]
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
import resource
import subprocess
import sys
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import impl  # noqa: E402

DT = torch.float64
TOL_ID = (1e-10, 1e-9)  # float64 identities
TOL_EXACT64 = (1e-12, 1e-12)  # the same float64 formula written by someone else
TOL_F32 = (1e-6, 1e-5)  # a reference that computes in float32 (frequencies, gate weights)
TOL_HF = (1e-5, 1e-5)  # whole transformers models: their RMSNorm and RoPE run in float32

MIN_COVERAGE = {
    "llama3_blend_dims": 1,  # Llama 3.1 frequencies in the blended band, not only kept or divided ones
    "yarn_partial_ramp_dims": 1,  # YaRN frequencies strictly between kept and interpolated
    "group_limit_changed_choice": 10,  # tokens whose experts differ from plain top-k because of the group limit
    "router_tokens": 400,
    "experts_checked_directly": 68,  # every routed expert of every model: 3 Mixtral (8+4+6) x 2 layers + 4 DeepSeek x 8
    "decode_single_token_steps": 50,
}


class Mismatch(Exception):
    pass


COUNT: Counter = Counter()
MAX_ERR: dict[str, float] = {}
MIN_GAP: dict[str, float] = {}  # smallest score gap between the k-th and (k+1)-th expert seen in routing
SECONDS: Counter = Counter()


def close(family: str, what: str, got: torch.Tensor, ref: torch.Tensor, tol) -> None:
    got, ref = torch.as_tensor(got, dtype=DT), torch.as_tensor(ref, dtype=DT)
    if got.shape != ref.shape:
        raise Mismatch(f"{family} {what}: shape {tuple(got.shape)} vs reference {tuple(ref.shape)}")
    if not (torch.isfinite(got).all() and torch.isfinite(ref).all()):
        raise Mismatch(f"{family} {what}: non-finite values")
    atol, rtol = tol
    ratio = ((got - ref).abs() / (atol + rtol * ref.abs())).max().item() if got.numel() else 0.0
    MAX_ERR[family] = max(MAX_ERR.get(family, 0.0), ratio)
    COUNT[f"compare_{family}"] += 1
    if ratio > 1:
        raise Mismatch(f"{family} {what}: max abs diff {(got - ref).abs().max().item():.3e} exceeds atol={atol} rtol={rtol}")


def equal(family: str, what: str, a, b) -> None:
    COUNT[f"compare_{family}"] += 1
    if a != b:
        raise Mismatch(f"{family} {what}: {a} != {b}")


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ================================================================================================
# RoPE
# ================================================================================================

LLAMA31 = impl.Rope(128, 500000.0, kind="llama3", factor=8.0, orig_max=8192)  # Llama 3.1 8B
DSV2 = impl.Rope(64, 10000.0, style="gptj", kind="yarn", factor=40.0, orig_max=4096, mscale=0.707,
                 mscale_all_dim=0.707)  # DeepSeek-V2
SMALL_LLAMA3 = impl.Rope(16, 10000.0, kind="llama3", factor=8.0, orig_max=256)
SMALL_YARN = impl.Rope(16, 10000.0, style="gptj", kind="yarn", factor=4.0, orig_max=64, mscale=0.707,
                       mscale_all_dim=0.707)
ROPES = [
    ("default d8", impl.Rope(8, 10000.0)),
    ("default d64 base 5e5", impl.Rope(64, 500000.0)),
    ("default d128", impl.Rope(128, 10000.0)),
    ("llama3 small", SMALL_LLAMA3),
    ("llama3.1 8B", LLAMA31),
    ("llama3 factor 4", impl.Rope(64, 500000.0, kind="llama3", factor=4.0, low_freq_factor=2.0,
                                  high_freq_factor=8.0, orig_max=4096)),
    ("yarn small", SMALL_YARN),
    ("yarn DeepSeek-V2", DSV2),
    ("yarn no mscale", impl.Rope(32, 10000.0, style="gptj", kind="yarn", factor=8.0, orig_max=2048)),
]


def check_rope_freq(oracle, name: str, r: impl.Rope) -> None:
    inv, factor = impl.rope_inv_freq(r)
    for ref_name, ref in (("transformers", oracle.hf_inv_freq(r)), ("vLLM", oracle.vllm_inv_freq(r))):
        close("rope_freq", f"{name} inv_freq vs {ref_name}", inv, ref[0], TOL_F32)
        close("rope_freq", f"{name} cos/sin factor vs {ref_name}", torch.tensor(factor), torch.tensor(float(ref[1])),
              TOL_F32)
    base = impl.base_inv_freq(r.dim, r.base)
    if r.kind == "llama3":
        wavelen = 2 * math.pi / base
        COUNT["llama3_blend_dims"] += int(((wavelen >= r.orig_max / r.high_freq_factor)
                                           & (wavelen <= r.orig_max / r.low_freq_factor)).sum())
    if r.kind == "yarn":
        ratio = inv / base
        COUNT["yarn_partial_ramp_dims"] += int(((ratio < 1 - 1e-12) & (ratio > 1 / r.factor + 1e-12)).sum())


def check_rope_apply(oracle, seed: int, r: impl.Rope, T: int, heads: int) -> None:
    g = gen(seed)
    pos = torch.randint(0, 3 * max(r.orig_max, 64), (T,), generator=g)
    x = torch.randn(1, heads, T, r.dim, generator=g, dtype=DT)
    cos, sin = impl.rope_cos_sin(r, pos)
    got = impl.apply_rope(x, cos, sin, r.style)
    tag = f"seed={seed} d={r.dim} kind={r.kind} style={r.style}"
    if r.style == "neox":
        close("rope_apply", f"vs transformers Llama {tag}", got, oracle.hf_apply_neox(x, cos, sin), TOL_EXACT64)
    else:
        close("rope_apply", f"vs transformers DeepSeek-V2 {tag}", got, oracle.hf_apply_gptj(x, cos, sin), TOL_F32)
    # vLLM indexes its cache by position: build one whose row p is position p.
    allpos = torch.arange(int(pos.max()) + 1)
    c_all, s_all = impl.rope_cos_sin(r, allpos)
    ref = oracle.vllm_apply(r, x[0].transpose(0, 1), pos, c_all, s_all).transpose(0, 1)[None]
    close("rope_apply", f"vs vLLM forward_static {tag}", got, ref, TOL_EXACT64)


def check_rope_identity(seed: int, r: impl.Rope) -> None:
    g = gen(seed)
    q, k = torch.randn(r.dim, generator=g, dtype=DT), torch.randn(r.dim, generator=g, dtype=DT)
    m, n, s = (int(v) for v in torch.randint(0, 4000, (3,), generator=g))
    tag = f"seed={seed} d={r.dim} kind={r.kind} style={r.style} m={m} n={n} shift={s}"

    def rot(x, p):
        cos, sin = impl.rope_cos_sin(r, torch.tensor([p]))
        return impl.apply_rope(x[None], cos, sin, r.style)[0]

    close("rope_identity", f"score depends only on m - n {tag}", rot(q, m) @ rot(k, n), rot(q, m + s) @ rot(k, n + s),
          TOL_ID)
    _, factor = impl.rope_inv_freq(r)
    close("rope_identity", f"rotation keeps the norm {tag}", rot(q, m).norm(), q.norm() * factor, TOL_ID)
    P = impl.neox_to_gptj(r.dim)
    cos, sin = impl.rope_cos_sin(r, torch.tensor([m]))
    close("rope_identity", f"gptj(x[P]) = neox(x)[P] {tag}", impl.apply_rope(q[P][None], cos, sin, "gptj")[0],
          impl.apply_rope(q[None], cos, sin, "neox")[0][P], TOL_ID)


# ================================================================================================
# Routing
# ================================================================================================


def by_id(w: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder each token's (weight, id) pairs by expert id, so order-free routers can be compared."""
    order = ids.long().argsort(dim=-1)
    return w.to(DT).gather(-1, order), ids.long().gather(-1, order)


def note_gap(family: str, scores: torch.Tensor, k: int) -> None:
    top = scores.topk(min(k + 1, scores.shape[-1]), dim=-1).values
    if top.shape[-1] > k:
        MIN_GAP[family] = min(MIN_GAP.get(family, math.inf), float((top[:, k - 1] - top[:, k]).min()))


def check_routing(oracle, seed: int, E: int, k: int, T: int = 64, H: int = 16) -> None:
    g = gen(seed)
    x = torch.randn(T, H, generator=g, dtype=DT)
    W = 2.0 * torch.randn(E, H, generator=g, dtype=DT) / H ** 0.5
    logits = x @ W.T
    COUNT["router_tokens"] += T
    note_gap("routing", torch.softmax(logits, -1), k)
    tag = f"seed={seed} E={E} k={k}"

    mix = impl.Router(W, k, renormalize=True)
    w, ids = by_id(*impl.route(mix, x))
    for ref_name, ref in (("transformers Mixtral", oracle.hf_mixtral_route(W, k, x)),
                          ("vLLM _softmax_topk", oracle.vllm_softmax_topk(logits, k, True))):
        rw, rids = by_id(*ref)
        equal("routing", f"renormalized experts vs {ref_name} {tag}", ids.tolist(), rids.tolist())
        close("routing", f"renormalized weights vs {ref_name} {tag}", w, rw, TOL_F32)
    w0, ids0 = by_id(*impl.route(impl.Router(W, k, renormalize=False), x))
    rw, rids = by_id(*oracle.vllm_softmax_topk(logits, k, False))
    equal("routing", f"plain experts vs vLLM _softmax_topk {tag}", ids0.tolist(), rids.tolist())
    close("routing", f"plain weights vs vLLM _softmax_topk {tag}", w0, rw, TOL_F32)
    rw, rids = by_id(*oracle.hf_deepseek_route(W, k, x, "greedy", 1, 1, 1.0))
    equal("routing", f"plain experts vs transformers DeepSeek-V2 greedy {tag}", ids0.tolist(), rids.tolist())
    close("routing", f"plain weights vs transformers DeepSeek-V2 greedy {tag}", w0, rw, TOL_F32)

    for n_group in (g_ for g_ in (2, 4, 8) if E % g_ == 0 and g_ < E):
        for topk_group in range(1, n_group):
            if topk_group * (E // n_group) < k:
                continue
            for scaling in (1.0, 16.0):
                r = impl.Router(W, k, False, scaling, n_group, topk_group)
                gw, gids = by_id(*impl.route(r, x))
                t2 = f"{tag} groups={n_group} keep={topk_group} scaling={scaling}"
                for ref_name, ref in (
                        ("transformers DeepSeek-V2", oracle.hf_deepseek_route(W, k, x, "group_limited_greedy", n_group,
                                                                               topk_group, scaling)),
                        ("vLLM _grouped_topk", oracle.vllm_grouped_topk(logits, k, False, n_group, topk_group, scaling))):
                    rw, rids = by_id(*ref)
                    equal("routing", f"group-limited experts vs {ref_name} {t2}", gids.tolist(), rids.tolist())
                    close("routing", f"group-limited weights vs {ref_name} {t2}", gw, rw, TOL_F32)
                rn = impl.Router(W, k, True, scaling, n_group, topk_group)
                nw, nids = by_id(*impl.route(rn, x))
                rw, rids = by_id(*oracle.vllm_grouped_topk(logits, k, True, n_group, topk_group, scaling))
                equal("routing", f"group-limited renormalized experts vs vLLM {t2}", nids.tolist(), rids.tolist())
                close("routing", f"group-limited renormalized weights vs vLLM {t2}", nw, rw, TOL_F32)
                COUNT["group_limit_changed_choice"] += int((gids != ids0).any(dim=-1).sum()) if scaling == 1.0 else 0
                check_group_limit(x, r, gids, t2)


def check_group_limit(x: torch.Tensor, r: impl.Router, ids: torch.Tensor, tag: str) -> None:
    """Every chosen expert lies in one of the topk_group groups with the best single expert."""
    scores = torch.softmax(x @ r.weight.T, dim=-1)
    E = scores.shape[-1]
    best = scores.view(-1, r.n_group, E // r.n_group).amax(-1)
    kept = best.topk(r.topk_group, dim=-1).indices
    group = ids // (E // r.n_group)
    inside = (group[:, :, None] == kept[:, None, :]).any(-1).all()
    equal("moe_identity", f"group-limited choice stays in the kept groups {tag}", bool(inside), True)


def check_moe_identity(seed: int, E: int, k: int, T: int = 32, H: int = 16, I: int = 12) -> None:
    g = gen(seed)
    x = torch.randn(T, H, generator=g, dtype=DT)
    W = 2.0 * torch.randn(E, H, generator=g, dtype=DT) / H ** 0.5
    logits = x @ W.T
    w, ids = impl.route(impl.Router(W, k, True), x)
    top, top_ids = logits.topk(k, dim=-1)
    a, ai = by_id(w, ids)
    b, bi = by_id(torch.softmax(top, dim=-1), top_ids)
    tag = f"seed={seed} E={E} k={k}"
    equal("moe_identity", f"softmax-top-k-renormalize picks the top-k logits {tag}", ai.tolist(), bi.tolist())
    close("moe_identity", f"softmax-top-k-renormalize = softmax over the top-k logits {tag}", a, b, TOL_ID)
    experts = [impl.MLP(torch.randn(I, H, generator=g, dtype=DT) / H ** 0.5, torch.randn(I, H, generator=g, dtype=DT) / H ** 0.5,
                        torch.randn(H, I, generator=g, dtype=DT) / I ** 0.5) for _ in range(E)]
    shared = impl.MLP(torch.randn(I, H, generator=g, dtype=DT) / H ** 0.5, torch.randn(I, H, generator=g, dtype=DT) / H ** 0.5,
                      torch.randn(H, I, generator=g, dtype=DT) / I ** 0.5)
    for router, sh in ((impl.Router(W, k, True), None), (impl.Router(W, k, False, 2.5, 2 if E % 2 == 0 else 1, 1), shared)):
        if router.n_group > 1 and (E // 2) < k:
            continue
        m = impl.MoE(router, experts, sh)
        got = impl.moe_forward(m, x)
        rw, rids = impl.route(router, x)
        dense = torch.zeros(T, E, dtype=DT).scatter(1, rids, rw)  # every expert's gate weight, 0 if not chosen
        want = sum(dense[:, e, None] * impl.swiglu(experts[e], x) for e in range(E))
        if sh is not None:
            want = want + impl.swiglu(sh, x)
        close("moe_identity", f"sparse MoE = dense weighted sum {tag} shared={sh is not None}", got, want, TOL_ID)
        want_load = [int((rids == e).sum()) for e in range(E)]
        equal("moe_identity", f"tokens per expert {tag}", impl.expert_load(rids, E).tolist(), want_load)


# ================================================================================================
# GQA identities
# ================================================================================================


def gqa_layer(seed: int, Hq: int, Hkv: int, d: int = 8, H: int = 32) -> impl.GQA:
    g = gen(seed)
    r = lambda rows: torch.randn(rows, H, generator=g, dtype=DT) / H ** 0.5
    return impl.GQA(r(Hq * d), r(Hkv * d), r(Hkv * d), torch.randn(H, Hq * d, generator=g, dtype=DT) / (Hq * d) ** 0.5,
                    Hq, Hkv, d, impl.Rope(d, 10000.0))


def check_gqa_identity(seed: int, Hq: int, Hkv: int, T: int = 12, d: int = 8, H: int = 32) -> None:
    a = gqa_layer(seed, Hq, Hkv, d, H)
    x = torch.randn(T, H, generator=gen(seed + 1000), dtype=DT)
    group = Hq // Hkv
    rep = lambda w: w.view(Hkv, d, H).repeat_interleave(group, dim=0).reshape(Hq * d, H)
    mha = impl.GQA(a.wq, rep(a.wk), rep(a.wv), a.wo, Hq, Hq, d, a.rope)
    tag = f"seed={seed} Hq={Hq} Hkv={Hkv}"
    close("gqa_identity", f"GQA = MHA with each key/value head repeated over its group {tag}",
          impl.gqa_forward(a, x, impl.KVCache()), impl.gqa_forward(mha, x, impl.KVCache()), TOL_ID)
    pooled = impl.mha_to_gqa(mha, Hkv)
    close("gqa_identity", f"mean-pooling heads that agree within each group changes nothing {tag}",
          impl.gqa_forward(pooled, x, impl.KVCache()), impl.gqa_forward(mha, x, impl.KVCache()), TOL_ID)
    raw = gqa_layer(seed + 7, Hq, Hq, d, H)  # distinct heads: pooled weights are the group means
    p = impl.mha_to_gqa(raw, Hkv)
    for which, got, src in (("key", p.wk, raw.wk), ("value", p.wv, raw.wv)):
        close("gqa_identity", f"pooled {which} weights are group means {tag}", got,
              src.view(Hkv, group, d, H).mean(1).reshape(Hkv * d, H), TOL_ID)
    c = impl.KVCache()
    impl.gqa_forward(a, x, c)
    equal("kv_size", f"GQA cache elements {tag}", c.elements(), impl.kv_elements_per_token(1, n_kv_heads=Hkv, head_dim=d) * T)


def check_bad_inputs() -> None:
    """Inputs that would otherwise give plausible wrong output are refused."""
    a = gqa_layer(0, 4, 2)
    m = impl.Model(torch.randn(8, 32, dtype=DT), [impl.Layer(torch.ones(32, dtype=DT), a, torch.ones(32, dtype=DT),
                                                             impl.MLP(torch.zeros(4, 32, dtype=DT), torch.zeros(4, 32, dtype=DT),
                                                                      torch.zeros(32, 4, dtype=DT)))] * 2,
                   torch.ones(32, dtype=DT), torch.randn(8, 32, dtype=DT), 1e-6)
    W = torch.randn(6, 16, dtype=DT)
    for what, call in (("one cache for two layers", lambda: impl.model_forward(m, [1, 2], m.new_cache()[:1])),
                       ("pool a GQA layer", lambda: impl.mha_to_gqa(a, 1)),
                       ("groups that do not split the experts", lambda: impl.route(impl.Router(W, 2, False, 1.0, 4, 1),
                                                                                    torch.randn(3, 16, dtype=DT))),
                       ("kept groups smaller than top_k", lambda: impl.route(impl.Router(W, 4, False, 1.0, 3, 1),
                                                                             torch.randn(3, 16, dtype=DT)))):
        try:
            call()
        except ValueError:
            COUNT["compare_inputs"] += 1
            continue
        raise Mismatch(f"inputs {what}: accepted instead of raising ValueError")


# ================================================================================================
# Whole models
# ================================================================================================

LLAMA = [(0, 4, 4, None), (1, 4, 2, None), (2, 4, 1, None), (3, 6, 3, None), (4, 8, 2, "llama3"), (5, 4, 1, "llama3")]
MIXTRAL = [(10, 8, 2), (11, 4, 1), (12, 6, 3)]
DEEPSEEK = [(20, 16, "group_limited_greedy", None), (21, None, "greedy", None), (22, 16, "greedy", "yarn"),
            (23, None, "group_limited_greedy", "yarn")]
TOKENS = 48
CHUNKINGS = [[48], [1] * 48, [5, 1, 1, 41], [17, 31]]


def models(oracle):
    for seed, Hq, Hkv, rope in LLAMA:
        d = 16 if rope else 8
        r = SMALL_LLAMA3 if rope else None
        hidden = 8 * Hq  # transformers requires hidden_size to be a multiple of the head count
        yield (f"llama Hq={Hq} Hkv={Hkv} rope={rope or 'default'}",
               oracle.tiny_llama(seed, Hq, Hkv, head_dim=d, rope=r, hidden=hidden))
    for seed, E, k in MIXTRAL:
        yield f"mixtral E={E} k={k}", oracle.tiny_mixtral(seed, experts=E, top_k=k)
    for seed, q_lora, method, rope in DEEPSEEK:
        r = SMALL_YARN if rope else None
        yield (f"deepseek q_lora={q_lora} {method} rope={rope or 'default'}",
               oracle.tiny_deepseek(seed, q_lora_rank=q_lora, method=method, rope=r))


def predicted_elements(m: impl.Model, tokens: int) -> int:
    total = 0
    for l in m.layers:
        a = l.attn
        if isinstance(a, impl.MLA):
            total += impl.kv_elements_per_token(1, d_c=a.d_c, d_rope=a.d_rope)
        else:
            total += impl.kv_elements_per_token(1, n_kv_heads=a.n_kv_heads, head_dim=a.head_dim)
    return total * tokens


def check_model(oracle, name: str, hf, seed: int) -> None:
    m = oracle.to_impl(hf)
    vocab = m.embed.shape[0]
    toks = torch.randint(0, vocab, (TOKENS,), generator=gen(seed)).tolist()
    want = oracle.hf_logits(hf, toks)
    got = impl.model_forward(m, toks, m.new_cache())
    close("model_hf", f"{name} logits", got, want, TOL_HF)
    check_each_expert(name, hf, m, seed)
    is_mla = any(isinstance(l.attn, impl.MLA) for l in m.layers)
    for chunks in CHUNKINGS:
        parts, start = [], 0
        for n in chunks:
            parts.append(toks[start:start + n])
            start += n
        tag = f"{name} chunks={chunks}"
        for absorbed in ((False, True) if is_mla else (False,)):
            m.absorbed = absorbed
            cache = m.new_cache()
            inc = torch.cat([impl.model_forward(m, p, cache) for p in parts])
            close("model_decode", f"{tag} absorbed={absorbed}: chunked = one-shot", inc, got, TOL_ID)
            equal("kv_size", f"{tag} absorbed={absorbed}: impl cache elements", impl.cache_elements(cache),
                  predicted_elements(m, TOKENS))
            COUNT["decode_single_token_steps"] += sum(1 for p in parts if len(p) == 1)
        m.absorbed = False
        hf_inc, hf_cache = oracle.hf_incremental(hf, parts)
        close("model_decode", f"{tag}: transformers chunked vs impl", inc, hf_inc, TOL_HF)
        equal("kv_size", f"{tag}: transformers cache elements", oracle.hf_cache_elements(hf_cache),
              predicted_elements(m, TOKENS))


def check_each_expert(name: str, hf, m: impl.Model, seed: int) -> None:
    """Routing need not reach every expert, so send all tokens to each expert in turn, through transformers'
    experts module and through impl's copy of that expert."""
    x = torch.randn(8, m.embed.shape[1], generator=gen(seed), dtype=DT)
    for i, (layer, hf_layer) in enumerate(zip(m.layers, hf.model.layers)):
        if not isinstance(layer.ffn, impl.MoE):
            continue
        for e, expert in enumerate(layer.ffn.experts):
            idx = torch.full((x.shape[0], 1), e, dtype=torch.long)
            ref = hf_layer.mlp.experts(x, idx, torch.ones(x.shape[0], 1, dtype=DT))
            close("model_hf", f"{name} layer {i} expert {e} alone", impl.swiglu(expert, x), ref, TOL_EXACT64)
            COUNT["experts_checked_directly"] += 1


# ================================================================================================
# KV size of real models
# ================================================================================================

# Configurations from the papers' own tables (see README section 3).
REAL = {
    "Llama 3 8B (GQA)": dict(layers=32, n_kv_heads=8, head_dim=128),
    "Llama 3 8B if it were MHA": dict(layers=32, n_kv_heads=32, head_dim=128),
    "Mixtral 8x7B (GQA)": dict(layers=32, n_kv_heads=8, head_dim=128),
    "DeepSeek-V2 (MLA)": dict(layers=60, d_c=512, d_rope=64),
    "DeepSeek-V2 if it were MHA": dict(layers=60, n_kv_heads=128, head_dim=128),  # Table 1's 2 n_h d_h l
}


def kv_table() -> dict:
    out = {}
    for name, cfg in REAL.items():
        n = impl.kv_elements_per_token(cfg["layers"], **{k: v for k, v in cfg.items() if k != "layers"})
        out[name] = {"elements_per_token": n, "kib_per_token_bf16": n * 2 / 1024}
    # Formula sanity against hand arithmetic, so the README table cannot drift from the code.
    equal("kv_size", "Llama 3 8B elements per token", out["Llama 3 8B (GQA)"]["elements_per_token"], 32 * 2 * 8 * 128)
    equal("kv_size", "DeepSeek-V2 elements per token", out["DeepSeek-V2 (MLA)"]["elements_per_token"], 60 * (512 + 64))
    # DeepSeek-V2 Table 1 and section 2.1.4: with d_c = 4 d_h and d_R = d_h / 2, MLA caches as much as GQA with 2.25 groups.
    # Groups must be whole, so compare four MLA caches with GQA at 4 * 2.25 = 9 groups.
    equal("kv_size", "DeepSeek-V2 MLA = GQA with 2.25 groups", 4 * out["DeepSeek-V2 (MLA)"]["elements_per_token"],
          impl.kv_elements_per_token(60, n_kv_heads=9, head_dim=128))
    return out


# ================================================================================================
# Driver
# ================================================================================================

ROUTING = [(s, E, k) for s, (E, k) in enumerate([(8, 1), (8, 2), (16, 2), (16, 4), (64, 6), (6, 2), (32, 8)])]
MOE_ID = [(s, E, k) for s, (E, k) in enumerate([(8, 2), (6, 1), (16, 4)])]
GQA_ID = [(s, Hq, Hkv) for s, (Hq, Hkv) in enumerate([(4, 4), (4, 2), (4, 1), (8, 2), (6, 3), (12, 4)])]
ROPE_APPLY = [(s, r) for s, (_, r) in enumerate(ROPES)]

EXPECTED_CASES = {
    "rope_freq": len(ROPES),
    "rope_apply": len(ROPE_APPLY),
    "rope_identity": 3 * len(ROPES),
    "routing": len(ROUTING),
    "moe_identity": len(MOE_ID),
    "gqa_identity": len(GQA_ID),
    "model": len(LLAMA) + len(MIXTRAL) + len(DEEPSEEK),
    "kv_table": 1,
    "inputs": 1,
}


CASES: Counter = Counter()  # filled as families finish, so a failure still reports what ran


def run_all(oracle) -> tuple[dict, dict]:
    cases = CASES
    last = [time.time()]

    def done(family: str) -> None:
        now = time.time()
        SECONDS[family] += now - last[0]
        last[0] = now
        cases[family] += 1

    for name, r in ROPES:
        check_rope_freq(oracle, name, r)
        done("rope_freq")
    for seed, r in ROPE_APPLY:
        check_rope_apply(oracle, seed, r, 24, 3)
        done("rope_apply")
    for seed in range(3):
        for _, r in ROPES:
            check_rope_identity(seed, r)
            done("rope_identity")
    for args in ROUTING:
        check_routing(oracle, *args)
        done("routing")
    for args in MOE_ID:
        check_moe_identity(*args)
        done("moe_identity")
    for args in GQA_ID:
        check_gqa_identity(*args)
        done("gqa_identity")
    for i, (name, hf) in enumerate(models(oracle)):
        check_model(oracle, name, hf, 100 + i)
        done("model")
    table = kv_table()
    done("kv_table")
    check_bad_inputs()
    done("inputs")
    return dict(cases), table


MUTATIONS = {
    "neox_rotation_sign": "neox rotation uses the wrong sign on the second half",
    "gptj_pairs_as_neox": "gptj rotation pairs (i, i + d/2) instead of adjacent dimensions",
    "inv_freq_exponent": "theta_i = base^(-i/d) instead of base^(-2i/d)",
    "llama3_no_blend": "Llama 3.1 divides the blended band by the full factor",
    "yarn_ramp_inverted": "YaRN interpolates the fast dimensions and keeps the slow ones",
    "yarn_no_attention_factor": "YaRN leaves cos and sin unscaled",
    "decode_position_reset": "each forward call starts its positions at 0",
    "kv_head_modulo": "query head h reads key/value head h mod Hkv instead of h // group",
    "causal_off_by_one": "a token cannot attend to itself",
    "pool_first_head": "MHA-to-GQA conversion keeps the first head of each group instead of the mean",
    "pool_values_first_head": "MHA-to-GQA conversion mean-pools keys but keeps the first value head",
    "expert_load_one_expert": "expert_load puts every assignment on expert 0",
    "cache_count_unchecked": "the model runs with fewer caches than layers, skipping the rest",
    "route_no_renormalize": "Mixtral routing does not renormalize the top-k weights",
    "route_logits_as_weights": "gate weights are the raw logits of the chosen experts",
    "group_score_sum": "group-limited routing ranks groups by their summed scores",
    "group_limit_ignored": "group-limited routing ignores the groups",
    "routed_scaling_dropped": "routed_scaling_factor is not applied",
    "shared_expert_dropped": "DeepSeek-V2's shared experts are left out",
    "latent_cached_before_norm": "MLA caches the latent before its RMSNorm",
    "mla_scale_no_mscale": "MLA's softmax scale ignores YaRN's mscale",
    "absorbed_drops_rope_score": "absorbed MLA scores only the latent part, not the shared rotary key",
    "kv_b_split_order": "kv_b_proj split as W_UV rows then W_UK rows",
    "kv_formula_keys_only": "the KV-size formula counts keys but not values",
}


def swap_source(fn, old: str, new: str) -> None:
    """Redefine impl function `fn` with one line of its source replaced."""
    src = inspect.getsource(fn)
    if src.count(old) != 1:
        raise RuntimeError(f"mutation no longer matches impl.{fn.__name__}: {old}")
    exec(compile(src.replace(old, new), impl.__file__, "exec"), impl.__dict__)


def apply_mutation(name: str) -> None:
    if name == "neox_rotation_sign":
        orig = impl.apply_rope

        def f(x, cos, sin, style):
            if style != "neox":
                return orig(x, cos, sin, style)
            half = x.shape[-1] // 2
            a, b = x[..., :half], x[..., half:]
            return torch.cat((a * cos + b * sin, b * cos - a * sin), dim=-1)

        impl.apply_rope = f
    elif name == "gptj_pairs_as_neox":
        orig = impl.apply_rope
        impl.apply_rope = lambda x, cos, sin, style: orig(x, cos, sin, "neox")
    elif name == "inv_freq_exponent":
        impl.base_inv_freq = lambda dim, base: 1.0 / base ** (torch.arange(0, dim, 2, dtype=DT) / dim / 2)
    elif name == "llama3_no_blend":
        def f(r):
            inv = impl.base_inv_freq(r.dim, r.base)
            wavelen = 2 * math.pi / inv
            return torch.where(wavelen < r.orig_max / r.high_freq_factor, inv, inv / r.factor)

        impl.llama3_inv_freq = f
    elif name == "yarn_ramp_inverted":
        swap_source(impl.yarn_inv_freq, "out = inv * (1 - ramp) + inv / r.factor * ramp",
                    "out = inv * ramp + inv / r.factor * (1 - ramp)")
    elif name == "yarn_no_attention_factor":
        orig = impl.yarn_inv_freq
        impl.yarn_inv_freq = lambda r: (orig(r)[0], 1.0)
    elif name == "decode_position_reset":
        impl.KVCache.length = property(lambda self: 0)
        impl.LatentCache.length = property(lambda self: 0)
    elif name == "kv_head_modulo":
        def attend(q, k, v, q_pos, k_pos, scale):
            group = q.shape[0] // k.shape[0]
            k = k.repeat(group, 1, 1)
            v = v.repeat(group, 1, 1)
            scores = (q @ k.transpose(-1, -2)) * scale
            scores = scores.masked_fill(k_pos[None, None, :] > q_pos[None, :, None], float("-inf"))
            return torch.softmax(scores, dim=-1) @ v

        impl.attend = attend
    elif name == "causal_off_by_one":
        def attend(q, k, v, q_pos, k_pos, scale):
            group = q.shape[0] // k.shape[0]
            k = k.repeat_interleave(group, dim=0)
            v = v.repeat_interleave(group, dim=0)
            scores = (q @ k.transpose(-1, -2)) * scale
            mask = k_pos[None, None, :] >= q_pos[None, :, None]
            mask[..., 0] = False  # keep the first key so no row is empty
            scores = scores.masked_fill(mask, float("-inf"))
            return torch.softmax(scores, dim=-1) @ v

        impl.attend = attend
    elif name == "pool_values_first_head":
        swap_source(impl.mha_to_gqa, "return GQA(a.wq, pool(a.wk), pool(a.wv), a.wo, a.n_heads, n_kv_heads, a.head_dim, a.rope)",
                    "return GQA(a.wq, pool(a.wk), a.wv.view(n_kv_heads, a.n_heads // n_kv_heads, a.head_dim, H)[:, 0]"
                    ".reshape(-1, H), a.wo, a.n_heads, n_kv_heads, a.head_dim, a.rope)")
    elif name == "expert_load_one_expert":
        impl.expert_load = lambda ids, n: torch.zeros(n, dtype=torch.long).index_fill(0, torch.tensor([0]), ids.numel())
    elif name == "cache_count_unchecked":
        swap_source(impl.model_forward, "if len(cache) != len(m.layers):", "if False:")
    elif name == "pool_first_head":
        def f(a, n_kv_heads):
            H = a.wk.shape[1]
            first = lambda w: w.view(n_kv_heads, a.n_heads // n_kv_heads, a.head_dim, H)[:, 0].reshape(-1, H)
            return impl.GQA(a.wq, first(a.wk), first(a.wv), a.wo, a.n_heads, n_kv_heads, a.head_dim, a.rope)

        impl.mha_to_gqa = f
    elif name in ("route_no_renormalize", "route_logits_as_weights", "group_score_sum", "group_limit_ignored",
                  "routed_scaling_dropped"):
        orig = impl.route

        def f(r, x, name=name):
            if name == "route_no_renormalize":
                return orig(impl.Router(**{**r.__dict__, "renormalize": False}), x)
            if name == "routed_scaling_dropped":
                return orig(impl.Router(**{**r.__dict__, "scaling": 1.0}), x)
            if name == "group_limit_ignored":
                return orig(impl.Router(**{**r.__dict__, "n_group": 1, "topk_group": 1}), x)
            if name == "route_logits_as_weights":
                w, ids = orig(r, x)
                return (x @ r.weight.T).gather(1, ids) * r.scaling, ids
            scores = torch.softmax(x @ r.weight.T, dim=-1)  # group_score_sum
            E = scores.shape[-1]
            if r.n_group > 1:
                best = scores.view(-1, r.n_group, E // r.n_group).sum(dim=-1)
                keep = torch.zeros_like(best, dtype=torch.bool)
                keep.scatter_(1, best.topk(r.topk_group, dim=-1).indices, True)
                scores = scores.masked_fill(~keep.repeat_interleave(E // r.n_group, dim=1), 0.0)
            w, ids = scores.topk(r.top_k, dim=-1)
            if r.renormalize:
                w = w / w.sum(dim=-1, keepdim=True)
            return w * r.scaling, ids

        impl.route = f
    elif name == "shared_expert_dropped":
        orig = impl.moe_forward
        impl.moe_forward = lambda m, x: orig(impl.MoE(m.router, m.experts, None), x)
    elif name == "latent_cached_before_norm":
        swap_source(impl._mla_project, "c = rms_norm(ckv[:, : a.d_c], a.kv_a_norm, a.eps)", "c = ckv[:, : a.d_c]")
    elif name == "mla_scale_no_mscale":
        impl.mla_scale = lambda a: (a.d_nope + a.d_rope) ** -0.5
    elif name == "absorbed_drops_rope_score":
        swap_source(impl.mla_forward_absorbed, "scores = (q_lat @ c.T + q_rope @ k_rope.T) * mla_scale(a)",
                    "scores = (q_lat @ c.T) * mla_scale(a)")
    elif name == "kv_b_split_order":
        def f(a):
            w = a.wkv_b.view(a.n_heads, a.d_nope + a.d_v, a.d_c)
            return w[:, a.d_v:], w[:, : a.d_v]

        impl.mla_split_kv_b = f
    elif name == "kv_formula_keys_only":
        swap_source(impl.kv_elements_per_token,
                    "return layers * n_kv_heads * (head_dim + (head_dim if v_head_dim is None else v_head_dim))",
                    "return layers * n_kv_heads * head_dim")
    else:
        raise SystemExit(f"unknown mutation {name}")


def self_test(out: str | None) -> int:
    base = subprocess.run([sys.executable, __file__], capture_output=True, text=True)
    if base.returncode != 0:  # a failing baseline would make every mutation look caught
        print(json.dumps({"check": "04 self-test", "passed": False,
                          "error": f"unmutated run failed (exit {base.returncode}); stdout tail: {base.stdout[-500:]}"}))
        return 1
    rows = []
    for name in MUTATIONS:
        p = subprocess.run([sys.executable, __file__, "--mutate", name], capture_output=True, text=True)
        try:
            res = json.loads(p.stdout)
        except json.JSONDecodeError:
            res = {"first_mismatch": None, "error": f"no JSON; stderr tail: {p.stderr[-300:]}"}
        caught = p.returncode == 1 and bool(res.get("first_mismatch")) and not res.get("error")
        rows.append({"mutation": name, "what": MUTATIONS[name], "exit": p.returncode, "caught": caught,
                     "first_mismatch": res.get("first_mismatch"), "error": res.get("error")})
    result = {"check": "04 self-test: every planted bug must be reported as a mismatch",
              "mutations": rows, "passed": all(r["caught"] for r in rows)}
    print(json.dumps(result, indent=2))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutate", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test(args.out)
    if args.mutate:
        apply_mutation(args.mutate)

    warnings.filterwarnings("ignore")
    torch.set_grad_enabled(False)
    import oracle

    versions = oracle.check_version()
    t0 = time.time()
    failure = error = None
    cases: dict = {}
    table: dict = {}
    try:
        cases, table = run_all(oracle)
    except Mismatch as e:
        failure = str(e)
        cases = dict(CASES)
    except Exception as e:  # a crash is a failure too, but not a caught mismatch
        frames = traceback.extract_tb(e.__traceback__)[-3:]
        where = " <- ".join(f"{Path(f.filename).name}:{f.lineno}" for f in reversed(frames))
        error = f"{type(e).__name__}: {e} at {where}"
        cases = dict(CASES)

    coverage = {k: COUNT.get(k, 0) for k in MIN_COVERAGE}
    short = [k for k, v in coverage.items() if v < MIN_COVERAGE[k]]
    missing = {k: (cases.get(k, 0), v) for k, v in EXPECTED_CASES.items() if cases.get(k, 0) != v}
    passed = failure is None and error is None and not short and not missing
    result = {
        "check": "04-transformer-variants vs transformers (Llama, Mixtral, DeepSeek-V2; RoPE init and apply; "
                 "routers) and vLLM (rotary classes, CPU routers)",
        **versions,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "mutation": args.mutate,
        "tolerances": {name: {"atol": t[0], "rtol": t[1]} for name, t in
                       (("impl_identity", TOL_ID), ("float64_reference", TOL_EXACT64),
                        ("float32_reference", TOL_F32), ("transformers_model", TOL_HF))},
        "cases": cases,
        "expected_cases": EXPECTED_CASES,
        "comparisons": {k: v for k, v in sorted(COUNT.items()) if k.startswith("compare_")},
        "max_error_over_tolerance": {k: float(f"{v:.3g}") for k, v in sorted(MAX_ERR.items())},
        "min_router_score_gap": {k: float(f"{v:.3g}") for k, v in MIN_GAP.items()},
        "seconds_per_family": {k: round(v, 1) for k, v in SECONDS.items()},
        "kv_per_token": table,
        "coverage": coverage,
        "min_coverage": MIN_COVERAGE,
        "coverage_short": short,
        "cases_missing": missing,
        "first_mismatch": failure,
        "error": error,
        "passed": passed,
        "seconds": round(time.time() - t0, 1),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

"""Check impl against transformers and vLLM, and against exact enumeration where no library can go.

Reference comparisons:
- hf_sampling:   impl.sample_round vs transformers' _speculative_sampling, same seed, so the same random
                 numbers: the accepted count and every emitted token must be identical.
- hf_assisted:   impl.speculative (greedy) driving two transformers Llama models through their own
                 DynamicCache (rolled back with cache.crop) vs generate(assistant_model=...) and vs plain
                 generate(); all three token sequences must be identical.
- vllm_greedy:   impl.verify_greedy vs vLLM's rejection_sample at temperature 0: identical accepted count
                 and tokens.
- vllm_sampled:  vLLM's rejection_sample over many seeds vs impl.outcome_distribution, the exact law of
                 (accepted count, emitted token) for fixed drafts. Every cell within 5 standard errors, and
                 no sample in a cell of probability 0. With draft logits and with one-hot drafts.

Identities inside impl:
- theorem:       one round of speculative sampling, continued by the target to a fixed length, has exactly
                 the target's distribution (enumerated over every sequence; context-dependent p and q,
                 including q = p, q with disjoint support, and tokens of probability 0).
- rollback:      a model that consumed extra tokens and rolled them back continues exactly like a fresh
                 model fed only the kept tokens (attention KV, Mamba state, transformers' DynamicCache).
- generation:    greedy speculative decoding equals greedy decoding, for attention and Mamba targets and
                 drafts and several k; sampled speculative decoding of a small vocabulary matches the
                 target's exact distribution over sequences (5 standard errors per sequence).

`--mutate NAME` plants one bug in impl; `--self-test` runs every mutation in a subprocess and passes only
if each is reported as a mismatch.

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
import weakref
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import impl  # noqa: E402

DT = torch.float64
TOL_ID = (1e-10, 1e-9)  # float64 identities
TOL_EXACT = (1e-12, 1e-10)  # enumerated probabilities
Z_MAX = 5.0  # sampled frequencies: standard errors allowed per cell

MIN_COVERAGE = {
    "hf_rejected_at_0": 20,  # first draft rejected
    "hf_rejected_later": 20,  # some drafts accepted, then one rejected
    "hf_all_accepted": 20,  # bonus token from the target
    "vllm_greedy_partial": 10,
    "vllm_greedy_all": 5,
    "assisted_partial_rounds": 20,  # greedy rounds that kept some but not all drafts
    "assisted_full_rounds": 5,
    "rollback_mamba_multi_call": 10,  # a Mamba rollback spanning several forward calls
    "generation_mamba_rounds": 20,
    "generation_sampled_excess": 20,
    "theorem_zero_prob_tokens": 5,
}


class Mismatch(Exception):
    pass


COUNT: Counter = Counter()
MAX_ERR: dict[str, float] = {}
MAX_Z: dict[str, float] = {}
MAX_CHI2: dict[str, float] = {}  # chi-square / its limit; must stay <= 1


def close(family: str, what: str, got: torch.Tensor, ref: torch.Tensor, tol) -> None:
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


def frequencies(family: str, what: str, counts: Counter, exact: dict, n: int) -> None:
    """Sampled counts vs exact probabilities: each cell within Z_MAX standard errors, none outside the support."""
    COUNT[f"compare_{family}"] += 1
    if abs(sum(exact.values()) - 1) > 1e-9:
        raise Mismatch(f"{family} {what}: exact probabilities sum to {sum(exact.values())}")
    for cell, c in counts.items():
        if exact.get(cell, 0.0) == 0.0:
            raise Mismatch(f"{family} {what}: {c} samples of {cell}, which has probability 0")
    worst = 0.0
    for cell, e in exact.items():
        se = math.sqrt(e * (1 - e) / n) if 0 < e < 1 else 0.0
        z = abs(counts.get(cell, 0) / n - e) / se if se > 0 else (0.0 if counts.get(cell, 0) / n == e else math.inf)
        worst = max(worst, z)
    MAX_Z[family] = max(MAX_Z.get(family, 0.0), worst)
    if worst > Z_MAX:
        raise Mismatch(f"{family} {what}: a cell is {worst:.1f} standard errors from its exact probability")
    # Pearson chi-square over all cells, against its 1 - 1e-6 quantile (Wilson-Hilferty). The per-cell rule
    # alone misses many small shifts that add up; neither can see a cell whose expected count is a few samples.
    cells = [c for c, e in exact.items() if e > 0]
    chi2 = sum((counts.get(c, 0) - n * exact[c]) ** 2 / (n * exact[c]) for c in cells)
    df = max(len(cells) - 1, 1)
    limit = df * (1 - 2 / (9 * df) + 4.75 * math.sqrt(2 / (9 * df))) ** 3
    MAX_CHI2[family] = max(MAX_CHI2.get(family, 0.0), chi2 / limit)
    if chi2 > limit:
        raise Mismatch(f"{family} {what}: chi-square {chi2:.1f} exceeds {limit:.1f} (df={df})")


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ================================================================================================
# transformers: _speculative_sampling, same random numbers
# ================================================================================================


def logits_pair(seed: int, k: int, V: int, scale: float, kind: str):
    g = gen(seed)
    tl = scale * torch.randn(k + 1, V, dtype=DT, generator=g)
    if kind == "same":  # q = p at the draft positions: nothing can be rejected
        dl = tl[:k].clone()
    elif kind == "close":
        dl = tl[:k] + 0.3 * torch.randn(k, V, dtype=DT, generator=g)
    else:
        dl = scale * torch.randn(k, V, dtype=DT, generator=g)
    return tl, dl


def check_hf_sampling(oracle, seed: int, k: int, V: int, scale: float, kind: str) -> None:
    tl, dl = logits_pair(seed, k, V, scale, kind)
    q = impl.probs(dl, 1.0)
    drafts = torch.stack([torch.multinomial(q[i], 1, generator=gen(seed * 7 + i))[0] for i in range(k)])
    torch.manual_seed(seed)
    ref_tokens, ref_n = oracle.hf_speculative_sampling(drafts, dl, tl)
    torch.manual_seed(seed)
    r = impl.sample_round(impl.probs(tl, 1.0), q, drafts)
    tag = f"seed={seed} k={k} V={V} scale={scale} kind={kind}"
    equal("hf_sampling", f"accepted {tag}", r.accepted, ref_n)
    equal("hf_sampling", f"tokens {tag}", r.tokens, ref_tokens)
    COUNT["hf_all_accepted" if r.accepted == k else "hf_rejected_at_0" if r.accepted == 0 else "hf_rejected_later"] += 1


# ================================================================================================
# transformers: assisted generation (greedy) and DynamicCache rollback
# ================================================================================================


def check_hf_assisted(oracle, models, prompt_seed: int, plen: int, k: int, n_new: int) -> None:
    target, draft = models
    prompt = torch.randint(0, target.config.vocab_size, (plen,), generator=gen(prompt_seed)).tolist()
    plain = oracle.hf_greedy(target, prompt, n_new)
    assisted = oracle.hf_greedy(target, prompt, n_new, assistant=draft, k=k)
    t, d = oracle.HFModel(target), oracle.HFModel(draft)
    res = impl.speculative(t, d, prompt, n_new, k, "greedy")
    tag = f"plen={plen} k={k} seed={prompt_seed}"
    equal("hf_assisted", f"assisted generate vs plain generate {tag}", assisted, plain)
    equal("hf_assisted", f"impl speculative vs plain generate {tag}", res.tokens, plain)
    for n in res.rounds:
        COUNT["assisted_full_rounds" if n == k else "assisted_partial_rounds" if n > 0 else "assisted_empty_rounds"] += 1


def check_rollback_model(kind: str, make, V: int, seed: int, t: int, extra: list[int], drop: int, cont: int,
                         calls: int = 1) -> None:
    """Consume t tokens, then `extra` in `calls` forward calls, roll back `drop`, continue with `cont` tokens.

    The continuation's logits and the final state must equal a fresh model fed only the kept tokens.
    """
    g = gen(seed * 31 + t)
    base = torch.randint(0, V, (t,), generator=g).tolist()
    more = torch.randint(0, V, (len(extra),), generator=g).tolist()
    tail = torch.randint(0, V, (cont,), generator=g).tolist()
    m = make()
    m.forward(base)
    step = max(1, len(more) // calls)
    sizes = []
    for i in range(0, len(more), step):
        m.forward(more[i : i + step])
        sizes.append(len(more[i : i + step]))
    m.rollback(drop)
    kept = base + more[: len(more) - drop]
    tag = f"{kind} seed={seed} t={t} extra={len(more)} drop={drop} calls={calls}"
    equal("rollback", f"consumed after rollback {tag}", m.consumed, len(kept))
    got = m.forward(tail)
    fresh = make()
    fresh.forward(kept)
    want = fresh.forward(tail)
    close("rollback", f"continuation logits {tag}", got, want, TOL_ID)
    for name, v in m.state().items():
        close("rollback", f"state {name} {tag}", v, fresh.state()[name], TOL_ID)
    if kind == "mamba" and sizes and drop > sizes[-1]:  # the rollback reaches back past the last call
        COUNT["rollback_mamba_multi_call"] += 1


# ================================================================================================
# vLLM: rejection_sample on CPU
# ================================================================================================


def check_vllm_greedy(oracle, seed: int, k: int, V: int, R: int, with_draft_logits: bool) -> None:
    g = gen(seed)
    tl = torch.randn(R, k + 1, V, generator=g).to(torch.float32).to(DT)
    best = tl.argmax(-1)
    keep = torch.randint(0, k + 1, (R,), generator=g)  # how many drafts agree with the target before a miss
    drafts = best[:, :k].clone()
    for r in range(R):
        if keep[r] < k:
            drafts[r, keep[r]] = (best[r, keep[r]] + 1 + torch.randint(0, V - 1, (1,), generator=g)) % V
            drafts[r, keep[r] + 1 :] = torch.randint(0, V, (k - keep[r] - 1,), generator=g)
    dl = torch.randn(R, k, V, generator=g) if with_draft_logits else None
    sampled, num = oracle.vllm_rejection_sample(tl, dl, drafts, 0.0, torch.arange(R))
    for r in range(R):
        want = impl.verify_greedy(tl[r], drafts[r])
        tag = f"seed={seed} k={k} V={V} row={r} draft_logits={with_draft_logits}"
        equal("vllm_greedy", f"accepted {tag}", int(num[r]) - 1, want.accepted)
        equal("vllm_greedy", f"tokens {tag}", sampled[r, : int(num[r])].tolist(), want.tokens)
        COUNT["vllm_greedy_all" if want.accepted == k else "vllm_greedy_partial"] += 1


def check_vllm_rows(oracle, seed: int, k: int, V: int, R: int) -> None:
    """Sampled verification where each request's outcome is certain, and different per request.

    Rows alternate between q = p (every draft accepted: ratio 1) and a first draft the target gives
    probability 0 (rejected at once). A sampler that mixed up rows, or read one row's draft logits for
    another, would get the accepted counts wrong.
    """
    g = gen(seed)
    tl = torch.randn(R, k + 1, V, generator=g)
    dl = torch.randn(R, k, V, generator=g)
    drafts = torch.randint(0, V, (R, k), generator=g)
    want = []
    for r in range(R):
        if r % 2 == 0:
            dl[r] = tl[r, :k]
            want.append(k)
        else:
            tl[r, 0, drafts[r, 0]] = -1e30
            want.append(0)
    sampled, num = oracle.vllm_rejection_sample(tl, dl, drafts, 1.0, 7 + torch.arange(R))
    for r in range(R):
        tag = f"seed={seed} k={k} V={V} row={r}"
        equal("vllm_rows", f"accepted {tag}", int(num[r]) - 1, want[r])
        equal("vllm_rows", f"kept drafts {tag}", sampled[r, : want[r]].tolist(), drafts[r, : want[r]].tolist())
        if want[r] == 0:
            equal("vllm_rows", f"resampled token avoids the rejected draft {tag}", int(sampled[r, 0]) != int(drafts[r, 0]), True)


def check_vllm_sampled(oracle, seed: int, k: int, V: int, temperature: float, with_draft_logits: bool,
                       n: int) -> None:
    g = gen(seed)
    tl = (1.5 * torch.randn(k + 1, V, generator=g)).to(torch.float32)
    dl = (tl[:k] + 0.8 * torch.randn(k, V, generator=g)).to(torch.float32)
    p = impl.probs(tl.to(DT), temperature)
    if with_draft_logits:
        q = impl.probs(dl.to(DT), temperature)
        drafts = torch.stack([torch.multinomial(q[i], 1, generator=g)[0] for i in range(k)])
    else:  # one-hot drafts: q puts all its mass on the proposed token
        drafts = torch.stack([torch.multinomial(p[i], 1, generator=g)[0] for i in range(k)])
        q = torch.nn.functional.one_hot(drafts, V).to(DT)
    exact = impl.outcome_distribution(p, q, drafts)
    sampled, num = oracle.vllm_rejection_sample(tl.expand(n, -1, -1), dl.expand(n, -1, -1) if with_draft_logits else None,
                                                drafts.expand(n, -1), temperature, 1000 * seed + torch.arange(n))
    counts: Counter = Counter()
    tag = f"seed={seed} k={k} V={V} T={temperature} draft_logits={with_draft_logits} n={n}"
    for r in range(n):
        acc = int(num[r]) - 1
        if sampled[r, :acc].tolist() != drafts[:acc].tolist():
            raise Mismatch(f"vllm_sampled kept prefix {tag} row={r}: {sampled[r, :acc].tolist()} vs drafts")
        counts[(acc, int(sampled[r, acc]))] += 1
    frequencies("vllm_sampled", tag, counts, exact, n)


# ================================================================================================
# Exact enumeration: the theorem
# ================================================================================================


def table_dist(seed: int, V: int, scale: float, zero: set[int] | None = None):
    """A context-dependent distribution: logits drawn from a generator seeded by the context."""
    cache: dict = {}

    def f(ctx: tuple[int, ...]) -> torch.Tensor:
        if ctx not in cache:
            h = hash((seed,) + ctx) & 0x7FFFFFFF
            lg = scale * torch.randn(V, dtype=DT, generator=gen(h))
            if zero:
                lg[list(zero)] = float("-inf")
            cache[ctx] = torch.softmax(lg, -1)
        return cache[ctx]

    return f


def check_theorem(seed: int, V: int, k: int, extra: int, kind: str) -> None:
    p_fn = table_dist(seed, V, 1.5, zero={0} if kind == "zeros" else None)
    if kind == "same":
        q_fn = p_fn
    elif kind == "disjoint":  # q only proposes token 0, which p never emits: every draft is rejected
        q_fn = lambda ctx: torch.nn.functional.one_hot(torch.tensor(0), V).to(DT)
        p_fn = table_dist(seed, V, 1.5, zero={0})
    elif kind == "zeros":  # p gives token 0 probability 0; q does not
        q_fn = table_dist(seed + 1, V, 1.0)
        COUNT["theorem_zero_prob_tokens"] += 1
    else:
        q_fn = table_dist(seed + 1, V, 1.0)
    ctx = (1,)
    L = k + 1 + extra
    spec = impl.extend_with_target(impl.round_distribution(p_fn, q_fn, ctx, k), p_fn, ctx, L)
    want = impl.target_distribution(p_fn, ctx, L)
    keys = sorted(set(spec) | set(want))
    tag = f"seed={seed} V={V} k={k} L={L} kind={kind}"
    close("theorem", tag, torch.tensor([spec.get(s, 0.0) for s in keys], dtype=DT),
          torch.tensor([want.get(s, 0.0) for s in keys], dtype=DT), TOL_EXACT)


# ================================================================================================
# Generation loops on impl's own models
# ================================================================================================


def attn_weights(seed: int, V: int, D: int, heads: int, kv_heads: int, head_dim: int, sharp: float) -> impl.AttnWeights:
    g = gen(seed)
    r = lambda *s, sc=1.0: sc * torch.randn(*s, dtype=DT, generator=g)
    return impl.AttnWeights(emb=r(V, D), wq=r(heads * head_dim, D, sc=0.5), wk=r(kv_heads * head_dim, D, sc=0.5),
                            wv=r(kv_heads * head_dim, D, sc=0.5), wo=r(D, heads * head_dim, sc=0.5),
                            head=r(V, D, sc=sharp / math.sqrt(D)), heads=heads, kv_heads=kv_heads, head_dim=head_dim)


def mamba_weights(seed: int, V: int, D: int, sharp: float) -> impl.MambaLMWeights:
    g = gen(seed)
    r = lambda *s, sc=1.0: sc * torch.randn(*s, dtype=DT, generator=g)
    I, N, K, R = 2 * D, 3, 4, 2
    mixer = impl.hybrid.MambaWeights(in_w=r(2 * I, D, sc=0.5), conv_w=r(I, K, sc=0.5), conv_b=r(I, sc=0.1),
                                     x_w=r(R + 2 * N, I, sc=0.5), dt_w=r(I, R, sc=0.5), dt_b=r(I, sc=0.5) - 1.0,
                                     A_log=torch.log(torch.arange(1, N + 1, dtype=DT)).repeat(I, 1) + r(I, N, sc=0.3),
                                     D=1 + r(I, sc=0.3), out_w=r(D, I, sc=0.5))
    return impl.MambaLMWeights(emb=r(V, D), mixer=mixer, head=r(V, D, sc=sharp / math.sqrt(D)))


def makers(V: int, k: int):
    """(name, target maker, draft maker). Drafts are different, smaller models."""
    ta, da = attn_weights(1, V, 16, 4, 2, 4, 6.0), attn_weights(2, V, 8, 2, 1, 4, 6.0)
    tm, dm = mamba_weights(3, V, 8, 6.0), mamba_weights(4, V, 6, 6.0)
    A = lambda W: (lambda: impl.AttnLM(W))
    M = lambda W: (lambda: impl.MambaLM(W, window=k + 1))
    return [("attn<-attn", A(ta), A(da)), ("mamba<-attn", M(tm), A(da)), ("attn<-mamba", A(ta), M(dm)),
            ("mamba<-mamba", M(tm), M(dm))]


def check_generation_greedy(V: int, k: int, plen: int, n_new: int, seed: int) -> None:
    prompt = torch.randint(0, V, (plen,), generator=gen(seed)).tolist()
    pick = lambda lg: int(lg.argmax())
    for name, T, Dm in makers(V, k):
        want = impl.autoregressive(T(), prompt, n_new, pick)
        t, d = T(), Dm()
        res = impl.speculative(t, d, prompt, n_new, k, "greedy")
        tag = f"{name} k={k} plen={plen} seed={seed}"
        equal("generation", f"greedy speculative vs greedy {tag}", res.tokens, want)
        # On return both models have consumed every returned token but the last, and nothing else.
        for role, m, make in (("target", t, T), ("draft", d, Dm)):
            equal("generation", f"{role} consumed on return {tag}", m.consumed, len(res.tokens) - 1)
            fresh = make()
            fresh.forward(res.tokens[:-1])
            for key, v in m.state().items():
                close("generation", f"{role} state {key} on return {tag}", v, fresh.state()[key], TOL_ID)
        if name.startswith("mamba") or name.endswith("mamba"):
            COUNT["generation_mamba_rounds"] += len(res.rounds)


def check_zero_probability_draft() -> None:
    """A draft the target gives probability 0 is rejected even when the uniform is exactly 0."""
    p = torch.tensor([[0.0, 0.5, 0.5], [0.2, 0.3, 0.5]], dtype=DT)
    q = torch.tensor([[0.6, 0.2, 0.2]], dtype=DT)
    r = impl.verify_sampled(p, q, torch.tensor([0]), torch.zeros(1, dtype=DT), lambda d: int(d.argmax()))
    equal("directed", "draft of target probability 0, uniform 0: accepted count", r.accepted, 0)
    equal("directed", "draft of target probability 0, uniform 0: resampled token is possible", p[0, r.tokens[0]] > 0, True)


def check_mamba_peak_history(window: int, prefill: int) -> None:
    """A long prefill keeps at most window + 1 snapshots at any time, not one per prompt token.

    Measured from outside the model, by object: every MambaState is registered as it is built, and the
    number still alive is counted at each construction and each per-token Mamba step. However the model
    stores or replaces its snapshots, a snapshot that exists is counted. The bound is window + 1 snapshots
    plus the model's current state.
    """
    State = impl.hybrid.MambaState
    refs: list[weakref.ref] = []  # MambaState is an unhashable dataclass, so no WeakSet

    def live() -> int:
        return sum(r() is not None for r in refs)
    seen: list[int] = []

    init = State.__init__

    def register(self, *a, **kw) -> None:
        init(self, *a, **kw)
        refs.append(weakref.ref(self))
        seen.append(live())

    step = impl.hybrid.mamba_forward

    def counted(*a, **kw):
        seen.append(live())
        return step(*a, **kw)

    State.__init__ = register
    impl.hybrid.mamba_forward = counted
    try:
        m = impl.MambaLM(mamba_weights(6, 12, 8, 3.0), window=window)
        m.forward(list(torch.randint(0, 12, (prefill,), generator=gen(prefill)).tolist()))
        seen.append(live())
    finally:
        impl.hybrid.mamba_forward = step
        State.__init__ = init
    equal("directed", f"states seen during a {prefill}-token prefill", len(seen) >= 2 * prefill, True)
    equal("directed", f"peak live states during a {prefill}-token prefill, window {window}", max(seen) <= window + 2,
          True)


def check_bad_inputs() -> None:
    """k = 0 and an empty prompt are refused, not run into an index error."""
    W = attn_weights(1, 8, 8, 2, 1, 4, 1.0)
    for what, call in (("k=0", lambda: impl.speculative(impl.AttnLM(W), impl.AttnLM(W), [1], 3, 0, "greedy")),
                       ("empty prompt", lambda: impl.speculative(impl.AttnLM(W), impl.AttnLM(W), [], 3, 2, "greedy")),
                       ("empty prompt, plain", lambda: impl.autoregressive(impl.AttnLM(W), [], 3, lambda lg: 0))):
        try:
            call()
        except ValueError:
            COUNT["compare_inputs"] += 1
            continue
        raise Mismatch(f"inputs {what}: accepted instead of raising ValueError")


def check_generation_sampled(V: int, k: int, L: int, runs: int, seed: int, temperature: float) -> None:
    """Sampled speculative decoding of L tokens vs the target's exact distribution over all V^L sequences."""
    prompt = [1]
    for name, T, Dm in makers(V, k):
        def p_fn(ctx, T=T):
            return torch.softmax(T().forward(list(ctx))[-1] / temperature, -1)

        want = impl.target_distribution(p_fn, tuple(prompt), L)
        g = gen(seed)
        counts: Counter = Counter()
        for run in range(runs):
            t, d = T(), Dm()
            res = impl.speculative(t, d, prompt, L, k, "sampled", temperature, g)
            counts[tuple(res.tokens[len(prompt) :])] += 1
            # The same invariant as greedy: on return each model holds every returned token but the last.
            excess = sum(a + 1 for a in res.rounds) - L
            if excess > 0:
                COUNT["generation_sampled_excess"] += 1
            for role, m, make in (("target", t, T), ("draft", d, Dm)):
                equal("generation_sampled", f"{role} consumed on return {name} run={run}", m.consumed, len(res.tokens) - 1)
                if run < 100 or excess > 0 and COUNT["generation_sampled_excess"] <= 400:
                    fresh = make()
                    fresh.forward(res.tokens[:-1])
                    for key, v in m.state().items():
                        close("generation_sampled", f"{role} state {key} on return {name} run={run}", v,
                              fresh.state()[key], TOL_ID)
        frequencies("generation_sampled", f"{name} V={V} k={k} L={L} T={temperature} runs={runs}", counts, want, runs)


# ================================================================================================
# Driver
# ================================================================================================

HF_SEEDS = range(40)
HF_KS = (1, 3, 5)
HF_VS = (4, 16, 64)
HF_SHAPES = ((0.5, "random"), (3.0, "random"), (3.0, "close"), (2.0, "same"))
THEOREM = [(s, V, k, extra, kind) for s in range(3) for V in (2, 3) for k in (1, 2, 3) for extra in (0, 1)
           for kind in ("random", "same", "disjoint", "zeros")]
ASSISTED = [(s, plen, k) for s in range(2) for plen in (1, 3, 7) for k in (1, 2, 4, 6)]
GREEDY_GEN = [(k, plen, s) for k in (1, 2, 3, 5) for plen in (1, 4) for s in range(3)]
VLLM_SAMPLED = [(0, 3, 8, 1.0, True), (1, 3, 8, 0.7, True), (2, 2, 8, 1.0, False), (3, 4, 6, 0.8, True)]
VLLM_SAMPLED_N = 4000

EXPECTED_CASES = {
    "hf_sampling": len(HF_SEEDS) * len(HF_KS) * len(HF_VS) * len(HF_SHAPES),
    "theorem": len(THEOREM),
    "hf_assisted": len(ASSISTED),
    "rollback": 0,  # filled in below
    "vllm_greedy": 6,
    "vllm_rows": 2,
    "vllm_sampled": len(VLLM_SAMPLED),
    "generation_greedy": len(GREEDY_GEN),
    "generation_sampled": 2,
    "inputs": 1,
    "directed": 1,
}
ROLLBACK = [(kind, s, t, extra, drop, calls) for kind in ("attn", "mamba", "hf") for s in range(2) for t in (1, 5)
            for extra, drop, calls in ((3, 0, 1), (3, 1, 1), (3, 3, 1), (4, 2, 2), (4, 4, 4), (5, 3, 5),
                                       (4, 3, 2), (5, 5, 5))]
EXPECTED_CASES["rollback"] = len(ROLLBACK)


SECONDS: Counter = Counter()


def run_all(oracle) -> dict:
    cases: Counter = Counter()
    real = cases.__class__.__setitem__

    class Timed(Counter):  # charge the time since the previous case to the family just finished
        last = time.time()

        def __setitem__(self, key, value):
            now = time.time()
            SECONDS[key] += now - Timed.last
            Timed.last = now
            real(self, key, value)

    cases = Timed()
    for seed in HF_SEEDS:
        for k in HF_KS:
            for V in HF_VS:
                for scale, kind in HF_SHAPES:
                    check_hf_sampling(oracle, seed, k, V, scale, kind)
                    cases["hf_sampling"] += 1
    for args in THEOREM:
        check_theorem(*args)
        cases["theorem"] += 1
    hf_models = (oracle.tiny_llama(10, 48, 32, 2, 4, 2), oracle.tiny_llama(10, 48, 32, 2, 4, 2, noise=0.3))
    for s, plen, k in ASSISTED:
        check_hf_assisted(oracle, hf_models, s, plen, k, 16)
        cases["hf_assisted"] += 1
    V = 12
    attn_w, mamba_w = attn_weights(5, V, 16, 4, 2, 4, 3.0), mamba_weights(6, V, 8, 3.0)
    make = {"attn": lambda: impl.AttnLM(attn_w), "mamba": lambda: impl.MambaLM(mamba_w, window=6),
            "hf": lambda: oracle.HFModel(hf_models[0])}
    for kind, s, t, extra, drop, calls in ROLLBACK:
        vocab = 48 if kind == "hf" else V
        check_rollback_model(kind, make[kind], vocab, s, t, list(range(extra)), drop, 3, calls)
        cases["rollback"] += 1
    for seed, (k, V_, R) in enumerate(((1, 16, 40), (3, 16, 60), (5, 64, 60))):
        for with_dl in (True, False):
            check_vllm_greedy(oracle, seed, k, V_, R, with_dl)
            cases["vllm_greedy"] += 1
    for seed, (k, V_, R) in enumerate(((2, 8, 10), (4, 32, 12))):
        check_vllm_rows(oracle, seed, k, V_, R)
        cases["vllm_rows"] += 1
    for args in VLLM_SAMPLED:
        check_vllm_sampled(oracle, *args, VLLM_SAMPLED_N)
        cases["vllm_sampled"] += 1
    for k, plen, s in GREEDY_GEN:
        check_generation_greedy(24, k, plen, 14, s)
        cases["generation_greedy"] += 1
    check_bad_inputs()
    cases["inputs"] += 1
    check_zero_probability_draft()
    check_mamba_peak_history(4, 60)
    cases["directed"] += 1
    for k, L, temp in ((2, 3, 1.0), (3, 4, 0.5)):
        check_generation_sampled(3, k, L, 3000, 100 + k, temp)
        cases["generation_sampled"] += 1
    return dict(cases)


MUTATIONS = {
    "ratio_inverted": "accept with probability q(x)/p(x) instead of p(x)/q(x)",
    "residual_abs": "resample from norm(|p - q|) instead of norm(max(p - q, 0))",
    "residual_is_p": "resample a rejected position from p instead of the residual",
    "bonus_wrong_row": "the bonus token is drawn from p_{k-1} instead of p_k",
    "continue_after_reject": "count every accepted draft instead of stopping at the first rejection",
    "greedy_shifted": "greedy compares draft i with the target's argmax at position i + 1",
    "uniforms_float32": "the acceptance uniforms are drawn in float32",
    "kv_rollback_keeps_one": "attention rollback forgets one token fewer than asked",
    "mamba_rollback_no_restore": "Mamba rollback trims its snapshots but keeps the latest state",
    "draft_drop_off_by_one": "after a rejection the draft forgets k - accepted tokens instead of k - 1 - accepted",
    "temperature_ignored": "probs() ignores the temperature",
    "accept_on_equal": "accept when u <= p/q, which accepts a draft of target probability 0 when u = 0",
    "sampled_skips_final_rollback": "sampled generation trims the excess tokens but does not roll the models back by them",
    "loop_ignores_temperature": "the sampled generation loop runs at temperature 1 whatever it is given",
    "mamba_trim_after_call": "Mamba keeps a snapshot per token of a forward call and trims only after it",
    "mamba_append_before_trim": "Mamba adds a snapshot before trimming, so it briefly holds window + 2",
    "mamba_append_then_slice": "Mamba adds a snapshot, then replaces the list with its last window + 1",
    "enumeration_keeps_q": "the exact round distribution accepts each draft with probability q(x), not min(p, q)",
}


def apply_mutation(name: str) -> None:
    if name == "ratio_inverted":
        impl.accept_ratio = lambda p, q, x: q[torch.arange(x.shape[0]), x] / p[torch.arange(x.shape[0]), x]
    elif name == "residual_abs":
        def f(p, q):
            r = (p - q).abs()
            return r / r.sum(-1, keepdim=True)

        impl.residual = f
    elif name == "residual_is_p":
        impl.residual = lambda p, q: p
    elif name == "bonus_wrong_row":
        orig = impl.verify_sampled

        def f(p, q, draft, uniforms, sample):
            p2 = p.clone()
            p2[-1] = p[-2]
            return orig(p2, q, draft, uniforms, sample)

        impl.verify_sampled = f
    elif name == "continue_after_reject":
        impl.first_rejection = lambda ratio, u: int((u <= ratio).sum())
    elif name == "greedy_shifted":
        def f(target_logits, draft):
            best = target_logits.argmax(dim=-1)
            n = 0
            while n < draft.shape[0] and int(draft[n]) == int(best[n + 1]):
                n += 1
            return impl.Round(n, [int(t) for t in draft[:n]] + [int(best[n])])

        impl.verify_greedy = f
    elif name == "uniforms_float32":
        def f(p, q, draft, gen=None):
            u = torch.rand(draft.shape[0], dtype=torch.float32, generator=gen).to(DT)
            return impl.verify_sampled(p, q, draft, u, lambda d: int(torch.multinomial(d[None], 1, generator=gen)))

        impl.sample_round = f
    elif name == "kv_rollback_keeps_one":
        orig = impl.AttnLM.rollback
        impl.AttnLM.rollback = lambda self, drop: orig(self, max(drop - 1, 0))
    elif name == "mamba_rollback_no_restore":
        def f(self, drop):
            if drop:
                del self.history[len(self.history) - drop :]

        impl.MambaLM.rollback = f
    elif name == "draft_drop_off_by_one":
        impl.rollback_counts = lambda k, n: (k - n, (k - n if n < k else -1))
    elif name == "temperature_ignored":
        impl.probs = lambda logits, temperature: torch.softmax(logits, dim=-1)
    elif name == "accept_on_equal":
        def f(ratio, u):
            rejected = u > ratio
            return int(rejected.int().argmax()) if bool(rejected.any()) else ratio.shape[0]

        impl.first_rejection = f
    elif name == "sampled_skips_final_rollback":
        # impl.speculative with the final rollback limited to greedy mode.
        src = inspect.getsource(impl.speculative)
        old = "        target.rollback(excess)\n        draft.rollback(excess)\n"
        if src.count(old) != 1:
            raise RuntimeError("mutation sampled_skips_final_rollback no longer matches impl.speculative")
        src = src.replace(old, "        if mode == \"greedy\":\n            target.rollback(excess)\n            draft.rollback(excess)\n")
        exec(compile(src, impl.__file__, "exec"), impl.__dict__)
    elif name == "loop_ignores_temperature":
        orig = impl.speculative
        impl.speculative = lambda t, d, prompt, n_new, k, mode, temperature=1.0, gen=None: orig(
            t, d, prompt, n_new, k, mode, 1.0, gen)
    elif name == "mamba_trim_after_call":
        def fwd(self, tokens):
            outs = []
            for t in tokens:
                x = self.W.emb[torch.tensor([[t]])]
                y = impl.hybrid.mamba_forward(self.W.mixer, x, self.st)
                outs.append((x + y)[0, 0] @ self.W.head.T)
                self.history.append(self.st.snapshot())
            del self.history[: max(0, len(self.history) - (self.window + 1))]
            return torch.stack(outs)

        impl.MambaLM.forward = fwd
    elif name == "mamba_append_then_slice":
        def fwd(self, tokens):
            outs = []
            for t in tokens:
                x = self.W.emb[torch.tensor([[t]])]
                y = impl.hybrid.mamba_forward(self.W.mixer, x, self.st)
                outs.append((x + y)[0, 0] @ self.W.head.T)
                self.history.append(self.st.snapshot())
                self.history = self.history[-(self.window + 1) :]
            return torch.stack(outs)

        impl.MambaLM.forward = fwd
    elif name == "mamba_append_before_trim":
        def fwd(self, tokens):
            outs = []
            for t in tokens:
                x = self.W.emb[torch.tensor([[t]])]
                y = impl.hybrid.mamba_forward(self.W.mixer, x, self.st)
                outs.append((x + y)[0, 0] @ self.W.head.T)
                self.history.append(self.st.snapshot())
                if len(self.history) > self.window + 1:
                    del self.history[0]
            return torch.stack(outs)

        impl.MambaLM.forward = fwd
    elif name == "enumeration_keeps_q":
        orig_min = torch.minimum

        def rd(p_fn, q_fn, ctx, k):
            torch.minimum = lambda a, b: b
            try:
                return orig_rd(p_fn, q_fn, ctx, k)
            finally:
                torch.minimum = orig_min

        orig_rd = impl.round_distribution
        impl.round_distribution = rd
    else:
        sys.exit(f"unknown mutation {name}; choose from {sorted(MUTATIONS)}")


def self_test(out: str | None) -> int:
    base = subprocess.run([sys.executable, __file__], capture_output=True, text=True)
    if base.returncode != 0:  # a failing baseline would make every mutation look caught
        print(json.dumps({"check": "03 self-test", "passed": False,
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
    result = {"check": "03 self-test: every planted bug must be reported as a mismatch",
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
    import transformers

    import oracle

    oracle.check_version()
    t0 = time.time()
    failure = error = None
    cases: dict = {}
    try:
        cases = run_all(oracle)
    except Mismatch as e:
        failure = str(e)
    except Exception as e:  # a crash is a failure too, but not a caught mismatch
        frames = traceback.extract_tb(e.__traceback__)[-3:]
        where = " <- ".join(f"{Path(f.filename).name}:{f.lineno}" for f in reversed(frames))
        error = f"{type(e).__name__}: {e} at {where}"

    coverage = {k: COUNT.get(k, 0) for k in MIN_COVERAGE}
    short = [k for k, v in coverage.items() if v < MIN_COVERAGE[k]]
    missing = {k: (cases.get(k, 0), v) for k, v in EXPECTED_CASES.items() if cases.get(k, 0) != v}
    passed = failure is None and error is None and not short and not missing
    import vllm
    import vllm._version

    result = {
        "check": "03-speculative-decoding vs transformers (_speculative_sampling, assisted generation, DynamicCache) "
                 "and vLLM (rejection_sample under the Triton interpreter)",
        "transformers": transformers.__version__,
        "vllm": [vllm.__version__, vllm._version.__commit_id__],
        "torch": torch.__version__,
        "python": platform.python_version(),
        "mutation": args.mutate,
        "tolerances": {"impl_identity": {"atol": TOL_ID[0], "rtol": TOL_ID[1]},
                       "enumeration": {"atol": TOL_EXACT[0], "rtol": TOL_EXACT[1]}, "z_max": Z_MAX},
        "cases": cases,
        "expected_cases": EXPECTED_CASES,
        "comparisons": {k: v for k, v in sorted(COUNT.items()) if k.startswith("compare_")},
        "max_error_over_tolerance": {k: float(f"{v:.3g}") for k, v in sorted(MAX_ERR.items())},
        "max_standard_errors": {k: float(f"{v:.3g}") for k, v in sorted(MAX_Z.items())},
        "seconds_per_family": {k: round(v, 1) for k, v in SECONDS.items()},
        "max_chi_square_over_limit": {k: float(f"{v:.3g}") for k, v in sorted(MAX_CHI2.items())},
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

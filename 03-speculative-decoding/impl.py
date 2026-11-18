"""Speculative decoding from scratch: draft and verify, the rejection rule that keeps the target
distribution, and rolling back KV caches and SSM states after a rejection.

Float64, torch only; no transformers or vLLM import. The Mamba mixer is the one from 02, already
checked against transformers there.

Notation (Leviathan et al., Algorithm 1; Chen et al., Algorithm 2):
    p_i = target distribution at draft position i      (i = 0 .. k, one more than the drafts)
    q_i = draft distribution at draft position i       (i = 0 .. k-1)
    x_i = the i-th draft token, sampled from q_i
    accept x_i with probability min(1, p_i(x_i) / q_i(x_i)), in order, stopping at the first rejection
    on a rejection at i: emit one token from norm(max(p_i - q_i, 0))
    if all k drafts are accepted: emit one more token from p_k (the "bonus" token)
"""
from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import torch

DT = torch.float64


def _load_hybrid():
    """02-hybrid-state/impl.py under its own module name, so it does not shadow this file."""
    path = Path(__file__).resolve().parent.parent / "02-hybrid-state" / "impl.py"
    spec = importlib.util.spec_from_file_location("hybrid_impl", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hybrid_impl"] = mod  # dataclasses look their module up while it executes
    spec.loader.exec_module(mod)
    return mod


hybrid = _load_hybrid()


# ================================================================================================
# The verification rule
# ================================================================================================


def probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=-1)


def residual(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """norm(max(p - q, 0)), the distribution a rejected position is resampled from.

    Its unnormalised mass, sum max(p - q, 0), equals the probability of a rejection at this position
    (the total variation distance between p and q). It is zero only if p == q, and then nothing is rejected.
    """
    r = (p - q).clamp_min(0)
    return r / r.sum(dim=-1, keepdim=True)


def accept_ratio(p: torch.Tensor, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """p(x) / q(x) for each draft position; accept when a uniform draw is <= this."""
    idx = torch.arange(x.shape[0])
    return p[idx, x] / q[idx, x]


@dataclass
class Round:
    accepted: int  # how many drafts were kept, 0 .. k
    tokens: list[int]  # the kept drafts plus one token: a resampled one or the bonus one


def verify_greedy(target_logits: torch.Tensor, draft: torch.Tensor) -> Round:
    """Greedy verification: keep drafts while each equals the target's argmax, then emit the argmax.

    target_logits [k+1, V] are the target's outputs at the k draft positions and one past them.
    """
    best = target_logits.argmax(dim=-1)
    n = 0
    while n < draft.shape[0] and int(draft[n]) == int(best[n]):
        n += 1
    return Round(n, [int(t) for t in draft[:n]] + [int(best[n])])


def first_rejection(ratio: torch.Tensor, uniforms: torch.Tensor) -> int:
    """Index of the first draft with uniform > ratio, or k if every draft is accepted."""
    rejected = uniforms > ratio
    return int(rejected.int().argmax()) if bool(rejected.any()) else ratio.shape[0]


def verify_sampled(p: torch.Tensor, q: torch.Tensor, draft: torch.Tensor, uniforms: torch.Tensor,
                   sample: Callable[[torch.Tensor], int]) -> Round:
    """Sampled verification (Leviathan et al. Algorithm 1). p [k+1, V], q [k, V], draft [k], uniforms [k].

    All k uniforms are drawn before any token is resampled; `sample` draws one token from a distribution.
    """
    n = first_rejection(accept_ratio(p[:-1], q, draft), uniforms)
    nxt = sample(residual(p[n], q[n])) if n < draft.shape[0] else sample(p[n])
    return Round(n, [int(t) for t in draft[:n]] + [nxt])


def sample_round(p: torch.Tensor, q: torch.Tensor, draft: torch.Tensor,
                 gen: torch.Generator | None = None) -> Round:
    """verify_sampled with its randomness: k float64 uniforms first, then one multinomial draw.

    This is the order transformers' _speculative_sampling uses, so with the same seed both pick the same
    tokens. gen=None draws from torch's global generator, as transformers does.
    """
    u = torch.rand(draft.shape[0], dtype=DT, generator=gen)
    return verify_sampled(p, q, draft, u, lambda d: int(torch.multinomial(d[None], 1, generator=gen)))


# ================================================================================================
# Exact distribution of one round, by enumeration
# ================================================================================================

Dist = Callable[[tuple[int, ...]], torch.Tensor]  # context -> distribution over the vocabulary


def round_distribution(p_fn: Dist, q_fn: Dist, ctx: tuple[int, ...], k: int) -> dict[tuple[int, ...], float]:
    """Probability of every token sequence one speculative round can emit after `ctx`.

    Drafts are sampled from q and verified against p; the target and draft can depend on the whole
    context, so a later draft position sees the accepted drafts before it.
    """
    out: dict[tuple[int, ...], float] = {}

    def go(emitted: tuple[int, ...], i: int, mass: float) -> None:
        here = ctx + emitted
        p = p_fn(here)
        if i == k:  # every draft accepted: one bonus token from the target
            for t in range(p.shape[0]):
                if p[t] > 0:
                    out[emitted + (t,)] = out.get(emitted + (t,), 0.0) + mass * float(p[t])
            return
        q = q_fn(here)
        keep = torch.minimum(p, q)  # q(x) * min(1, p(x)/q(x)) = min(p(x), q(x)) for every draft token x
        for x in range(q.shape[0]):
            if keep[x] > 0:
                go(emitted + (x,), i + 1, mass * float(keep[x]))
        reject = float(1 - keep.sum())  # = sum max(p - q, 0)
        if reject > 0:
            r = residual(p, q)
            for t in range(r.shape[0]):
                if r[t] > 0:
                    out[emitted + (t,)] = out.get(emitted + (t,), 0.0) + mass * reject * float(r[t])

    go((), 0, 1.0)
    return out


def extend_with_target(dist: dict[tuple[int, ...], float], p_fn: Dist, ctx: tuple[int, ...],
                       length: int) -> dict[tuple[int, ...], float]:
    """Continue every sequence with target samples up to `length` tokens, exactly."""
    out: dict[tuple[int, ...], float] = {}
    for seq, mass in dist.items():
        frontier = {seq: mass}
        for _ in range(length - len(seq)):
            nxt: dict[tuple[int, ...], float] = {}
            for s, m in frontier.items():
                p = p_fn(ctx + s)
                for t in range(p.shape[0]):
                    if p[t] > 0:
                        nxt[s + (t,)] = nxt.get(s + (t,), 0.0) + m * float(p[t])
            frontier = nxt
        for s, m in frontier.items():
            out[s] = out.get(s, 0.0) + m
    return out


def target_distribution(p_fn: Dist, ctx: tuple[int, ...], length: int) -> dict[tuple[int, ...], float]:
    return extend_with_target({(): 1.0}, p_fn, ctx, length)


def outcome_distribution(p: torch.Tensor, q: torch.Tensor, draft: torch.Tensor) -> dict[tuple[int, int], float]:
    """For fixed drafts: probability of (accepted count n, emitted token) under sampled verification."""
    k = draft.shape[0]
    ratio = accept_ratio(p[:-1], q, draft).clamp(max=1.0)
    out: dict[tuple[int, int], float] = {}
    alive = 1.0
    for i in range(k):
        rej = alive * (1 - float(ratio[i]))
        r = residual(p[i], q[i]) if rej > 0 else None
        for t in range(p.shape[1]):
            if r is not None and r[t] > 0:
                out[(i, t)] = rej * float(r[t])
        alive *= float(ratio[i])
    for t in range(p.shape[1]):
        if p[k, t] > 0 and alive > 0:
            out[(k, t)] = alive * float(p[k, t])
    return out


# ================================================================================================
# Models with a cache that can be rolled back
# ================================================================================================


class Model(Protocol):
    def forward(self, tokens: list[int]) -> torch.Tensor: ...  # logits [len(tokens), V] for new tokens

    def rollback(self, drop: int) -> None: ...  # forget the last `drop` consumed tokens

    @property
    def consumed(self) -> int: ...


@dataclass
class AttnWeights:
    emb: torch.Tensor  # [V, D]
    wq: torch.Tensor  # [H*d, D]
    wk: torch.Tensor  # [Hkv*d, D]
    wv: torch.Tensor  # [Hkv*d, D]
    wo: torch.Tensor  # [D, H*d]
    head: torch.Tensor  # [V, D]
    heads: int
    kv_heads: int
    head_dim: int
    rope_base: float = 10000.0


def rope(x: torch.Tensor, pos: torch.Tensor, base: float) -> torch.Tensor:
    """Rotate pairs (i, i + d/2) of x [H, L, d] by angle pos * base^(-2i/d)."""
    d = x.shape[-1]
    inv = base ** (-torch.arange(0, d // 2, dtype=DT) * 2 / d)
    ang = pos[:, None].to(DT) * inv[None, :]
    cos, sin = ang.cos(), ang.sin()
    a, b = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class AttnLM:
    """One causal attention layer with a residual, between an embedding and an output head. KV cached."""

    def __init__(self, W: AttnWeights):
        self.W = W
        self.keys = torch.zeros(W.kv_heads, 0, W.head_dim, dtype=DT)
        self.values = torch.zeros(W.kv_heads, 0, W.head_dim, dtype=DT)

    @property
    def consumed(self) -> int:
        return self.keys.shape[1]

    def forward(self, tokens: list[int]) -> torch.Tensor:
        W, L, n = self.W, len(tokens), self.consumed
        x = W.emb[torch.tensor(tokens)]
        pos = torch.arange(n, n + L)
        q = rope((x @ W.wq.T).view(L, W.heads, W.head_dim).transpose(0, 1), pos, W.rope_base)
        k = rope((x @ W.wk.T).view(L, W.kv_heads, W.head_dim).transpose(0, 1), pos, W.rope_base)
        v = (x @ W.wv.T).view(L, W.kv_heads, W.head_dim).transpose(0, 1)
        self.keys = torch.cat([self.keys, k], dim=1)
        self.values = torch.cat([self.values, v], dim=1)
        rep = W.heads // W.kv_heads
        kk, vv = self.keys.repeat_interleave(rep, 0), self.values.repeat_interleave(rep, 0)
        scores = q @ kk.transpose(1, 2) / math.sqrt(W.head_dim)
        allowed = torch.arange(n + L)[None, :] <= pos[:, None]
        scores = scores.masked_fill(~allowed, float("-inf"))
        attn = torch.softmax(scores, dim=-1) @ vv  # [H, L, d]
        h = x + attn.transpose(0, 1).reshape(L, -1) @ W.wo.T
        return h @ W.head.T

    def rollback(self, drop: int) -> None:
        """A KV cache rolls back by truncation: entry t depends only on tokens <= t."""
        keep = self.consumed - drop
        self.keys = self.keys[:, :keep].clone()
        self.values = self.values[:, :keep].clone()

    def state(self) -> dict[str, torch.Tensor]:
        return {"keys": self.keys, "values": self.values}


@dataclass
class MambaLMWeights:
    emb: torch.Tensor  # [V, D]
    mixer: "hybrid.MambaWeights"
    head: torch.Tensor  # [V, D]


class MambaLM:
    """Embedding, one Mamba mixer with a residual, output head.

    The state is fixed-size and cannot be truncated: it sums over every past token. So rollback restores a
    snapshot. The model keeps a snapshot after each of its last `window` tokens (and the one before
    them); speculative decoding never drops more than k tokens, so window = k + 1 suffices.
    """

    def __init__(self, W: MambaLMWeights, window: int):
        self.W, self.window = W, window
        self.st = hybrid.mamba_new_state(W.mixer, 1)
        self.history = [self.st.snapshot()]  # history[-1 - j] = state before the last j tokens

    @property
    def consumed(self) -> int:
        return self.st.consumed

    def forward(self, tokens: list[int]) -> torch.Tensor:
        outs = []
        for t in tokens:  # one token at a time, so a snapshot exists after each one
            x = self.W.emb[torch.tensor([[t]])]
            y = hybrid.mamba_forward(self.W.mixer, x, self.st)
            outs.append((x + y)[0, 0] @ self.W.head.T)
            self.history.append(self.st.snapshot())
        del self.history[: max(0, len(self.history) - (self.window + 1))]
        return torch.stack(outs)

    def rollback(self, drop: int) -> None:
        """Restore the state as it was `drop` tokens ago."""
        if drop == 0:
            return
        if drop >= len(self.history):
            raise ValueError(f"cannot drop {drop} tokens; snapshots cover the last {len(self.history) - 1}")
        del self.history[len(self.history) - drop :]
        self.st = self.history[-1].snapshot()

    def state(self) -> dict[str, torch.Tensor]:
        return self.st.tensors()


# ================================================================================================
# The generation loop
# ================================================================================================


@dataclass
class Generation:
    tokens: list[int]  # prompt + generated
    rounds: list[int] = field(default_factory=list)  # accepted drafts per round


def autoregressive(model: Model, prompt: list[int], n_new: int, pick: Callable[[torch.Tensor], int]) -> list[int]:
    """Plain decoding: one target forward per token. pick(logits [V]) -> token."""
    if not prompt:
        raise ValueError("need a non-empty prompt")
    out = list(prompt)
    logits = model.forward(prompt)[-1]
    for _ in range(n_new):
        out.append(pick(logits))
        if len(out) - len(prompt) < n_new:
            logits = model.forward([out[-1]])[-1]
    return out


def rollback_counts(k: int, accepted: int) -> tuple[int, int]:
    """Tokens each model forgets after a round: (target, draft); a draft count of -1 means the draft
    must instead consume the last draft token, because all k were accepted."""
    return k - accepted, (k - 1 - accepted if accepted < k else -1)


def speculative(target: Model, draft: Model, prompt: list[int], n_new: int, k: int, mode: str,
                temperature: float = 1.0, gen: torch.Generator | None = None) -> Generation:
    """Speculative decoding with draft length k. mode "greedy" or "sampled".

    Invariant at the start of each round: both models have consumed every emitted token except the
    newest one, `last`. A round
      1. drafts k tokens: the draft model consumes last, x_0 .. x_{k-2}  (k forwards, one token each)
      2. verifies: the target consumes last, x_0 .. x_{k-1} in one forward, giving p_0 .. p_k
      3. keeps n drafts and one more token, then rolls both models back to the invariant.
    The target rolls back k - n tokens (the rejected drafts). The draft consumed last and x_0 .. x_{k-2};
    it keeps last and the accepted x_0 .. x_{n-1}, so it rolls back k - 1 - n tokens, or consumes
    x_{k-1} when all k were accepted.
    """
    if k < 1 or not prompt or n_new < 0:
        raise ValueError(f"need k >= 1, a non-empty prompt and n_new >= 0 (k={k}, prompt={len(prompt)}, n_new={n_new})")
    out = list(prompt)
    if len(prompt) > 1:
        target.forward(prompt[:-1])
        draft.forward(prompt[:-1])
    res = Generation(out)
    while len(out) - len(prompt) < n_new:
        last = out[-1]
        drafts, qs, tok = [], [], last
        for _ in range(k):
            dl = draft.forward([tok])[-1]
            if mode == "greedy":
                tok = int(dl.argmax())
            else:
                q = probs(dl, temperature)
                tok = int(torch.multinomial(q, 1, generator=gen))
                qs.append(q)
            drafts.append(tok)
        tl = target.forward([last] + drafts)  # [k+1, V]: p_0 .. p_k
        d = torch.tensor(drafts)
        if mode == "greedy":
            r = verify_greedy(tl, d)
        else:
            r = sample_round(probs(tl, temperature), torch.stack(qs), d, gen)
        res.rounds.append(r.accepted)
        target_drop, draft_drop = rollback_counts(k, r.accepted)
        target.rollback(target_drop)
        if draft_drop < 0:
            draft.forward([drafts[-1]])
        else:
            draft.rollback(draft_drop)
        out.extend(r.tokens)
    # The last round can emit up to k tokens too many. Drop them and roll both models back by the same
    # amount, so the invariant holds on return: each has consumed every returned token but the last.
    excess = len(out) - (len(prompt) + n_new)
    if excess > 0:
        del out[len(prompt) + n_new :]
        target.rollback(excess)
        draft.rollback(excess)
    return res


def enumerate_sequences(V: int, length: int):
    return itertools.product(range(V), repeat=length)

"""Check impl against Hugging Face transformers, and against itself where the reference cannot go.

Reference comparisons (tolerance TOL_REF, because the reference casts softmax and scan steps to float32):
- window:     impl.WindowKV vs a one-layer Mistral with sliding_window=w and DynamicCache. Per chunk: the
              attention output, the retained keys/values (shape and values), the cache's token count.
- longformer: impl.sliding_attention(causal=False) vs LongformerSelfAttention, local attention only.
- mamba:      impl.mamba_forward vs MambaMixer with its cache. Per chunk: output, SSM state, conv history.
              The reference can continue a cache only one token at a time (a multi-token continuation
              restarts its scan from a zero state), so its schedules are a prefill then single tokens.
- mamba2:     impl.mamba2_forward, both token-by-token and chunked SSD, vs Mamba2Mixer with its cache,
              including multi-token continuations, two group counts and two chunk sizes.
- ssd core:   impl.ssd_chunked with a finite dt limit vs the library's mamba2_chunk_scan, from a nonzero
              initial state, with limits chosen so both bounds actually clamp.
- vllm norm:  impl.gated_rms_norm with groups = n_groups vs vLLM's Mixer2RMSNormGated.forward_native.

Diagnostics (recorded, not pass/fail): the library's one-token Mamba-2 path ignores a finite dt limit,
which impl applies on every path; and the library's whole-width gated norm vs vLLM's per-group one.

Identities inside impl (tolerance TOL_ID, float64 end to end):
- any chunk partition of the input gives the same outputs and final state as one call;
- chunked SSD equals token-by-token SSD;
- dense masked window attention equals the rolling WindowKV;
- checkpoint and restore: snapshot after t tokens, advance the live state, resume two different suffixes
  from the snapshot; each must equal a fresh run over prefix + suffix, and the snapshot must be untouched.

State sizes: every impl state's payload bytes must equal impl.expected_bytes, and the reference cache
tensors must have the sizes the README states (the library keeps K conv columns, impl keeps K - 1).

A pass also needs every case in the matrix to run and the coverage counters to reach their minimums.
Snapshot isolation is checked bit for bit, and token counters exactly.

`--mutate NAME` plants one bug in impl; `--self-test` runs every mutation in a subprocess and passes only
if each is reported as a mismatch (not a crash, not a pass).

Usage:  python check.py [--mutate NAME] [--out results/check.json]
        python check.py --self-test [--out results/self_test.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import platform
import random
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
TOL_REF = (2e-6, 2e-5)  # (atol, rtol) against the reference; the largest measured error is about 1e-6
TOL_ID = (1e-10, 1e-9)  # impl against itself, float64 throughout

SEEDS = (0, 1, 2)
BATCHES = (1, 2)
LENGTHS = (1, 3, 4, 5, 7, 8, 9, 17, 33)
WINDOWS = (2, 4, 8)
ONE_SIDED = (2, 3, 4)  # one_sided=1 (attention_window=2) fails inside the library's chunked form
GROUPS = (1, 2)
CHUNKS = (4, 8)

MIN_COVERAGE = {
    "window_wraps": 50,  # decode steps taken after the window was already full
    "ssd_multi_chunk_calls": 50,  # chunked SSD calls with more than one chunk
    "ssd_short_last_chunk": 50,  # the last chunk shorter than chunk_size
    "ssd_nonzero_carry_in": 50,  # a chunked call that started from a nonzero state
    "dt_clamped_low": 20,  # dt values the finite limit actually raised
    "dt_clamped_high": 20,  # dt values the finite limit actually lowered
    "two_group_cases": 20,
    "resumed_suffixes": 50,  # suffixes resumed from a snapshot (not the live state)
    "mixer_dt_clamped_low": 20,
    "mixer_dt_clamped_high": 20,
    "window_reference_continuations": 50,  # a multi-token chunk after the first, run through Mistral
    "mamba2_reference_continuations": 50,  # the same, run through Mamba2Mixer
    "impl_continuations": 50,  # the same, impl-only partition checks
}


class Mismatch(Exception):
    pass


COUNT: Counter = Counter()
MAX_ERR: dict[str, float] = {}  # family -> max of |a - b| / (atol + rtol |b|); must stay <= 1


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
        diff = (got - ref).abs().max().item()
        raise Mismatch(f"{family} {what}: max abs diff {diff:.3e} exceeds atol={atol} rtol={rtol}")


def identical(family: str, what: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    """Bit-for-bit equality (shape, dtype, every element)."""
    COUNT[f"compare_{family}"] += 1
    if got.shape != ref.shape or got.dtype != ref.dtype or not torch.equal(got, ref):
        n = int((got != ref).sum()) if got.shape == ref.shape else -1
        raise Mismatch(f"{family} {what}: not identical ({n} elements differ)")


def compact(what: str, tensors: dict[str, torch.Tensor]) -> None:
    """Each state tensor owns exactly its payload: no larger buffer kept alive behind a view."""
    for k, t in tensors.items():
        equal("sizes", f"storage of {k} {what}", t.untyped_storage().nbytes(), t.numel() * t.element_size())


def equal(family: str, what: str, a, b) -> None:
    COUNT[f"compare_{family}"] += 1
    if a != b:
        raise Mismatch(f"{family} {what}: {a} != {b}")


def partitions(T: int, rng: random.Random, prefill_then_ones: bool = True, irregular: bool = True) -> list[list[int]]:
    """Ways to feed T tokens: all at once, half then one at a time, and a random irregular split."""
    out = [[T]]
    if prefill_then_ones and T > 1:
        p = (T + 1) // 2
        out.append([p] + [1] * (T - p))
    if irregular and T > 2:
        parts, left = [], T
        while left:
            n = rng.randint(1, min(left, 6))
            parts.append(n)
            left -= n
        out.append(parts)
    return out


def note_continuations(sched: list[int], counter: str) -> None:
    COUNT[counter] += sum(1 for n in sched[1:] if n > 1)


# ================================================================================================
# Sliding window
# ================================================================================================


def check_window(oracle, seed: int, w: int, B: int, T: int, rng: random.Random) -> None:
    ref = oracle.MistralRef(w, seed)
    ids = torch.randint(0, 32, (B, T), generator=torch.Generator().manual_seed(seed * 1000 + T))
    for sched in partitions(T, rng):
        note_continuations(sched, "window_reference_continuations")
        steps = ref.run(ids, sched)
        kv = impl.WindowKV(w)
        qs, ks, vs, outs = [], [], [], []
        for i, s in enumerate(steps):
            q, k, v = ref.qkv(s["hidden"], s["cos"], s["sin"])
            if kv.consumed >= w and sched[i] == 1:
                COUNT["window_wraps"] += 1
            a = kv.forward(q, k, v)
            tag = f"seed={seed} w={w} B={B} T={T} sched={sched} chunk={i}"
            close("window", f"output {tag}", ref.o_proj(a), s["out"], TOL_REF)
            close("window_state", f"retained keys {tag}", kv.keys, s["keys"], TOL_ID)
            close("window_state", f"retained values {tag}", kv.values, s["values"], TOL_ID)
            equal("window_state", f"consumed {tag}", kv.consumed, s["seq_length"])
            compact(tag, kv.tensors())
            equal("sizes", f"window bytes {tag}", impl.state_bytes(kv.tensors()),
                  impl.expected_bytes("window", batch=B, elem=8, consumed=kv.consumed, w=w,
                                      kv_heads=ref.kv_heads, head_dim=ref.head_dim))
            equal("sizes", f"reference window numel {tag}", s["keys"].numel(),
                  B * ref.kv_heads * ref.head_dim * min(kv.consumed, w - 1))
            qs.append(q), ks.append(k), vs.append(v), outs.append(a)
        dense = impl.sliding_attention(torch.cat(qs, 2), torch.cat(ks, 2), torch.cat(vs, 2), w, causal=True)
        close("window_identity", f"dense vs rolling seed={seed} w={w} T={T} sched={sched}", torch.cat(outs, 2),
              dense, TOL_ID)


def check_longformer(oracle, seed: int, r: int, B: int, blocks: int) -> None:
    ref = oracle.LongformerRef(r, seed)
    L = 2 * r * blocks
    h = torch.randn(B, L, 16, dtype=DT, generator=torch.Generator().manual_seed(seed * 7 + L))
    out = ref.run(h)
    q, k, v = ref.qkv(h)
    a = impl.sliding_attention(q, k, v, r, causal=False)
    close("longformer", f"seed={seed} r={r} B={B} L={L}", a.transpose(1, 2).reshape(B, L, -1), out, TOL_REF)


# ================================================================================================
# Mamba and Mamba-2
# ================================================================================================


def run_impl(kind: str, W, x: torch.Tensor, sched: list[int], st=None, chunk_size=None):
    if st is None:
        st = impl.mamba_new_state(W, x.shape[0]) if kind == "mamba" else impl.mamba2_new_state(W, x.shape[0])
    outs, pos = [], 0
    for n in sched:
        piece = x[:, pos : pos + n]
        if kind == "mamba":
            outs.append(impl.mamba_forward(W, piece, st))
        else:
            if chunk_size is not None:
                COUNT["ssd_multi_chunk_calls"] += n > chunk_size
                COUNT["ssd_short_last_chunk"] += n % chunk_size != 0
                COUNT["ssd_nonzero_carry_in"] += bool(st.ssm.abs().max() > 0)
            outs.append(impl.mamba2_forward(W, piece, st, chunk_size=chunk_size))
        pos += n
    return outs, st


def check_mamba(oracle, seed: int, B: int, T: int) -> None:
    ref = oracle.MambaRef(seed)
    W = ref.weights()
    I, K = W.conv_w.shape
    N = W.A_log.shape[1]
    x = torch.randn(B, T, 16, dtype=DT, generator=torch.Generator().manual_seed(seed * 31 + T))
    for sched in partitions(T, random.Random(0), irregular=False):
        steps = ref.run(x, sched)
        st = impl.mamba_new_state(W, B)
        pos = 0
        for i, (n, s) in enumerate(zip(sched, steps)):
            y = impl.mamba_forward(W, x[:, pos : pos + n], st)
            pos += n
            tag = f"seed={seed} B={B} T={T} sched={sched} chunk={i}"
            close("mamba", f"output {tag}", y, s["out"], TOL_REF)
            close("mamba", f"ssm state {tag}", st.ssm, s["ssm"], TOL_REF)
            # The library keeps K conv columns; the oldest one is never read again.
            close("mamba_state", f"conv history {tag}", st.conv, s["conv"][..., 1:], TOL_ID)
            equal("mamba_state", f"consumed {tag}", st.consumed, pos)
            compact(tag, st.tensors())
            equal("sizes", f"mamba bytes {tag}", impl.state_bytes(st.tensors()),
                  impl.expected_bytes("mamba", batch=B, elem=8, I=I, K=K, N=N))
            equal("sizes", f"reference mamba numel {tag}", (s["conv"].numel(), s["ssm"].numel()),
                  (B * I * K, B * I * N))


def check_mamba2(oracle, seed: int, G: int, cs: int, B: int, T: int, rng: random.Random) -> None:
    ref = oracle.Mamba2Ref(seed, G, cs)
    W = ref.weights()
    H, P, N = W.num_heads, W.head_dim, W.state_size
    C, K = W.conv_w.shape
    COUNT["two_group_cases"] += G == 2
    x = torch.randn(B, T, 16, dtype=DT, generator=torch.Generator().manual_seed(seed * 37 + T))
    for sched in partitions(T, rng):
        note_continuations(sched, "mamba2_reference_continuations")
        steps = ref.run(x, sched)
        for mode in (None, cs):
            st = impl.mamba2_new_state(W, B)
            pos = 0
            for i, (n, s) in enumerate(zip(sched, steps)):
                if mode is not None:
                    COUNT["ssd_multi_chunk_calls"] += n > cs
                    COUNT["ssd_short_last_chunk"] += n % cs != 0
                    COUNT["ssd_nonzero_carry_in"] += bool(st.ssm.abs().max() > 0)
                y = impl.mamba2_forward(W, x[:, pos : pos + n], st, chunk_size=mode)
                pos += n
                tag = f"seed={seed} G={G} cs={cs} B={B} T={T} sched={sched} chunk={i} mode={mode}"
                close("mamba2", f"output {tag}", y, s["out"], TOL_REF)
                close("mamba2", f"ssm state {tag}", st.ssm, s["ssm"], TOL_REF)
                close("mamba2_state", f"conv history {tag}", st.conv, s["conv"][..., 1:], TOL_ID)
                equal("mamba2_state", f"consumed {tag}", st.consumed, pos)
                compact(tag, st.tensors())
                equal("sizes", f"mamba2 bytes {tag}", impl.state_bytes(st.tensors()),
                      impl.expected_bytes("mamba2", batch=B, elem=8, H=H, P=P, G=G, N=N, K=K))
                equal("sizes", f"reference mamba2 numel {tag}", (s["conv"].numel(), s["ssm"].numel()),
                      (B * C * K, B * H * P * N))


def check_ssd_core(oracle, seed: int, G: int, cs: int, T: int) -> None:
    """Finite dt limits, nonzero initial state, the SSD core alone, against the library's chunk scan."""
    g = torch.Generator().manual_seed(seed * 101 + T * 7 + G)
    Bsz, H, P, N = 2, 4, 8, 3
    x = torch.randn(Bsz, T, H, P, dtype=DT, generator=g)
    dt_raw = torch.randn(Bsz, T, H, dtype=DT, generator=g)
    dt_bias = 0.5 * torch.randn(H, dtype=DT, generator=g)
    A = -torch.exp(0.5 * torch.randn(H, dtype=DT, generator=g))
    Bg = torch.randn(Bsz, T, G, N, dtype=DT, generator=g)
    Cg = torch.randn(Bsz, T, G, N, dtype=DT, generator=g)
    D = torch.randn(H, dtype=DT, generator=g)
    S0 = torch.randn(Bsz, H, P, N, dtype=DT, generator=g)
    soft = torch.nn.functional.softplus(dt_raw + dt_bias)
    lim = (soft.quantile(0.25).item(), soft.quantile(0.75).item())
    COUNT["dt_clamped_low"] += int((soft < lim[0]).sum())
    COUNT["dt_clamped_high"] += int((soft > lim[1]).sum())
    y_ref, S_ref = oracle.mamba2_chunk_scan(x, dt_raw, A, Bg, Cg, chunk_size=cs, D=D, dt_bias=dt_bias,
                                            initial_states=S0.clone(), dt_softplus=True, dt_limit=lim,
                                            return_final_states=True)
    dt = impl.mamba2_dt(dt_raw, dt_bias, lim)
    S = S0.clone()
    COUNT["ssd_multi_chunk_calls"] += T > cs
    COUNT["ssd_short_last_chunk"] += T % cs != 0
    COUNT["ssd_nonzero_carry_in"] += 1
    y = impl.ssd_chunked(x, dt, A, impl.heads_from_groups(Bg, H), impl.heads_from_groups(Cg, H), D, S, cs)
    tag = f"seed={seed} G={G} cs={cs} T={T} limit=({lim[0]:.3f},{lim[1]:.3f})"
    close("ssd_core", f"output {tag}", y, y_ref.to(DT), TOL_REF)
    close("ssd_core", f"final state {tag}", S, S_ref.to(DT), TOL_REF)
    S2 = S0.clone()
    y2 = impl.ssd_recurrent(x, dt, A, impl.heads_from_groups(Bg, H), impl.heads_from_groups(Cg, H), D, S2)
    close("ssd_identity", f"chunked vs recurrent {tag}", y, y2, TOL_ID)
    close("ssd_identity", f"chunked vs recurrent state {tag}", S, S2, TOL_ID)


def check_mamba2_mixer_variants(oracle, seed: int, G: int, cs: int, T: int) -> None:
    """Whole-mixer checks for the two options the plain mamba2 family runs with defaults only.

    1. A finite dt limit, chosen from this input so both bounds clamp, against the library's one-call
       (chunk-scan) path, which applies the limit. impl must match it in every mode and schedule.
    2. norm_groups = G against the library's mixer with its norm replaced by vLLM's grouped norm.
    """
    x = torch.randn(1, T, 16, dtype=DT, generator=torch.Generator().manual_seed(seed * 43 + T + G))
    base = oracle.Mamba2Ref(seed, G, cs)
    W0 = base.weights()
    H = W0.num_heads
    soft = torch.nn.functional.softplus((x @ W0.in_w.T)[..., -H:] + W0.dt_bias)
    lim = (soft.quantile(0.25).item(), soft.quantile(0.75).item())
    COUNT["mixer_dt_clamped_low"] += int((soft < lim[0]).sum())
    COUNT["mixer_dt_clamped_high"] += int((soft > lim[1]).sum())
    variants = (("dt_limit", oracle.Mamba2Ref(seed, G, cs, dt_limit=lim), {"dt_limit": lim}),
                ("vllm_norm", oracle.Mamba2Ref(seed, G, cs, vllm_norm=True), {"norm_groups": G}))
    for label, ref, change in variants:
        W = dataclasses.replace(ref.weights(), **change)
        for k in ("in_w", "conv_w", "dt_bias", "A_log", "D", "norm_w", "out_w"):
            identical("mixer_variant", f"{label} weights {k} seed={seed}", getattr(W, k), getattr(W0, k))
        want = ref.run(x, [T])[0]
        for sched in ([T], [1] * T, partitions(T, random.Random(T))[-1]):
            for mode in (None, cs):
                outs, st = run_impl("mamba2", W, x, sched, chunk_size=mode)
                tag = f"{label} seed={seed} G={G} cs={cs} T={T} sched={sched} mode={mode}"
                close("mixer_variant", f"output {tag}", torch.cat(outs, 1), want["out"], TOL_REF)
                close("mixer_variant", f"final ssm {tag}", st.ssm, want["ssm"], TOL_REF)


def check_ssd_directed() -> None:
    """A huge step next to a small one: a = dt * A = [-1e16, -1] in one chunk.

    Token 1 injects dt*B*x = 1e16 * 1e-16 = 1; token 2 decays it by exp(-1) and reads it with C = 1.
    The exact output at token 2 is exp(-1). Subtracting running sums gives s_2 - s_1 = 0 and output 1.
    """
    x = torch.ones(1, 2, 1, 1, dtype=DT)
    dt = torch.tensor([[[1e16], [1.0]]], dtype=DT)
    A = torch.tensor([-1.0], dtype=DT)
    Bh = torch.tensor([1e-16, 0.0], dtype=DT).view(1, 2, 1, 1)
    Ch = torch.ones(1, 2, 1, 1, dtype=DT)
    D = torch.zeros(1, dtype=DT)
    want = torch.tensor([1.0, math.exp(-1)], dtype=DT).view(1, 2, 1, 1)
    for mode in ("recurrent", "chunked"):
        S = torch.zeros(1, 1, 1, 1, dtype=DT)
        if mode == "recurrent":
            y = impl.ssd_recurrent(x, dt, A, Bh, Ch, D, S)
        else:
            y = impl.ssd_chunked(x, dt, A, Bh, Ch, D, S, 4)
        close("ssd_directed", f"large dt {mode}", y, want, TOL_ID)
        close("ssd_directed", f"large dt {mode} final state", S, torch.full_like(S, math.exp(-1)), TOL_ID)

    # One chunk of a production size (Mamba-Codestral uses 256, Mamba-2 checkpoints up to 1024).
    # The segment sums must stay quadratic in the chunk length for this to run under the 4 GiB cap.
    g = torch.Generator().manual_seed(7)
    L, H, P, N = 1024, 2, 2, 3
    x = torch.randn(1, L, H, P, dtype=DT, generator=g)
    dt = 0.01 + 0.1 * torch.rand(1, L, H, dtype=DT, generator=g)
    A = -torch.rand(H, dtype=DT, generator=g)
    Bh, Ch = (torch.randn(1, L, H, N, dtype=DT, generator=g) for _ in range(2))
    D = torch.randn(H, dtype=DT, generator=g)
    S0 = torch.randn(1, H, P, N, dtype=DT, generator=g)
    S_rec, S_chk = S0.clone(), S0.clone()
    y_rec = impl.ssd_recurrent(x, dt, A, Bh, Ch, D, S_rec)
    y_chk = impl.ssd_chunked(x, dt, A, Bh, Ch, D, S_chk, L)
    close("ssd_directed", f"one chunk of {L} tokens", y_chk, y_rec, TOL_ID)
    close("ssd_directed", f"one chunk of {L} tokens final state", S_chk, S_rec, TOL_ID)


NORM_GROUPS = (1, 2, 4)


def check_vllm_norm(oracle, seed: int, G: int) -> float:
    """impl's grouped gated norm vs vLLM's own code. Returns how far the transformers form is from vLLM."""
    g = torch.Generator().manual_seed(seed * 71 + G)
    y = torch.randn(2, 5, 32, dtype=DT, generator=g)
    z = torch.randn(2, 5, 32, dtype=DT, generator=g)
    w = 1 + 0.3 * torch.randn(32, dtype=DT, generator=g)
    ref = oracle.vllm_gated_norm(y, z, w, 1e-5, G)
    close("vllm_norm", f"seed={seed} n_groups={G}", impl.gated_rms_norm(y, z, w, 1e-5, G), ref, TOL_REF)
    return (impl.gated_rms_norm(y, z, w, 1e-5, 1) - ref).abs().max().item()


def reference_step_dt_limit(oracle) -> dict:
    """Diagnostic, not part of pass/fail: does the library's single-token path apply a finite dt limit?

    Its chunk scan clamps dt and its token update does not, so with a finite limit a prefill-then-decode
    run and a one-call run of the same tokens disagree. impl clamps on both paths.
    """
    x = torch.randn(1, 9, 16, dtype=DT, generator=torch.Generator().manual_seed(5))
    out = {}
    for lim in ((0.0, float("inf")), (0.05, 0.2)):
        ref = oracle.Mamba2Ref(0, 1, 4, dt_limit=lim)
        one = torch.cat([s["out"] for s in ref.run(x, [9])], 1)
        dec = torch.cat([s["out"] for s in ref.run(x, [1] * 9)], 1)
        W = ref.weights()
        i_one = torch.cat(run_impl("mamba2", W, x, [9], chunk_size=4)[0], 1)
        i_dec = torch.cat(run_impl("mamba2", W, x, [1] * 9)[0], 1)
        out[str(lim)] = {"reference_one_call_vs_token_steps": (one - dec).abs().max().item(),
                         "impl_one_call_vs_token_steps": (i_one - i_dec).abs().max().item()}
    return out


# ================================================================================================
# Identities: partitions, checkpoints
# ================================================================================================


def state_clone(st) -> dict[str, torch.Tensor]:
    """An independent copy made with torch directly, so it cannot share a bug with snapshot()."""
    return {k: v.clone() for k, v in st.tensors().items()}


def check_isolation(live, snap, pristine, tag: str) -> None:
    """Write into every live tensor in place; the snapshot must not see it. Then restore the live state.

    Forward passes mostly replace state tensors, so a snapshot that shares a tensor can survive a
    continuation unnoticed and still be corrupted later by any in-place update. This writes directly.
    """
    for v in live.tensors().values():
        v.add_(1.0)
    for k, v in snap.tensors().items():
        identical("checkpoint", f"snapshot isolated from in-place writes {k} {tag}", v, pristine[k])
    for k, v in live.tensors().items():
        v.copy_(pristine[k])


def check_checkpoint(kind: str, W, seed: int, B: int, t: int, cs=None) -> None:
    """Snapshot at t; advance the live state with suffix A; then resume suffix B from the snapshot itself.

    The baseline is cloned from the live state with torch before snapshot() runs, so a snapshot that
    copies wrongly (shares, zeroes, transforms) differs from it, however snapshot() is written.
    """
    g = torch.Generator().manual_seed(seed * 53 + t)
    prefix = torch.randn(B, t, 16, dtype=DT, generator=g)
    sa = torch.randn(B, 5, 16, dtype=DT, generator=g)
    sb = torch.randn(B, 7, 16, dtype=DT, generator=g)
    rng = random.Random(seed * 59 + t)

    def fresh(x):
        return torch.cat(run_impl(kind, W, x, [x.shape[1]], chunk_size=cs)[0], 1)

    _, live = run_impl(kind, W, prefix, [t], chunk_size=cs)
    pristine = state_clone(live)
    snap = live.snapshot()
    tag = f"{kind} seed={seed} B={B} t={t} cs={cs}"
    for k, v in snap.tensors().items():
        identical("checkpoint", f"snapshot equals live state {k} {tag}", v, pristine[k])
    equal("checkpoint", f"snapshot consumed {tag}", snap.consumed, t)
    check_isolation(live, snap, pristine, tag)
    for name, suffix, target in (("A live", sa, live), ("B from snapshot", sb, snap)):
        sched = partitions(suffix.shape[1], rng)[-1]
        if target is snap:
            for k, v in snap.tensors().items():  # advancing the live state must not have moved it
                identical("checkpoint", f"snapshot untouched {k} {tag}", v, pristine[k])
            COUNT["resumed_suffixes"] += 1
        outs, _ = run_impl(kind, W, suffix, sched, st=target, chunk_size=cs)
        whole = fresh(torch.cat([prefix, suffix], 1))
        close("checkpoint", f"output after {name} sched={sched} {tag}", torch.cat(outs, 1), whole[:, t:], TOL_ID)
        equal("checkpoint", f"consumed after {name} {tag}", target.consumed, t + suffix.shape[1])


def check_window_checkpoint(seed: int, w: int, t: int) -> None:
    g = torch.Generator().manual_seed(seed * 61 + t + w)
    mk = lambda L, H: torch.randn(1, H, L, 4, dtype=DT, generator=g)
    q, k, v = mk(t + 6, 4), mk(t + 6, 2), mk(t + 6, 2)
    live = impl.WindowKV(w)
    live.forward(q[:, :, :t], k[:, :, :t], v[:, :, :t])
    pristine = state_clone(live)
    snap = live.snapshot()
    tag = f"window seed={seed} w={w} t={t}"
    for name, val in snap.tensors().items():
        identical("checkpoint", f"snapshot equals live state {name} {tag}", val, pristine[name])
    equal("checkpoint", f"snapshot consumed {tag}", snap.consumed, t)
    check_isolation(live, snap, pristine, tag)
    live.forward(q[:, :, t:], k[:, :, t:], v[:, :, t:])
    for name, val in snap.tensors().items():
        identical("checkpoint", f"snapshot untouched {name} {tag}", val, pristine[name])
    outs = [snap.forward(q[:, :, i : i + 1], k[:, :, i : i + 1], v[:, :, i : i + 1]) for i in range(t, t + 6)]
    dense = impl.sliding_attention(q, k, v, w)
    COUNT["resumed_suffixes"] += 1
    close("checkpoint", f"resumed output {tag}", torch.cat(outs, 2), dense[:, :, t:], TOL_ID)
    equal("checkpoint", f"resumed consumed {tag}", snap.consumed, t + 6)


def check_partitions(kind: str, W, seed: int, B: int, T: int, rng, cs=None) -> None:
    x = torch.randn(B, T, 16, dtype=DT, generator=torch.Generator().manual_seed(seed * 67 + T))
    one, st1 = run_impl(kind, W, x, [T], chunk_size=cs)
    for sched in partitions(T, rng)[1:]:
        note_continuations(sched, "impl_continuations")
        outs, st = run_impl(kind, W, x, sched, chunk_size=cs)
        tag = f"{kind} seed={seed} B={B} T={T} sched={sched} cs={cs}"
        close("partition", f"outputs {tag}", torch.cat(outs, 1), one[0], TOL_ID)
        for k, v in st.tensors().items():
            close("partition", f"final {k} {tag}", v, st1.tensors()[k], TOL_ID)
        equal("partition", f"consumed {tag}", st.consumed, T)


# ================================================================================================
# Driver
# ================================================================================================


def run_all(oracle) -> dict:
    rng = random.Random(0)
    cases = Counter()
    check_ssd_directed()
    cases["ssd_directed"] += 1
    for seed in SEEDS:
        for w in WINDOWS:
            for B in BATCHES:
                for T in LENGTHS:
                    check_window(oracle, seed, w, B, T, rng)
                    cases["window"] += 1
            for t in (1, w - 1, w, w + 3):
                check_window_checkpoint(seed, w, t)
                cases["window_checkpoint"] += 1
        for r in ONE_SIDED:
            for B in BATCHES:
                for blocks in (1, 2, 3, 5):
                    check_longformer(oracle, seed, r, B, blocks)
                    cases["longformer"] += 1
        Wm = oracle.MambaRef(seed).weights()
        for B in BATCHES:
            for T in LENGTHS:
                check_mamba(oracle, seed, B, T)
                check_partitions("mamba", Wm, seed, B, T, rng)
                cases["mamba"] += 1
            for t in (1, 3, 4, 9):
                check_checkpoint("mamba", Wm, seed, B, t)
                cases["mamba_checkpoint"] += 1
        for G in GROUPS:
            for cs in CHUNKS:
                W2 = oracle.Mamba2Ref(seed, G, cs).weights()
                for B in BATCHES:
                    for T in LENGTHS:
                        check_mamba2(oracle, seed, G, cs, B, T, rng)
                        check_partitions("mamba2", W2, seed, B, T, rng, cs=cs)
                        cases["mamba2"] += 1
                    for t in (1, cs - 1, cs, cs + 3):
                        for mode in (None, cs):
                            check_checkpoint("mamba2", W2, seed, B, t, cs=mode)
                            cases["mamba2_checkpoint"] += 1
                for T in LENGTHS:
                    check_ssd_core(oracle, seed, G, cs, T)
                    cases["ssd_core"] += 1
                for T in (5, 9, 17):
                    check_mamba2_mixer_variants(oracle, seed, G, cs, T)
                    cases["mamba2_mixer_variants"] += 1
        for G in NORM_GROUPS:
            gap = check_vllm_norm(oracle, seed, G)
            NORM_GAP[G] = max(NORM_GAP.get(G, 0.0), gap)
            cases["vllm_norm"] += 1
    return dict(cases)


NORM_GAP: dict[int, float] = {}  # n_groups -> max |transformers-style norm - vLLM norm|


EXPECTED_CASES = {
    "window": len(SEEDS) * len(WINDOWS) * len(BATCHES) * len(LENGTHS),
    "window_checkpoint": len(SEEDS) * len(WINDOWS) * 4,
    "longformer": len(SEEDS) * len(ONE_SIDED) * len(BATCHES) * 4,
    "mamba": len(SEEDS) * len(BATCHES) * len(LENGTHS),
    "mamba_checkpoint": len(SEEDS) * len(BATCHES) * 4,
    "mamba2": len(SEEDS) * len(GROUPS) * len(CHUNKS) * len(BATCHES) * len(LENGTHS),
    "mamba2_checkpoint": len(SEEDS) * len(GROUPS) * len(CHUNKS) * len(BATCHES) * 4 * 2,
    "ssd_core": len(SEEDS) * len(GROUPS) * len(CHUNKS) * len(LENGTHS),
    "vllm_norm": len(SEEDS) * len(NORM_GROUPS),
    "mamba2_mixer_variants": len(SEEDS) * len(GROUPS) * len(CHUNKS) * 3,
    "ssd_directed": 1,
}


# ================================================================================================
# Mutations
# ================================================================================================

MUTATIONS = {
    "window_includes_w_back": "causal window also lets query i see key i - w (w + 1 tokens)",
    "window_evicts_early": "rolling KV keeps w - 2 past tokens instead of w - 1",
    "longformer_one_sided": "symmetric local mask made causal (only past neighbours)",
    "conv_taps_reversed": "convolution applies its taps in reverse time order",
    "conv_keeps_k": "convolution history keeps K columns instead of K - 1",
    "mamba_dt_no_bias": "Mamba dt ignores the dt_proj bias",
    "mamba_no_d_skip": "Mamba drops the D skip connection",
    "mamba2_no_dt_clamp": "Mamba-2 dt ignores the dt limit",
    "norm_before_gate": "Mamba-2 output norm applied before the gate instead of after",
    "norm_groups_ignored": "grouped gated norm always normalises the whole width",
    "groups_tiled": "B/C groups assigned to heads round-robin instead of in contiguous blocks",
    "ssd_drops_carry_in": "a chunk ignores the state carried in from earlier chunks in its outputs",
    "ssd_state_no_decay": "the carried-in state is not decayed across a chunk",
    "snapshot_shallow": "Mamba-2 snapshot shares tensors with the live state",
    "snapshot_shares_conv": "Mamba-2 snapshot shares only the convolution history with the live state",
    "mamba_snapshot_shallow": "Mamba snapshot shares tensors with the live state",
    "window_snapshot_shallow": "WindowKV snapshot shares its keys and values with the live state",
    "snapshot_drops_conv": "Mamba-2 snapshot resets the convolution history",
    "window_snapshot_resets_position": "WindowKV snapshot forgets how many tokens were consumed",
    "snapshot_negates": "Mamba snapshot stores the negated state (a second snapshot would undo it)",
    "consumed_overwritten": "Mamba and Mamba-2 set consumed to the last chunk length instead of adding it",
    "mixer_ignores_dt_limit": "Mamba-2 mixer runs with an unlimited dt whatever the weights say",
    "mixer_ignores_norm_groups": "Mamba-2 mixer always normalises the whole width",
    "ssd_running_sum_subtraction": "SSD chunk decays computed as exp(s_t - s_u) from running sums",
    "conv_history_view": "convolution history kept as a view of the whole history ++ x buffer",
}


def apply_mutation(name: str) -> None:
    if name == "window_includes_w_back":
        orig = impl.window_allowed

        def f(q, k, w, causal):
            return orig(q, k, w + 1, causal) if causal else orig(q, k, w, causal)

        impl.window_allowed = f
    elif name == "window_evicts_early":
        impl.window_retained = lambda n, w: max(min(n, w - 2), 0)
    elif name == "longformer_one_sided":
        orig = impl.window_allowed

        def f(q, k, w, causal):
            return orig(q, k, w + 1, True) if not causal else orig(q, k, w, causal)

        impl.window_allowed = f
    elif name == "conv_taps_reversed":
        orig = impl.causal_conv
        impl.causal_conv = lambda h, x, w, b: orig(h, x, w.flip(1), b)
    elif name == "conv_keeps_k":
        orig = impl.causal_conv

        def f(h, x, w, b):
            out, _ = orig(h, x, w, b)
            full = torch.cat([h, x], dim=2)
            K = w.shape[1]
            return out, (full[:, :, -K:] if full.shape[2] >= K else torch.nn.functional.pad(full, (K - full.shape[2], 0)))

        impl.causal_conv = f
    elif name == "mamba_dt_no_bias":
        impl.mamba_dt = lambda W, low: torch.nn.functional.softplus(low @ W.dt_w.T)
    elif name == "mamba_no_d_skip":
        orig = impl.mamba_forward
        impl.mamba_forward = lambda W, x, st: orig(dataclasses.replace(W, D=torch.zeros_like(W.D)), x, st)
    elif name == "mamba2_no_dt_clamp":
        impl.mamba2_dt = lambda raw, bias, lim: torch.nn.functional.softplus(raw + bias)
    elif name == "norm_before_gate":
        def f(y, z, weight, eps, groups):
            ys = y.reshape(*y.shape[:-1], groups, y.shape[-1] // groups)
            ys = ys * torch.rsqrt(ys.pow(2).mean(-1, keepdim=True) + eps)
            return ys.reshape(y.shape) * torch.nn.functional.silu(z) * weight

        impl.gated_rms_norm = f
    elif name == "norm_groups_ignored":
        orig = impl.gated_rms_norm
        impl.gated_rms_norm = lambda y, z, weight, eps, groups: orig(y, z, weight, eps, 1)
    elif name == "groups_tiled":
        impl.heads_from_groups = lambda t, H: t.repeat(1, 1, H // t.shape[2], 1)
    elif name == "ssd_drops_carry_in":
        orig = impl.ssd_chunk

        def f(x, dt, A, Bh, Ch, D, S):
            y = orig(x, dt, A, Bh, Ch, D, torch.zeros_like(S))  # outputs as if starting from zero
            S_in = S.clone()
            S.zero_()
            orig(x, dt, A, Bh, Ch, D, S)
            a = torch.cumsum(dt * A, dim=1)
            S.add_(torch.exp(a[:, -1])[:, :, None, None] * S_in)  # state still carried correctly
            return y

        impl.ssd_chunk = f
    elif name == "ssd_state_no_decay":
        orig = impl.ssd_chunk

        def f(x, dt, A, Bh, Ch, D, S):
            S_in = S.clone()
            y = orig(x, dt, A, Bh, Ch, D, S)
            a = torch.cumsum(dt * A, dim=1)
            S.add_((1 - torch.exp(a[:, -1]))[:, :, None, None] * S_in)  # undo the decay of S_in
            return y

        impl.ssd_chunk = f
    elif name == "snapshot_shallow":
        impl.Mamba2State.snapshot = lambda self: impl.Mamba2State(self.conv, self.ssm, self.consumed)
    elif name == "snapshot_shares_conv":
        impl.Mamba2State.snapshot = lambda self: impl.Mamba2State(self.conv, self.ssm.clone(), self.consumed)
    elif name == "mamba_snapshot_shallow":
        impl.MambaState.snapshot = lambda self: impl.MambaState(self.conv, self.ssm, self.consumed)
    elif name == "window_snapshot_shallow":
        def f(self):
            s = impl.WindowKV(self.w)
            s.keys, s.values, s.consumed = self.keys, self.values, self.consumed
            return s

        impl.WindowKV.snapshot = f
    elif name == "snapshot_drops_conv":
        impl.Mamba2State.snapshot = lambda self: impl.Mamba2State(torch.zeros_like(self.conv), self.ssm.clone(),
                                                                  self.consumed)
    elif name == "window_snapshot_resets_position":
        orig = impl.WindowKV.snapshot

        def f(self):
            s = orig(self)
            s.consumed = 0
            return s

        impl.WindowKV.snapshot = f
    elif name == "snapshot_negates":
        impl.MambaState.snapshot = lambda self: impl.MambaState(-self.conv, -self.ssm, self.consumed)
    elif name == "consumed_overwritten":
        for fn in ("mamba_forward", "mamba2_forward"):
            orig = getattr(impl, fn)

            def f(W, x, st, *a, _orig=orig, **kw):
                y = _orig(W, x, st, *a, **kw)
                st.consumed = x.shape[1]
                return y

            setattr(impl, fn, f)
    elif name == "mixer_ignores_dt_limit":
        orig = impl.mamba2_forward
        impl.mamba2_forward = lambda W, x, st, chunk_size=None: orig(
            dataclasses.replace(W, dt_limit=(0.0, float("inf"))), x, st, chunk_size)
    elif name == "mixer_ignores_norm_groups":
        orig = impl.mamba2_forward
        impl.mamba2_forward = lambda W, x, st, chunk_size=None: orig(
            dataclasses.replace(W, norm_groups=1), x, st, chunk_size)
    elif name == "ssd_running_sum_subtraction":
        def seg(a):
            s = torch.cumsum(a, dim=1)
            l = a.shape[1]
            causal = torch.ones(l, l, dtype=torch.bool).tril()
            d = s[:, :, None, :] - s[:, None, :, :]
            return d.masked_fill(~causal[None, :, :, None], float("-inf"))

        impl.segment_sums = seg
    elif name == "conv_history_view":
        def f(h, x, w, b):
            K = w.shape[1]
            full = torch.cat([h, x], dim=2)
            out = torch.zeros_like(x)
            for k in range(K):
                out = out + w[None, :, k, None] * full[:, :, k : k + x.shape[2]]
            if b is not None:
                out = out + b[None, :, None]
            return out, full[:, :, full.shape[2] - (K - 1) :]

        impl.causal_conv = f
    else:
        sys.exit(f"unknown mutation {name}; choose from {sorted(MUTATIONS)}")


def self_test(out: str | None) -> int:
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
    result = {"check": "02 self-test: every planted bug must be reported as a mismatch",
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
    diagnostic = None
    try:
        cases = run_all(oracle)
        diagnostic = reference_step_dt_limit(oracle)
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
    result = {
        "check": "02-hybrid-state vs transformers (Mistral sliding window, Longformer, Mamba, Mamba-2) and vLLM (Mamba-2 gated norm)",
        "vllm": oracle.VLLM_VERSION,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "mutation": args.mutate,
        "tolerances": {"reference": {"atol": TOL_REF[0], "rtol": TOL_REF[1]},
                       "impl_identity": {"atol": TOL_ID[0], "rtol": TOL_ID[1]}},
        "cases": cases,
        "expected_cases": EXPECTED_CASES,
        "comparisons": {k: v for k, v in sorted(COUNT.items()) if k.startswith("compare_")},
        "max_error_over_tolerance": {k: float(f"{v:.3g}") for k, v in sorted(MAX_ERR.items())},
        "coverage": coverage,
        "min_coverage": MIN_COVERAGE,
        "coverage_short": short,
        "cases_missing": missing,
        "diagnostic_reference_step_dt_limit": diagnostic,
        "diagnostic_transformers_vs_vllm_norm_max_abs_diff": {str(k): float(f"{v:.3g}") for k, v in NORM_GAP.items()},
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

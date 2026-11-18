"""Per-request state of the layers a hybrid model mixes: sliding-window attention, Mamba, Mamba-2.

A full-attention layer keeps every past key and value, so its state grows with the sequence. The
three layers here keep a bounded state instead:

- Sliding-window attention (Longformer's local attention; Mistral's causal form) keeps the last w-1
  keys and values.
- A Mamba layer (selective SSM) keeps a short convolution history and one recurrent state matrix.
- A Mamba-2 layer keeps the same two things, with the recurrence restricted to a scalar decay per
  head (state space duality, SSD), which lets a whole chunk be computed as a masked matrix product.

Everything runs in float64 on CPU with plain torch ops. Weights are plain tensors so the check can copy
them from a Hugging Face module; the computation here never calls transformers.

Layout conventions: batch first. x is [batch, length, hidden]. Attention heads are [batch, heads,
length, head_dim]. Recurrent states are updated in place, as an engine would, so a checkpoint has to
be an explicit copy (`snapshot`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ================================================================================================
# Sliding-window attention
# ================================================================================================


def window_allowed(q_pos: torch.Tensor, k_pos: torch.Tensor, w: int, causal: bool) -> torch.Tensor:
    """Which keys each query may see. Returns a bool matrix [len(q_pos), len(k_pos)].

    Causal (Mistral): query i sees key j when 0 <= i - j <= w - 1, that is itself and w - 1 earlier
    tokens, w tokens in all.
    Symmetric (Longformer local attention): query i sees key j when |i - j| <= w, where w is the
    one-sided width, 2w + 1 tokens in all.
    """
    d = q_pos[:, None] - k_pos[None, :]
    if causal:
        return (d >= 0) & (d <= w - 1)
    return d.abs() <= w


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Scaled dot-product attention with grouped KV heads.

    q is [B, Hq, Lq, d]; k and v are [B, Hkv, Lk, d] with Hq a multiple of Hkv. Query head h reads KV
    head h // (Hq // Hkv). Every query must be allowed at least one key.
    """
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def sliding_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: int, causal: bool = True) -> torch.Tensor:
    """Dense reference form: full score matrix, masked to the window."""
    pos = torch.arange(q.shape[2])
    return attend(q, k, v, window_allowed(pos, pos, w, causal))


def window_retained(num_tokens: int, w: int) -> int:
    """How many past tokens a causal window-w layer must keep to answer the next query: w - 1."""
    return min(num_tokens, w - 1)


class WindowKV:
    """Rolling KV state of one causal sliding-window layer for a batch of requests.

    After consuming t tokens it holds the keys and values of the last min(t, w - 1) of them. That is
    all the next query can see besides its own key, so the state stops growing at w - 1 tokens.
    """

    def __init__(self, w: int):
        self.w = w
        self.keys: torch.Tensor | None = None  # [B, Hkv, kept, d]
        self.values: torch.Tensor | None = None
        self.consumed = 0  # absolute position of the next token

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Consume L new tokens (prefill if L > 1, decode if L == 1) and return their outputs."""
        L = q.shape[2]
        if self.keys is not None:
            k_all = torch.cat([self.keys, k], dim=2)
            v_all = torch.cat([self.values, v], dim=2)
        else:
            k_all, v_all = k, v
        kept_before = k_all.shape[2] - L
        q_pos = torch.arange(self.consumed, self.consumed + L)
        k_pos = torch.arange(self.consumed - kept_before, self.consumed + L)
        out = attend(q, k_all, v_all, window_allowed(q_pos, k_pos, self.w, causal=True))
        keep = window_retained(k_all.shape[2], self.w)
        # Slice by start index: k_all[:, :, -0:] would keep everything when keep == 0.
        self.keys = k_all[:, :, k_all.shape[2] - keep :].clone()
        self.values = v_all[:, :, v_all.shape[2] - keep :].clone()
        self.consumed += L
        return out

    def snapshot(self) -> "WindowKV":
        s = WindowKV(self.w)
        s.keys = None if self.keys is None else self.keys.clone()
        s.values = None if self.values is None else self.values.clone()
        s.consumed = self.consumed
        return s

    def tensors(self) -> dict[str, torch.Tensor]:
        return {} if self.keys is None else {"keys": self.keys, "values": self.values}


# ================================================================================================
# Short causal convolution, shared by Mamba and Mamba-2
# ================================================================================================


def causal_conv(history: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
    """Depthwise causal convolution over time, continuing from `history`.

    history is [B, C, K-1], the last K-1 inputs already seen (zeros at the start); x is [B, C, L];
    weight is [C, K]. Output t is sum_k weight[:, k] * window[t + k] over the window
    history ++ x, so weight[:, K-1] multiplies the current input. Returns (output, new history).
    """
    K = weight.shape[1]
    full = torch.cat([history, x], dim=2)
    L = x.shape[2]
    out = torch.zeros_like(x)
    for k in range(K):
        out = out + weight[None, :, k, None] * full[:, :, k : k + L]
    if bias is not None:
        out = out + bias[None, :, None]
    # Copy the tail: a slice would keep the whole history ++ x buffer alive, so the state would
    # grow with the prefill length even though only K - 1 columns are meaningful.
    return out, full[:, :, full.shape[2] - (K - 1) :].clone()


def gated_rms_norm(y: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float, groups: int) -> torch.Tensor:
    """Mamba-2's output norm: gate first, then RMS-normalise each of `groups` equal slices of the width.

    groups=1 normalises the whole expanded width at once (what transformers does); groups=n_groups
    normalises each group's slice separately (what vLLM and the original Mamba-2 code do). The two
    agree only when n_groups is 1.
    """
    g = y * F.silu(z)
    gs = g.reshape(*g.shape[:-1], groups, g.shape[-1] // groups)
    gs = gs * torch.rsqrt(gs.pow(2).mean(-1, keepdim=True) + eps)
    return gs.reshape(g.shape) * weight


# ================================================================================================
# Mamba (selective SSM)
# ================================================================================================


@dataclass
class MambaWeights:
    in_w: torch.Tensor  # [2I, hidden]: input branch x and gate z
    conv_w: torch.Tensor  # [I, K]
    conv_b: torch.Tensor | None  # [I]
    x_w: torch.Tensor  # [R + 2N, I]: low-rank dt, B, C from the convolved input
    dt_w: torch.Tensor  # [I, R]
    dt_b: torch.Tensor  # [I]
    A_log: torch.Tensor  # [I, N]; A = -exp(A_log) is negative, so the state decays
    D: torch.Tensor  # [I], skip connection around the SSM
    out_w: torch.Tensor  # [hidden, I]
    eps: float = 0.0  # unused by Mamba; kept so both weight classes look alike


@dataclass
class MambaState:
    conv: torch.Tensor  # [B, I, K-1] last K-1 pre-convolution inputs
    ssm: torch.Tensor  # [B, I, N]
    consumed: int = 0

    def snapshot(self) -> "MambaState":
        return MambaState(self.conv.clone(), self.ssm.clone(), self.consumed)

    def tensors(self) -> dict[str, torch.Tensor]:
        return {"conv": self.conv, "ssm": self.ssm}


def mamba_new_state(W: MambaWeights, batch: int) -> MambaState:
    I, K = W.conv_w.shape
    N = W.A_log.shape[1]
    z = dict(dtype=W.in_w.dtype)
    return MambaState(torch.zeros(batch, I, K - 1, **z), torch.zeros(batch, I, N, **z))


def mamba_dt(W: MambaWeights, dt_low_rank: torch.Tensor) -> torch.Tensor:
    """Step size per channel: softplus(dt_proj(dt_low_rank) + bias), always positive."""
    return F.softplus(dt_low_rank @ W.dt_w.T + W.dt_b)


def mamba_forward(W: MambaWeights, x: torch.Tensor, st: MambaState) -> torch.Tensor:
    """Consume x [B, L, hidden] from state `st` (updated in place) and return [B, L, hidden].

    The recurrence per channel c and state index n, with input-dependent dt, B, C (the "selection"):
        h[c, n] <- exp(dt[c] * A[c, n]) * h[c, n] + dt[c] * B[n] * u[c]
        y[c]    =  sum_n C[n] * h[c, n] + D[c] * u[c]
    followed by the gate y * silu(z). This is the discretization the reference code uses (zero-order
    hold for A, Euler for B), not exact zero-order hold for B.
    """
    I, K = W.conv_w.shape
    N = W.A_log.shape[1]
    u, z = (x @ W.in_w.T).split(I, dim=-1)
    u, st.conv = causal_conv(st.conv, u.transpose(1, 2), W.conv_w, W.conv_b)
    u = F.silu(u).transpose(1, 2)  # [B, L, I]
    R = W.x_w.shape[0] - 2 * N
    dt_low, Bm, Cm = (u @ W.x_w.T).split([R, N, N], dim=-1)
    dt = mamba_dt(W, dt_low)  # [B, L, I]
    A = -torch.exp(W.A_log)  # [I, N]
    ys = []
    for t in range(x.shape[1]):
        decay = torch.exp(dt[:, t, :, None] * A)
        st.ssm = decay * st.ssm + dt[:, t, :, None] * Bm[:, t, None, :] * u[:, t, :, None]
        ys.append((st.ssm @ Cm[:, t, :, None]).squeeze(-1) + W.D * u[:, t])
    y = torch.stack(ys, dim=1) * F.silu(z)
    st.consumed += x.shape[1]
    return y @ W.out_w.T


# ================================================================================================
# Mamba-2 (SSD)
# ================================================================================================


@dataclass
class Mamba2Weights:
    in_w: torch.Tensor  # [I + (I + 2GN) + H, hidden]: gate z, conv input (x, B, C), dt
    conv_w: torch.Tensor  # [I + 2GN, K]
    conv_b: torch.Tensor | None
    dt_bias: torch.Tensor  # [H]
    A_log: torch.Tensor  # [H]: one scalar decay rate per head
    D: torch.Tensor  # [H]
    norm_w: torch.Tensor  # [I]
    out_w: torch.Tensor  # [hidden, I]
    num_heads: int
    head_dim: int
    n_groups: int
    state_size: int
    eps: float
    dt_limit: tuple[float, float] = (0.0, float("inf"))
    norm_groups: int = 1  # 1 = transformers semantics; n_groups = vLLM / original Mamba-2 semantics


@dataclass
class Mamba2State:
    conv: torch.Tensor  # [B, I + 2GN, K-1]
    ssm: torch.Tensor  # [B, H, P, N]
    consumed: int = 0

    def snapshot(self) -> "Mamba2State":
        return Mamba2State(self.conv.clone(), self.ssm.clone(), self.consumed)

    def tensors(self) -> dict[str, torch.Tensor]:
        return {"conv": self.conv, "ssm": self.ssm}


def mamba2_new_state(W: Mamba2Weights, batch: int) -> Mamba2State:
    C, K = W.conv_w.shape
    z = dict(dtype=W.in_w.dtype)
    return Mamba2State(torch.zeros(batch, C, K - 1, **z),
                       torch.zeros(batch, W.num_heads, W.head_dim, W.state_size, **z))


def heads_from_groups(t: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[B, L, G, N] -> [B, L, H, N]: B and C are shared by H/G consecutive heads."""
    return t.repeat_interleave(num_heads // t.shape[2], dim=2)


def ssd_recurrent(x, dt, A, Bh, Ch, D, S):
    """Token-by-token SSD. x [B,L,H,P], dt [B,L,H], A [H], Bh/Ch [B,L,H,N], D [H], S [B,H,P,N] (in place).

        S_h <- exp(dt_h * A_h) * S_h + dt_h * x_h (outer) B_h
        y_h  = S_h C_h + D_h * x_h
    """
    ys = []
    for t in range(x.shape[1]):
        decay = torch.exp(dt[:, t] * A)[:, :, None, None]
        S.mul_(decay).add_(dt[:, t, :, None, None] * x[:, t, :, :, None] * Bh[:, t, :, None, :])
        ys.append(torch.einsum("bhpn,bhn->bhp", S, Ch[:, t]) + D[:, None] * x[:, t])
    return torch.stack(ys, dim=1)


def segment_sums(a: torch.Tensor) -> torch.Tensor:
    """seg[b, t, u, h] = a[b, u+1, h] + ... + a[b, t, h] for u <= t (0 on the diagonal), -inf above it.

    Summing only the terms between u and t, instead of subtracting two running sums s_t - s_u, keeps a
    small decay exact next to a huge one: with a = [-1e16, -1], s_2 - s_1 rounds to 0, not -1.
    """
    l = a.shape[1]
    idx = torch.arange(l)
    # between[t, u, k] = u < k <= t
    between = (idx[None, None, :] > idx[None, :, None]) & (idx[None, None, :] <= idx[:, None, None])
    seg = torch.einsum("tuk,bkh->btuh", between.to(a.dtype), a)
    causal = idx[:, None] >= idx[None, :]
    return seg.masked_fill(~causal[None, :, :, None], float("-inf"))


def ssd_chunk(x, dt, A, Bh, Ch, D, S):
    """One chunk of SSD as matrix products. Same arguments as ssd_recurrent; S is updated in place.

    With a_t = dt_t * A and cumulative sums s_t = a_1 + ... + a_t inside the chunk:
        y_t   = sum_{u <= t} exp(s_t - s_u) (C_t . B_u) dt_u x_u      within the chunk ("diagonal block")
              + exp(s_t) C_t S_in                                     carried-in state ("off-diagonal")
              + D x_t
        S_out = exp(s_last) S_in + sum_u exp(s_last - s_u) dt_u x_u (outer) B_u
    The first term is attention-like: a causal L x L matrix of scores (C_t . B_u) times a decay mask.
    The decays exp(s_t - s_u) are computed from segment sums, not from that subtraction (see segment_sums).
    """
    a = dt * A  # [B, l, H], all <= 0
    s = torch.cumsum(a, dim=1)  # decay from the chunk start, used for the carried-in state
    seg = segment_sums(a)  # [B, t, u, H]; exp(-inf) = 0 masks the future
    scores = torch.einsum("bthn,buhn->btuh", Ch, Bh) * torch.exp(seg)
    xdt = x * dt[..., None]
    y = torch.einsum("btuh,buhp->bthp", scores, xdt)
    y = y + torch.exp(s)[..., None] * torch.einsum("bhpn,bthn->bthp", S, Ch)
    y = y + D[:, None] * x
    to_end = torch.exp(seg[:, -1])  # [B, l, H]: decay from each token to the chunk end
    S.mul_(torch.exp(s[:, -1])[:, :, None, None])
    S.add_(torch.einsum("blh,blhp,blhn->bhpn", to_end, xdt, Bh))
    return y


def ssd_chunked(x, dt, A, Bh, Ch, D, S, chunk_size: int):
    """Split the sequence into chunks; carry the state across chunk boundaries. The last chunk may be short."""
    ys = [ssd_chunk(x[:, i : i + chunk_size], dt[:, i : i + chunk_size], A,
                    Bh[:, i : i + chunk_size], Ch[:, i : i + chunk_size], D, S)
          for i in range(0, x.shape[1], chunk_size)]
    return torch.cat(ys, dim=1)


def mamba2_dt(dt_raw: torch.Tensor, dt_bias: torch.Tensor, dt_limit: tuple[float, float]) -> torch.Tensor:
    """Step size per head: softplus(raw + bias), clamped to dt_limit on every path (prefill and decode)."""
    return F.softplus(dt_raw + dt_bias).clamp(min=dt_limit[0], max=dt_limit[1])


def mamba2_forward(W: Mamba2Weights, x: torch.Tensor, st: Mamba2State, chunk_size: int | None = None) -> torch.Tensor:
    """Consume x [B, L, hidden] from `st` (in place). chunk_size None means token-by-token SSD."""
    H, P, G, N = W.num_heads, W.head_dim, W.n_groups, W.state_size
    I = H * P
    Bsz, L, _ = x.shape
    z, xbc, dt_raw = (x @ W.in_w.T).split([I, I + 2 * G * N, H], dim=-1)
    xbc, st.conv = causal_conv(st.conv, xbc.transpose(1, 2), W.conv_w, W.conv_b)
    xbc = F.silu(xbc).transpose(1, 2)
    u, Bg, Cg = xbc.split([I, G * N, G * N], dim=-1)
    u = u.reshape(Bsz, L, H, P)
    Bh = heads_from_groups(Bg.reshape(Bsz, L, G, N), H)
    Ch = heads_from_groups(Cg.reshape(Bsz, L, G, N), H)
    dt = mamba2_dt(dt_raw, W.dt_bias, W.dt_limit)
    A = -torch.exp(W.A_log)
    if chunk_size is None:
        y = ssd_recurrent(u, dt, A, Bh, Ch, W.D, st.ssm)
    else:
        y = ssd_chunked(u, dt, A, Bh, Ch, W.D, st.ssm, chunk_size)
    y = y.reshape(Bsz, L, I)
    y = gated_rms_norm(y, z, W.norm_w, W.eps, W.norm_groups)
    st.consumed += L
    return y @ W.out_w.T


# ================================================================================================
# State sizes
# ================================================================================================


def state_bytes(tensors: dict[str, torch.Tensor]) -> int:
    """Payload bytes of a state: elements times element size, nothing about allocator overhead."""
    return sum(t.numel() * t.element_size() for t in tensors.values())


def expected_bytes(kind: str, *, batch: int, elem: int, consumed: int = 0, **c) -> int:
    """Closed-form per-layer state size, written independently of the classes above.

    full attention:  2 * t * Hkv * d            (grows with t)
    sliding window:  2 * min(t, w - 1) * Hkv * d
    mamba:           I * (K - 1) + I * N
    mamba2:          (I + 2 G N) * (K - 1) + H * P * N
    """
    if kind == "full":
        n = 2 * consumed * c["kv_heads"] * c["head_dim"]
    elif kind == "window":
        n = 2 * min(consumed, c["w"] - 1) * c["kv_heads"] * c["head_dim"]
    elif kind == "mamba":
        n = c["I"] * (c["K"] - 1) + c["I"] * c["N"]
    elif kind == "mamba2":
        n = (c["H"] * c["P"] + 2 * c["G"] * c["N"]) * (c["K"] - 1) + c["H"] * c["P"] * c["N"]
    else:
        raise ValueError(kind)
    return batch * n * elem

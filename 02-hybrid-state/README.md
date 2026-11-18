# 02-hybrid-state

## 1. What this covers

- Sliding-window attention: the causal form (Mistral) and Longformer's symmetric local form, and the
  rolling KV state that stops growing at `w − 1` tokens.
- Mamba's selective state space layer: a short convolution history plus one recurrent state, with
  step size and input/output projections that depend on the input.
- Mamba-2's state space duality (SSD): the same recurrence restricted to one scalar decay per head, so
  a whole chunk can be computed as a masked matrix product, with state carried across chunks.
- Per-request state: what each layer must keep between steps, how big it is, and how to checkpoint it
  at token `t` and resume.

Questions you should be able to answer when done:

1. Mistral's `sliding_window=4096` and Longformer's `attention_window=512`: how many tokens does one
   query see in each?
2. Why does a sliding-window layer keep `w − 1` past keys, not `w`?
3. In Mamba, which quantities depend on the input and which do not? Why does that rule out a
   convolution-only (S4-style) implementation?
4. Why does SSD need the decay to be a scalar per head? What would break with Mamba's per-channel,
   per-state `A`?
5. A checkpoint of a Mamba-2 layer at token `t` holds two tensors. Which two, and what goes wrong if
   only one of them is restored?
6. How large is the per-request state of Mamba-Codestral-7B, and how many tokens of Mistral-7B KV is
   that?

## 2. Concepts

- **Window `w` (causal).** Query at position `i` attends to keys `j` with `0 ≤ i − j ≤ w − 1`: itself
  plus `w − 1` earlier tokens. This is Mistral's `sliding_window` and what transformers implements.
- **One-sided width `r` (symmetric).** Longformer's local attention: `|i − j| ≤ r`, `2r + 1` tokens. The
  library's `attention_window` is `2r` and must be even.
- **SSM state.** A fixed-size matrix carried from token to token. It replaces the growing list of keys
  and values.
- **Selective.** The step size `Δ` and the projections `B`, `C` are computed from the current input, so
  the layer can choose what to keep and what to forget. `A` and `D` are learned constants.
- **Discretization.** This is how the continuous parameters become a per-token update. The reference
  code uses `exp(Δ·A)` for the decay and `Δ·B` for the input. That is zero-order hold for `A` and an
  Euler step for `B`, not exact zero-order hold for both.
- **Convolution history.** Both Mamba layers run a short depthwise causal convolution, width `K`
  (`K = 4` here), before the SSM. Continuing it needs the last `K − 1` inputs of the convolution, and
  those are part of the state.
- **Chunk.** A block of consecutive tokens that SSD processes as one matrix product. The last chunk
  may be shorter.
- **Checkpoint at `t`.** A copy of every state tensor after `t` tokens have been consumed. The next
  input is token `t`.

## 3. How it works

### Sliding window

```
allowed(i, j) = 0 ≤ i − j ≤ w − 1                      causal (Mistral)
allowed(i, j) = |i − j| ≤ r                            symmetric (Longformer, local part)
out_i = softmax_j( q_i·k_j / √d  over allowed j ) · v
```

To answer query `t`, the layer needs keys `t − w + 1 … t`. Key `t` arrives with the query, so the
state between steps is the last `w − 1` keys and values (`WindowKV`). Decoding drops the oldest pair
once that many are held. Prefill computes the dense masked product and then keeps the tail. Both give
the same outputs, and `check.py` compares them.

Longformer's symmetric form needs future tokens, so it applies to encoders and to prefill, not to
token-by-token decoding. Its global-attention tokens are not covered here.

### Mamba

With `u` the convolved input (after SiLU), and per channel `c` and state index `n`:

```
Δ      = softplus(W_Δ · low_rank(u) + b_Δ)         per channel, input dependent
B, C   = linear(u)                                  per token, input dependent, shared by channels
A      = −exp(A_log)                                per channel and state index, constant, negative
h[c,n] ← exp(Δ[c]·A[c,n]) · h[c,n] + Δ[c]·B[n]·u[c]
y[c]   = Σ_n C[n]·h[c,n] + D[c]·u[c]
out    = W_out · (y ⊙ silu(z))                      z: the gate branch of the input projection
```

`A` is negative, so the decay is below 1. A large `Δ` resets the state toward the current input and a
small `Δ` keeps the past. That choice, made per token, is the "selection". Because `Δ`, `B` and `C`
change every token, the layer is not time-invariant and cannot be computed as one long convolution
the way S4 can. It is computed as a scan instead.

State per layer and request: `I × (K − 1)` for the convolution plus `I × N` for `h`, where `I` is the
expanded width.

### Mamba-2 (SSD)

Mamba-2 makes `A` one scalar per head and groups `B`, `C`: `G` groups, each shared by `H/G`
consecutive heads. Per head, the state `S` is `P × N`:

```
S ← exp(Δ·A) · S + Δ · x ⊗ B
y = S·C + D·x
```

With a scalar decay, the contribution of token `u` to the output at `t` inside one chunk is
`exp(s_t − s_u) · (C_t·B_u) · Δ_u x_u`, where `s` is the running sum of `Δ·A`. The whole chunk is
then a causal `L × L` score matrix `(C_t·B_u)` times a decay mask. That is the duality: attention-like
inside the chunk, recurrent between chunks. `impl.ssd_chunk` writes out the three terms: inside the
chunk, the carried-in state, and the `D` skip. It also writes out the state handed to the next chunk.
`check.py` checks that it equals the token-by-token form.

The decay `exp(s_t − s_u)` is computed as `exp(a_{u+1} + … + a_t)`, summing only the terms between
`u` and `t`. Subtracting two running sums loses a small decay that sits next to a huge one: with
`a = [−1e16, −1]`, `s_2 − s_1` rounds to 0 instead of −1. The library uses the same masked
segment-sum form (`segment_sum` in `modeling_mamba2.py`).

After the scan: `out = W_out · RMSNorm(y ⊙ silu(z))`. The gate comes before the norm. Which slice the
norm spans differs between implementations (§6).

State per layer and request: `(I + 2GN) × (K − 1)` for the convolution plus `H × P × N` for `S`.

### Worked numbers

These are per request, from the published configs at the revisions in §7. They count only the tensor
payload; block rounding and allocator behaviour are out of scope.

| Model | Layer state | Size |
|---|---|---|
| Mistral-7B-v0.1, bf16 KV (32 layers, 8 KV heads × 128) | 2 × 8 × 128 × 2 B × 32 = **128 KiB per token** | full attention at 32,768 tokens: 4 GiB. Window 4,096 keeps 4,095 tokens: **511.875 MiB**, constant after that |
| mamba-2.8b (64 layers, `I` = 5,120, `N` = 16, `K` = 4) | 5,120 × 3 + 5,120 × 16 = 97,280 elements per layer | fp32: **23.75 MiB**, at any length |
| Mamba-Codestral-7B (64 layers, `H` = 128, `P` = 64, `N` = 128, `G` = 8) | conv (8,192 + 2 × 8 × 128) × 3 = 30,720; SSM 128 × 64 × 128 = 1,048,576 | fp32: **263.5 MiB**, at any length. That equals the KV of about 2,108 Mistral-7B tokens |

The SSM state does not grow, but it is not small. For short requests a pure-attention model can hold
less state than an SSM.

## 4. Code and how to run it

| File | What it does |
|---|---|
| `impl.py` | `WindowKV`, `sliding_attention`, `causal_conv`, `mamba_forward`, `ssd_recurrent`, `ssd_chunk`/`ssd_chunked`, `mamba2_forward`, `gated_rms_norm`, `snapshot()` on every state, `state_bytes`/`expected_bytes`. Float64, torch only, no transformers. |
| `oracle.py` | Tiny random one-layer Mistral, Longformer self-attention, Mamba and Mamba-2 mixers from transformers 5.22.2, run with the library's own caches; vLLM 0.34.0's gated-norm code. Copies weights into impl's plain classes. |
| `check.py` | Every comparison below; writes `results/check.json`. `--self-test` plants 28 bugs and requires each to be reported. |

```bash
scripts/run-capped --mem 4G --cpus 4 -- .venv/bin/python 02-hybrid-state/check.py --out 02-hybrid-state/results/check.json
scripts/run-capped --mem 4G --cpus 4 -- .venv/bin/python 02-hybrid-state/check.py --self-test --out 02-hybrid-state/results/self_test.json
```

The two commands take about 6 s and a little over a minute, with a peak RSS of about 600 MiB. The check pins
transformers 5.22.2 and vLLM 0.34.0 (`g62d4bc3c7`) and refuses to run on other versions.

## 5. Checks and expected results

**References.** transformers 5.22.2 (tag `v5.22.2`; the installed files compared byte-for-byte with
the tag) on CPU, where every Mamba op falls back to the library's pure-torch implementation. vLLM 0.34.0
for the Mamba-2 gated norm. Random tiny models (hidden 16; Mamba `I` = 32, `N` = 3, `K` = 4; Mamba-2
4 heads × 8, `N` = 3), with every mixer parameter moved off its structured init, so that a skipped
`D`, a dropped bias or swapped channels cannot cancel out.

**Tolerances.**
- Against the reference: `atol 2e-6, rtol 2e-5`. The reference casts the softmax, the scan steps and
  the norm to float32, so it is not a float64 oracle. The largest measured error is 23 % of this
  tolerance, in the SSD core (`max_error_over_tolerance` in the JSON). An error below the tolerance in
  every element, for example a constant bias of `1e-6`, would still pass.
- impl against itself: `1e-10 / 1e-9`, because it is float64 throughout.
- Snapshot isolation: bit for bit (`torch.equal`); token counters: exactly.

| Family | What is compared | Matrix |
|---|---|---|
| window | Per chunk: attention output vs Mistral, retained keys/values (shape and values) vs `DynamicSlidingWindowLayer`, token count; dense masked = rolling | `w` ∈ {2, 4, 8}, lengths 1–33, one-shot / prefill + decode / irregular chunks |
| window `w = 1` | impl only (the reference mishandles it, below): every output equals the token's own value, nothing is retained, exact token count | lengths 1–33, one-shot / prefill + decode / irregular chunks |
| longformer | Local attention output vs `LongformerSelfAttention` | `r` ∈ {2, 3, 4}, 1–5 windows |
| mamba | Per chunk: output, SSM state, convolution history vs `MambaMixer` + cache | prefill + single-token decode |
| mamba2 | Per chunk: output, SSM state, convolution history vs `Mamba2Mixer` + cache, impl in both token and chunked mode | `G` ∈ {1, 2}, chunk ∈ {4, 8}, multi-token continuations |
| ssd core | `ssd_chunked` vs the library's `mamba2_chunk_scan`, nonzero initial state, finite Δ limits that clamp on both sides | 108 cases |
| mixer variants | Whole Mamba-2 mixer with (a) a finite Δ limit that clamps on both sides, vs the library's one-call path, which applies it; (b) `norm_groups = G`, vs the library's mixer with its norm replaced by vLLM's. Each in token and chunked mode, one-shot / token-by-token / irregular | 36 cases |
| ssd directed | A Δ of `1e16` next to a Δ of 1 in one chunk; the exact output is `exp(−1)`. One 1,024-token chunk vs the token-by-token form (the segment sums must stay `L²`, not `L³`) | 1 case |
| vllm norm | `gated_rms_norm(groups=G)` vs vLLM `Mixer2RMSNormGated.forward_native` | `G` ∈ {1, 2, 4} |
| partition | Prefill + decode and an irregular split = one call (outputs and final state) | all lengths |
| checkpoint | Snapshot at `t` must equal a torch clone of the live state taken before `snapshot()` runs, and `snapshot()` must leave the live state (tensors, window, token count) as it was. Add 1 in place to every live tensor: the snapshot must not change; then restore the live state. Advance the live state with one suffix, check the snapshot did not move, then resume another suffix from the snapshot itself. Both continuations must equal a fresh run over prefix + suffix, with exact token counts | `t` around window and chunk edges; windows `w` ∈ {1, 2, 4, 8} |
| sizes | `state_bytes` = `expected_bytes`; reference cache tensor sizes; each state tensor's storage is exactly its payload (no view keeping a larger buffer alive) | every step |

A pass requires all 976 cases and minimum counts of window wraps, multi-chunk calls, short last chunks,
nonzero carried-in states, Δ values clamped low and high (in the core and in the mixer), two-group
cases, suffixes resumed from a snapshot, and
multi-token continuations counted separately for the Mistral reference, the Mamba-2 reference and the
impl-only partition checks.

**Result** (`results/check.json`): passed. The run makes about 60,000 comparisons and finds no
mismatch.

**Planted bugs** (`results/self_test.json`): all 28 are reported as mismatches.

| Mutation | First reported by |
|---|---|
| window includes key `i − w` | window output |
| rolling KV keeps `w − 2` | retained-KV shape |
| rolling KV sliced with `[-keep:]`, which keeps everything at `w = 1` | retained tokens at `w = 1` |
| Longformer mask made one-sided | longformer output |
| convolution taps reversed | mamba output |
| convolution history keeps `K` columns | conv-history shape |
| Mamba Δ without bias | mamba output |
| Mamba without `D` skip | mamba output |
| Mamba-2 Δ limit ignored | ssd core (finite limits) |
| norm before gate | mamba2 output |
| norm groups ignored | mixer with vLLM's norm |
| B/C groups assigned round-robin | mamba2 output with `G = 2` |
| chunk ignores carried-in state in its outputs | ssd directed (1,024-token chunk, nonzero initial state) |
| carried-in state not decayed across a chunk | ssd directed (1,024-token chunk, final state) |
| shallow snapshot (Mamba-2) | in-place write to the live state |
| Mamba-2 snapshot shares only the convolution history | in-place write to the live state |
| shallow snapshot (Mamba) | in-place write to the live state |
| window snapshot shares its keys and values | in-place write to the live state |
| snapshot zeroes the live state after copying it | live state unchanged by `snapshot()` |
| window snapshot resets the live window and token count | live token count unchanged by `snapshot()` (at `w = 1`) |
| snapshot drops the convolution history | snapshot equals the live state |
| window snapshot forgets its token count | checkpoint token count (at `w = 1`) |
| snapshot stores the negated state | snapshot equals the live state |
| `consumed` overwritten instead of added | token count after a continuation |
| mixer ignores the Δ limit | mixer with a finite Δ limit |
| mixer ignores `norm_groups` | mixer with vLLM's norm |
| SSD decays from running-sum subtraction | ssd directed |
| convolution history kept as a view of the whole buffer | storage size |

**Things the reference does that a test has to work around.**

- A multi-token call to `MambaMixer` with a non-empty cache starts its scan from a zero state
  ([`modeling_mamba.py#L251`](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/mamba/modeling_mamba.py#L251)).
  Only the one-token path reads the cached state
  ([#L442](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/mamba/modeling_mamba.py#L442)).
  So reference schedules for Mamba are a prefill followed by single tokens. impl continues correctly
  from multi-token chunks, which the partition check covers.
- The Mamba-2 chunk scan clamps Δ to `time_step_limit`
  ([#L276](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/mamba2/modeling_mamba2.py#L276)).
  The one-token update does not
  ([#L192](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/mamba2/modeling_mamba2.py#L192)).
  With the default limit `(0, ∞)` this changes nothing. With a finite limit, one call and
  token-by-token decoding of the same 9 tokens differ by 0.17 (diagnostic in the JSON; impl: 3e-15).
- The library's caches keep `K` convolution columns. The oldest one is never read again, and impl
  keeps `K − 1`.
- `DynamicSlidingWindowLayer` keeps `w − 1` tokens
  ([`cache_utils.py#L245`](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/cache_utils.py#L245))
  although its docstring says `sliding_window`. `get_seq_length()` reports all tokens seen, not the
  number held.
- At `sliding_window = 1` the same slice is `[-0:]`, which keeps every token. Decoding then attends to
  all of them: token by token, the output matches full attention (6e-10) and is 0.017 from the correct
  self-only output. A chunk of several tokens after the first fails with a shape mismatch. Both are
  recorded in `diagnostic_reference_window_one`. So `w = 1` is checked against its closed form instead.
- `LongformerSelfAttention` fails for `attention_window = 2` (`r = 1`): at
  [#L794](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/longformer/modeling_longformer.py#L794)
  the slice `1 - window_overlap :` becomes `0 :` and the shapes no longer match. So the matrix uses
  `r ≥ 2`.

## 6. In the engines

vLLM v0.34.0 (commit `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`):

| Topic | vLLM code | Agreement |
|---|---|---|
| State shapes, conv history `K − 1` | [`mamba_utils.py` `mamba1_state_shape`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/layers/mamba/mamba_utils.py#L179), [`mamba2_state_shape`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/layers/mamba/mamba_utils.py#L194) | Same `K − 1` as impl's `expected_bytes` (read from source, not executed) |
| Mamba-2 gated norm, per group | [`mamba_mixer2.py` `Mixer2RMSNormGated`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/layers/mamba/mamba_mixer2.py#L88) | impl with `norm_groups = n_groups` matches `forward_native` run on CPU, alone (3 group counts × 3 seeds) and inside the whole mixer |
| Δ limit | [`mamba_mixer2.py` prefill passes `dt_limit=(0.0, inf)`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/layers/mamba/mamba_mixer2.py#L884) | vLLM applies no finite limit on either path, so the transformers inconsistency in §5 does not arise there (read from source) |
| Sliding window, tokens skipped | [`single_type_kv_cache_manager.py` `SlidingWindowManager.get_num_skipped_tokens`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/core/single_type_kv_cache_manager.py#L1153) | `t − w + 1` tokens are dropped before token `t` is computed, which is the same `w − 1` past tokens as impl (read from source; block-level freeing is out of scope) |

**Difference found: transformers' `Mamba2` normalizes over the whole width.** transformers'
`MambaRMSNormGated`
([`modeling_mamba2.py#L105`](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/mamba2/modeling_mamba2.py#L105))
normalizes the full expanded width at once. vLLM normalizes each group's `I/G` slice separately. So
does the original Mamba-2 code:
[`RMSNormGated(..., group_size=self.d_ssm // ngroups)`](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba2.py#L144-L145).
The two forms agree only when `n_groups = 1`.

On the same inputs, the full-width form differs from vLLM's by up to 2.6 in absolute value for
`n_groups` = 2 and 4 (diagnostic in the JSON). Mamba-Codestral-7B uses `n_groups = 8`. In the same
library, `NemotronH` and `Zamba2` use the grouped form. This repository records the difference. It
does not report it upstream.

**Out of scope here:** vLLM's block-level sliding-window freeing and Mamba prefix caching (allocator
accounting), Triton kernels, speculative decoding with SSM state (see 03), and Longformer's global
attention.

## 7. References

- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150v2)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752v2)
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060v1)
- transformers v5.22.2 and vLLM v0.34.0 source, linked above. The original Mamba code is at
  state-spaces/mamba tag `v2.2.4`.
- Model configurations used in §3, from Hugging Face:
  - Mistral-7B-v0.1 (`config.json` at revision `27d67f1b5f`)
  - mamba-2.8b-hf (`96c48e0292`)
  - Mamba-Codestral-7B-v0.1 (`4f086c08c1`)

## 8. Self-check questions

1. A Mistral layer has `w = 4`. After 10 tokens, which positions does `WindowKV` hold, and which keys
   does query 10 see?
2. Write the Mamba update for one channel. Which terms come from the current token?
3. Why does setting `Δ` large act like a reset of the state?
4. In `ssd_chunk`, what are the three output terms? Which of them would be missing if a chunk ignored
   the state carried in?
5. `impl.ssd_chunked` lets the last chunk be short; the library pads it to a full chunk instead. What
   must a padded step's log decay and input be so that the final state is unchanged?
6. A checkpoint of a Mamba-2 layer stores `conv` and `ssm`. Resume with only `ssm` restored: which
   outputs are wrong, and for how many tokens?
7. With `n_groups = 8`, why do transformers' `Mamba2` and vLLM produce different outputs from the same
   weights? Which of them matches the original code?
8. Mamba-Codestral-7B's state is about 263.5 MiB per request. Below what prompt length does Mistral-7B
   need less KV than that?

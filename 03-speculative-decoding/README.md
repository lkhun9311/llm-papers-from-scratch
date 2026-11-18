# 03-speculative-decoding

## 1. What this covers

- Draft and verify: a small model proposes `k` tokens, the large model scores all of them in one
  forward pass, and a rule decides how many to keep.
- The verification rule that makes the output distribution exactly the large model's, not an
  approximation of it: accept with probability `min(1, p/q)`, resample a rejection from
  `norm(max(p − q, 0))`, and add one token from the target when every draft is kept.
- Rolling back after a rejection: a KV cache is truncated; a recurrent (SSM) state cannot be, so it is
  restored from a snapshot.

Questions you should be able to answer when done:

1. Why is the output of speculative sampling distributed exactly like the target's, for any draft model?
2. What is the probability that a draft position is rejected, in terms of `p` and `q`?
3. Why does a round emit up to `k + 1` tokens, not `k`?
4. Draft acceptance rate `α = 0.8`, `k = 4`, draft cost 5% of a target forward: what speed-up does the
   paper's model predict? And at `α = 0.6`?
5. After a rejection, how many tokens does the target forget, and how many does the draft forget?
6. Why can transformers' assisted generation crop an attention cache but not a Mamba state, and what does
   vLLM keep instead?

## 2. Concepts

- **Target `p`, draft `q`.** Next-token distributions of the large and the small model, given the same
  context.
- **Draft length `k`.** Tokens proposed per round. The target scores them in one forward over `k + 1`
  positions: the token before the drafts, then each draft.
- **Round.** Draft `k` tokens, verify, keep `n ≤ k` drafts, emit one more token.
- **Bonus token.** When all `k` drafts are kept, the target's distribution at the position after them
  is already computed, so one more token is sampled from it for free.
- **Residual.** `norm(max(p − q, 0))`: the part of `p` that `q` under-proposes. A rejected position is
  resampled from it.
- **Rollback.** Making a model forget the tokens it consumed for rejected drafts, so its cache matches
  the emitted sequence again.

## 3. How it works

**Verification (sampling).** For draft position `i` with draft token `x_i ~ q_i`:

```
accept x_i      with probability  min(1, p_i(x_i) / q_i(x_i))      (stop at the first rejection)
on rejection at i:  emit one token from  norm(max(p_i − q_i, 0))
all k accepted:     emit one token from  p_k                        (the bonus token)
```

Why the output is exactly `p`: the probability of emitting `x` at a position is the chance it is
proposed and kept, `q(x)·min(1, p(x)/q(x)) = min(p(x), q(x))`, plus the chance of a rejection times the
residual. The rejection chance is `1 − Σ min(p, q) = Σ max(p − q, 0)`, and the residual is
`max(p − q, 0)` divided by that same sum, so the second term is `max(p(x) − q(x), 0)`. Together:
`min(p, q) + max(p − q, 0) = p(x)`. The draft only changes how fast tokens come, not which tokens come.
`impl.round_distribution` enumerates every outcome of a round, and `check.py` confirms that the round,
continued by the target to a fixed length, has exactly the target's law over every sequence.

**Verification (greedy).** Keep drafts while each equals the target's argmax, then emit the argmax.
The output equals greedy decoding of the target.

**Cost.** If each draft is accepted with probability `α` independently, a round emits on average
`(1 − α^{k+1}) / (1 − α)` tokens (Leviathan et al.). One round costs one target forward plus `k`
draft forwards. With the draft at cost `c` of a target forward, the predicted speed-up is
`(1 − α^{k+1}) / ((1 − α)(k·c + 1))`:

| `α` | `k` | `c` | Tokens per round | Speed-up |
|---|---|---|---|---|
| 0.8 | 4 | 0.05 | 3.36 | **2.80×** |
| 0.6 | 4 | 0.05 | 2.31 | **1.92×** |

The target forward over `k + 1` positions is assumed to cost the same as over one. That holds when
decoding is memory-bound, as for one sequence on a GPU.

**Rollback.** The invariant at the start of a round: both models have consumed every emitted token
except the newest one. A round with `n` drafts kept:

| Model | Consumed this round | Keeps | Forgets |
|---|---|---|---|
| target | the previous token + `k` drafts | the previous token + `n` drafts | `k − n` |
| draft | the previous token + drafts `0 … k−2` | the previous token + `n` drafts | `k − 1 − n` (if `n = k`, it consumes the last draft instead) |

An attention KV cache rolls back by truncation: the entry for token `t` depends only on tokens `≤ t`.
An SSM state is a sum over every past token and cannot be truncated. `impl.MambaLM` keeps a snapshot
after each of its last `k + 1` tokens and restores one.

What a rollback has to hold, for one sequence and `k = 4`:

| Model | Extra memory for rollback |
|---|---|
| Mistral-7B-v0.1, bf16 KV (128 KiB per token, from 02) | the `k` draft positions written by the verify pass: 4 × 128 KiB = **512 KiB**, dropped by truncation |
| Mamba-Codestral-7B, fp32 state (263.5 MiB, from 02) | one state per draft position: 4 × 263.5 MiB = **1,054 MiB** |

That gap is why engines look for other ways to roll back an SSM. vLLM's default keeps one state slot
per speculative token. Its ReplaySSM option instead caches recent SSM inputs and recomputes from one
checkpoint (§6).

## 4. Code and how to run it

| File | What it does |
|---|---|
| `impl.py` | `verify_greedy`, `verify_sampled`, `sample_round` (transformers' random-number order), `residual`; `round_distribution`, `outcome_distribution` (exact laws by enumeration); `AttnLM` (KV cache, rollback by truncation), `MambaLM` (02's Mamba mixer, rollback by snapshot); `speculative` and `autoregressive` loops. Float64, torch only. |
| `oracle.py` | transformers 5.22.2: `_speculative_sampling`, a tiny Llama pair, `generate()` with and without an assistant, `DynamicCache.crop`. vLLM 0.34.0: `rejection_sample` on CPU through Triton's interpreter. |
| `check.py` | Every comparison below; writes `results/check.json`. `--self-test` plants 12 bugs and requires each to be reported, after an unmutated run passes. |

```bash
scripts/run-capped --mem 4G --cpus 4 -- .venv/bin/python 03-speculative-decoding/check.py --out 03-speculative-decoding/results/check.json
scripts/run-capped --mem 4G --cpus 4 -- .venv/bin/python 03-speculative-decoding/check.py --self-test --out 03-speculative-decoding/results/self_test.json
```

The check takes about 5 minutes with a peak RSS of about 540 MiB; 258 s of that is vLLM's sampler in
the Triton interpreter. The check pins transformers 5.22.2 and vLLM 0.34.0 (`g62d4bc3c7`).

**Running vLLM's sampler on CPU.** The sampler is Triton GPU kernels. With `TRITON_INTERPRET=1`, Triton
runs the same kernel source on CPU with numpy. The interpreter has no `libdevice`, and the resample
kernel calls `libdevice.log1p`, so `oracle.py` supplies numpy's `log1p` for it. Nothing in vLLM's code
is changed.

## 5. Checks and expected results

**References.** transformers 5.22.2 (tag `v5.22.2`) and vLLM 0.34.0 (commit `62d4bc3c7e`). The installed
files compared byte for byte with those revisions on GitHub.

| Family | What is compared | Matrix |
|---|---|---|
| hf sampling | `impl.sample_round` vs transformers' `_speculative_sampling`, same seed: accepted count and every emitted token identical | `k` ∈ {1, 3, 5}, `V` ∈ {4, 16, 64}, 4 shapes of `p, q` (flat, sharp, close, equal), 40 seeds |
| hf assisted | `impl.speculative` (greedy) driving a transformers Llama target and a draft made from it with perturbed weights, through `DynamicCache` and `crop`, vs `generate(assistant_model=…)` vs plain `generate()`: identical tokens | prompt lengths 1, 3, 7; `k` ∈ {1, 2, 4, 6} |
| vllm greedy | `impl.verify_greedy` vs `rejection_sample` at temperature 0: accepted count and tokens | `k` ∈ {1, 3, 5}, with and without draft logits, 320 rows |
| vllm rows | Each request's outcome is certain and differs from its neighbour (q = p, or a first draft of probability 0): exact accepted counts and kept drafts per row | 22 rows |
| vllm sampled | `rejection_sample` over 4,000 seeds vs `impl.outcome_distribution`, the exact law of (accepted count, emitted token): every cell within 5 standard errors, nothing in a cell of probability 0, Pearson chi-square under its `1 − 10⁻⁶` quantile | temperatures 1.0, 0.7, 0.8; with draft logits and one-hot drafts |
| theorem | One round continued by the target to `k + 1` or `k + 2` tokens equals the target's law over every sequence (enumerated) | `V` ∈ {2, 3}, `k` ∈ {1, 2, 3}; `q` random, `q = p`, `q` disjoint from `p`, `p` with a zero-probability token |
| rollback | Consume tokens, consume more in 1–5 calls, roll back, continue: logits and state equal a fresh model fed the kept tokens | impl attention, impl Mamba, transformers `DynamicCache` |
| generation | Greedy speculative = greedy, and on return both models hold exactly the returned tokens but the last; sampled speculative of a 3-token vocabulary vs the target's exact law over all sequences (3,000 runs) | attention and Mamba targets and drafts in all four pairings, `k` ∈ {1, 2, 3, 5} |
| inputs | `k = 0` and an empty prompt raise `ValueError` | |

Tolerances: impl identities `1e-10 + 1e-9·|x|`; enumerated probabilities `1e-12 + 1e-10·|x|`. The
statistical tests use fixed seeds, so a run is reproducible. Their limit: a cell expected only a few
times in 4,000 samples can vanish without failing either test.

A pass also needs all 1,743 cases and minimum counts of: first draft rejected, rejection after some
acceptances, all accepted (bonus), partial and full greedy rounds in assisted generation, Mamba
rollbacks reaching back past the last forward call, and Mamba generation rounds.

**Result** (`results/check.json`): passed. About 4,900 comparisons. Largest error over tolerance:
`2.3e-5`. Largest cell deviation: 3.4 standard errors; largest chi-square: 0.37 of its limit.

**Planted bugs** (`results/self_test.json`): all 12 are reported as mismatches, after an unmutated run
passes.

| Mutation | First reported by |
|---|---|
| accept with probability `q/p` instead of `p/q` | hf sampling (accepted count) |
| resample from `norm(abs(p − q))` | hf sampling (tokens) |
| resample a rejection from `p` | hf sampling (tokens) |
| bonus token from `p_{k−1}` instead of `p_k` | hf sampling (tokens) |
| keep counting acceptances after a rejection | hf sampling (accepted count) |
| greedy compares draft `i` with the argmax at `i + 1` | vllm greedy |
| acceptance uniforms drawn in float32 | hf sampling (tokens) |
| attention rollback forgets one token fewer | rollback (token count) |
| Mamba rollback trims snapshots but keeps the latest state | rollback (token count) |
| draft forgets `k − n` tokens instead of `k − 1 − n` | generation (draft state on return) |
| temperature ignored | vllm sampled (17 standard errors at `T = 0.7`) |
| exact round law accepts with `q(x)` instead of `min(p, q)` | theorem |

**Things the references do that a test has to work around.**

- vLLM's sampler draws its random numbers from a counter-based generator keyed by (seed, position). Its
  samples cannot be matched draw for draw, so they are compared by distribution.
- vLLM accepts when `p(x) > u·q(x)` with `u` in `(0, 1]`; the paper and transformers accept when
  `u ≤ p(x)/q(x)` with `u` in `[0, 1)`. The two differ only on a set of probability zero.
- transformers' cropping can't put back a recurrent state: `is_croppable` is false once a
  linear-attention layer holds one
  ([`cache_utils.py#L1107`](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/cache_utils.py#L1107)).
  So the Mamba rollback is checked against a fresh run instead.

## 6. In the engines

| Piece | vLLM 0.34.0 | Agrees? |
|---|---|---|
| Acceptance | `target_logprob > log(u) + draft_logprob` ([#L716](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py#L716)), `u` from `tl_rand32(seed, pos)` ([#L616](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py#L616)) | yes, in distribution |
| Greedy | accept iff the draft equals the target argmax ([#L634](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py#L634)) | yes, exactly |
| Residual | computed in log space, `log p + log1p(−q/p)` where `q < p`, then a Gumbel-max sample ([#L986](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py#L986)) | yes, in distribution |
| One-hot drafts (no draft logits, e.g. n-gram) | `log q = 0` ([#L201](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py#L201)), so accept with probability `p(x)` and resample from `p` without `x` | yes, in distribution |
| Temperature | target logits divided before the sampler ([`sampler.py#L260`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/worker/gpu/sample/sampler.py#L260)); draft logits cached before temperature and divided in the kernel | yes |
| Mamba state for drafts | one state slot per speculative token (`num_speculative_blocks`, [`abstract.py#L94`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/layers/mamba/abstract.py#L94)), unless ReplaySSM is on: it caches recent SSM inputs and verifies the window from one checkpoint ([`cache.py#L332`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/config/cache.py#L332)) | same idea as `MambaLM` (a state per position); ReplaySSM not implemented here |

In transformers 5.22.2, `_speculative_sampling`
([`generation/utils.py#L4359`](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/generation/utils.py#L4359))
accepts when `r_i <= p_i/q_i`
([#L4394](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/generation/utils.py#L4394)),
and assisted generation crops the cache by `candidate_length − n_matches`
([#L4161](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/generation/utils.py#L4161)).
impl matches both exactly.

## 7. References

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192v2)
  (Leviathan, Kalman, Matias). Algorithm 1, the expected tokens per round, and the speed-up model.
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318v1)
  (Chen et al.). The same rule, with the proof that it recovers the target distribution.
- transformers v5.22.2 and vLLM `62d4bc3c7e` source, linked above.

## 8. Self-check questions

1. Write the probability of emitting token `x` at the first draft position as a sum of two terms, and
   show it equals `p(x)`.
2. Show that the rejection probability at a position is `Σ max(p − q, 0)`, the total variation distance.
3. If `q = p`, what is the acceptance rate, and how many tokens does a round emit?
4. If `q` only proposes tokens `p` gives probability 0, what happens each round, and is the output still
   distributed like `p`?
5. `k = 4`, two drafts kept. How many tokens does the target forget? The draft?
6. Why can't a Mamba layer roll back by truncation, and what does keeping one state per draft position
   cost for Mamba-Codestral-7B at `k = 4`?
7. vLLM's sampler cannot be matched draw for draw with any other implementation. What does `check.py`
   compare instead, and what is the smallest effect that comparison can miss?

# 04-transformer-variants

## 1. What this covers

- RoPE: rotating query and key pairs by position, and the two ways to stretch it past the trained
  context (Llama 3.1's frequency bands, YaRN as DeepSeek-V2 uses it).
- Grouped-query attention (GQA): fewer key/value heads than query heads, and converting an MHA
  checkpoint by mean-pooling heads.
- Mixture-of-experts routing: Mixtral's softmax over the top-k, DeepSeek-V2's unnormalized gates,
  scaling factor, group-limited routing and shared experts.
- Multi-head latent attention (MLA): caching one small latent per token instead of per-head keys and
  values, and the "absorbed" form that attends to the latent directly.
- Incremental decode: feeding a sequence in pieces through a cache gives the same logits as feeding it
  at once, for every variant above.

Questions you should be able to answer when done:

1. Why does the score between two RoPE-rotated vectors depend only on their distance?
2. What do "neox" and "gptj" RoPE styles differ in, and why can a checkpoint be converted between them?
3. Llama 3 8B has 32 query heads and 8 key/value heads. How much KV cache per token does that save over
   MHA, and what does a 128K-token context cost in bf16?
4. Mixtral writes its gate as `Softmax(TopK(logits))`; transformers computes softmax, then top-k, then
   renormalizes. Why are they the same?
5. Why does DeepSeek-V2 need a separate rotary key `k_R` next to the latent `c`?
6. In MLA, why does vLLM run prefill the "naive" way and decode the "absorbed" way?

## 2. Concepts

- **Head dimension `d`, heads `H_q`, key/value heads `H_kv`.** Each query head attends with its own
  `d`-wide query; key/value heads are shared by `H_q / H_kv` query heads (a group).
- **MHA, GQA, MQA.** `H_kv = H_q`, `1 < H_kv < H_q`, and `H_kv = 1`.
- **KV cache.** The keys and values of past tokens, kept so a new token does not recompute them. Its
  size per token is what limits batch size and context length in serving.
- **RoPE.** Rotary position embedding: rotate each pair of query and key dimensions by
  `position × θ_i`. Positions enter only through those angles.
- **Context extension.** Running a model past the length it was trained at by slowing some
  frequencies `θ_i` down (interpolation) while keeping others.
- **Expert, router, gate.** An expert is one feed-forward block; the router scores every expert for a
  token; the gate weights are how much each chosen expert's output counts.
- **Latent `c` (MLA).** A `d_c`-wide vector per token from which every head's key and value are
  recovered by fixed up-projections `W_UK`, `W_UV`.
- **Absorbed MLA.** Folding `W_UK` into the query and `W_UV` into the output so attention runs on
  the latents themselves.

## 3. How it works

### RoPE

For a pair `(x_a, x_b)` at position `m` and frequency `θ_i = base^(−2i/d)`:

```
(x_a, x_b)  ->  (x_a cos mθ_i − x_b sin mθ_i,   x_b cos mθ_i + x_a sin mθ_i)
```

A rotation by `m` followed by the inverse of a rotation by `n` is a rotation by `m − n`, so
`⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩` depends only on the distance
(RoFormer eq. 16). `check.py` confirms it for every RoPE kind
below at positions up to 8,000.

Which dimensions form a pair is a convention. RoFormer pairs neighbours `(x_{2i}, x_{2i+1})`; vLLM calls
that "gptj" style and DeepSeek-V2 uses it. Llama implementations pair `(x_i, x_{i+d/2})`, "neox"
style. The two differ by a fixed permutation of dimensions (`impl.neox_to_gptj`), which a checkpoint
converter can fold into the query and key weights; a model must be run with the style it was
trained with.

**Stretching the context.** Dimension `i` turns `L·θ_i / 2π` times over a context of `L` tokens.
Fast dimensions have turned through every angle many times during training; slow ones have not, so at
longer contexts they produce angles the model never saw. Both methods slow down only the slow ones:

| Method | Rule | Real setting |
|---|---|---|
| Llama 3.1 | wavelength `2π/θ_i` below `L_orig / high` (2,048): keep; above `L_orig / low` (8,192): divide by `factor`; between: blend linearly | `factor` 8, `low` 1, `high` 4, `L_orig` 8,192, `d` 128, base 500,000 → 29 dimensions kept, 6 blended, 29 divided |
| YaRN (DeepSeek-V2) | dimensions turning more than `β_fast` (32) times over `L_orig` are kept, fewer than `β_slow` (1) times are divided by `s`, a linear ramp between; cos and sin scale by `mscale(s, a)/mscale(s, b)` and the softmax scale by `mscale(s, b)²`, with `mscale(s, x) = 0.1·x·ln s + 1` | `s` 40, `L_orig` 4,096, `d_R` 64, `a = b = 0.707` → 11 dimensions kept, 12 on the ramp, 9 divided; softmax scale × 1.590 |

The Llama 3.1 values are those transformers notes as "the original implementation"
([`modeling_rope_utils.py#L644`](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/modeling_rope_utils.py#L644)).
The YaRN values are from DeepSeek-V2 section 3.1.4, which states
the length factor as `0.0707 ln s + 1`.

### Grouped-query attention

Query head `h` reads key/value head `h // (H_q / H_kv)`. Only the `H_kv` key/value heads are cached, so the
cache shrinks by `H_q / H_kv`. GQA converts an MHA checkpoint by
mean-pooling each group's key and value projections (section 2.2), then trains for 5% of the original
compute. Mean pooling beat keeping one head or starting from random heads (their Figure 4).

| Llama 3 8B, bf16 | Elements per token | Per token | 128K tokens |
|---|---|---|---|
| as trained: 32 layers, 8 KV heads, `d` 128 | 32 · 2 · 8 · 128 = 65,536 | **128 KiB** | **16 GiB** |
| if it were MHA (32 KV heads) | 262,144 | 512 KiB | 64 GiB |

Configurations: Llama 3 Table 3. Mixtral
8x7B (Table 1: 32 layers, 8 KV heads, `d` 128) caches the same 128 KiB per token.

### Mixture of experts

| | Mixtral | DeepSeek-V2 |
|---|---|---|
| Scores | `softmax(logits)` | `softmax(logits)` |
| Choice | top-k | top-k within the `topk_group` groups whose best expert scores highest |
| Gate | chosen scores renormalized to sum 1 | chosen scores as they are, × `routed_scaling_factor` |
| Always on | | shared experts |
| Real setting | 8 experts, top 2 | 160 routed + 2 shared, top 6, experts on 8 devices, at most 3 devices per token |

Mixtral's paper writes the gate as `Softmax(TopK(x·W_g))` with the rest set to `−∞`. Softmax over all
experts, then renormalizing the top k, gives the same numbers: the shared denominator cancels.
transformers computes the second form, vLLM's CPU router the first; `check.py` confirms they agree.

DeepSeek-V2 bounds how many devices a token's experts sit on (section 2.2.2). Engines implement this as
groups: experts split into `n_group` groups (one per device), and only the `topk_group` groups with the
best single expert are eligible.

Each token runs only `k` experts, so compute per token is that of a `k`-expert model while the weights
are those of all of them. Mixtral: each expert is 3 · 4,096 · 14,336 = 176 M weights, 45.1 B over 8
experts and 32 layers, of which 11.3 B run per token. DeepSeek-V2 runs 8 of 162 experts per MoE layer
(4.9%). These count expert weights only, not attention or embeddings.

### Multi-head latent attention

Per token, DeepSeek-V2 caches the latent `c = RMSNorm(W_DKV h)` (`d_c` = 512) and one rotated key
`k_R = RoPE(W_KR h)` shared by all heads (`d_R` = 64). Each head's key is `[W_UK^h c ; k_R]` and its
value `W_UV^h c` (section 2.1.2–2.1.3).

Why a separate `k_R`: the absorbed form needs `q · (W_UK c)` to equal `(W_UKᵀ q) · c` with one fixed
`W_UK`. RoPE puts a position-dependent rotation between them, which cannot be folded into a fixed
matrix. So the latent carries no position, and position enters only through the small shared `k_R`.

| DeepSeek-V2, bf16 | Elements per token | Per token | 128K tokens |
|---|---|---|---|
| MLA: 60 layers · (512 + 64) | 34,560 | **67.5 KiB** | **8.4 GiB** |
| MHA with the same 128 heads, `d` 128 (paper Table 1: `2 n_h d_h l`) | 1,966,080 | 3,840 KiB | 480 GiB |

MLA caches 1.76% of that MHA (57× less). The paper puts it as GQA with 2.25 groups (Table 1);
`check.py` confirms `4 × 34,560` equals GQA with 9 groups.

**Three ways to decode with MLA.** Per layer, one new token, `S` cached tokens, multiply-adds
(arithmetic from the shapes, not a measurement):

| Option | Cache per token | Work at `S` = 4,096 |
|---|---|---|
| cache the expanded keys and values | 3,840 KiB (the MHA row above) | attention only |
| cache `c`, expand all `S` keys and values every step | 67.5 KiB | 6.9 · 10¹⁰ |
| cache `c`, absorbed: attend to `c` directly | 67.5 KiB | 5.9 · 10⁸ (117× less) |

For prefill, where every position is a query, the ranking flips: absorbed attention works in `d_c + d_R`
= 576 dimensions per score instead of 192, and at `S` = 4,096 costs 3.2× the naive form. That is why
vLLM's MLA runs the naive ("compute friendly") form for prefill and the absorbed ("data-movement
friendly") form for decode (§6). `impl.mla_forward` and `impl.mla_forward_absorbed` give the same
logits within the float64 identity tolerance (§5).

### Incremental decode

Each variant caches what the next token needs: rotated keys and values per key/value head, or `c` and
`k_R`. A token's cache entry depends only on that token and its position, so feeding a sequence in any
split (one shot, one token at a time, or chunks) gives the same logits and the same cache.

## 4. Code and how to run it

| File | What it does |
|---|---|
| `impl.py` | RoPE (`rope_inv_freq` for default, Llama 3.1 and YaRN; `apply_rope` in both styles); `gqa_forward` with `KVCache`, `mha_to_gqa`; `route` (Mixtral and DeepSeek-V2 routing), `moe_forward`; `mla_forward` and `mla_forward_absorbed` with `LatentCache`; `kv_elements_per_token`; a decoder (`Model`, `model_forward`) mixing them. Float64, torch only. |
| `oracle.py` | Small transformers Llama, Mixtral and DeepSeek-V2 models (eager attention, eager experts, float64) with their weights copied into impl; transformers' and vLLM's RoPE and router functions. Pins transformers 5.22.2 and vLLM 0.34.0. |
| `check.py` | Every comparison below; writes `results/check.json`. `--self-test` plants 24 bugs and requires each to be reported, after an unmutated run passes. |

```bash
scripts/run-capped --mem 4G --cpus 4 -- .venv/bin/python 04-transformer-variants/check.py --out 04-transformer-variants/results/check.json
scripts/run-capped --mem 4G --cpus 4 -- .venv/bin/python 04-transformer-variants/check.py --self-test --out 04-transformer-variants/results/self_test.json
```

The check takes about 1 second after a few seconds of imports, with a peak RSS of about 720 MiB (most of
it importing vLLM).

## 5. Checks and expected results

**References.** transformers 5.22.2 (tag `v5.22.2`) and vLLM 0.34.0 (commit `62d4bc3c7e`).

| Family | What is compared | Matrix |
|---|---|---|
| rope freq | impl's frequencies and cos/sin factor vs transformers' RoPE init and vLLM's rotary classes | default (`d` 8, 64, 128), Llama 3.1 (small, the real 8B setting, another factor), YaRN (small, DeepSeek-V2's, without mscale) |
| rope apply | `impl.apply_rope` vs transformers' Llama `apply_rotary_pos_emb` and vLLM's `forward_static`, both given impl's cos and sin; vs DeepSeek-V2's `apply_rotary_emb` | the same 9 settings, 24 positions up to 3× the trained context, 3 heads |
| rope identity | score depends only on distance; norms kept (times the cos/sin factor); gptj = neox under the permutation | 9 settings × 3 seeds |
| routing | `impl.route` vs transformers' Mixtral and DeepSeek-V2 routers and vLLM's `_softmax_topk` and `_grouped_topk`: same experts per token, same gate weights | `E` ∈ {6, 8, 16, 32, 64}, `k` 1–8, 64 tokens; every valid group split with 1 … `n_group − 1` kept groups, scaling 1 and 16, renormalized or not |
| moe identity | softmax-top-k-renormalize = softmax over the top-k logits; sparse MoE = dense weighted sum (with shared experts); tokens per expert; group-limited choices stay inside the kept groups | 3 router shapes |
| gqa identity | GQA = MHA with key/value heads repeated per group; mean-pooling heads that agree within a group changes nothing; pooled key and value weights are group means | `(H_q, H_kv)` ∈ (4,4), (4,2), (4,1), (8,2), (6,3), (12,4) |
| model hf | impl's decoder vs transformers' whole model, logits of 48 tokens; and every routed expert alone vs transformers' experts module | Llama: MHA, GQA, MQA, default and Llama 3.1 RoPE; Mixtral: (8, 2), (4, 1), (6, 3); DeepSeek-V2: with and without the query low-rank path, greedy and group-limited routing, default RoPE and YaRN |
| model decode | the same 48 tokens in one shot, one at a time, `[5, 1, 1, 41]` and `[17, 31]`: impl chunked = impl one-shot; MLA absorbed = naive; transformers through `DynamicCache` = impl | all 13 models |
| kv size | cache elements of impl and of transformers' `DynamicCache` = `kv_elements_per_token`; the real-model numbers above; MLA = GQA with 2.25 groups | |
| inputs | too few caches for the layers, pooling a non-MHA layer, groups that do not split the experts, kept groups smaller than `k`: all raise `ValueError` | |

Tolerances: impl identities `1e-10 + 1e-9·|x|`; a float64 reference given impl's cos and sin
`1e-12 + 1e-12·|x|`; references that compute in float32 (RoPE frequencies, router softmax, DeepSeek-V2's
`apply_rotary_emb`) `1e-6 + 1e-5·|x|`; whole transformers models `1e-5 + 1e-5·|x|`, because their RMSNorm
and RoPE run in float32 even for a float64 model. The float64 identities cover what the looser
tolerances could hide.

A pass also needs all 76 cases and minimum counts of: Llama 3.1 dimensions in the blended band, YaRN
dimensions on the ramp, tokens whose experts change because of the group limit, routed tokens, experts
checked one by one (all 68), and single-token decode steps.

**Result** (`results/check.json`): passed. 1,281 comparisons. Largest error over tolerance: `0.18`
(decode against transformers). The closest pair of expert scores across the routing tests was `3.0e-5`
apart, so the float32 routers' choices are not decided by rounding.

**Planted bugs** (`results/self_test.json`): all 24 are reported as mismatches, after an unmutated run
passes.

| Mutation | First reported by |
|---|---|
| neox rotation with the wrong sign on the second half | rope apply (transformers) |
| gptj rotation pairs `(i, i + d/2)` | rope apply (DeepSeek-V2) |
| `θ_i = base^(−i/d)` | rope freq |
| Llama 3.1 divides the blended band by the full factor | rope freq |
| YaRN interpolates the fast dimensions and keeps the slow ones | rope freq |
| YaRN leaves cos and sin unscaled | rope freq (factor) |
| each forward call starts its positions at 0 | model decode |
| query head `h` reads key/value head `h mod H_kv` | gqa identity |
| a token cannot attend to itself | model hf |
| MHA-to-GQA keeps the first head of each group | gqa identity (keys) |
| MHA-to-GQA pools keys but keeps the first value head | gqa identity (values) |
| tokens-per-expert puts everything on expert 0 | moe identity |
| a model runs with fewer caches than layers | inputs |
| Mixtral gates not renormalized | routing (transformers) |
| gate weights are raw logits | routing (transformers) |
| groups ranked by summed instead of best score | routing (transformers) |
| group limit ignored | routing (transformers) |
| `routed_scaling_factor` not applied | routing (transformers) |
| shared experts left out | moe identity |
| MLA caches the latent before its RMSNorm | model hf |
| MLA softmax scale without YaRN's `mscale²` | model hf |
| absorbed MLA scores only the latent, not `k_R` | model decode (absorbed vs naive) |
| `kv_b_proj` split as `W_UV` then `W_UK` | model hf |
| KV-size formula counts keys but not values | kv size |

**Things the references do that a test has to work around.**

- transformers' RMSNorm and RoPE cast to float32 even in a float64 model, so its logits agree with impl
  to about `1e-6`, not `1e-15`. The float64 identities and the float64 RoPE comparisons cover that gap.
- transformers' default experts path (`grouped_mm`) refuses float64; the small models select the
  eager experts path.
- A transformers config needs `hidden_size` divisible by the head count even when `head_dim` is set.
- `DeepseekV2TopkRouter` deletes the `norm_topk_prob` it inherits, so DeepSeek-V2 gates are never
  renormalized whatever the config says
  ([`modular_deepseek_v2.py#L166`](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/deepseek_v2/modular_deepseek_v2.py#L166)).
  DeepSeek-V2 sets it to false, so this matches the model. Recorded here; not reported upstream.

## 6. In the engines

| Piece | vLLM 0.34.0 | Agrees? |
|---|---|---|
| RoPE apply | `forward_static` rotates the first `rotary_dim` dimensions, neox or gptj ([`base.py#L161`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/layers/rotary_embedding/base.py#L161)) | yes, exactly (given the same cos/sin) |
| Llama 3.1 frequencies | [`llama3_rope.py#L33`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/layers/rotary_embedding/llama3_rope.py#L33) | yes (float32 there) |
| YaRN frequencies and factor | [`deepseek_scaling_rope.py#L73`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py#L73); the model passes the original context length as `max_position_embeddings` | yes (float32 there) |
| Top-k routing (CPU) | `_softmax_topk`: top-k of the logits, then softmax ([`cpu_router.py#L138`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/layers/fused_moe/router/cpu_router.py#L138)) | yes |
| Group-limited routing (CPU) | `_grouped_topk` ([`cpu_router.py#L26`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/layers/fused_moe/router/cpu_router.py#L26)) | yes |
| MLA prefill vs decode | compute-friendly form for prefill, data-movement-friendly (absorbed) for decode ([`mla_attention.py#L12`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/layers/attention/mla_attention.py#L12)) | same two forms as impl; not run |
| MLA absorption | `W_UK_T` and `W_UV` split from `kv_b_proj` once at load ([`#L1428`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/layers/attention/mla_attention.py#L1428)); decode multiplies the query by `W_UK_T` ([`#L1362`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/layers/attention/mla_attention.py#L1362)) and the latent output by `W_UV` ([`#L1633`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/model_executor/layers/attention/mla_attention.py#L1633)) | same algebra as `impl.mla_forward_absorbed`; not run (GPU kernels) |

In transformers 5.22.2: Llama RoPE `rotate_half` and `apply_rotary_pos_emb`
([`modeling_llama.py#L119`](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/llama/modeling_llama.py#L119)),
Mixtral's router
([`modeling_mixtral.py#L95`](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/mixtral/modeling_mixtral.py#L95)),
DeepSeek-V2's router
([`modeling_deepseek_v2.py#L85`](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py#L85)),
and its attention, which caches the normed latent and `k_R` and expands them on every call
([`modeling_deepseek_v2.py#L395`](https://github.com/huggingface/transformers/blob/v5.22.2/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py#L395)).
impl matches all of them to the tolerances above.

## 7. References

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v5)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245v3)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088v1)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434v5)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783v3)
- transformers [v5.22.2](https://github.com/huggingface/transformers/tree/v5.22.2), vLLM
  [`62d4bc3c7e`](https://github.com/vllm-project/vllm/tree/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca)

## 8. Self-check questions

1. Rotate `q` by position 1,000 and `k` by 990, then both by 5,000 more. Why is `q·k` unchanged?
2. A checkpoint trained with gptj-style RoPE is loaded into a neox-style engine. What goes wrong, and
   what permutation fixes it?
3. Mixtral 8x7B serving 32 sequences of 32K tokens: how much KV cache in bf16? What if it were MHA?
4. DeepSeek-V2 with `n_group = 8`, `topk_group = 3`, top 6: what is the largest number of devices one
   token's routed experts can sit on?
5. Why does MLA's absorbed form need `k_R` to be shared across heads and kept out of the latent?
6. At what ratio of new tokens to cached tokens would you switch MLA from the absorbed to the naive form,
   by the multiply-add counts in section 3?

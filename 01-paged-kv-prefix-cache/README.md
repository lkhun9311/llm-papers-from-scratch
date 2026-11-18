# 01 · Paged KV cache and prefix caching

A block manager that pages the KV cache and shares cached prefixes between requests, checked step by
step against vLLM v0.34.0's `KVCacheManager` on 900 random traces (about 180,000 operations).

## 1. What this covers

- Splitting each sequence's KV cache into fixed-size blocks, and the block table that maps a sequence's
  logical blocks to physical ones.
- Reference counts, so several sequences can point at one physical block.
- Prefix caching: giving each full block an identity that covers its whole prefix, so a new request
  can reuse blocks another request already computed.
- Eviction: which free block gets overwritten first, and why the order matters.

Questions you should be able to answer when done:

1. Why does paging remove external fragmentation but not all internal fragmentation?
2. Why must a block's identity include every token before it, not only its own tokens?
3. A 32-token prompt is fully cached and the block size is 16. How many tokens are recomputed, and why?
4. Why can a finished request's blocks stay useful after it releases them?
5. Why are a finished request's blocks released tail-first?

## 2. Concepts

| Term | Meaning |
|---|---|
| Block | Storage for the K and V vectors of `B` consecutive tokens, in every layer. Here it is only an id. |
| Block table | Per sequence: logical block `i` → physical block id. Append-only while the sequence runs. |
| Reference count | Number of live sequences whose block table contains the block. |
| Identity (block hash) | `h_i = H(h_{i-1}, tokens of block i)`. Only full blocks have one. |
| Prefix hit | Leading blocks of a new request whose identities are already cached. |
| Free queue | Blocks with reference count 0, ordered by eviction priority. A free block may still be cached. |
| Null block | Physical block 0, reserved as a placeholder and never allocated (a vLLM convention). |

## 3. How it works

### Paging

Without paging, a server reserves a contiguous region for each sequence's maximum length, because the
final length is unknown. With paging, a sequence holds `ceil(T / B)` blocks for `T` tokens, taken from
anywhere in the pool. The only waste is the unfilled tail of the last block:

```
waste_paged      = C · Σ_i (B · ceil(T_i / B) − T_i)   <  n · B · C
waste_contiguous = C · Σ_i (R_i − T_i)                  (+ gaps between regions)
```

`C` is the KV bytes per token across all layers, `R_i` is the reserved length, and `n` is the number of
sequences. Uniform blocks also remove gaps between regions inside the pool.

### Identity and sharing

Block `i` of two requests holds the same K/V only if all tokens up to the end of block `i` are equal,
because attention at every position depends on every earlier token. So the identity chains the parent:
`h_i = H(h_{i-1}, block_i)`. `impl.py` uses BLAKE2b over the parent digest and the block's tokens. vLLM
uses SHA-256 with extra keys (LoRA id, multimodal ids, cache salt) that this directory leaves out.

A trailing partial block has no identity. It is private to its sequence, so the sequence can keep
writing into it, and it never needs copy-on-write.

### The rules borrowed from vLLM

The PagedAttention paper defines paging and sharing. These policy details come from vLLM V1 instead,
so the two implementations can be compared operation by operation:

| Rule | Why |
|---|---|
| A request may hit at most `T − 1` tokens. Hits are whole blocks, so a fully cached block-aligned prompt recomputes its whole last block. | The last token must go through the model to produce logits. |
| Lookup is read-only. Allocation takes the hits first, then allocates new blocks. If the pool is short, nothing changes. | A refused request must leave the cache exactly as it was. |
| A hit on a free block counts against free capacity. | The hit takes that block out of the free queue. |
| New blocks come from the head of the free queue. Reusing a cached block drops its identity. | The head holds the blocks least likely to be hit again. |
| Release is tail-first. Blocks with no identity go to the head; cached blocks go to the tail. | Later requests share early blocks far more often than late ones. An uncached block can never be hit, so it should be reused first. |
| Duplicate cached blocks with the same identity are kept, and a hit returns the oldest. | Merging them would rewrite a running request's block table, which must stay append-only. |

### Memory cost

KV bytes per token per layer, with element size `s` bytes:

| Attention | Bytes per token per layer |
|---|---|
| Multi-head (MHA) | `2 · H_q · d · s` |
| Grouped-query (GQA) | `2 · H_kv · d · s` |
| Multi-head latent (MLA, latent cached) | `(r + d_rope) · s` |

Worked example, Qwen2.5-7B in BF16 (28 layers, 4 KV heads, head size 128):

- `C = 2 · 28 · 4 · 128 · 2 = 57,344` bytes, which is **56 KiB per token**. With all 28 heads as KV
  heads (MHA), it would be 392 KiB.
- 4,097 tokens with `B = 16` need 257 blocks, which is **224.875 MiB**. Of that, 840 KiB (15 tokens) is
  tail waste.
- Reserving 8,192 tokens contiguously for the same request takes **448 MiB**.
- An 8 GiB KV budget holds about 149,796 tokens.

DeepSeek-V2 with MLA (60 layers, `r = 512`, `d_rope = 64`, BF16) caches 69,120 bytes, which is **67.5 KiB
per token**. Its 128 heads as plain MHA would need 4.69 MiB per token, 71× more.

Compute: a prefix hit of `k` blocks skips `k · B` tokens of prefill. Lookup costs one hash per full
block and one dictionary probe per hit.

## 4. Code and how to run it

| File | What it does |
|---|---|
| `impl.py` | The block manager: `lookup`, `allocate`, `free_seq` |
| `oracle.py` | Drives vLLM's `KVCacheManager` with a synthetic `Request` on CPU. No model, no tensors. Refuses any vLLM other than 0.34.0. |
| `check.py` | Random traces fed to both sides, compared after every operation, plus a self-test |

```bash
uv venv -p 3.12 .venv
VIRTUAL_ENV=.venv uv pip install \
  https://github.com/vllm-project/vllm/releases/download/v0.34.0/vllm-0.34.0+cpu-cp38-abi3-manylinux_2_39_x86_64.whl \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

cd 01-paged-kv-prefix-cache
../scripts/run-capped --mem 4G --cpus 4 -- ../.venv/bin/python check.py --out results/check.json
../scripts/run-capped --mem 4G --cpus 4 -- ../.venv/bin/python check.py --self-test --out results/self_test.json
```

## 5. Checks and expected results

**Reference:** vLLM 0.34.0 CPU wheel (tag `v0.34.0`, commit `62d4bc3c7e`), torch 2.14.0+cpu, Python 3.12.

**Traces:** three pools, `(B, N)` = (4, 12), (4, 40) and (16, 20), with 300 traces each and 200
operations per trace. Each operation is one of admit (a new or waiting request), decode one token,
finish, or preempt. Prompts reuse three random prefixes and are cut at random points, so prefix hits,
divergence inside a block, block-aligned prompts, and pool exhaustion all happen.

**Compared after every operation (exact equality):** admission success, hit length, hit block ids,
every live block table, every reference count, the full free-queue order, and the cached blocks grouped
by identity in hit-priority order. Identities are compared as groups of block ids, not as hash values,
so the two sides share no hashing code.

**Invariants checked on `impl.py` alone:** a block's reference count equals the number of block tables
that contain it; reference count 0 if and only if the block is in the free queue; a block shared by two
sequences has an identity; every cached block is listed under its own identity; every sequence has
enough blocks for its computed tokens.

**Coverage required for a pass:** every trace runs to the end. Prefix hits, evictions, duplicate cached
blocks, and last-token-capped full hits each happen at least 20 times.

**Result** ([results/check.json](results/check.json)): **passed**. 900 of 900 traces ran: 51,848
admissions (14,626 refused for lack of blocks), 68,389 decodes, 29,750 finishes, and 15,387 preemptions.
Coverage: 174,808 hit tokens, 51,725 evictions, 497 duplicate blocks, and 412 capped full hits. The run
took 6.7 s with a peak RSS of 534 MiB.

**Self-test** ([results/self_test.json](results/self_test.json)): **passed**. Each planted bug was
reported as a mismatch:

| Planted bug | Caught by |
|---|---|
| Release head-first | free-queue order |
| No last-token rule | hit length |
| Hit the newest duplicate | hit block ids |
| Uncached blocks released to the tail | free-queue order |
| Hits on free blocks cost nothing at admission | admission success |

## 6. In the engines

vLLM v0.34.0 (commit `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`):

| Rule | vLLM code |
|---|---|
| Hit cap of `T − 1` tokens | [`kv_cache_manager.py` `get_computed_blocks`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/core/kv_cache_manager.py#L288), [`request.py` `get_max_cache_hit_length`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/request.py#L352) |
| Admission counts free hits | [`single_type_kv_cache_manager.py` `get_num_blocks_to_allocate`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/core/single_type_kv_cache_manager.py#L181) |
| Allocate, then cache full blocks | [`kv_cache_manager.py` `allocate_slots`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/core/kv_cache_manager.py#L401) |
| Pop from the head, drop the identity | [`block_pool.py` `get_new_blocks`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/core/block_pool.py#L681) |
| Take a free block back on a hit | [`block_pool.py` `touch`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/core/block_pool.py#L774) |
| Tail-first release, uncached to the head | [`single_type_kv_cache_manager.py` `free`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/core/single_type_kv_cache_manager.py#L596), [`block_pool.py` `free_blocks`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/core/block_pool.py#L798) |
| Duplicates kept, oldest returned | [`block_pool.py` `BlockHashToBlockMap`](https://github.com/vllm-project/vllm/blob/62d4bc3c7e33b1b6c906e233e6af2d3506c482ca/vllm/v1/core/block_pool.py#L38) |

**Agreement:** exact on every compared quantity in the traced scope. No difference was found, so
nothing was reported upstream.

**Out of scope here:** sliding-window and Mamba groups (see 02), speculative decoding, chunked prefill,
KV connectors and offloading, multimodal, LoRA and cache-salt extra keys, and the watermark.

## 7. References

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180v1)
- vLLM v0.34.0 source, pinned above.
- Model configurations used in §3: Qwen2.5-7B (`config.json` at revision `d149729398`) and DeepSeek-V2
  (`config.json` at revision `4461458f18`), from Hugging Face.

## 8. Self-check questions

1. A request with prompt length 17 and `B = 16` is admitted with no hits, and then one token is decoded.
   How many blocks does it hold, and how many of them have an identity?
2. Two requests share a 20-token prefix and `B = 16`. Which of their blocks can be shared?
3. Why can a refused admission not touch any of the hit blocks?
4. Where does a block with no identity go when it is freed, and why there?
5. Give a case where the same identity is held by two physical blocks. Which one does the next hit return?
6. How much KV memory per token would Qwen2.5-7B need if it used MHA instead of GQA?

# llm-papers-from-scratch

Small, readable implementations of LLM inference papers, each checked numerically against a reference
implementation (Hugging Face `transformers`, vLLM, or the paper's own code).

The goal is understanding that holds up when reading a production engine: every directory ends with the
question "where does vLLM or SGLang do this, and does my version agree with it?"

## Study order

The order follows what an inference-engine runtime has to get right first: cache memory, then hybrid
state, then speculative decoding, then the model-side variants and kernels.

| # | Directory | Topic | Papers | Status |
|---|---|---|---|---|
| 01 | [`01-paged-kv-prefix-cache/`](01-paged-kv-prefix-cache/) | Paged KV blocks, block tables, prefix-cache hashing, eviction and reference counts | [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180v1) | done |
| 02 | [`02-hybrid-state/`](02-hybrid-state/) | Sliding-window KV and recurrent (SSM) state next to full attention: sizes, checkpoints, reuse, invalidation | [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150v2), [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752v2), [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060v1) | done |
| 03 | [`03-speculative-decoding/`](03-speculative-decoding/) | Draft-and-verify, rejection sampling that preserves the target distribution, KV and SSM-state rollback | [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192v2), [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318v1) | done |
| 04 | [`04-transformer-variants/`](04-transformer-variants/) | GQA, RoPE, MoE top-k routing, MLA KV-size arithmetic; prefill vs incremental decode equivalence | [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245v3), [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v5), [Mixtral of Experts](https://arxiv.org/abs/2401.04088v1), [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434v5) | done |
| 05 | `05-flash-attention-numerics/` | Tiling and online softmax; floating-point error against naive attention | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135v2), [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691v1) | planned |

Later, if time allows: quantization (LLM.int8, SmoothQuant, GPTQ, AWQ) and GRPO.

## Layout of every directory

Each directory follows [TEMPLATE.md](TEMPLATE.md): code, a check against a reference with its result saved
as JSON, and a README that works as a study note.

## Running

This repository runs on CPU. Every model run goes through a memory cap so an experiment cannot take the
machine down:

```bash
scripts/run-capped --mem 8G --cpus 8 -- python 01-paged-kv-prefix-cache/check.py
```

`run-capped` refuses to start without `MemAvailable >= cap + 4 GiB`, runs the command in a systemd scope
with `MemoryMax` and no swap, pins it to the first `--cpus` cores, and reports exit 137 as a likely OOM kill.

## Link checks

```bash
scripts/check_links.py            # arXiv titles and versions, pinned GitHub permalinks (network)
scripts/check_links.py --offline  # format rules only
```

arXiv links must be versioned `abs` links whose text equals the paper title. GitHub code links must be
pinned to a commit or a version tag.

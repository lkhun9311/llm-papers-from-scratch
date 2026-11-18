"""Reference side of check.py: transformers' speculative sampling and assisted generation, and vLLM's
rejection sampler run on CPU through Triton's interpreter.

vLLM's sampler is Triton GPU kernels. With TRITON_INTERPRET=1 Triton executes the same kernel source on
CPU with numpy. One shim is needed, outside vLLM's code: the interpreter has no libdevice, and the
resample kernel calls libdevice.log1p, so numpy's log1p is supplied for it.
The sampler draws its random numbers from a counter-based generator keyed by (seed, position), so its
samples cannot be matched draw for draw; check.py compares their distribution over many seeds instead.
"""
from __future__ import annotations

import os

os.environ["TRITON_INTERPRET"] = "1"  # before triton is imported anywhere

import numpy as np
import torch
import transformers
import triton.language as tl
import triton.language.extra.libdevice as libdevice
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM
from transformers.generation.utils import _speculative_sampling
from triton.runtime import interpreter

TRANSFORMERS_VERSION = "5.22.2"
VLLM_VERSION = ("0.34.0", "g62d4bc3c7")
DT = torch.float64


def _log1p(x, _semantic=None):
    h = x.handle
    return tl.core.tensor(interpreter.TensorHandle(np.log1p(h.data), h.dtype.scalar), x.type)


libdevice.log1p = _log1p

from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample  # noqa: E402


def check_version() -> None:
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise SystemExit(f"oracle pinned to transformers {TRANSFORMERS_VERSION}, found {transformers.__version__}")
    import vllm
    import vllm._version

    if (vllm.__version__, vllm._version.__commit_id__) != VLLM_VERSION:
        raise SystemExit(f"oracle pinned to vLLM {VLLM_VERSION}, found {vllm.__version__} {vllm._version.__commit_id__}")


# ------------------------------------------------------------------------------------------------
# transformers
# ------------------------------------------------------------------------------------------------


def hf_speculative_sampling(drafts: torch.Tensor, draft_logits: torch.Tensor, target_logits: torch.Tensor):
    """transformers' _speculative_sampling for one sequence. drafts [k], draft_logits [k, V], target_logits [k+1, V].

    Returns (tokens emitted, accepted count). It draws from torch's global generator.
    """
    k = drafts.shape[0]
    ids = torch.cat([torch.zeros(1, dtype=torch.long), drafts])[None]  # the function reads the last k
    tokens, n = _speculative_sampling(ids, draft_logits[None], k, target_logits[None])
    return [int(t) for t in tokens[0]], int(n)


def tiny_llama(seed: int, vocab: int, hidden: int, layers: int, heads: int, kv_heads: int,
               noise: float = 0.0) -> LlamaForCausalLM:
    """A random Llama. With noise > 0, the same model with every weight perturbed: a draft that agrees
    with the target often but not always."""
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=kv_heads, max_position_embeddings=256,
                      bos_token_id=None, eos_token_id=None, pad_token_id=None, attn_implementation="eager")
    m = LlamaForCausalLM(cfg).to(DT).eval()
    with torch.no_grad():  # sharpen the output so drafts are sometimes right and sometimes wrong
        m.lm_head.weight.mul_(8.0)
        g = torch.Generator().manual_seed(seed + 1000)
        for prm in m.parameters():
            prm.add_(noise * prm.abs().mean() * torch.randn(prm.shape, generator=g, dtype=DT))
    return m


class HFModel:
    """A transformers causal LM behind impl's Model interface: DynamicCache, rolled back with cache.crop."""

    def __init__(self, model: LlamaForCausalLM):
        self.model = model
        self.cache = DynamicCache(config=model.config)

    @property
    def consumed(self) -> int:
        return self.cache.get_seq_length()

    def forward(self, tokens: list[int]) -> torch.Tensor:
        with torch.inference_mode():
            out = self.model(input_ids=torch.tensor([tokens]), past_key_values=self.cache, use_cache=True)
        return out.logits[0].detach().clone()

    def rollback(self, drop: int) -> None:
        self.cache.crop(-drop)

    def state(self) -> dict[str, torch.Tensor]:
        return {f"{n}{i}": getattr(layer, n).detach().clone() for i, layer in enumerate(self.cache.layers)
                for n in ("keys", "values")}


def hf_greedy(model, prompt: list[int], n_new: int, assistant=None, k: int | None = None) -> list[int]:
    """generate() with do_sample=False, optionally with an assistant model drafting k tokens a round."""
    kw = {}
    if assistant is not None:
        assistant.generation_config.num_assistant_tokens = k
        assistant.generation_config.num_assistant_tokens_schedule = "constant"
        assistant.generation_config.assistant_confidence_threshold = 0.0
        kw["assistant_model"] = assistant
    with torch.inference_mode():
        out = model.generate(torch.tensor([prompt]), max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False,
                             attention_mask=torch.ones(1, len(prompt), dtype=torch.long), **kw)
    return [int(t) for t in out[0]]


# ------------------------------------------------------------------------------------------------
# vLLM
# ------------------------------------------------------------------------------------------------


def vllm_rejection_sample(target_logits: torch.Tensor, draft_logits: torch.Tensor | None, drafts: torch.Tensor,
                          temperature: float, seeds: torch.Tensor, first_pos: int = 100):
    """vLLM's rejection_sample for R requests that each verify k drafts.

    target_logits [R, k+1, V], draft_logits [R, k, V] or None (one-hot drafts, as from n-gram lookup),
    drafts [R, k], seeds [R]. temperature 0 means greedy. Returns (sampled [R, k+1], num_sampled [R]);
    only the first num_sampled entries of a row are meaningful.

    vLLM's sampler divides target logits by the temperature before rejection_sample
    (Sampler.apply_sampling_params); the draft logits are cached before temperature and divided inside
    the kernels. The adapter does the first division the same way.
    """
    R, k1, V = target_logits.shape
    k = k1 - 1
    tl_ = target_logits.to(torch.float32)
    if temperature > 0:
        tl_ = tl_ / temperature
    cache = None
    if draft_logits is not None:
        cache = DraftLogitsCache(R, k, V, dtype=torch.float32, fill=0.0, device="cpu")
        cache.values.copy_(draft_logits.to(torch.float32))
    # draft_sampled[start + i + 1] is the i-th draft of a request; slot `start` holds the token before them.
    draft_sampled = torch.cat([torch.zeros(R, 1, dtype=torch.long), drafts], dim=1).reshape(-1)
    sampled, num = rejection_sample(
        tl_.reshape(R * k1, V).contiguous(), cache, draft_sampled,
        cu_num_logits=torch.arange(R + 1, dtype=torch.int32) * k1,
        pos=(first_pos + torch.arange(k1)).repeat(R),
        idx_mapping=torch.arange(R, dtype=torch.int32),
        expanded_idx_mapping=torch.arange(R, dtype=torch.int32).repeat_interleave(k1),
        expanded_local_pos=torch.arange(k1, dtype=torch.int32).repeat(R),
        temperature=torch.full((R,), float(temperature)),
        seed=seeds.to(torch.int64),
        num_speculative_steps=k,
    )
    return sampled.clone(), num.clone()

"""Thin adapters around Hugging Face transformers (and one vLLM function), the reference side of check.py.

Each builder makes a tiny randomly initialised one-layer model in float64, and exposes:
- the weights, copied into impl's plain weight classes (so impl never imports transformers), and
- a runner that feeds chunks through the real module with the library's own cache and returns the
  layer outputs and the cache tensors after every chunk.

The reference casts some intermediates to float32 internally (scan steps, the gated norm), so it is
not a pure float64 oracle; check.py compares against it with a calibrated tolerance and keeps the
impl-vs-impl identities at float64 tolerance.
"""
from __future__ import annotations

import torch
import transformers
from transformers import (DynamicCache, LongformerConfig, Mamba2Config, Mamba2ForCausalLM, MambaConfig,
                          MambaForCausalLM, MistralConfig, MistralForCausalLM)
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention
from transformers.models.mamba2 import modeling_mamba2
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

import impl

TRANSFORMERS_VERSION = "5.22.2"
VLLM_VERSION = ("0.34.0", "g62d4bc3c7")
DT = torch.float64


def check_version() -> None:
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise SystemExit(f"oracle pinned to transformers {TRANSFORMERS_VERSION}, found {transformers.__version__}")
    import vllm
    import vllm._version

    if (vllm.__version__, vllm._version.__commit_id__) != VLLM_VERSION:
        raise SystemExit(f"oracle pinned to vLLM {VLLM_VERSION}, found {vllm.__version__} {vllm._version.__commit_id__}")


def vllm_gated_norm(y: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float, n_groups: int):
    """vLLM's Mixer2RMSNormGated.forward_native, run on CPU with one tensor-parallel rank.

    Building the layer needs an initialised tensor-parallel group, so the method is called on a stub
    holding exactly the attributes it reads. The code that runs is vLLM's own.
    """
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.mamba_mixer2 import Mixer2RMSNormGated

    width = y.shape[-1]
    stub = SimpleNamespace(tp_size=1, tp_rank=0, full_hidden_size=width, group_size=width // n_groups,
                           per_rank_hidden_size=width, n_groups=n_groups, variance_epsilon=eps,
                           use_rms_norm=True, weight=weight)
    return _det(Mixer2RMSNormGated.forward_native(stub, y, z))


def _det(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone()


# ------------------------------------------------------------------------------------------------
# Mistral: causal sliding window
# ------------------------------------------------------------------------------------------------


class MistralRef:
    """One-layer Mistral with sliding_window=w, eager attention, DynamicCache."""

    def __init__(self, w: int, seed: int, hidden=16, heads=4, kv_heads=2, head_dim=4):
        torch.manual_seed(seed)
        cfg = MistralConfig(vocab_size=32, hidden_size=hidden, intermediate_size=32, num_hidden_layers=1,
                            num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim,
                            sliding_window=w, max_position_embeddings=256, attn_implementation="eager")
        self.model = MistralForCausalLM(cfg).to(DT).eval()
        self.attn = self.model.model.layers[0].self_attn
        self.w = w
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim

    def run(self, ids: torch.Tensor, chunks: list[int]):
        """Feed ids [B, T] in chunks. Per chunk: attention inputs/outputs and the cache's K/V after it."""
        cache = DynamicCache(config=self.model.config)
        rec: dict = {}

        def hook(mod, args, kwargs, out):
            rec["hidden"] = _det(kwargs["hidden_states"] if "hidden_states" in kwargs else args[0])
            cos, sin = kwargs["position_embeddings"]
            rec["cos"], rec["sin"] = _det(cos), _det(sin)
            rec["out"] = _det(out[0])

        h = self.attn.register_forward_hook(hook, with_kwargs=True)
        steps, pos = [], 0
        try:
            with torch.inference_mode():
                for n in chunks:
                    self.model(input_ids=ids[:, pos : pos + n], past_key_values=cache, use_cache=True)
                    layer = cache.layers[0]
                    steps.append(dict(rec, keys=_det(layer.keys), values=_det(layer.values),
                                      seq_length=cache.get_seq_length(), layer_class=type(layer).__name__))
                    pos += n
        finally:
            h.remove()
        return steps

    def qkv(self, hidden, cos, sin):
        """Projections plus RoPE using the library's own helper. RoPE itself is covered in 04."""
        B, L, _ = hidden.shape
        a = self.attn
        q = (hidden @ a.q_proj.weight.T).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = (hidden @ a.k_proj.weight.T).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)
        v = (hidden @ a.v_proj.weight.T).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def o_proj(self, attn: torch.Tensor) -> torch.Tensor:
        B, H, L, d = attn.shape
        return attn.transpose(1, 2).reshape(B, L, H * d) @ self.attn.o_proj.weight.T


# ------------------------------------------------------------------------------------------------
# Longformer: symmetric local attention (no global tokens)
# ------------------------------------------------------------------------------------------------


class LongformerRef:
    """LongformerSelfAttention with attention_window = 2 * one_sided, local attention only."""

    def __init__(self, one_sided: int, seed: int, hidden=16, heads=4):
        torch.manual_seed(seed)
        cfg = LongformerConfig(hidden_size=hidden, num_attention_heads=heads, attention_window=[2 * one_sided],
                               num_hidden_layers=1, attention_probs_dropout_prob=0.0)
        self.attn = LongformerSelfAttention(cfg, layer_id=0).to(DT).eval()
        self.one_sided, self.heads, self.head_dim = one_sided, heads, hidden // heads

    def run(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [B, L, hidden], L a multiple of the window. Returns the attention output [B, L, hidden]."""
        B, L, _ = hidden.shape
        local = torch.zeros(B, L, dtype=DT)  # 0 = attend locally (-10000 masked, +10000 global)
        with torch.inference_mode():
            out = self.attn(hidden, attention_mask=local,
                            is_index_masked=torch.zeros(B, L, dtype=torch.bool),
                            is_index_global_attn=torch.zeros(B, L, dtype=torch.bool), is_global_attn=False)
        return _det(out[0])

    def qkv(self, hidden):
        B, L, _ = hidden.shape
        a = self.attn
        f = lambda lin: (hidden @ lin.weight.T + lin.bias).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        return f(a.query), f(a.key), f(a.value)


# ------------------------------------------------------------------------------------------------
# Mamba and Mamba-2 mixers
# ------------------------------------------------------------------------------------------------


def _mixer_runner(model, mixer, x: torch.Tensor, chunks: list[int]):
    """Run mixer over x [B, T, hidden] in chunks with the model's cache; outputs and states per chunk."""
    cache = DynamicCache(config=model.config)
    steps, pos = [], 0
    with torch.inference_mode():
        for n in chunks:
            y = mixer(x[:, pos : pos + n], cache_params=cache)
            layer = cache.layers[0]
            steps.append(dict(out=_det(y), conv=_det(layer.conv_states[0]), ssm=_det(layer.recurrent_states[0]),
                              layer_class=type(layer).__name__))
            pos += n
    return steps


class MambaRef:
    def __init__(self, seed: int, hidden=16, state=3, conv=4, expand=2, dt_rank=2):
        torch.manual_seed(seed)
        cfg = MambaConfig(vocab_size=32, hidden_size=hidden, state_size=state, conv_kernel=conv, expand=expand,
                          num_hidden_layers=1, time_step_rank=dt_rank, use_mambapy=False)
        self.model = MambaForCausalLM(cfg).to(DT).eval()
        self.mixer = self.model.backbone.layers[0].mixer
        _perturb(self.mixer)

    def weights(self) -> impl.MambaWeights:
        m = self.mixer
        return impl.MambaWeights(
            in_w=_det(m.in_proj.weight), conv_w=_det(m.conv1d.weight[:, 0]), conv_b=_det(m.conv1d.bias),
            x_w=_det(m.x_proj.weight), dt_w=_det(m.dt_proj.weight), dt_b=_det(m.dt_proj.bias),
            A_log=_det(m.A_log), D=_det(m.D), out_w=_det(m.out_proj.weight))

    def run(self, x, chunks):
        return _mixer_runner(self.model, self.mixer, x, chunks)


class Mamba2Ref:
    def __init__(self, seed: int, n_groups: int, chunk_size: int, dt_limit=(0.0, float("inf")),
                 hidden=16, heads=4, head_dim=8, state=3, conv=4):
        torch.manual_seed(seed)
        cfg = Mamba2Config(vocab_size=32, hidden_size=hidden, num_heads=heads, head_dim=head_dim, state_size=state,
                           conv_kernel=conv, expand=heads * head_dim // hidden, n_groups=n_groups,
                           chunk_size=chunk_size, num_hidden_layers=1, time_step_limit=tuple(dt_limit))
        self.model = Mamba2ForCausalLM(cfg).to(DT).eval()
        self.mixer = self.model.backbone.layers[0].mixer
        _perturb(self.mixer)
        self.cfg = cfg

    def weights(self) -> impl.Mamba2Weights:
        m, c = self.mixer, self.cfg
        return impl.Mamba2Weights(
            in_w=_det(m.in_proj.weight), conv_w=_det(m.conv1d.weight[:, 0]), conv_b=_det(m.conv1d.bias),
            dt_bias=_det(m.dt_bias), A_log=_det(m.A_log), D=_det(m.D), norm_w=_det(m.norm.weight),
            out_w=_det(m.out_proj.weight), num_heads=c.num_heads, head_dim=c.head_dim, n_groups=c.n_groups,
            state_size=c.state_size, eps=m.norm.variance_epsilon, dt_limit=tuple(c.time_step_limit))

    def run(self, x, chunks):
        return _mixer_runner(self.model, self.mixer, x, chunks)


def mamba2_chunk_scan(*args, **kwargs):
    """The library's pure-torch chunked SSD, for checks of the core alone (e.g. finite dt limits)."""
    return modeling_mamba2.mamba2_chunk_scan(*args, **kwargs)


def _perturb(mixer) -> None:
    """Move every parameter away from its structured init.

    The default init makes D all ones, A_log the same log(1..N) row in every Mamba channel (log(1..H)
    across Mamba-2 heads), and norm weights all ones, so a bug that swaps channels, skips D or ignores a
    bias can cancel out. Random offsets make
    each of those visible. Scales keep dt and the decays in a range where the recurrence matters.
    """
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in mixer.named_parameters():
            noise = torch.randn(p.shape, generator=g, dtype=p.dtype)
            if name in ("A_log", "D"):
                p.add_(0.5 * noise)
            elif name.endswith("bias") and "dt" in name:
                p.add_(0.5 * noise)
            elif name == "norm.weight":
                p.copy_(1.0 + 0.3 * noise)
            else:
                p.add_(0.3 * noise)

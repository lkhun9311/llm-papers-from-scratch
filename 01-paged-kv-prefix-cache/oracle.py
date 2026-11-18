"""Drive vLLM's own KVCacheManager on CPU with the same operations as impl.PagedKVCache.

Nothing here reimplements vLLM logic: it builds a synthetic Request and calls the public manager
methods, then reads the resulting state. No model and no GPU memory are involved
(`kv_cache_tensors=[]`, one tiny FullAttentionSpec), as in vLLM's tests/v1/core/test_prefix_caching.py.
"""
from __future__ import annotations

import torch
import vllm
from packaging.version import Version
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_block_hash, get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request


EXPECTED_VLLM = "0.34.0"  # the +cpu local build suffix is allowed, nothing else
if Version(vllm.__version__).public != EXPECTED_VLLM:
    raise SystemExit(f"oracle expects vLLM {EXPECTED_VLLM}, found {vllm.__version__}")


class VllmKVCache:
    def __init__(self, num_blocks: int, block_size: int, max_model_len: int = 4096):
        init_none_hash(sha256)
        self.block_size = block_size
        config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32),
                )
            ],
        )
        self.mgr = KVCacheManager(
            config,
            max_model_len=max_model_len,
            scheduler_block_size=block_size,
            hash_block_size=block_size,
            enable_caching=True,
        )
        self.pool = self.mgr.block_pool
        # vLLM block hash -> token prefix, learned from each request's own block_hashes. Used only to
        # express vLLM's cache in terms of token prefixes, so it can be compared with impl.py.
        self.prefix_by_hash: dict[bytes, tuple[int, ...]] = {}

    def new_request(self, req_id: str, tokens: list[int]) -> Request:
        sp = SamplingParams(max_tokens=1 << 20)
        sp.update_from_generation_config({}, eos_token_id=None)
        req = Request(
            request_id=req_id,
            prompt_token_ids=list(tokens),
            mm_features=None,
            sampling_params=sp,
            pooling_params=None,
            block_hasher=get_request_block_hasher(self.block_size, sha256, 0),
        )
        return req

    def lookup(self, req: Request):
        blocks, num_hit_tokens, _ = self.mgr.get_computed_blocks(req)
        return blocks, num_hit_tokens

    def allocate(self, req: Request, num_new_tokens: int, blocks, num_hit_tokens: int):
        out = self.mgr.allocate_slots(req, num_new_tokens, num_hit_tokens, blocks)
        tokens = req.all_token_ids
        bs = self.block_size
        for i, h in enumerate(req.block_hashes):
            self.prefix_by_hash.setdefault(h, tuple(tokens[: (i + 1) * bs]))
        return out

    def free_seq(self, req: Request) -> None:
        self.mgr.free(req)

    # ---- state views, same shape as impl.PagedKVCache ----

    def block_table(self, req_id: str) -> list[int]:
        return list(self.mgr.get_block_ids(req_id)[0])

    def ref_cnt(self) -> list[int]:
        return [b.ref_cnt for b in self.pool.blocks]

    def free_order(self) -> list[int]:
        return [b.block_id for b in self.pool.free_block_queue.get_all_free_blocks()]

    def cache_view(self) -> list[tuple[tuple[int, ...], list[int]]]:
        """(token prefix, holders in hit-priority order) for every cached hash. Reads
        BlockHashToBlockMap's two dicts: `_cache` holds the block a hit returns, `_dup_ids` all
        holders in insertion order."""
        m = self.pool.cached_block_hash_to_block
        out = []
        for key, first in m._cache.items():
            dups = m._dup_ids.get(key)
            holders = list(dups) if dups is not None else [first]
            assert holders[0] == first, "primary holder is not first in insertion order"
            out.append((self.prefix_by_hash[get_block_hash(key)], holders))
        return sorted(out)

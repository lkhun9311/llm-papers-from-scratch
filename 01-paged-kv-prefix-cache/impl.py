"""Paged KV-cache block manager with prefix caching.

Paging (fixed-size blocks, per-sequence block tables, reference counts) comes from the PagedAttention
paper. Prefix caching and its policies are not in that paper: hashed block identities, the null
block, last-token recompute, oldest-duplicate hits, free-queue eviction order. Those follow vLLM V1
on purpose, so the two can be compared step by step. The code is written separately; README.md lists
each borrowed policy and where vLLM implements it.

Scope: one full-attention KV cache group, synchronous scheduling, text tokens only, no speculation.

No model and no tensors: a "block" is only an id plus metadata. What is simulated is the bookkeeping
that decides which physical block holds which tokens.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

NULL_BLOCK = 0  # block 0 is reserved as a placeholder and never handed out


def chain_hashes(tokens: list[int], block_size: int) -> list[bytes]:
    """Identity of each full block: digest(parent identity, this block's tokens).

    Two blocks are interchangeable only when every earlier token is identical too, so the identity
    must cover the whole prefix. Chaining the parent digest does that in fixed size per block.
    A trailing partial block has no identity and is never shared.
    """
    out: list[bytes] = []
    parent = b""
    for start in range(0, len(tokens) - block_size + 1, block_size):
        h = hashlib.blake2b(parent, digest_size=16)
        h.update(repr(tokens[start : start + block_size]).encode())
        parent = h.digest()
        out.append(parent)
    return out


@dataclass
class Seq:
    """One request: its tokens, how many have KV computed, and its block table."""

    seq_id: str
    tokens: list[int]
    num_computed: int = 0
    block_table: list[int] = field(default_factory=list)
    num_cached_blocks: int = 0  # leading blocks of block_table already registered in the cache


class PagedKVCache:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.ref_cnt = [0] * num_blocks
        self.identity: list[bytes | None] = [None] * num_blocks
        # Free queue: head is evicted first. An OrderedDict gives O(1) removal from the middle,
        # which happens when a free but still cached block gets a prefix hit.
        self.free: OrderedDict[int, None] = OrderedDict((b, None) for b in range(1, num_blocks))
        # identity -> block ids holding it, oldest first. Duplicates are kept on purpose: merging
        # them would change a running request's block table, and block tables are append-only.
        self.cached: dict[bytes, list[int]] = {}
        # Coverage counters, so a check can prove the interesting paths actually ran.
        self.stats = {"evictions": 0, "duplicates": 0, "capped_full_hits": 0}

    # ---- lookup -------------------------------------------------------------------------------

    def lookup(self, seq: Seq) -> list[int]:
        """Longest cached prefix, in whole blocks. Read-only.

        At most len(tokens) - 1 tokens may hit: the last token must run through the model to
        produce logits, and because hits are whole blocks a fully cached prompt recomputes its
        whole last block.
        """
        max_blocks = max(len(seq.tokens) - 1, 0) // self.block_size
        idents = chain_hashes(seq.tokens, self.block_size)
        hit: list[int] = []
        for ident in idents[:max_blocks]:
            holders = self.cached.get(ident)
            if not holders:
                break
            hit.append(holders[0])
        if len(hit) == max_blocks < len(idents) and idents[max_blocks] in self.cached:
            self.stats["capped_full_hits"] += 1  # the last-token rule cost a whole cached block
        return hit

    # ---- allocation ---------------------------------------------------------------------------

    def allocate(self, seq: Seq, num_new_tokens: int, hit: list[int]) -> list[int] | None:
        """Give `seq` slots for `num_new_tokens` more tokens, adopting the `hit` blocks first.

        Returns the newly allocated block ids, or None with no state changed when the pool is short.
        """
        bs = self.block_size
        total = seq.num_computed + len(hit) * bs + num_new_tokens
        need = max(-(-total // bs) - len(seq.block_table) - len(hit), 0)
        # A hit on a free block takes that block out of the free pool, so it costs capacity too.
        hits_from_free = sum(1 for b in hit if self.ref_cnt[b] == 0)
        if need + hits_from_free > len(self.free):
            return None

        for b in hit:
            if self.ref_cnt[b] == 0:
                del self.free[b]
            self.ref_cnt[b] += 1
        seq.block_table.extend(hit)
        if hit:
            seq.num_cached_blocks = len(seq.block_table)

        new_blocks = []
        for _ in range(need):
            b, _ = self.free.popitem(last=False)
            self._uncache(b)  # evicting: its old contents are about to be overwritten
            self.ref_cnt[b] = 1
            new_blocks.append(b)
        seq.block_table.extend(new_blocks)

        # Register blocks that are now full. Draft tokens do not exist here, so every token
        # counted in `total` is final.
        self._cache_full_blocks(seq, min(total, len(seq.tokens)))
        return new_blocks

    def _cache_full_blocks(self, seq: Seq, num_tokens: int) -> None:
        idents = chain_hashes(seq.tokens[:num_tokens], self.block_size)
        for i in range(seq.num_cached_blocks, len(idents)):
            b = seq.block_table[i]
            self.identity[b] = idents[i]
            holders = self.cached.setdefault(idents[i], [])
            if holders:
                self.stats["duplicates"] += 1
            holders.append(b)
        seq.num_cached_blocks = max(seq.num_cached_blocks, len(idents))

    def _uncache(self, b: int) -> None:
        ident = self.identity[b]
        if ident is None:
            return
        self.stats["evictions"] += 1
        holders = self.cached[ident]
        holders.remove(b)
        if not holders:
            del self.cached[ident]
        self.identity[b] = None

    # ---- release ------------------------------------------------------------------------------

    def free_seq(self, seq: Seq) -> None:
        """Drop `seq`'s references. Blocks reaching zero go back to the free queue.

        The tail is released first: a later request is more likely to share an early block than a
        late one, so late blocks should be evicted sooner. Blocks with no identity can never be hit
        again and go to the head; cached ones go to the tail and stay reusable until evicted.
        """
        to_head, to_tail = [], []
        for b in reversed(seq.block_table):
            self.ref_cnt[b] -= 1
            if self.ref_cnt[b] == 0:
                (to_tail if self.identity[b] is not None else to_head).append(b)
        head = OrderedDict((b, None) for b in to_head)
        head.update(self.free)
        head.update((b, None) for b in to_tail)
        self.free = head
        seq.block_table = []
        seq.num_cached_blocks = 0

    # ---- views used by the checks -------------------------------------------------------------

    def free_order(self) -> list[int]:
        return list(self.free)

    def cache_classes(self) -> list[list[int]]:
        """Cached block ids grouped by identity, in hit-priority order (the first one is what a hit
        returns). Comparable with vLLM without comparing hash values."""
        return sorted(list(v) for v in self.cached.values())

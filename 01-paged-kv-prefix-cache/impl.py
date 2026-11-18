"""Paged KV-cache block manager with prefix caching.

The PagedAttention paper introduces paging (fixed-size blocks, per-sequence block tables, reference
counts) and also describes caching shared prefixes (its section 4.4). Finding shared prefixes
automatically by hashing every full block, and the exact policies used here (null block, last-token
recompute, oldest-duplicate hits, free-queue eviction order), follow vLLM V1 at the pinned commit, so
the two can be compared step by step. The code is written separately; README.md lists each borrowed
policy and where vLLM implements it.

Scope: one full-attention KV cache group, hash block size equal to block size, synchronous
scheduling, text tokens only, no speculation, no chunked prefill.

No model and no tensors: a "block" is only an id plus metadata. What is simulated is the bookkeeping
that decides which physical block holds which tokens.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

NULL_BLOCK = 0  # block 0 is reserved as a placeholder and never handed out


def block_digest(parent: bytes, block_tokens: list[int]) -> bytes:
    """Identity of one full block: digest(parent identity, this block's tokens).

    Two blocks may be reused for each other only when every earlier token is identical too, so the
    identity must cover the whole prefix. Chaining the parent digest does that in fixed size.
    """
    h = hashlib.blake2b(parent, digest_size=16)
    h.update(repr(block_tokens).encode())
    return h.digest()


@dataclass
class Seq:
    """One request: its tokens, how many have KV computed, and its block table."""

    seq_id: str
    tokens: list[int]
    num_computed: int = 0
    block_table: list[int] = field(default_factory=list)
    num_cached_blocks: int = 0  # leading blocks of block_table already registered in the cache
    idents: list[bytes] = field(default_factory=list)  # identities of full blocks, built incrementally


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
        self.stats = {"hit_lookups": 0, "capped_full_hit_lookups": 0, "evictions": 0, "duplicates": 0}

    def _idents(self, seq: Seq) -> list[bytes]:
        """Identities of all full blocks of seq.tokens. Tokens are append-only, so this only extends."""
        bs = self.block_size
        while (len(seq.idents) + 1) * bs <= len(seq.tokens):
            i = len(seq.idents)
            parent = seq.idents[-1] if seq.idents else b""
            seq.idents.append(block_digest(parent, seq.tokens[i * bs : (i + 1) * bs]))
        return seq.idents

    # ---- lookup -------------------------------------------------------------------------------

    def lookup(self, seq: Seq) -> list[int]:
        """Longest cached prefix, in whole blocks. Does not change the cache.

        Probes blocks in order until the first miss or the hit limit. At most len(tokens) - 1 tokens
        may hit: the last token must run through the model to produce logits, and because hits are
        whole blocks a fully cached block-aligned prompt recomputes its whole last block.
        """
        max_blocks = max(len(seq.tokens) - 1, 0) // self.block_size
        idents = self._idents(seq)
        hit: list[int] = []
        for ident in idents[:max_blocks]:
            holders = self.cached.get(ident)
            if not holders:
                break
            hit.append(holders[0])
        if hit:
            self.stats["hit_lookups"] += 1
        if len(hit) == max_blocks < len(idents) and idents[max_blocks] in self.cached:
            self.stats["capped_full_hit_lookups"] += 1  # the last-token rule gave up a cached block
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
        bs = self.block_size
        idents = self._idents(seq)[: num_tokens // bs]
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

    def cache_holders(self) -> dict[bytes, list[int]]:
        """identity -> holder block ids in hit-priority order (the first one is what a hit returns)."""
        return {k: list(v) for k, v in self.cached.items()}

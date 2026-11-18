"""Differential check: impl.PagedKVCache against vLLM's KVCacheManager on identical random traces.

Each trace is a sequence of scheduler-like operations (admit, decode, finish, preempt). After every
operation the two sides must agree on: admission success, prefix-hit length and block ids, every
live block table, every request's token list and computed-token count (running and waiting), every
block's reference count, the full free-queue order, and the cache as (token prefix -> holder block ids
in hit-priority order). A refused allocation must leave each side exactly as it was. Independent
invariants are checked on impl alone.

A pass also requires that every trace runs to the end and that prefix-hit lookups, capped full-hit
lookups, evictions and duplicate cached blocks each happen at least MIN_COVERAGE times.

`--mutate NAME` plants a known bug in impl. `--self-test` runs every mutation in a subprocess and
passes only if each one is reported as a mismatch (not a crash, not a pass).

Usage:  python check.py [--traces 300] [--ops 200] [--mutate NAME] [--out results/check.json]
        python check.py --self-test [--out results/self_test.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import impl
from impl import PagedKVCache, Seq
from oracle import VllmKVCache

CONFIGS = [  # (block_size, num_blocks): small pools force eviction; 16 matches vLLM's default
    (4, 12),
    (4, 40),
    (16, 20),
]
MAX_TRACES, MAX_OPS = 2000, 2000  # keeps a run to minutes and well under the memory cap
MIN_COVERAGE = 20


class Mismatch(Exception):
    pass


def compare(step: str, a, b) -> None:
    if a != b:
        raise Mismatch(f"{step}: impl={a!r} vllm={b!r}")


def gen_token(rng: random.Random) -> int:
    """Mostly a tiny alphabet so prefixes collide often; 0..7 also makes t and t+4 differ only above
    the low bits, and some real-vocabulary ids catch hashing that drops information."""
    r = rng.random()
    if r < 0.85:
        return rng.randint(0, 7)
    return rng.randint(1000, 151_000)


def gen_prompt(rng: random.Random, block_size: int, prefixes: list[list[int]]) -> list[int]:
    """Prompts share prefixes often, so hits, partial hits and mid-block divergence all occur."""
    base = list(rng.choice(prefixes)) if rng.random() < 0.7 else []
    cut = rng.randint(0, len(base))  # may cut inside a block: divergence mid-block
    tail = [gen_token(rng) for _ in range(rng.randint(0, 2 * block_size + 1))]
    return (base[:cut] + tail) or [gen_token(rng)]


def check_invariants(kv: PagedKVCache, live: dict[str, Seq]) -> None:
    """Properties that must hold whatever the eviction policy; written without reference to vLLM."""
    n = len(kv.ref_cnt)
    bs = kv.block_size
    owners = [0] * n
    for s in live.values():
        for b in s.block_table:
            owners[b] += 1
    for b in range(1, n):
        if kv.ref_cnt[b] != owners[b]:
            raise Mismatch(f"invariant: block {b} ref_cnt {kv.ref_cnt[b]} != owners {owners[b]}")
        if (kv.ref_cnt[b] == 0) != (b in kv.free):
            raise Mismatch(f"invariant: block {b} ref_cnt {kv.ref_cnt[b]} but in_free={b in kv.free}")
        if owners[b] > 1 and kv.identity[b] is None:
            raise Mismatch(f"invariant: block {b} shared by {owners[b]} sequences but has no identity")
        ident = kv.identity[b]
        if ident is not None and b not in kv.cached.get(ident, ()):
            raise Mismatch(f"invariant: block {b} has an identity but is not listed in the cache")
    for ident, holders in kv.cached.items():
        for b in holders:
            if kv.identity[b] != ident:
                raise Mismatch(f"invariant: block {b} listed under the wrong identity")
    for s in live.values():
        if len(s.block_table) < -(-s.num_computed // bs):
            raise Mismatch(f"invariant: {s.seq_id} has {len(s.block_table)} blocks for {s.num_computed} tokens")
        for i, b in enumerate(s.block_table[: s.num_cached_blocks]):
            if kv.prefix_of.get(kv.identity[b]) != tuple(s.tokens[: (i + 1) * bs]):
                raise Mismatch(f"invariant: {s.seq_id} block {i} (id {b}) is cached under another prefix")


def snapshot(mine: PagedKVCache, ref: VllmKVCache, seqs: list[tuple[Seq, object]]):
    return (
        copy.deepcopy((mine.ref_cnt, list(mine.free), mine.cache_view(),
                       [(s.block_table, s.num_computed, s.num_cached_blocks, len(s.tokens)) for s, _ in seqs])),
        copy.deepcopy((ref.ref_cnt(), ref.free_order(), ref.cache_view(),
                       [(ref.block_table(r.request_id) if s.block_table else [], r.num_computed_tokens, r.num_tokens)
                        for s, r in seqs])),
    )


def run_trace(seed: int, block_size: int, num_blocks: int, num_ops: int) -> dict:
    rng = random.Random(seed)
    mine = PagedKVCache(num_blocks, block_size)
    ref = VllmKVCache(num_blocks, block_size)
    prefixes = [[gen_token(rng) for _ in range(rng.randint(block_size, 3 * block_size))] for _ in range(3)]
    running: dict[str, tuple[Seq, object]] = {}
    waiting: list[tuple[Seq, object]] = []
    counts = {"admit_attempts": 0, "admit_ok": 0, "admit_refused": 0, "hit_tokens": 0,
              "decode_ok": 0, "decode_refused": 0, "finish": 0, "preempt": 0}
    next_id = 0

    def all_seqs() -> list[tuple[Seq, object]]:
        return list(running.values()) + waiting

    def try_allocate(step, s, req, n_new, hit, blocks, ref_hit_tokens):
        before = snapshot(mine, ref, all_seqs() + [(s, req)])
        got = mine.allocate(s, n_new, hit)
        got_ref = ref.allocate(req, n_new, blocks, ref_hit_tokens)
        compare(f"{step} success", got is not None, got_ref is not None)
        if got is None:  # a refusal must change nothing, on either side
            after = snapshot(mine, ref, all_seqs() + [(s, req)])
            compare(f"{step} impl state after refusal", after[0], before[0])
            compare(f"{step} vllm state after refusal", after[1], before[1])
        return got is not None

    def sync_state(step: str) -> None:
        compare(f"{step} ref_cnt", mine.ref_cnt[1:], ref.ref_cnt()[1:])
        compare(f"{step} free order", mine.free_order(), ref.free_order())
        compare(f"{step} cache", mine.cache_view(), ref.cache_view())
        for s, req in all_seqs():
            compare(f"{step} tokens {s.seq_id}", s.tokens, list(req.all_token_ids))
            compare(f"{step} computed {s.seq_id}", s.num_computed, req.num_computed_tokens)
        for sid, (s, _) in running.items():
            compare(f"{step} block table {sid}", s.block_table, ref.block_table(sid))
        check_invariants(mine, {sid: s for sid, (s, _) in running.items()})

    def append_token(s, req):
        tok = gen_token(rng)
        s.tokens.append(tok)
        req.append_output_token_ids(tok)

    def release(sid, requeue):
        s, req = running.pop(sid)
        mine.free_seq(s)
        ref.free_seq(req)
        if requeue:
            s.num_computed = 0
            req.num_computed_tokens = 0
            waiting.append((s, req))

    for op_i in range(num_ops):
        r = rng.random()
        step = f"seed={seed} B={block_size} N={num_blocks} op#{op_i}"
        if r < 0.35 or not running:
            if waiting and rng.random() < 0.5:
                s, req = waiting.pop(0)
            else:
                tokens = gen_prompt(rng, block_size, prefixes)
                s = Seq(f"r{next_id}", tokens)
                req = ref.new_request(f"r{next_id}", tokens)
                next_id += 1
            counts["admit_attempts"] += 1
            hit = mine.lookup(s)
            blocks, ref_hit_tokens = ref.lookup(req)
            compare(f"{step} admit {s.seq_id} hit tokens", len(hit) * block_size, ref_hit_tokens)
            compare(f"{step} admit {s.seq_id} hit blocks", hit, list(blocks.get_block_ids()[0]))
            n_new = len(s.tokens) - len(hit) * block_size
            if try_allocate(f"{step} admit {s.seq_id}", s, req, n_new, hit, blocks, ref_hit_tokens):
                counts["admit_ok"] += 1
                counts["hit_tokens"] += len(hit) * block_size
                s.num_computed = len(s.tokens)
                req.num_computed_tokens = len(s.tokens)
                append_token(s, req)
                running[s.seq_id] = (s, req)
            else:
                counts["admit_refused"] += 1
                waiting.append((s, req))
        elif r < 0.75:
            sid = rng.choice(sorted(running))
            s, req = running[sid]
            if try_allocate(f"{step} decode {sid}", s, req, 1, [], None, 0):
                counts["decode_ok"] += 1
                s.num_computed += 1
                req.num_computed_tokens += 1
                append_token(s, req)
            else:
                counts["decode_refused"] += 1  # out of blocks: preempt it, as a scheduler would
                counts["preempt"] += 1
                release(sid, requeue=True)
        elif r < 0.92:
            counts["finish"] += 1
            release(rng.choice(sorted(running)), requeue=False)
        else:
            counts["preempt"] += 1
            release(rng.choice(sorted(running)), requeue=True)
        sync_state(step)
    counts.update(mine.stats)
    return counts


MUTATIONS = {
    # Each plants one plausible bug. The check must report a mismatch for every one of them.
    "free_head_first": "release blocks head-first instead of tail-first",
    "no_last_token_rule": "allow the whole prompt to hit (no recompute of the last token)",
    "newest_duplicate": "hit the newest duplicate block instead of the oldest",
    "uncached_to_tail": "put identity-less freed blocks at the tail instead of the head",
    "free_hits_cost_nothing": "do not count hits on free blocks against free capacity at admission",
    "hash_low_bits": "hash only token % 4, so different blocks collide",
    "refusal_side_effect": "a refused allocation still bumps the computed-token count",
}


def apply_mutation(name: str) -> None:
    K = impl.PagedKVCache
    if name == "free_head_first":
        orig = K.free_seq

        def free_seq(self, seq):
            seq.block_table = list(reversed(seq.block_table))
            orig(self, seq)

        K.free_seq = free_seq
    elif name == "no_last_token_rule":

        def lookup(self, seq):
            hit = []
            for ident in self._idents(seq):
                holders = self.cached.get(ident)
                if not holders:
                    break
                hit.append(holders[0])
            return hit

        K.lookup = lookup
    elif name == "newest_duplicate":
        orig_lookup = K.lookup

        def lookup(self, seq):
            hit = orig_lookup(self, seq)
            return [self.cached[seq.idents[i]][-1] for i in range(len(hit))]

        K.lookup = lookup
    elif name == "uncached_to_tail":
        orig_free = K.free_seq

        def free_seq(self, seq):
            before = set(self.free)
            orig_free(self, seq)
            for b in [b for b in self.free if b not in before and self.identity[b] is None]:
                self.free.move_to_end(b)

        K.free_seq = free_seq
    elif name == "free_hits_cost_nothing":
        orig_alloc = K.allocate

        def allocate(self, seq, num_new_tokens, hit):
            bs = self.block_size
            total = seq.num_computed + len(hit) * bs + num_new_tokens
            need = max(-(-total // bs) - len(seq.block_table) - len(hit), 0)
            if need > len(self.free):
                return None
            hits_from_free = sum(1 for b in hit if self.ref_cnt[b] == 0)
            if need + hits_from_free > len(self.free):
                # The real check refuses here; the buggy one admits with fewer new tokens.
                return orig_alloc(self, seq, max(num_new_tokens - hits_from_free * bs, 0), hit) or []
            return orig_alloc(self, seq, num_new_tokens, hit)

        K.allocate = allocate
    elif name == "hash_low_bits":
        orig_digest = impl.block_digest
        impl.block_digest = lambda parent, toks: orig_digest(parent, [t % 4 for t in toks])
    elif name == "refusal_side_effect":
        orig_alloc = K.allocate

        def allocate(self, seq, num_new_tokens, hit):
            out = orig_alloc(self, seq, num_new_tokens, hit)
            if out is None:
                seq.num_computed += 123
            return out

        K.allocate = allocate
    else:
        sys.exit(f"unknown mutation {name}; choose from {sorted(MUTATIONS)}")


def self_test(out: str | None) -> int:
    """Each mutation must make the check fail through a Mismatch, not a crash or a pass."""
    rows = []
    for name in MUTATIONS:
        p = subprocess.run(
            [sys.executable, __file__, "--mutate", name, "--traces", "50", "--ops", "100"],
            capture_output=True, text=True,
        )
        try:
            res = json.loads(p.stdout)
        except json.JSONDecodeError:
            res = {"first_mismatch": None, "error": f"no JSON; stderr tail: {p.stderr[-300:]}"}
        caught = p.returncode == 1 and bool(res.get("first_mismatch")) and not res.get("error")
        rows.append({"mutation": name, "what": MUTATIONS[name], "exit": p.returncode, "caught": caught,
                     "first_mismatch": res.get("first_mismatch"), "error": res.get("error")})
    result = {"check": "01 self-test: every planted bug must be reported as a mismatch",
              "mutations": rows, "passed": all(r["caught"] for r in rows)}
    print(json.dumps(result, indent=2))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=int, default=300)
    ap.add_argument("--ops", type=int, default=200)
    ap.add_argument("--mutate", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test(args.out)
    if not (1 <= args.traces <= MAX_TRACES and 1 <= args.ops <= MAX_OPS):
        print(f"--traces must be 1..{MAX_TRACES} and --ops 1..{MAX_OPS}", file=sys.stderr)
        return 2
    if args.mutate:
        apply_mutation(args.mutate)

    import torch
    import vllm

    t0 = time.time()
    totals: dict[str, int] = {}
    failure = error = None
    traces_run = 0
    for cfg_i, (bs, nb) in enumerate(CONFIGS):
        for t in range(args.traces):
            seed = cfg_i * 100_000 + t
            try:
                counts = run_trace(seed, bs, nb, args.ops)
            except Mismatch as e:
                failure = str(e)
                break
            except Exception as e:  # a crash is a failure too, but not a caught mismatch
                error = f"seed={seed} B={bs} N={nb}: {type(e).__name__}: {e}"
                break
            traces_run += 1
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
        if failure or error:
            break

    expected_traces = len(CONFIGS) * args.traces
    coverage = {k: totals.get(k, 0) for k in ("hit_lookups", "capped_full_hit_lookups", "evictions", "duplicates")}
    short = [k for k, v in coverage.items() if v < MIN_COVERAGE]
    passed = failure is None and error is None and traces_run == expected_traces and not short
    result = {
        "check": "01-paged-kv-prefix-cache differential vs vLLM KVCacheManager",
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "configs": [{"block_size": bs, "num_blocks": nb} for bs, nb in CONFIGS],
        "traces_per_config": args.traces,
        "ops_per_trace": args.ops,
        "traces_run": traces_run,
        "traces_expected": expected_traces,
        "mutation": args.mutate,
        "op_totals": totals,
        "coverage": coverage,
        "coverage_minimum": MIN_COVERAGE,
        "coverage_short": short,
        "compared_each_op": ["admission success", "hit tokens", "hit block ids", "block tables",
                             "tokens and computed counts (running and waiting)", "ref counts",
                             "free-queue order", "cache as token prefix -> ordered holders",
                             "no state change on refusal", "impl invariants"],
        "first_mismatch": failure,
        "error": error,
        "passed": passed,
        "seconds": round(time.time() - t0, 2),
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

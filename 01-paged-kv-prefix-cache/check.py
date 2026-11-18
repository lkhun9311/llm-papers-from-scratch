"""Differential check: impl.PagedKVCache against vLLM's KVCacheManager on identical random traces.

Each trace is a sequence of scheduler-like operations (admit, decode, finish, preempt). After every
operation the two sides must agree on: admission success, prefix-hit length, every live block table,
every block's reference count, the full free-queue order, and which blocks share a cache identity.
Independent invariants are checked on impl alone. `--mutate NAME` plants a known bug in impl to show
that the check catches it (the run must then FAIL).

A pass also requires coverage: every config must run in full, and prefix hits, evictions, duplicate
cached blocks, and last-token-capped full hits must each have happened at least MIN_COVERAGE times.
`--self-test` runs every mutation in a subprocess and passes only if each one is caught by a
mismatch (not by a crash).

Usage:  python check.py [--traces 300] [--ops 200] [--mutate NAME] [--out results/check.json]
        python check.py --self-test [--out results/self_test.json]
"""
from __future__ import annotations

import argparse
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


def gen_prompt(rng: random.Random, block_size: int, prefixes: list[list[int]]) -> list[int]:
    """Prompts share prefixes often, so hits, partial hits and mid-block divergence all occur."""
    base = list(rng.choice(prefixes)) if rng.random() < 0.7 else []
    cut = rng.randint(0, len(base))  # may cut inside a block: divergence mid-block
    tail = [rng.randint(0, 3) for _ in range(rng.randint(0, 2 * block_size + 1))]
    tokens = base[:cut] + tail
    return tokens or [rng.randint(0, 3)]


MAX_TRACES, MAX_OPS = 2000, 2000  # keeps a run to minutes and well under the memory cap
MIN_COVERAGE = 20


class Mismatch(Exception):
    pass


def compare(step: str, a, b) -> None:
    if a != b:
        raise Mismatch(f"{step}: impl={a!r} vllm={b!r}")


def check_invariants(kv: PagedKVCache, live: dict[str, Seq]) -> None:
    """Properties that must hold whatever the eviction policy; written without reference to vLLM."""
    n = len(kv.ref_cnt)
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
    for ident, holders in kv.cached.items():
        for b in holders:
            if kv.identity[b] != ident:
                raise Mismatch(f"invariant: block {b} listed under the wrong identity")
    for s in live.values():
        need = -(-s.num_computed // kv.block_size)
        if len(s.block_table) < need:
            raise Mismatch(f"invariant: {s.seq_id} has {len(s.block_table)} blocks for {s.num_computed} tokens")


def run_trace(seed: int, block_size: int, num_blocks: int, num_ops: int) -> dict:
    rng = random.Random(seed)
    mine = PagedKVCache(num_blocks, block_size)
    ref = VllmKVCache(num_blocks, block_size)
    prefixes = [[rng.randint(0, 3) for _ in range(rng.randint(block_size, 3 * block_size))] for _ in range(3)]
    running: dict[str, tuple[Seq, object]] = {}
    waiting: list[tuple[Seq, object]] = []
    counts = {"admit": 0, "admit_fail": 0, "hit_tokens": 0, "decode": 0, "decode_fail": 0, "finish": 0, "preempt": 0}
    next_id = 0

    def sync_state(step: str) -> None:
        compare(f"{step} ref_cnt", mine.ref_cnt[1:], ref.ref_cnt()[1:])
        compare(f"{step} free order", mine.free_order(), ref.free_order())
        compare(f"{step} cache classes", mine.cache_classes(), ref.cache_classes())
        for sid, (s, _) in running.items():
            compare(f"{step} block table {sid}", s.block_table, ref.block_table(sid))
        check_invariants(mine, {sid: s for sid, (s, _) in running.items()})

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
            hit = mine.lookup(s)
            blocks, ref_hit_tokens = ref.lookup(req)
            compare(f"{step} admit {s.seq_id} hit tokens", len(hit) * block_size, ref_hit_tokens)
            compare(f"{step} admit {s.seq_id} hit blocks", hit, list(blocks.get_block_ids()[0]))
            n_new = len(s.tokens) - len(hit) * block_size
            got = mine.allocate(s, n_new, hit)
            got_ref = ref.allocate(req, n_new, blocks, ref_hit_tokens)
            compare(f"{step} admit {s.seq_id} success", got is not None, got_ref is not None)
            if got is None:
                counts["admit_fail"] += 1
                waiting.append((s, req))
            else:
                counts["admit"] += 1
                counts["hit_tokens"] += len(hit) * block_size
                s.num_computed = len(s.tokens)
                req.num_computed_tokens = len(s.tokens)
                tok = rng.randint(0, 3)
                s.tokens.append(tok)
                req.append_output_token_ids(tok)
                running[s.seq_id] = (s, req)
        elif r < 0.75:
            sid = rng.choice(sorted(running))
            s, req = running[sid]
            got = mine.allocate(s, 1, [])
            got_ref = ref.allocate(req, 1, None, 0)
            compare(f"{step} decode {sid} success", got is not None, got_ref is not None)
            if got is None:
                counts["decode_fail"] += 1  # out of blocks: preempt it, as a scheduler would
                mine.free_seq(s)
                ref.free_seq(req)
                s.num_computed = 0
                req.num_computed_tokens = 0
                del running[sid]
                waiting.append((s, req))
                counts["preempt"] += 1
            else:
                counts["decode"] += 1
                s.num_computed += 1
                req.num_computed_tokens += 1
                tok = rng.randint(0, 3)
                s.tokens.append(tok)
                req.append_output_token_ids(tok)
        elif r < 0.92:
            sid = rng.choice(sorted(running))
            s, req = running.pop(sid)
            mine.free_seq(s)
            ref.free_seq(req)
            counts["finish"] += 1
        else:
            sid = rng.choice(sorted(running))
            s, req = running.pop(sid)
            mine.free_seq(s)
            ref.free_seq(req)
            s.num_computed = 0
            req.num_computed_tokens = 0
            waiting.append((s, req))
            counts["preempt"] += 1
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
            for ident in impl.chain_hashes(seq.tokens, self.block_size):
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
            idents = impl.chain_hashes(seq.tokens, self.block_size)
            return [self.cached[idents[i]][-1] for i in range(len(hit))]

        K.lookup = lookup
    elif name == "uncached_to_tail":
        orig_free = K.free_seq

        def free_seq(self, seq):
            before = set(self.free)
            orig_free(self, seq)
            fresh_uncached = [b for b in self.free if b not in before and self.identity[b] is None]
            for b in fresh_uncached:
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
                # The real check would refuse here; the buggy one admits and must shrink `need`.
                return orig_alloc(self, seq, max(num_new_tokens - hits_from_free * bs, 0), hit) or []
            return orig_alloc(self, seq, num_new_tokens, hit)

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
            res = {"first_mismatch": None, "error": p.stderr[-500:]}
        caught = p.returncode == 1 and bool(res.get("first_mismatch")) and not res.get("error")
        rows.append({"mutation": name, "what": MUTATIONS[name], "exit": p.returncode, "caught": caught,
                     "first_mismatch": res.get("first_mismatch"), "error": res.get("error")})
    result = {"check": "01 self-test: every planted bug must be caught", "mutations": rows,
              "passed": all(r["caught"] for r in rows)}
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
        sys.exit(f"--traces must be 1..{MAX_TRACES} and --ops 1..{MAX_OPS}")
    if args.mutate:
        apply_mutation(args.mutate)

    import torch
    import vllm

    t0 = time.time()
    totals: dict[str, int] = {}
    failure = None
    error = None
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
    coverage = {k: totals.get(k, 0) for k in ("hit_tokens", "evictions", "duplicates", "capped_full_hits")}
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
        "compared_each_op": ["admission success", "hit tokens", "hit block ids", "block tables",
                             "ref counts", "free-queue order", "cache identity classes", "impl invariants"],
        "coverage": coverage,
        "coverage_minimum": MIN_COVERAGE,
        "coverage_short": short,
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

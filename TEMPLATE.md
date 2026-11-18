# NN-topic-name

<!-- Copy this file to NN-topic-name/README.md and fill every section. Delete these comments. -->

## 1. What this covers

- Items studied here (one line each).
- Questions you should be able to answer when done:
  1. ...
  2. ...

## 2. Concepts

Plain definitions of the terms the rest of the note uses.

## 3. How it works

- The equations, with every symbol defined.
- Why it is designed this way: the problem it removes and what it costs instead.
- Memory and compute cost, worked through with concrete numbers for one real model configuration.

## 4. Code and how to run it

| File | What it does |
|---|---|
| `impl.py` | The implementation |
| `check.py` | Compares against the reference and writes `results/*.json` |

```bash
scripts/run-capped --mem 8G --cpus 8 -- python NN-topic-name/check.py
```

## 5. Checks and expected results

- Reference: library, version, and the pinned commit or tag.
- What is compared, and the tolerance (and why that tolerance).
- Boundary cases covered (block edges, position offsets, empty or single-token inputs, ...).
- Result file: `results/<name>.json` with versions, seed, config, tolerance, max error, pass/fail, peak memory.

## 6. In the engines

Where vLLM and SGLang implement this, as pinned permalinks, and whether this implementation agrees with
them. Note any difference found and what happened to it (issue, PR, or "behaves as designed").

## 7. References

- Papers: versioned arXiv `abs` links, link text equal to the paper title.
- Code: GitHub permalinks pinned to a commit or a version tag.

## 8. Self-check questions

1. ...
2. ...

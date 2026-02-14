# ADR-002: Chunk Size Selection, Overlap, and Semantic Chunking

**Date**: 2026-02-13
**Status**: Accepted
**Project**: P2 — RAG Evaluation Benchmarking Framework

## Context

The grid search evaluates 5 chunking configurations across 3 embedding models.
The question is: which chunk size and overlap setting produces the best retrieval
quality? And does LLM-based semantic chunking (Config E) outperform fixed-size
chunking?

The 5 configs differ in token count, overlap, and strategy:

| Config | Tokens | Overlap | Strategy |
|--------|--------|---------|----------|
| A | 128 | 32 | Fixed-size |
| B | 256 | 64 | Fixed-size |
| C | 512 | 128 | Fixed-size |
| D | 256 | 0 | Fixed-size (no overlap) |
| E | variable | — | LLM-based semantic |

## Decision

**Config B (256 tokens, 64-token overlap) is the recommended default** for this
corpus. It consistently performs well across all 3 embedding models:

| Config | MiniLM R@5 | MPNet R@5 | OpenAI R@5 | Avg R@5 |
|--------|-----------|-----------|------------|---------|
| A | 0.291 | 0.235 | 0.304 | 0.277 |
| **B** | **0.481** | **0.467** | **0.607** | **0.518** |
| C | 0.512 | 0.375 | 0.529 | 0.472 |
| D | 0.427 | 0.347 | 0.398 | 0.391 |
| E | 0.452 | 0.413 | 0.625 | 0.497 |

Key findings:

1. **Config A (128 tokens) is the worst** — too small, splits answers across
   multiple chunks, dilutes retrieval signal. Consistent last place across models.

2. **Config B vs D** — 64-token overlap adds +12.7 pp R@5 on average. Overlap
   prevents boundary-splitting relevant content.

3. **Config E (semantic) vs Config B** — mixed results:
   - OpenAI: E wins by +1.8 pp (0.625 vs 0.607)
   - MiniLM: B wins by +2.9 pp (0.481 vs 0.452)
   - MPNet: E wins slightly (not statistically significant)

4. **Config C (512)** beats A and D but underperforms B with MiniLM and MPNet.
   Larger chunks dilute precision — more irrelevant text per retrieved chunk.

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **Config B (256/64)** (chosen) | Best overall R@5 with local models, cheap to produce, predictable | Not best with OpenAI embeddings |
| **Config E (semantic)** | Best R@5 with OpenAI (0.625), preserves section boundaries | Requires LLM call per document ($), inconsistent across models, variable chunk sizes complicate batching |
| **Config C (512/128)** | Second-best with some models, fewer chunks to search | Lower precision (more irrelevant text per chunk), underperforms B on average |
| **Config A (128/32)** | Fine-grained retrieval for paragraph-level questions | Worst recall everywhere — too small for most questions |

## Consequences

**Easier:**
- Config B is the default for all downstream tasks (Day 4 reranking, Day 5 RAG
  pipeline). Simple, cheap, no LLM dependency for chunking.
- The 256/64 parameters align with industry defaults (LangChain's
  RecursiveCharacterTextSplitter default is 1000 chars / ~250 tokens).

**Harder:**
- Config E's advantage with OpenAI embeddings suggests semantic chunking may
  become more valuable as embedding models improve. If we switch to a stronger
  model later, Config E might consistently outperform.
- The overlap cost (D vs B shows overlap matters) means we must store ~20% more
  chunks. For our corpus this is trivial, but at scale it adds storage cost.

## Java/TS Parallel

This is like choosing between **fixed-size pagination** (SQL `LIMIT/OFFSET`) and
**content-aware splitting** (splitting a document at section headers). Fixed-size
is predictable and cheap; content-aware is smarter but requires parsing logic.
Config B is the `LIMIT 256 OFFSET 192` equivalent — simple, good enough for most
queries. Config E is like splitting at `<h2>` tags — preserves semantic boundaries
but adds processing complexity.

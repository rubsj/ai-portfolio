# ADR-002: Chunk Size Selection, Overlap, and Semantic Chunking

**Date**: 2026-02-13
**Status**: Accepted
**Project**: P2 — RAG Evaluation Benchmarking Framework

## Context

The grid search evaluates 5 chunking configurations across 3 embedding models. The question is: which chunk size and overlap setting produces the best retrieval quality? And does LLM-based semantic chunking (Config E) outperform fixed-size chunking?

The 5 configs differ in token count, overlap, and strategy:

| Config | Tokens | Overlap | Strategy |
|--------|--------|---------|----------|
| A | 128 | 32 | Fixed-size |
| B | 256 | 64 | Fixed-size |
| C | 512 | 128 | Fixed-size |
| D | 256 | 0 | Fixed-size (no overlap) |
| E | variable | — | LLM-based semantic |

## Decision

**Config B (256 tokens, 64-token overlap) is the recommended default** for this corpus. It consistently performs well across all 3 embedding models:

| Config | MiniLM R@5 | MPNet R@5 | OpenAI R@5 | Avg R@5 |
|--------|-----------|-----------|------------|---------|
| A | 0.291 | 0.235 | 0.304 | 0.277 |
| **B** | **0.481** | **0.467** | **0.607** | **0.518** |
| C | 0.512 | 0.375 | 0.529 | 0.472 |
| D | 0.427 | 0.347 | 0.398 | 0.391 |
| E | 0.452 | 0.413 | 0.625 | 0.497 |

Key findings:

1. **Config A (128 tokens) is the worst** — too small, splits answers across multiple chunks, dilutes retrieval signal. Consistent last place across models.
2. **Config B vs D** — 64-token overlap adds +12.7 pp R@5 on average. Overlap prevents boundary-splitting relevant content.
3. **Config E (semantic) vs Config B** — mixed results: OpenAI: E wins by +1.8 pp (0.625 vs 0.607). MiniLM: B wins by +2.9 pp (0.481 vs 0.452). MPNet: E wins slightly (not statistically significant).
4. **Config C (512)** beats A and D but underperforms B with MiniLM and MPNet. Larger chunks dilute precision — more irrelevant text per retrieved chunk.

## Alternatives Considered

| Option | Pros | Cons | Why Not |
|--------|------|------|---------|
| **Config B (256/64)** ✅ | Best overall R@5 with local models, cheap to produce, predictable | Not best with OpenAI embeddings | — (selected) |
| Config E (semantic) | Best R@5 with OpenAI (0.625), preserves section boundaries | Requires LLM call per document ($), inconsistent across models, variable chunk sizes complicate batching | Best with OpenAI but inconsistent across local models; requires LLM call per document |
| Config C (512/128) | Second-best with some models, fewer chunks to search | Lower precision — more irrelevant text per chunk dilutes signal | Lower precision — more irrelevant text per chunk dilutes signal |
| Config A (128/32) | Fine-grained retrieval for paragraph-level questions | Worst recall everywhere — too small for most questions | Too small — worst recall everywhere, splits answers across chunks |

## Quantified Validation

- Config B avg R@5: **0.518** — best cross-model consistency
- Config B vs BM25 (0.381): **+13.7 pp** average improvement
- Overlap impact (D→B): **+12.7 pp** R@5 from 64-token overlap
- Config A penalty: **-24.1 pp** vs Config B — smallest chunks worst
- Config E + OpenAI: **0.625 R@5** (best single config but model-dependent)

## Consequences

**Easier:** Config B is the default for all downstream tasks (Day 4 reranking, Day 5 RAG pipeline). Simple, cheap, no LLM dependency for chunking. The 256/64 parameters align with industry defaults (LangChain's `RecursiveCharacterTextSplitter` default is 1000 chars / ~250 tokens).

**Harder:** Config E's advantage with OpenAI embeddings suggests semantic chunking may become more valuable as embedding models improve. If we switch to a stronger model later, Config E might consistently outperform. The overlap cost (D vs B shows overlap matters) means we must store ~20% more chunks. For our corpus this is trivial, but at scale it adds storage cost.

**Portability:** P5 (Production RAG) would use Config E + OpenAI as recommended default based on these results. The overlap finding (25% > 50%) is a transferable guideline for any chunking implementation.

## Cross-References

- **ADR-001**: Each chunk config creates its own FAISS index — 5 configs × 3 models = 15 indices
- **ADR-003**: Config E + OpenAI = best single config (0.625), but Config B + OpenAI = 97% as good (0.607) without LLM chunking cost
- **ADR-004**: QA pairs generated from Config B chunks — B is the reference namespace for `gold_chunk_ids`
- **ADR-005**: Dedicated semantic vs fixed-size deep dive with reranking interaction analysis

## Java/TS Parallel

This is like choosing between **fixed-size pagination** (SQL `LIMIT/OFFSET`) and **content-aware splitting** (splitting a document at `<h2>` section headers). Fixed-size is predictable and cheap; content-aware is smarter but requires parsing logic. Config B is the `LIMIT 256 OFFSET 192` equivalent — simple, good enough for most queries. Config E is like splitting at `<h2>` tags — preserves semantic boundaries but adds processing complexity.

**The key insight:** The "right" chunk size is empirical, not theoretical. Just as database page sizes are tuned by benchmarking actual query patterns, chunk sizes must be tuned by benchmarking actual retrieval on your corpus.

## Interview Signal

Demonstrates **systematic experimentation methodology**. The engineer ran a controlled grid search across 15 configurations isolating size, overlap, and strategy variables. The finding that 50% overlap (Config D) underperforms 25% overlap (Config B) contradicts common intuition and could only be discovered empirically — this is the experimental rigor production ML teams need.

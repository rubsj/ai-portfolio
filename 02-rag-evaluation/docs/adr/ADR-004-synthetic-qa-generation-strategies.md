# ADR-004: Synthetic QA Generation — 5 Strategies with Instructor Validation

**Date**: 2026-02-13
**Status**: Accepted
**Project**: P2 — RAG Evaluation Benchmarking Framework

## Context

Evaluating 16 retrieval configurations requires a gold-standard question set where each question has known `gold_chunk_ids` — the chunks that contain the answer. We need questions that cover different cognitive levels (factual recall → multi-hop reasoning), span different retrieval scopes (single paragraph → full page), test edge cases like chunk boundary overlap regions, are answerable from the corpus (not hallucinated), and have verifiable gold chunk mappings for metric computation. Manually writing 50+ questions for a 589-chunk corpus is impractical.

## Decision

**Use 5 complementary generation strategies** to produce 56 synthetic QA pairs from Config B chunks (256 tokens, 64-token overlap). Each strategy targets a different retrieval challenge. All LLM calls use GPT-4o-mini via Instructor for structured output validation.

### Strategy Distribution

| # | Strategy | Questions | Gold Chunks/Q | Hierarchy | Purpose |
|---|----------|-----------|---------------|-----------|---------|
| 1 | Per-Chunk Chain | 24 | 1 | paragraph | Baseline: can the system find a single chunk? |
| 2 | Multi-Chunk | 10 | 2-4 | section | Cross-chunk: does retrieval find semantically related chunks? |
| 3 | Overlap Region | 8 | 2 | paragraph | Boundary test: content split across chunk boundaries |
| 4 | Hierarchical | 8 | 1-5+ | mixed | Scope test: paragraph vs section vs page-level questions |
| 5 | Academic Pattern | 6 | 1 | paragraph | Format diversity: definition, comparison, application patterns |

### Question Type Coverage

| Type | Count | Source Strategies |
|------|-------|-------------------|
| Factual | 21 | S1 (factual tier), S3, S5 |
| Multi-hop | 19 | S1 (connective tier), S2, S3 |
| Analytical | 12 | S1 (analytical tier), S4, S5 |
| Summarization | 3 | S4 (page-level) |
| Comparative | 1 | S5 |

### Key Design Choices

1. **Config B as reference** — all `gold_chunk_ids` are in Config B namespace (e.g., `B_0_42`). When evaluating other configs, gold IDs are mapped via character position overlap in `grid_search.map_gold_chunks()`. Config B was chosen because BM25 uses the same chunks, so BM25 evaluation needs no mapping.
2. **Instructor for structured output** — every LLM call returns a Pydantic model. Instructor auto-retries on validation failure (up to 3 attempts). This eliminates JSON parsing errors.
3. **Semantic similarity from pre-built FAISS** — Strategy 2 finds related chunks by extracting embeddings from `minilm_B.faiss` via `index.reconstruct_n()`. No embedding model loaded at QA generation time.
4. **LLM response caching** — every Instructor call is cached via `cache.compute_cache_key()`. Regenerating QA pairs costs $0 after the first run.
5. **Diverse chunk sampling** — `_sample_diverse_chunks()` spreads selections across all 3 documents and samples from beginning/middle/end positions within each document.

## Alternatives Considered

| Option | Pros | Cons | Why Not |
|--------|------|------|---------|
| **5-strategy approach** ✅ | Covers all question types and hierarchy levels, tests chunk boundaries explicitly, 56 Qs exceeds 50 threshold | More complex generation code (~350 lines), some strategies produce similar questions | — (selected) |
| Single-strategy (per-chunk only) | Simple, one prompt template, easy to scale | All questions are single-chunk factual, misses multi-hop and boundary cases entirely | Misses multi-hop and boundary cases entirely |
| Manual question writing | Highest quality, perfect gold mappings | Impractical for 50+ questions, doesn't scale, subjective bias | Impractical for 50+, doesn't scale |
| RAGAS-style automatic generation | Battle-tested library, handles multi-hop | Black box — can't control strategy mix, harder to verify gold chunk mappings, adds dependency | Black box — can't control strategy mix or verify gold chunk mappings |

### Why not more questions?

56 is sufficient for a benchmarking framework that compares 16 configs. Statistical significance requires ~30+ observations per group. With 56 questions evaluated across 16 configs, we get 896 data points. Diminishing returns set in quickly — the top-3 ranking would not change with 100 or 200 questions on this corpus.

## Quantified Validation

- **56 QA pairs** across 5 strategies (exceeds 50 threshold)
- **100% Instructor validation** — zero rejected questions
- **5 question types**: factual 21, multi-hop 19, analytical 12, summarization 3, comparative 1
- **12.4% chunk coverage** (73 of 589 chunks) — acknowledged gap
- **Total cost: ~$0.08** first run, $0.00 thereafter (cached)
- **896 data points**: 56 questions × 16 configs

## Consequences

**Easier:** Gold chunk IDs are deterministic — each strategy explicitly assigns them from the source chunks used in generation. No post-hoc annotation needed. The 5-strategy mix catches retrieval weaknesses that single-strategy evaluation would miss. Caching means QA pairs are generated once and reused across all grid search runs.

**Harder:** 12.4% chunk coverage means 87.6% of chunks have no gold question. If a retrieval bug only affects uncovered chunks, we won't detect it. Strategy 2 (multi-chunk) depends on pre-built FAISS indices existing — this coupling creates an implicit ordering dependency (Day 2 before Day 3). The `comparative` question type has only 1 question — too few to draw per-type conclusions.

**Portability:** P4 adopted multi-strategy generation (controlled fit levels instead of question strategies). "Generate from reference config, evaluate across all configs" is a reusable RAG benchmarking pattern.

## Cross-References

- **ADR-001**: Strategy 2 uses FAISS `reconstruct_n()` for chunk similarity
- **ADR-002**: All `gold_chunk_ids` reference Config B namespace — B is the reference config
- **ADR-003**: Strategy 2 uses MiniLM FAISS index specifically
- **ADR-005**: Config E outperforms Config B despite QA pairs being generated from Config B — proves evaluation isn't biased toward reference config
- **P1 ADR-004**: Same "multi-strategy evaluation" principle — P1 used 6 failure modes, P2 uses 5 QA strategies

## Java/TS Parallel

This is like **JUnit test generation with different test categories**. Strategy 1 is unit tests (one chunk = one class), Strategy 2 is integration tests (multiple chunks = multiple services), Strategy 3 is boundary tests (chunk overlap = API contract boundaries), Strategy 4 is scope tests (paragraph → section → page = method → class → module), and Strategy 5 is pattern tests (academic templates = test patterns from a cookbook). The Instructor validation is like using `@Valid` on a Spring Controller DTO — the LLM can return whatever it wants, but Pydantic rejects malformed output and retries automatically.

**The key insight:** Test diversity matters as much as test quantity. Just as a JUnit suite with 100 unit tests but no integration tests gives false confidence, 100 factual-only QA pairs would make every chunk config look equally good and miss multi-hop retrieval failures.

## Interview Signal

Demonstrates **evaluation design thinking**. The engineer recognized that single-strategy QA generation creates measurement bias and designed 5 complementary strategies stress-testing different retrieval capabilities. The honest 12.4% coverage acknowledgment shows intellectual honesty about limitations — hiring managers value this over polished-but-incomplete claims.

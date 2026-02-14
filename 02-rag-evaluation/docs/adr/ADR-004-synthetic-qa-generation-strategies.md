# ADR-004: Synthetic QA Generation — 5 Strategies with Instructor Validation

**Date**: 2026-02-13
**Status**: Accepted
**Project**: P2 — RAG Evaluation Benchmarking Framework

## Context

Evaluating 16 retrieval configurations requires a gold-standard question set
where each question has known `gold_chunk_ids` — the chunks that contain the
answer. We need questions that:

1. Cover different cognitive levels (factual recall → multi-hop reasoning)
2. Span different retrieval scopes (single paragraph → full page)
3. Test edge cases like chunk boundary overlap regions
4. Are answerable from the corpus (not hallucinated questions)
5. Have verifiable gold chunk mappings for metric computation

Manually writing 50+ questions for a 589-chunk corpus is impractical.
LLM-generated questions risk being shallow (all factual) or disconnected
from actual chunk boundaries. We need a structured generation approach.

## Decision

**Use 5 complementary generation strategies** to produce 56 synthetic QA pairs
from Config B chunks (256 tokens, 64-token overlap). Each strategy targets a
different retrieval challenge. All LLM calls use GPT-4o-mini via Instructor
for structured output validation.

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

1. **Config B as reference** — all gold_chunk_ids are in Config B namespace
   (e.g., `B_0_42`). When evaluating other configs (A/C/D/E), gold IDs are
   mapped via character position overlap in `grid_search.map_gold_chunks()`.
   Config B was chosen because BM25 uses the same chunks, so BM25 evaluation
   needs no mapping.

2. **Instructor for structured output** — every LLM call returns a Pydantic
   model (`QuestionChainResponse`, `MultiChunkQuestionResponse`, etc.).
   Instructor auto-retries on validation failure (up to 3 attempts). This
   eliminates JSON parsing errors and ensures every question has required
   fields.

3. **Semantic similarity from pre-built FAISS** — Strategy 2 finds related
   chunks by extracting embeddings from `minilm_B.faiss` via
   `index.reconstruct_n()`. No embedding model loaded at QA generation time.
   This reuses Day 2 infrastructure instead of adding a dependency.

4. **LLM response caching** — every Instructor call is cached via
   `cache.compute_cache_key("gpt-4o-mini", prompt)`. Regenerating QA pairs
   costs $0 after the first run. Total first-run cost: ~$0.08 (56 calls ×
   ~250 tokens each × GPT-4o-mini pricing).

5. **Diverse chunk sampling** — `_sample_diverse_chunks()` spreads selections
   across all 3 documents and samples from beginning/middle/end positions
   within each document. This prevents question clustering.

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **5-strategy approach** (chosen) | Covers all question types and hierarchy levels, tests chunk boundaries explicitly, 56 Qs exceeds 50 threshold | More complex generation code (~350 lines), some strategies produce similar questions |
| **Single-strategy (per-chunk only)** | Simple, one prompt template, easy to scale | All questions are single-chunk factual, misses multi-hop and boundary cases entirely |
| **Manual question writing** | Highest quality, perfect gold mappings | Impractical for 50+ questions, doesn't scale, subjective bias |
| **RAGAS-style automatic generation** | Battle-tested library, handles multi-hop | Black box — can't control strategy mix, harder to verify gold chunk mappings, adds dependency |

### Why not more questions?

56 is sufficient for a benchmarking framework that compares 16 configs.
Statistical significance requires ~30+ observations per group. With 56
questions evaluated across 16 configs, we get 896 data points. Diminishing
returns set in quickly — the top-3 ranking would not change with 100 or 200
questions on this corpus.

## Consequences

**Easier:**
- Gold chunk IDs are deterministic — each strategy explicitly assigns them
  from the source chunks used in generation. No post-hoc annotation needed.
- The 5-strategy mix catches retrieval weaknesses that single-strategy
  evaluation would miss. A system that aces factual questions but fails
  multi-hop would look falsely strong with Strategy 1 alone.
- Caching means QA pairs are generated once and reused across all grid
  search runs. Iterating on evaluation code doesn't re-incur LLM costs.

**Harder:**
- 12.4% chunk coverage means 87.6% of chunks have no gold question.
  If a retrieval bug only affects uncovered chunks, we won't detect it.
  Acceptable for benchmarking; insufficient for production QA testing.
- Strategy 2 (multi-chunk) depends on pre-built FAISS indices existing.
  If indices are deleted, QA generation fails. This coupling is documented
  but creates an implicit ordering dependency (Day 2 before Day 3).
- The `comparative` question type has only 1 question — too few to draw
  per-type conclusions. Future work could add a Strategy 6 targeting
  comparisons explicitly.

## Java/TS Parallel

This is like **JUnit test generation with different test categories**. Strategy
1 is unit tests (one chunk = one class), Strategy 2 is integration tests
(multiple chunks = multiple services), Strategy 3 is boundary tests (chunk
overlap = API contract boundaries), Strategy 4 is scope tests (paragraph →
section → page = method → class → module), and Strategy 5 is pattern tests
(academic templates = test patterns from a cookbook). The Instructor validation
is like using a JSON Schema validator on API responses — the LLM can return
whatever it wants, but Pydantic rejects malformed output and retries
automatically, just like `@Valid` on a Spring Controller DTO.

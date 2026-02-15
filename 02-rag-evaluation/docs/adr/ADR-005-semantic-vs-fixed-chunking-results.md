# ADR-005: Semantic vs Fixed-Size Chunking — Experimental Results

**Date**: 2026-02-14
**Status**: Accepted
**Project**: P2 — RAG Evaluation Benchmarking Framework

## Context

Day 3 grid search evaluated 16 retrieval configurations (5 chunk strategies ×
3 embedding models + BM25 baseline) across 56 synthetic QA pairs. The top-3
configs by Recall@5 were:

| Rank | Config | Strategy | Embedding | Recall@5 | MRR@5 |
|------|--------|----------|-----------|----------|-------|
| 1 | E-openai | Semantic (Markdown headers, 512 subdivision) | text-embedding-3-small | 0.625 | 0.578 |
| 2 | B-openai | Fixed 256/64 | text-embedding-3-small | 0.607 | 0.618 |
| 3 | D-openai | Fixed 256/128 | text-embedding-3-small | 0.529 | 0.494 |

Day 4 added three evaluation layers: Cohere reranking, RAGAS generation
evaluation, and LLM-as-Judge. This ADR documents the combined findings and
production recommendation.

## Decision

**Use Config E (semantic chunking) + OpenAI embeddings + Cohere reranking as
the recommended retrieval pipeline.** Post-reranking, E-openai achieves
Recall@5 = 0.747 — the highest across all configurations.

## Evidence

### 1. Retrieval: Semantic vs Fixed-Size (Day 3)

E-openai beat B-openai by 1.8 percentage points on Recall@5 (0.625 vs 0.607).
The margin is narrow, but E-openai wins on per-question-type recall across all
answerable categories:

| Question Type | E-openai R@5 | B-openai R@5 | Delta |
|---------------|-------------|-------------|-------|
| Factual (21) | 0.667 | 0.643 | +2.4 pp |
| Multi-hop (19) | 0.645 | 0.632 | +1.3 pp |
| Analytical (12) | 0.653 | 0.639 | +1.4 pp |
| Summarization (3) | 0.300 | 0.278 | +2.2 pp |

B-openai has higher MRR (0.618 vs 0.578), meaning it places the first
relevant result higher. This is because fixed-size chunks are more uniform in
scope — when they match, they match precisely. Semantic chunks vary in length,
which can dilute the top-1 rank even when more total relevant chunks appear in
the top-5.

### 2. Reranking Impact (Day 4 — Cohere rerank-v3.5)

FAISS retrieved top-20 candidates; Cohere cross-encoder reranked to top-5.

| Config | R@5 Before | R@5 After | Improvement |
|--------|-----------|-----------|-------------|
| E-openai | 0.625 | 0.747 | +19.5% |
| B-openai | 0.607 | 0.667 | +9.8% |
| D-openai | 0.529 | 0.670 | +26.6% |

Key findings:
- **Reranking widened E-openai's lead** over B-openai from 1.8 pp to 8.0 pp.
  Semantic chunks give the cross-encoder more coherent passages to score,
  amplifying the reranking benefit.
- **D-openai benefited most** (+26.6% recall, +40.9% MRR). The high-overlap
  config (256/128) produced many near-duplicate chunks that confused FAISS
  cosine similarity but were easy for the cross-encoder to disambiguate.
- **MRR improvements** were even larger: E-openai +21.3%, D-openai +40.9%.
  Reranking is especially effective at promoting the single best chunk to rank 1.

### 3. Generation Quality (Day 4 — RAGAS, GPT-4o-mini)

RAGAS scores for E-openai (best config), using manual GPT-4o-mini fallback
due to RAGAS library Pydantic V1/V2 conflict:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Faithfulness | 0.511 | ~51% of generated claims are grounded in retrieved context |
| Answer Relevancy | 0.563 | ~56% of answers address the question well |
| Context Recall | 0.713 | ~71% of gold chunks appear in retrieved context |
| Context Precision | 0.734 | ~73% of retrieved chunks are relevant to the answer |

Context metrics (recall 0.713, precision 0.734) are strong — the retriever
finds relevant material. Faithfulness (0.511) is the bottleneck: the generator
fabricates details not in the context, or refuses to answer when context is
present but ambiguous.

### 4. Judge Verdicts (Day 4 — GPT-4o judges)

Four judges evaluated all 56 generated answers for E-openai:

| Metric | Count | Rate |
|--------|-------|------|
| Correct answers | 18/56 | 32.1% |
| Hallucinated answers | 41/56 | 73.2% |
| Thumbs up (correct AND no hallucination) | 7/56 | 12.5% |
| "I don't have enough context" refusals | 22/56 | 39.3% |

**Judge calibration issue**: 21 of 22 refusal answers were marked as
hallucinations. A refusal ("I don't have enough context") is not a
hallucination — it's a retrieval failure. The true hallucination rate on
substantive answers is 20/34 (58.8%), not 73.2%.

**Bloom taxonomy breakdown** reveals that Analyze-level questions account for
68.2% of refusals but only 29.4% of substantive answers. Factual recall
(Remember) questions succeed at 2× the rate of analytical questions.

### 5. Cross-Layer Synthesis

| Layer | What It Measures | E-openai Score |
|-------|-----------------|----------------|
| Retrieval (Day 3) | Can we find relevant chunks? | R@5 = 0.625 |
| Reranking (Day 4) | Can we promote the best chunks? | R@5 = 0.747 (+19.5%) |
| RAGAS (Day 4) | Is the generated answer grounded? | Faithfulness = 0.511 |
| Judges (Day 4) | Is the answer correct and useful? | 32.1% correct, 12.5% thumbs-up |

The pipeline degrades at each layer: retrieval finds ~63% of relevant chunks,
reranking boosts to ~75%, but generation only faithfully uses ~51% of what it
retrieves, and only 32% of final answers are correct. The biggest gap is
between retrieval quality and generation quality — the retriever does its job,
but the generator doesn't fully exploit the retrieved context.

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **E-openai + Cohere reranking** (chosen) | Highest R@5 (0.747), reranking amplifies semantic chunk advantage, coherent passages for cross-encoder | API cost ($0.01/query embed + free Cohere tier), 6.5s/query rate limit on trial key |
| **B-openai (fixed 256/64)** | Higher MRR pre-reranking (0.618), simpler chunking logic, no section-detection dependency | Lower R@5 (0.607), gains less from reranking (+9.8% vs +19.5%), uniform chunks lose document structure |
| **D-openai (fixed 256/128)** | Highest reranking gain (+26.6%), good post-reranking R@5 (0.670) | 22% more chunks than B (717 vs 589), pre-reranking R@5 is worst of top-3, high overlap adds noise |

## Consequences

**Easier:**
- Semantic chunking preserves document structure (section headers, paragraph
  boundaries). This gives the generator more coherent context passages, which
  should improve faithfulness in production with better prompt engineering.
- Cohere reranking provides a +19.5% recall boost at zero marginal cost
  (free tier, 1000 calls/month). For a benchmarking framework, this is
  sufficient.
- The 3-layer evaluation (retrieval → reranking → generation) establishes
  baselines for each pipeline stage. Future improvements can target the
  weakest layer (generation faithfulness at 0.511).

**Harder:**
- Semantic chunking depends on Markdown header detection. Documents without
  clear structure (e.g., OCR'd PDFs, free-form text) would fall back to
  fixed-size subdivision, losing the advantage.
- The 39.3% refusal rate ("I don't have enough context") on analytical
  questions suggests the RAG prompt is too conservative. Tuning the prompt
  to allow inference from partial context would improve coverage but risk
  hallucination.
- Judge calibration needs fixing: refusals should not be marked as
  hallucinations. A production pipeline should add a pre-judge filter that
  classifies refusals separately from substantive answers.

## Java/TS Parallel

This is like **choosing between a fixed-size thread pool and a work-stealing
pool** (e.g., `Executors.newFixedThreadPool(n)` vs `ForkJoinPool`). Fixed-size
chunking (Config B) is the fixed thread pool — predictable, uniform, easy to
reason about. Semantic chunking (Config E) is the ForkJoinPool — it adapts to
the work structure (document sections), is slightly more complex, but handles
heterogeneous workloads better. The reranking step is like adding a priority
queue on top of the thread pool's task queue — it doesn't change what gets
scheduled, just which tasks run first.

The judge calibration issue maps to **test assertion specificity**. A JUnit
test that asserts `assertNotNull(result)` when it should assert
`assertEquals(expected, result)` gives false confidence. Our hallucination
judge is too broad — it flags refusals as failures when it should only flag
fabricated content. Like tightening test assertions, we need to narrow the
judge's scope to distinguish "wrong answer" from "no answer."

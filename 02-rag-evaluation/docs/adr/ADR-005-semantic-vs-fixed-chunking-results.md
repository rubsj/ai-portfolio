# ADR-005: Semantic vs Fixed-Size Chunking — Experimental Results

**Date**: 2026-02-14
**Status**: Accepted
**Project**: P2 — RAG Evaluation Benchmarking Framework

## Context

Day 3 grid search evaluated 16 retrieval configurations across 56 synthetic QA pairs. Day 4 added three evaluation layers: Cohere reranking, RAGAS generation evaluation, and LLM-as-Judge. This ADR synthesizes the full cross-layer analysis — the pipeline degrades at each stage, revealing that retrieval quality does not guarantee generation quality:

| Layer | What It Measures | E-openai Score |
|-------|-----------------|----------------|
| Retrieval (Day 3) | Can we find relevant chunks? | R@5 = 0.625 |
| Reranking (Day 4) | Can we promote the best chunks? | R@5 = 0.747 (+19.5%) |
| RAGAS (Day 4) | Is the generated answer grounded? | Faithfulness = 0.511 |
| Judges (Day 4) | Is the answer correct and useful? | 32.1% correct, 12.5% thumbs-up |

## Decision

**Use Config E (semantic chunking) + OpenAI embeddings + Cohere reranking as the recommended retrieval pipeline.** Post-reranking, E-openai achieves Recall@5 = 0.747 — the highest across all configurations.

### Retrieval: Semantic vs Fixed-Size

E-openai beat B-openai by 1.8 pp on Recall@5 (0.625 vs 0.607). E-openai wins on per-question-type recall across all answerable categories:

| Question Type | E-openai R@5 | B-openai R@5 | Delta |
|---------------|-------------|-------------|-------|
| Factual (21) | 0.667 | 0.643 | +2.4 pp |
| Multi-hop (19) | 0.645 | 0.632 | +1.3 pp |
| Analytical (12) | 0.653 | 0.639 | +1.4 pp |
| Summarization (3) | 0.300 | 0.278 | +2.2 pp |

B-openai has higher MRR (0.618 vs 0.578), meaning it places the first relevant result higher — fixed-size chunks are more uniform in scope, so when they match, they match precisely.

### Generation Quality (RAGAS, GPT-4o-mini)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Faithfulness | 0.511 | ~51% of generated claims grounded in context |
| Answer Relevancy | 0.563 | ~56% of answers address the question well |
| Context Recall | 0.713 | ~71% of gold chunks appear in retrieved context |
| Context Precision | 0.734 | ~73% of retrieved chunks are relevant |

### Judge Verdicts (GPT-4o)

**Judge calibration issue**: 21 of 22 refusal answers ("I don't have enough context") were marked as hallucinations. True hallucination rate on substantive answers is **20/34 (58.8%)**, not 73.2%. Bloom taxonomy breakdown reveals Analyze-level questions account for 68.2% of refusals but only 29.4% of substantive answers.

## Alternatives Considered

| Option | Pros | Cons | Why Not |
|--------|------|------|---------|
| **E-openai + Cohere reranking** ✅ | Highest R@5 (0.747), reranking amplifies semantic chunk advantage, coherent passages for cross-encoder | API cost ($0.01/query embed + free Cohere tier), 6.5s/query rate limit on trial key | — (selected) |
| B-openai (fixed 256/64) | Higher MRR pre-reranking (0.618), simpler chunking logic, no section-detection dependency | Lower R@5 (0.607), gains less from reranking (+9.8% vs +19.5%) | 97% of E's pre-rerank quality but gains less from reranking; simpler but lower ceiling |
| D-openai (fixed 256/128) | Highest reranking gain (+26.6%), good post-reranking R@5 (0.670) | 22% more chunks than B (717 vs 589), pre-reranking R@5 is worst of top-3 | Highest reranking gain but worst pre-reranking of top-3; 22% more chunks from high overlap |

## Quantified Validation

- E-openai pre-reranking: **0.625 R@5** (best single config)
- E-openai post-reranking: **0.747 R@5** (+19.5%)
- Reranking widened semantic lead: **1.8 pp → 8.0 pp** gap over B-openai

| Config | R@5 Before | R@5 After | Improvement |
|--------|-----------|-----------|-------------|
| E-openai | 0.625 | 0.747 | +19.5% |
| B-openai | 0.607 | 0.667 | +9.8% |
| D-openai | 0.529 | 0.670 | +26.6% |

- E-openai vs BM25: **+36.6 pp** R@5 post-reranking
- Faithfulness bottleneck: **0.511** — retrieval quality (0.747) does NOT guarantee generation quality
- Judge calibration finding: True hallucination rate **58.8%** (not 73.2%) — refusals inflated the count

## Consequences

**Easier:** Semantic chunking preserves document structure (section headers, paragraph boundaries), giving the generator more coherent context passages. Cohere reranking provides a +19.5% recall boost at zero marginal cost (free tier, 1000 calls/month). The 3-layer evaluation (retrieval → reranking → generation) establishes baselines for each pipeline stage — future improvements can target the weakest layer (generation faithfulness at 0.511).

**Harder:** Semantic chunking depends on Markdown header detection. Documents without clear structure (OCR'd PDFs, free-form text) fall back to fixed-size subdivision, losing the advantage. The 39.3% refusal rate on analytical questions suggests the RAG prompt is too conservative — tuning it risks hallucination. Judge calibration needs fixing: refusals should not be marked as hallucinations.

**Portability:** P5 (Production RAG) would use Config E + OpenAI + Cohere as default for structured documents. The faithfulness gap (0.511) identifies the next optimization target: generation prompt engineering, not retrieval improvement.

## Cross-References

- **ADR-001**: FAISS provides the first-stage retrieval (top-20) that feeds Cohere reranking
- **ADR-002**: Config E vs Config B is the central comparison — ADR-002 established Config B as the fixed-size champion
- **ADR-003**: OpenAI embeddings amplify semantic chunking's advantage — E-minilm (0.452) underperforms B-minilm (0.481)
- **ADR-004**: QA evaluation isn't biased toward Config B despite gold chunks being generated from Config B — Config E still wins
- **P1 ADR-003**: Same judge calibration lesson — positivity bias requires explicit strictness. Here, refusal misclassification inflated hallucination rate by 14.4 pp

## Java/TS Parallel

This is like choosing between a **fixed-size thread pool** (`Executors.newFixedThreadPool(n)`) and a **work-stealing pool** (`ForkJoinPool`). Fixed-size chunking (Config B) is the fixed thread pool — predictable, uniform, easy to reason about. Semantic chunking (Config E) is the ForkJoinPool — it adapts to the work structure (document sections), is slightly more complex, but handles heterogeneous workloads better. The reranking step is like adding a priority queue on top of the thread pool's task queue — it doesn't change what gets scheduled, just which tasks run first. The judge calibration issue maps to **test assertion specificity** — a JUnit test that asserts `assertNotNull(result)` when it should assert `assertEquals(expected, result)` gives false confidence.

**The key insight:** Document structure is free metadata — headers are the author's own chunking decisions. Ignoring them with fixed-size splitting is like ignoring package boundaries in a Java codebase and splitting files at arbitrary byte offsets.

## Interview Signal

Demonstrates **multi-layer evaluation thinking** and the ability to synthesize results across pipeline stages. The key finding — that reranking amplifies semantic chunking's advantage from 1.8 pp to 8.0 pp — is a second-order interaction effect that only emerges from systematic experimentation. The faithfulness bottleneck discovery (0.511 vs 0.747 retrieval) proves the engineer evaluates end-to-end rather than optimizing a single component in isolation.

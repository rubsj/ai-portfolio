# ADR-003: Embedding Model Comparison — Local vs API, Cost vs Quality

**Date**: 2026-02-13
**Status**: Accepted
**Project**: P2 — RAG Evaluation Benchmarking Framework

## Context

The grid search evaluates 3 embedding models across 5 chunking configurations.
The question is: does paying for an API embedding model (OpenAI) produce
meaningfully better retrieval than free, local models (MiniLM, MPNet)?

The 3 models differ in architecture, dimension, and cost:

| Model | Dimensions | Parameters | Cost | Runs On |
|-------|-----------|------------|------|---------|
| MiniLM (all-MiniLM-L6-v2) | 384 | 22M | Free (local) | CPU/GPU |
| MPNet (all-mpnet-base-v2) | 768 | 109M | Free (local) | CPU/GPU |
| OpenAI (text-embedding-3-small) | 1536 | Unknown | $0.02/1M tokens | API |

Hypothesis going in: OpenAI should win because it has 4x the dimensions of
MiniLM and is trained on vastly more data. The interesting question is *by
how much* — is the gap worth the cost and API dependency?

## Decision

**OpenAI text-embedding-3-small is the best model for this corpus**, but
**MiniLM is the recommended default** for development and cost-sensitive
deployments. MPNet underperforms despite being 5x larger than MiniLM.

### Head-to-head: Config B (controlled comparison)

| Model | R@1 | R@3 | R@5 | P@1 | MRR@5 |
|-------|-----|-----|-----|-----|-------|
| MiniLM | 0.238 | 0.423 | 0.481 | 0.393 | 0.492 |
| MPNet | 0.146 | 0.347 | 0.467 | 0.268 | 0.398 |
| **OpenAI** | **0.317** | **0.537** | **0.607** | **0.518** | **0.618** |

### Average across all 5 chunk configs

| Model | Avg R@5 | Avg MRR@5 | Best Config | Best R@5 |
|-------|---------|-----------|-------------|----------|
| MiniLM | 0.415 | 0.404 | B-minilm | 0.481 |
| MPNet | 0.367 | 0.332 | B-mpnet | 0.467 |
| **OpenAI** | **0.515** | **0.500** | **E-openai** | **0.625** |

Key findings:

1. **OpenAI dominates** — +12.6 pp R@5 over MiniLM on Config B, +10.0 pp
   averaged across all configs. The top 4 configs in the grid search are all
   OpenAI. The gap is consistent, not a fluke of one chunk size.

2. **MPNet is the worst model** — despite having 5x the parameters of MiniLM
   (109M vs 22M) and 2x the dimensions (768 vs 384). On Config B, MPNet
   trails MiniLM by -1.4 pp R@5 and -9.4 pp MRR@5. More parameters ≠ better
   retrieval for this task.

3. **MiniLM is the efficiency champion** — 22M params, 384 dims, <50ms per
   query, zero cost. Gets 79% of OpenAI's R@5 performance (0.481 vs 0.607)
   at 0% of the cost.

4. **OpenAI benefits most from semantic chunking** — E-openai (0.625) is the
   overall #1 config, while E-minilm (0.452) and E-mpnet (0.415) show
   smaller gains. Better embeddings amplify better chunking.

5. **All 3 models beat BM25** on their best configs — MiniLM by +10.0 pp,
   MPNet by +8.6 pp, OpenAI by +24.4 pp. Embeddings add value over keywords
   for this corpus.

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **MiniLM (recommended default)** | Free, fast (<50ms), no API dependency, 79% of OpenAI quality, 22M params fit on any machine | Lower recall than OpenAI, especially at R@1 |
| **OpenAI (best quality)** | Best R@5 (0.607), best MRR@5 (0.618), benefits most from semantic chunking | $0.02/1M tokens, API latency (~200ms), requires internet, vendor lock-in |
| **MPNet** | Higher dimension (768) than MiniLM, widely recommended in tutorials | Worst of the 3 on this corpus, 5x MiniLM params for worse results, slower inference |

### Why not larger local models?

Models like `bge-large-en-v1.5` (335M params, 1024 dims) or `e5-large-v2`
exist but weren't tested because:
- Day 2 scope was limited to 3 models (keep grid search tractable: 3×5=15)
- MiniLM vs MPNet already showed that bigger local ≠ better
- Adding models is easy — the infrastructure supports it via `EmbeddingModel` enum

## Consequences

**Easier:**
- MiniLM as default means zero-cost development iteration. No API key needed
  for chunking experiments, test runs, or CI.
- The 3-model comparison validates that our evaluation infrastructure works
  correctly — if all models scored the same, something would be wrong.
- Clear recommendation for Day 4+: use MiniLM for development, switch to
  OpenAI for production evaluation runs.

**Harder:**
- MPNet's poor showing is surprising and may not generalize. On a different
  corpus (e.g., code, legal text), the ranking could change. This ADR's
  conclusions are specific to the Home DIY repair corpus.
- OpenAI's API dependency means production RAG needs a fallback strategy.
  If the API is down, MiniLM can serve as a degraded-mode alternative.
- The 1536-dim OpenAI vectors require 4x the FAISS storage vs 384-dim
  MiniLM. At our scale (<1K vectors) this is irrelevant, but at 1M+ vectors
  the memory difference matters.

## Java/TS Parallel

This is like choosing between **embedded H2** (MiniLM — free, fast, local,
good enough for most queries), **managed PostgreSQL** (OpenAI — better query
planning, costs money, external dependency), and **embedded Derby** (MPNet —
more features than H2 on paper but slower in practice). For development and
tests you use H2; for production you switch to PostgreSQL; and Derby just
isn't worth the overhead. The key insight: benchmark your actual workload,
don't trust parameter counts or marketing — `all-mpnet-base-v2` has great
MTEB scores but underperforms MiniLM on *our* retrieval task.

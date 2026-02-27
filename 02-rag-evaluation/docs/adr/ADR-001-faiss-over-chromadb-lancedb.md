# ADR-001: FAISS over ChromaDB/LanceDB for Benchmarking

**Date**: 2026-02-13
**Status**: Accepted
**Project**: P2 — RAG Evaluation Benchmarking Framework

## Context

P2 is a benchmarking framework evaluating 16 retrieval configurations (5 chunking strategies × 3 embedding models + BM25 baseline). Each config requires its own vector index with direct access to raw similarity scores for Recall@K, Precision@K, and MRR@K computation. The corpus is ~500–1200 chunks per config (<1K vectors), so approximate nearest neighbor, metadata filtering, and server processes are unnecessary overhead.

## Decision

Use **faiss-cpu with IndexFlatIP** (brute-force inner product) as the vector store, wrapped in a thin `FAISSVectorStore` class (~90 lines in `src/vector_store.py`).

1. **IndexFlatIP** computes exact inner product. Since our embedder L2-normalizes all vectors, inner product = cosine similarity. No approximation error.
2. **Brute-force is fast at our scale** — <1ms per query for <1K vectors. Approximate indices (IVF, HNSW) add complexity with no benefit under ~10K vectors.
3. **Two-file persistence** — `.faiss` binary (FAISS native `write_index`) + `.json` sidecar (chunk ID list). The JSON sidecar is human-readable for debugging and maps FAISS integer positions back to chunk IDs.
4. **Validation on `add()`** — dimension mismatch and ID count mismatch raise immediately, not silently at search time. The same `search(query, k)` return signature is shared with `BM25Retriever`.

## Alternatives Considered

| Option | Pros | Cons | Why Not |
|--------|------|------|---------|
| **FAISS IndexFlatIP** ✅ | Raw scores for metric computation, exact results, <1ms queries, no server process, 1 dependency (faiss-cpu) | Manual chunk ID mapping via JSON sidecar, no built-in metadata filtering | — (selected) |
| ChromaDB | Built-in persistence + metadata filtering, auto-generates IDs, popular in RAG tutorials | Abstraction hides raw similarity scores (returns distances not similarities), manages its own SQLite/DuckDB storage, harder to create/load 15 separate collections cleanly | Abstraction hides the scores we need for fair benchmarking |
| LanceDB | Columnar on-disk format, embedded (no server), scales well for large datasets | Overkill for <1K vectors, adds columnar format dependency, abstracts away index internals we need to inspect | Over-engineered for our scale |

## Quantified Validation

- **15 separate indices** built and loaded independently with zero conflicts
- **<1ms per query** for <1K vectors — brute-force is faster than ANN index setup at this scale
- **~90 LOC** for FAISSVectorStore wrapper — fully auditable, no hidden behavior
- **Two-file persistence** (`.faiss` + `.json` sidecar): human-inspectable chunk ID mapping enabled debugging cross-config gold chunk resolution
- **Zero approximation error**: IndexFlatIP + L2-normalized vectors = exact cosine similarity

## Consequences

**Easier:** Raw similarity scores flow directly into metric computation. Each config's index is a pair of files that can be inspected, moved, or deleted independently. FAISSVectorStore is small enough to understand completely — no hidden behavior from a managed store. Same FAISS library that production systems (Meta, Spotify) use at scale.

**Harder:** Chunk ID mapping is manual — we maintain a parallel list and a JSON sidecar file. No metadata filtering (not needed for benchmarking). If we needed 1M+ vectors, we'd need to switch to IndexIVFFlat or HNSW and add training steps.

**Portability:** P3 reused FAISS IndexFlatIP for embedding similarity evaluation. P4 switched to ChromaDB for production (metadata filtering, live API) — confirming FAISS was right specifically for benchmarking but insufficient for serving.

## Cross-References

- **ADR-002**: Chunk configs define what goes INTO each FAISS index — 5 configs × 3 models = 15 indices
- **ADR-003**: Embedding model determines vector dimensions (384/768/1536) stored in FAISS
- **ADR-004**: QA evaluation runs `search()` against all 15 FAISS indices; Strategy 2 uses `reconstruct_n()` for chunk similarity
- **ADR-005**: Post-reranking results (0.747 R@5) still flow through FAISS retrieval as the first stage
- **P4 ADR-005**: P4 switched to ChromaDB for live API needs — validates that FAISS was correct for benchmarking but insufficient for serving

## Java/TS Parallel

FAISS is like using **raw JDBC** instead of **Hibernate/JPA**. You write the SQL (index operations) yourself, you see every result set (raw scores), and you manage the connection lifecycle (create/save/load). A managed vector store like ChromaDB is the Hibernate equivalent — convenient for CRUD applications, but when you're benchmarking query performance, you want the raw driver so nothing is hidden. The `FAISSVectorStore` class is our thin DAO layer — it adds ID mapping and validation but doesn't abstract away FAISS's behavior.

**The key insight:** Choose the lowest-abstraction tool that satisfies your requirements. Benchmarking needs raw scores and full control; production needs managed lifecycle and metadata filtering. The right tool depends on the job, not the popularity.

## Interview Signal

Demonstrates **context-dependent tool selection**. The engineer chose FAISS for benchmarking (raw scores, full control) and later switched to ChromaDB for production serving (P4), proving the decision wasn't dogmatic but fit-for-purpose. This signals understanding that infrastructure choices should be driven by use case constraints, not convention or tutorial defaults.

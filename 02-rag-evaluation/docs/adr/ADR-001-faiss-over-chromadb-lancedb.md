# ADR-001: FAISS over ChromaDB/LanceDB for Benchmarking

**Date**: 2026-02-13
**Status**: Accepted
**Project**: P2 — RAG Evaluation Benchmarking Framework

## Context

P2 is a benchmarking framework that evaluates 16 retrieval configurations:
5 chunking strategies × 3 embedding models = 15 vector search configs + 1 BM25
lexical baseline. For each config, we build a separate vector index, run 50+
queries, and compare raw similarity scores across configs to determine which
RAG setup produces the best retrieval quality.

This means we need a vector store that gives us:

1. **Direct access to similarity scores** — we compute Recall@K, Precision@K,
   and MRR@K from raw scores. An abstraction layer that hides or transforms
   scores would complicate metric computation.
2. **Full control over index lifecycle** — we create 15 separate indices, save
   each to disk, and load them independently during evaluation. We need to
   control exactly when vectors are added and how indices are persisted.
3. **Transparent persistence** — debugging requires inspecting what's stored.
   We need to verify that chunk ID mappings are correct across configs.
4. **Minimal overhead** — our corpus is ~500–1200 chunks per config (<1K
   vectors). We don't need approximate nearest neighbor, metadata filtering,
   or a server process.

## Decision

Use **`faiss-cpu`** with **`IndexFlatIP`** (brute-force inner product) as the
vector store, wrapped in a thin `FAISSVectorStore` class (~90 lines in
`src/vector_store.py`).

Key implementation details:

- **IndexFlatIP** computes exact inner product. Since our embedder L2-normalizes
  all vectors (`_l2_normalize()` in `embedder.py`), inner product = cosine
  similarity. No approximation error.
- **Brute-force is fast at our scale** — <1ms per query for <1K vectors.
  Approximate indices (IVF, HNSW) add complexity with no benefit under ~10K
  vectors.
- **Two-file persistence** — `.faiss` binary (FAISS native `write_index`) +
  `.json` sidecar (chunk ID list). The JSON sidecar is human-readable for
  debugging and maps FAISS integer positions back to chunk IDs like `"B_42"`.
- **Validation on add()** — dimension mismatch and ID count mismatch raise
  immediately, not silently at search time.

The same `search(query, k) -> list[tuple[str, float]]` return signature is
shared with `BM25Retriever`, so the retrieval evaluator treats both backends
identically.

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **FAISS IndexFlatIP** (chosen) | Raw scores for metric computation, exact results, <1ms queries, no server process, 1 dependency (`faiss-cpu`) | Manual chunk ID mapping via JSON sidecar, no built-in metadata filtering |
| **ChromaDB** | Built-in persistence + metadata filtering, auto-generates IDs, popular in RAG tutorials | Abstraction hides raw similarity scores (returns `distances` not `scores`), manages its own SQLite + DuckDB storage, harder to create/load 15 separate collections cleanly |
| **LanceDB** | Columnar on-disk format, embedded (no server), good for large datasets | Overkill for <1K vectors, adds Lance columnar format dependency, abstracts away index internals we need to inspect |

## Consequences

**Easier:**
- Raw similarity scores flow directly into metric computation — no score
  transformation or reverse-engineering an abstraction layer.
- Each config's index is a pair of files (`{config}.faiss` + `{config}.json`)
  that can be inspected, moved, or deleted independently.
- `FAISSVectorStore` is small enough to understand completely — no hidden
  behavior from a managed store.
- Same FAISS library that production systems (Meta, Spotify) use at scale —
  the skills transfer directly.

**Harder:**
- Chunk ID mapping is manual — we maintain a parallel `list[str]` and a JSON
  sidecar file. ChromaDB handles this automatically with its built-in document
  store.
- No metadata filtering (e.g., "search only chunks from section X"). Not needed
  for benchmarking, but would require post-filtering if added later.
- If we needed 1M+ vectors, we'd need to switch to `IndexIVFFlat` or HNSW and
  add training steps. Not a concern at our scale.

## Java/TS Parallel

FAISS is like using **raw JDBC** instead of **Hibernate/JPA**. You write the
SQL (index operations) yourself, you see every result set (raw scores), and you
manage the connection lifecycle (create/save/load). A managed vector store like
ChromaDB is the Hibernate equivalent — convenient for CRUD applications, but
when you're benchmarking query performance, you want the raw driver so nothing
is hidden. The `FAISSVectorStore` class is our thin DAO layer — it adds ID
mapping and validation but doesn't abstract away FAISS's behavior.

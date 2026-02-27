# ADR-005: ChromaDB over FAISS for P4 Resume Vector Search

**Date**: 2026-02-26
**Status**: Accepted
**Project**: P4 — Resume Coach

## Context

P4 requires semantic search over 250 resumes: given a free-text job description or skill query, return the top-k most similar candidates by embedding cosine similarity. This powers the `/search/similar-candidates` API endpoint and the "Search Similar" Streamlit page.

In P2 (RAG Evaluation), we used FAISS directly via LangChain for document retrieval. P4 has different requirements:
- **Persistence across restarts**: The API process restarts frequently during development; rebuilding 250 embeddings (~2s on M2 each run) is unacceptable
- **Metadata filtering**: Filter candidates by `fit_level` ("excellent", "good", "partial") without post-processing all results in Python
- **REST API integration**: The vector store must be queryable from a long-lived FastAPI process, not just a batch script
- **Inspection**: Ability to count documents, inspect metadata, rebuild the index without full data reload

## Decision

Use **ChromaDB** (`chromadb` library with `PersistentClient`) for P4 resume vector indexing and search.

Key design choices:
1. **PersistentClient at `data/chromadb/`**: Index survives process restarts. `build_resume_index()` (run once after pipeline) → `get_collection()` (every API/Streamlit startup) pattern.
2. **Module-level `_ef` singleton** (`_SentenceTransformerEF`): Loaded once at `get_collection()` time, reused for all queries. Avoids the ~2s `SentenceTransformer` load on every search request.
3. **`query_embeddings` instead of `query_texts`**: ChromaDB 1.5 raises `ValueError` if a custom `EmbeddingFunction` is passed to `get_collection()` when the collection was built without one. We manage the EF ourselves and pass pre-computed embeddings directly.
4. **`where={"fit_level": fit_level}` metadata filter**: ChromaDB's native filtering, no post-processing.
5. **Cosine space HNSW**: `hnsw:space = cosine` matches `all-MiniLM-L6-v2`'s training objective (cosine similarity).

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **ChromaDB** (chosen) | Persistent; metadata filtering; HTTP client available; simple Python API | Adds a dependency; slightly slower than FAISS on raw ANN throughput (irrelevant at 250 docs) |
| FAISS (used in P2) | Fastest ANN search; in-memory; no dependencies beyond numpy | No persistence — must rebuild on every restart; no native metadata filtering; not a database |
| Pinecone | Fully managed; production-grade | Paid service; overkill for 250 docs; no offline use |
| Qdrant | Better production story than ChromaDB | Requires Docker or separate process; adds operational complexity |
| pgvector | SQL + vector in one database | Requires PostgreSQL; far too heavy for a file-based project |

## Consequences

**Easier**:
- `build_resume_index()` runs once after the Day 1/2 pipeline; API and Streamlit start instantly with `get_collection()`
- `fit_level` filtering is one `where` parameter — no Python-side post-filtering needed
- `collection.count()` gives total index size for the API health check and search response
- Different tools for different jobs: FAISS (P2 batch retrieval) vs ChromaDB (P4 live API search)

**Harder**:
- ChromaDB 1.5 has a known `EmbeddingFunction` conflict when calling `get_collection()` on a collection built without an EF — worked around by managing `_ef` as a module-level singleton and passing `query_embeddings` directly
- Collection must be deleted + recreated on re-index to avoid duplicate IDs (handled in `build_resume_index()`)
- ChromaDB's `PersistentClient` uses SQLite under the hood — not suitable for multi-process writes (fine for P4's single-writer, multi-reader pattern)

## Java/TS Parallel

ChromaDB occupies the same architectural position as an **embedded H2 database** in a Spring Boot app: it's a lightweight, file-backed store that starts with the process, survives restarts, and is queried via a simple API — no separate server process needed. FAISS is more like an in-memory `HashMap<String, float[]>`: blazing fast, but ephemeral. Choosing ChromaDB over FAISS for P4 is the same decision as choosing H2 with persistence over a `ConcurrentHashMap` when you need durability without full PostgreSQL overhead.

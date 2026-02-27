# ADR-005: ChromaDB over FAISS for P4 Resume Vector Search

**Date**: 2026-02-26
**Project**: P4 — Resume Coach

## Status

Accepted

## Context

P4 requires semantic search over 250 resumes: given a free-text job description or skill query, return the top-k most similar candidates ranked by embedding cosine similarity. This powers the `/search/similar-candidates` API endpoint and the "Search Similar" page in the Streamlit demo.

Three requirements made the choice non-trivial:

1. **Persistence across restarts**: The FastAPI process restarts frequently during development (`uvicorn --reload`). Encoding 250 resumes with `all-MiniLM-L6-v2` takes ~2 seconds on the M2 MacBook Air. Rebuilding the index on every restart means a 2-second delay on the first request, which feels broken in a demo. The index must survive process restarts.

2. **Metadata filtering**: The API supports filtering search results by `fit_level` ("excellent", "good", "partial", "poor", "mismatch"). Without native filtering, every query would return all 250 results and the API would post-filter in Python — wasting computation and complicating pagination.

3. **Live API integration**: The vector store must be queryable from a long-lived FastAPI process, not just from a batch script that builds an index and exits. It needs `count()` for the health endpoint, `query()` for search, and `get_or_create_collection()` for startup — a database API, not a numpy array.

In P2 (RAG Evaluation), we used FAISS via LangChain for batch document retrieval. FAISS is an in-memory index — blazing fast, but ephemeral. It met P2's needs (build index, run evaluation, exit). P4's live-serving requirements are fundamentally different.

## Decision

Use **ChromaDB** (`PersistentClient` at `data/chromadb/`) with `all-MiniLM-L6-v2` embeddings (384 dimensions, cosine space).

Key design choices:

1. **Two-phase lifecycle**: `build_resume_index()` runs once after the pipeline completes — loads the SentenceTransformer model, encodes all 250 resumes in batch, stores embeddings + metadata in ChromaDB, then `del model; gc.collect()` to free ~90MB on the 8GB M2. `get_collection()` runs at every API/Streamlit startup — opens the persisted index with zero model loading, instant startup.

2. **Module-level `_ef` singleton**: A `_SentenceTransformerEF` class wrapping `SentenceTransformer.encode()` is loaded once at first query time and stays in memory. This avoids the ~2s model reload on every search request while keeping startup instant (model loads lazily on first search, not at import time).

3. **`query_embeddings` instead of `query_texts`**: ChromaDB 1.5 has a known conflict: passing a custom `EmbeddingFunction` to `get_collection()` on a collection that was built without one raises `ValueError`. The workaround: manage `_ef` as our own singleton, call `_ef([query_text])` to get the embedding, and pass it to `collection.query(query_embeddings=...)` directly. This bypasses ChromaDB's internal EF dispatch entirely.

4. **`where={"fit_level": fit_level}` metadata filter**: Each resume is stored with 4 metadata fields: `fit_level`, `writing_style`, `name`, `skills` (comma-separated). ChromaDB's native `where` clause filters at the index level — no post-processing needed. The API endpoint passes the filter straight through.

5. **Cosine space HNSW**: `metadata={"hnsw:space": "cosine"}` on the collection matches `all-MiniLM-L6-v2`'s training objective. Similarity scores are `1 - distance`, ranging from 0 (unrelated) to 1 (identical).

## Alternatives Considered

**FAISS (used in P2)**: The fastest approximate nearest neighbor library, purpose-built for billion-scale vector search. At 250 documents, FAISS's raw ANN throughput advantage is unmeasurable — both FAISS and ChromaDB return results in <50ms. FAISS's fatal flaw for P4 is that it's an in-memory index with no persistence. `faiss.write_index()` / `faiss.read_index()` exist but require manual serialization of metadata alongside the index file, manual ID-to-document mapping, and manual rebuild logic. ChromaDB handles all of this as a database — `PersistentClient` persists both vectors and metadata to SQLite automatically. For P2's batch pattern (build → evaluate → exit), FAISS's simplicity was an advantage. For P4's live API pattern (start → serve queries → restart → serve again), the lack of persistence is a dealbreaker.

**Pinecone**: A fully managed vector database with production-grade features: automatic scaling, backups, monitoring. But P4 has 250 documents — Pinecone's free tier would work, but it adds a network dependency (every query is an HTTP round-trip to a remote server) and requires an API key. The Streamlit demo and API must work offline for local recruiter demos. Pinecone solves problems that don't exist at P4's scale.

**Qdrant**: Better production story than ChromaDB — supports distributed deployment, has a Rust engine, and offers a more mature filtering DSL. But Qdrant requires either Docker (`qdrant/qdrant` container) or a separate server process. P4 runs as a single Python process (FastAPI or Streamlit). Adding a Docker dependency to a portfolio project means the recruiter needs Docker installed to run the demo. ChromaDB's `PersistentClient` embeds inside the Python process — no separate infrastructure.

**pgvector**: SQL + vector search in one database, using PostgreSQL's `VECTOR` type. This is the production choice for applications that already have PostgreSQL. P4 has no database — all data lives in JSONL files on disk. Installing PostgreSQL, running migrations, and loading JSONL into tables to serve 250 vectors is engineering malpractice. pgvector solves the "I already have Postgres, how do I add vectors?" problem. P4 has the opposite problem: "I have vectors, I need the simplest possible store."

## Consequences

### What This Enabled

API and Streamlit start instantly via `get_collection()` — no index rebuild, no model load, no 2-second delay on the first request. The startup penalty from P2's FAISS pattern (rebuild index every process restart) is completely eliminated because ChromaDB's `PersistentClient` persists both vectors and metadata to SQLite on disk. `fit_level` filtering is a single `where` parameter on the query call: the `/search/similar-candidates?fit_level=excellent` endpoint passes it straight through to ChromaDB's HNSW index with no Python-side post-filtering, narrowing results to the ~50 excellent-fit resumes without touching the other 200. `collection.count()` powers the `/health` endpoint's `vector_store_ready` field and the search response's `total_in_index` count — direct database introspection that FAISS would require manual tracking to replicate. The portfolio now demonstrates different vector store tools chosen for different deployment contexts: FAISS for P2's batch retrieval benchmarks where raw ANN speed and index control matter, ChromaDB for P4's live API search where persistence and metadata filtering matter.

### Accepted Trade-offs

- ChromaDB 1.5's `EmbeddingFunction` conflict required the `query_embeddings` workaround — 5 extra lines of code in `search_similar()` to manage the EF singleton manually instead of letting ChromaDB handle it. A minor but real API wart
- Collection must be deleted + recreated on re-index (`build_resume_index()` calls `client.delete_collection()` then `client.create_collection()`) to avoid duplicate IDs. No upsert-on-conflict for the full-rebuild case
- ChromaDB's `PersistentClient` uses SQLite under the hood — not suitable for multi-process concurrent writes. Fine for P4's single-writer (pipeline builds index) / multi-reader (API + Streamlit query it) pattern, but would need Qdrant or Pinecone for true multi-writer production use
- ChromaDB adds a transitive dependency on `onnxruntime` (~50MB) even though P4 uses its own SentenceTransformer for encoding. The default embedding model is never used, but the dependency is pulled in

## Cross-Project Context

P2 (RAG Evaluation) chose FAISS for its retrieval benchmarks because it needed raw ANN speed and direct index control for evaluating retrieval quality — persistence didn't matter since the benchmark script ran once end-to-end. When P4's requirements shifted to live API serving with persistence and metadata filtering, reusing FAISS would have meant building a persistence layer, a metadata store, and a query API on top of it — essentially reimplementing ChromaDB poorly. Using both across projects demonstrates tool selection by deployment requirement, not habit.

## Java/TS Parallel

ChromaDB occupies the same architectural position as an embedded H2 database in a Spring Boot application: a lightweight, file-backed store that starts with the JVM process, survives restarts, and is queried via a simple API — no separate server, no Docker, no infrastructure. FAISS is more like an in-memory `ConcurrentHashMap<String, float[]>`: blazing fast for lookups, but everything evaporates when the process exits. The progression from FAISS (P2) to ChromaDB (P4) maps to the Java decision of moving from a `ConcurrentHashMap` cache to H2 with persistence when your service needs to survive restarts.

## Validation

The vector store indexes 250 resumes with 4 metadata fields each. `search_similar("Python machine learning data science", top_k=5)` returns relevant candidates ranked by cosine similarity in <50ms. Metadata filtering works: `where={"fit_level": "excellent"}` narrows results to the ~50 excellent-fit resumes without touching the other 200. The `/health` endpoint reports `vector_store_ready: true` and the correct document count. The Streamlit "Search Similar" page queries the same index interactively. The 2-second startup penalty from P2's FAISS pattern is eliminated.

## Reversibility

**Medium** — Switching back to FAISS would require: (1) building a persistence layer (save/load index + metadata to disk), (2) implementing metadata filtering in Python post-query, (3) replacing `collection.count()` with manual tracking, (4) updating the API health check and search endpoint. The embedding model stays the same; the storage and query layer changes. Estimated effort: ~2 hours to implement, plus re-testing 26 API tests and the Streamlit search page.

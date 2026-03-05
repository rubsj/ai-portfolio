# ADR-001: FAISS over ChromaDB for Vector Index

**Project:** P5: ShopTalk Knowledge Agent
**Category:** Tool Choice
**Status:** Accepted
**Date:** 2026-03-03

---

## Context

P5 is an experiment platform comparing 35+ RAG configurations: 5 chunking strategies × 3 embedding models × 3 retriever types × hyperparameter sweeps. Each config needs its own isolated index - different embedding dimensions (MiniLM=384, MPNet=768, OpenAI=1536), different chunk sets, and persistence that makes results reproducible.

We used ChromaDB in P4 (Resume Coach) for its REST API. That burned us in two ways:
1. ChromaDB has a module-level `EmbeddingFunction` singleton. When you try to run multiple collection configs in the same process, they stomp on each other.
2. ChromaDB hides embedding injection behind `query_texts`/`query_embeddings`, and the normalization behavior is subtle - it calls `normalize_embeddings()` internally and hands back `list[np.ndarray]` instead of `list[list[float]]`. Debugging that cost real time.

What we actually need here is pretty simple: create an index with a specific dimension, add vectors, swap the index out, compare retrieval quality, and persist each config separately. The tool shouldn't fight us on any of that.

---

## Decision

**FAISS** (`faiss-cpu`) with `IndexFlatIP` (inner product on L2-normalised vectors = cosine similarity).

Each experiment config gets its own FAISS index:
```python
import faiss
import numpy as np

# Create index for embedding dimension 384 (MiniLM)
index = faiss.IndexFlatIP(384)

# Add normalised vectors
embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
index.add(embeddings_norm.astype(np.float32))

# Search: returns (distances, indices) - distances = cosine similarity
scores, indices = index.search(query_norm.reshape(1, -1).astype(np.float32), k=5)

# Persist: two files per index
faiss.write_index(index, "index.faiss")
# Companion JSON stores chunk metadata keyed by FAISS integer ID
```

Each index saves to `{experiment_id}.faiss` + `{experiment_id}_chunks.json`. To load it back, it's just `faiss.read_index(path)`. No server, no daemon, no port conflicts.

---

## Alternatives Considered

**ChromaDB** - Metadata support built-in, Python-native, solid for production APIs. But: the module-level EF singleton breaks multi-config experiments. We hit this in P4. Also, it abstracts away too much of the embedding pipeline for what we're doing here. When you're running 35 configs and need to know exactly what's happening at each step, that abstraction works against you.

**Qdrant** - Rich filtering, production-grade. But it needs a running server process, and we're not building a production service. That's ops overhead we don't want for an experiment platform.

**Raw numpy + cosine_similarity** - Zero dependencies, which is nice. But it's O(n) scan with no indexing. 35 configs × 4 papers × ~500 chunks each - that's too slow once you're actually iterating.

---

## Quantified Validation

- The P4 ChromaDB bug (`normalize_embeddings()` return type changed silently) took about 2 hours to track down.
- FAISS `IndexFlatIP` search on 2000 vectors (4 papers × ~500 chunks): under 1ms per query. Latency differences across configs are measurement noise, not index overhead.
- Persistence cost: writing `.faiss` binary + JSON metadata for 2000 vectors takes about 12ms total.

---

## Consequences

We get full isolation per experiment - separate files, explicit dimension at construction, and dimension mismatches blow up immediately instead of failing silently. No server to babysit.

The tradeoff: we have to handle ID→chunk mapping ourselves. FAISS gives you integer IDs; mapping those back to `Chunk` objects is on us. And if we want metadata filtering (like "only chunks from page 3"), that's post-retrieval filtering - FAISS doesn't do that natively.

For future projects: P6–P9 can import the `FAISSVectorStore` class directly. The `BaseVectorStore` ABC means if we ever move to Qdrant for a production use case, we only swap the implementation class.

---

## Takeaway

ChromaDB and FAISS solve different problems. ChromaDB is built for production APIs - metadata, filtering, nice ergonomics. FAISS gives you a bare index you control completely. We're not building a production service; we're running 35+ experiment configs and need to iterate fast. The P4 singleton bug made this concrete - using a production-oriented tool for experiment work meant fighting abstractions that weren't built for our use case.
